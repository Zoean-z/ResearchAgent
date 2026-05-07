from __future__ import annotations

import re


_INSUFFICIENT_EVIDENCE_TEXT = "无法基于当前论文内容稳定生成该字段。"


class IngestSummaryPolicy:
    """Shared summary and fallback text policy for ingest analysis."""

    def summary_text_or_unavailable(self, text: str | None, *, paper_title: str | None = None) -> str:
        candidate = (text or "").strip()
        if candidate and not self.looks_like_summary_noise(candidate):
            if paper_title is not None:
                title_candidate = paper_title.strip()
                if (
                    candidate == title_candidate
                    or candidate.startswith("Imported local PDF")
                    or candidate.startswith("Imported arXiv PDF")
                ):
                    return _INSUFFICIENT_EVIDENCE_TEXT
            return candidate[:240]
        return _INSUFFICIENT_EVIDENCE_TEXT

    def summary_items_or_unavailable(self, items: tuple[str, ...], *, max_items: int = 2) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in items:
            normalized = self.normalize_summary_item(item)
            if normalized is None or normalized in cleaned:
                continue
            if self.looks_like_generic_new_idea(normalized) or self.looks_like_generic_suggestion(normalized):
                continue
            cleaned.append(normalized)
            if len(cleaned) >= max_items:
                break
        return tuple(cleaned) if cleaned else (_INSUFFICIENT_EVIDENCE_TEXT,)

    def sanitize_summary_text(self, text: str | None, *fallback_candidates: str | None) -> str:
        candidate = (text or "").strip()
        if not candidate:
            return self.first_clean_text(*fallback_candidates)
        if self.looks_like_summary_noise(candidate):
            return self.first_clean_text(*fallback_candidates)
        return candidate[:240]

    def first_clean_text(self, *candidates: str | None) -> str:
        for candidate in candidates:
            cleaned = (candidate or "").strip()
            if cleaned and not self.looks_like_summary_noise(cleaned):
                return cleaned[:240]
        for candidate in candidates:
            cleaned = (candidate or "").strip()
            if cleaned:
                return cleaned[:240]
        return ""

    def sanitize_summary_items(
        self,
        items: tuple[str, ...],
        fallback_items: tuple[str, ...],
        *,
        max_items: int = 2,
    ) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in items:
            normalized = self.normalize_summary_item(item)
            if normalized is None or normalized in cleaned:
                continue
            cleaned.append(normalized)
            if len(cleaned) >= max_items:
                break
        if cleaned:
            return tuple(cleaned)
        fallback_cleaned: list[str] = []
        for item in fallback_items:
            normalized = self.normalize_summary_item(item)
            if normalized is None or normalized in fallback_cleaned:
                continue
            fallback_cleaned.append(normalized)
            if len(fallback_cleaned) >= max_items:
                break
        return tuple(fallback_cleaned)

    def normalize_summary_item(self, item: str) -> str | None:
        candidate = item.strip()
        if not candidate or self.looks_like_summary_noise(candidate):
            return None
        return candidate[:240]

    def looks_like_summary_noise(self, text: str) -> bool:
        lowered = text.lower()
        noise_markers = (
            "proceedings of",
            "et al.",
            "doi:",
            "http://",
            "https://",
            "page ",
            "vol.",
            "conference",
            "workshop",
            "in proceedings",
            "tacas",
            "etap",
            "bibliography",
            "references",
        )
        if any(marker in lowered for marker in noise_markers):
            return True
        if "..." in text or re.search(r"\.{4,}", text):
            return True
        if re.match(r"^\s*\d+\s+\d+\.\d+\s+", text):
            return True
        if len(text.split()) > 28 and any(char.isdigit() for char in text):
            return True
        return False

    def is_placeholder_source_title(self, title: str) -> bool:
        lowered = title.strip().lower()
        return lowered.startswith("imported local pdf") or lowered.startswith("imported arxiv pdf")

    def summary_seed_text(self, *values: str | None) -> str:
        return " ".join(value.strip() for value in values if value and value.strip())

    def contains_cjk(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def looks_like_generic_new_idea(self, text: str) -> bool:
        normalized = text.strip()
        generic_markers = (
            "本文采用",
            "本文通过",
            "本文提出",
            "我们提出",
            "研究显示",
            "实验结果表明",
            "方式展开分析",
        )
        return any(marker in normalized for marker in generic_markers)

    def looks_like_generic_suggestion(self, text: str) -> bool:
        normalized = text.strip()
        generic_markers = (
            "继续回读原文",
            "补充更多证据",
            "开展更大规模实验",
            "继续追问",
            "重新阅读原文",
            "Run larger-scale experiments",
            "insufficient evidence",
        )
        return any(marker in normalized for marker in generic_markers)

    def fallback_topic_text(self, seed_text: str, title: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "long context", "position-aware", "positional", "position sensitive")):
            return "本文主要讨论长上下文场景中的位置敏感性，以及它对检索和推理评估的影响。"
        if "retrieval" in lowered and ("reasoning" in lowered or "evaluation" in lowered):
            return "本文主要讨论检索与推理评估中的差异及其位置偏差。"
        if "benchmark" in lowered and "evaluation" in lowered:
            return "本文主要讨论相关基准上的评估方法及其局限。"
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return "本文主要讨论模型在分布偏移下的鲁棒性表现。"
        if self.is_placeholder_source_title(title):
            return "本文主要围绕论文中的核心研究问题展开。"
        return f"本文主要围绕《{title}》的核心研究问题展开。"

    def fallback_problem_text(self, seed_text: str, title: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "long context", "position-aware", "positional", "position sensitive")):
            return "本文试图解决长上下文评估中位置变化带来的偏差问题，并分析检索和推理场景中的脆弱性。"
        if "retrieval" in lowered and ("reasoning" in lowered or "evaluation" in lowered):
            return "本文试图解决检索与推理评估中的差异和位置偏差问题。"
        if "benchmark" in lowered and "evaluation" in lowered:
            return "本文试图补足相关基准评估中的系统性分析。"
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return "本文试图提升模型在分布偏移下的稳定性。"
        if self.is_placeholder_source_title(title):
            return "本文试图解决论文提出的核心问题，并验证相关方法的效果。"
        return f"本文试图解决《{title}》提出的核心问题，并验证相关方法的效果。"

    def fallback_method_text(self, seed_text: str) -> str:
        lowered = seed_text.lower()
        if "retrieval" in lowered and "evaluation" in lowered:
            return "本文采用检索与评估结合的方式展开分析。"
        if "benchmark" in lowered:
            return "本文采用基准评估与对比实验来验证结论。"
        return "本文采用文中的方法设计和实验流程来验证上述问题。"

    def fallback_novelty_text(self, seed_text: str) -> str:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "position-aware", "positional")):
            return "本文提出了位置敏感性的分析视角。"
        if "retrieval" in lowered and "reasoning" in lowered:
            return "本文提出了检索与推理评估的对照分析。"
        return "本文提出了新的方法框架或评估视角。"

    def fallback_idea_texts(self, seed_text: str) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("long-context", "position-aware", "positional")):
            return ("本文提出了位置敏感性的分析视角。", "本文比较了位置变化对不同评估设置的影响。")
        if "retrieval" in lowered and "reasoning" in lowered:
            return ("本文提出了检索与推理评估的对照分析。", "本文强调了两类场景中的位置偏差差异。")
        if "benchmark" in lowered and "evaluation" in lowered:
            return ("本文补充了相关基准上的系统评估。", "本文揭示了现有评估方法的局限。")
        return ("本文提出了新的方法框架或评估视角。", "本文通过实验验证了核心结论。")

    def fallback_key_result_texts(self, seed_text: str) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if "long-context" in lowered or "position-aware" in lowered or "positional" in lowered:
            return ("研究显示，位置变化会显著影响长上下文评估结果。",)
        if "retrieval" in lowered and "reasoning" in lowered:
            return ("研究显示，检索与推理场景中的位置偏差表现不同。",)
        if "benchmark" in lowered and "evaluation" in lowered:
            return ("研究结果强调了基准评估中需要额外关注位置偏差。",)
        return ("实验结果表明该方法在相关设置下取得了改进。",)

    def fallback_limitation_texts(self, seed_text: str, why_open: tuple[str, ...] = ()) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("当前结论仍需要在更大规模实验中继续验证。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("当前结论仍需要在分布偏移场景下进一步验证。",)
        if why_open and any(self.contains_cjk(item) for item in why_open):
            return tuple(why_open[:2])
        return ("当前结论仍依赖现有实验设置，后续还需要更多证据。",)

    def fallback_why_open_texts(self, seed_text: str, chunk_count: int) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("当前结果仍需要在更大规模设置中继续验证。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("当前结果在分布偏移下的稳定性仍需继续检验。",)
        return (f"已解析 {chunk_count} 个文本分块，但尚未抽取到足够清晰的局限性。",)

    def fallback_suggestion_texts(self, seed_text: str, title: str, possible_followup: tuple[str, ...] = ()) -> tuple[str, ...]:
        lowered = seed_text.lower()
        if possible_followup and any(self.contains_cjk(item) for item in possible_followup):
            return tuple(possible_followup[:2])
        if any(keyword in lowered for keyword in ("scal", "large")):
            return ("后续可以开展更大规模实验。",)
        if any(keyword in lowered for keyword in ("robust", "distribution shift", "shift")):
            return ("后续可以继续评估方法在分布偏移下的表现。",)
        if self.is_placeholder_source_title(title):
            return ("后续可以继续回读原文，并补充更多证据。",)
        return (f"后续可以围绕《{title}》中仍未验证的部分继续追问。",)
