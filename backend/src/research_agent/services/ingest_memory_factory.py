from __future__ import annotations

import re

from research_agent.domain.enums import RelationType
from research_agent.domain.models import Chunk, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SourceRef
from research_agent.domain.value_objects import ConfidenceScore

from research_agent.services.ingest_summary_policy import IngestSummaryPolicy


_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class IngestMemoryFactory:
    """Build memory domain objects from cleaned ingest context."""

    def __init__(self, summary_policy: IngestSummaryPolicy) -> None:
        self._summary_policy = summary_policy

    def build_paper_memory(self, paper: Paper, artifact_id: str, chunks: list[Chunk], context_text: str) -> PaperMemory:
        sentences = self._sentences(context_text)
        source_refs = self._build_source_refs(paper.id, artifact_id, chunks, context_text)
        if self._summary_policy.is_placeholder_source_title(paper.title):
            seed_text = self._summary_policy.summary_seed_text(context_text, *sentences[:3])
            return PaperMemory(
                paper_id=paper.id,
                problem=self._summary_policy.fallback_topic_text(seed_text, paper.title),
                method=self._summary_policy.fallback_method_text(seed_text),
                key_results=self._summary_policy.fallback_key_result_texts(seed_text),
                limitations=self._summary_policy.fallback_limitation_texts(seed_text),
                novelty_claim=self._summary_policy.fallback_novelty_text(seed_text),
                source_refs=source_refs,
                confidence=self._paper_confidence(chunks, [], []),
            )
        problem = self._first_sentence(sentences, self._problem_keywords()) or paper.title
        method = self._first_sentence(sentences, self._method_keywords())
        key_results = self._collect_sentences(sentences, self._result_keywords(), limit=3)
        limitations = self._collect_sentences(sentences, self._limitation_keywords(), limit=3)
        novelty_claim = self._first_sentence(sentences, self._novelty_keywords())
        confidence = self._paper_confidence(chunks, key_results, limitations)
        if not key_results and chunks:
            key_results = [chunks[0].text[:240]]
        return PaperMemory(
            paper_id=paper.id,
            problem=problem,
            method=method,
            key_results=key_results,
            limitations=limitations,
            novelty_claim=novelty_claim,
            source_refs=source_refs,
            confidence=confidence,
        )

    def build_relation_memory(self, paper: Paper, related_paper: Paper, context_text: str) -> RelationMemory:
        relation_type = self._infer_relation_type(context_text)
        evidence = self._collect_sentences(self._sentences(context_text), self._relation_keywords(), limit=2)
        if not evidence:
            evidence = [f"导入时已将来源内容与《{related_paper.title}》进行关系比较。"]
        summary = self._build_relation_summary(paper.title, related_paper.title, relation_type)
        return RelationMemory(
            source_paper=paper.id,
            target_paper=related_paper.id,
            relation_type=relation_type,
            summary=summary,
            evidence=evidence,
            confidence=ConfidenceScore(value=0.6 if evidence else 0.5),
        )

    def build_open_question_memory(
        self,
        paper: Paper,
        chunks: list[Chunk],
        context_text: str,
        related_paper: Paper | None,
    ) -> OpenQuestionMemory:
        sentences = self._sentences(context_text)
        limitation = self._first_sentence(sentences, self._limitation_keywords())
        if self._summary_policy.is_placeholder_source_title(paper.title):
            unresolved_question = self._limitation_to_question(limitation or context_text, prefer_chinese=True)
        else:
            unresolved_question = self._limitation_to_question(limitation) if limitation else f"《{paper.title}》中还有哪些结论尚未充分验证？"
        why_open = self._collect_sentences(sentences, self._limitation_keywords(), limit=3)
        if not why_open:
            why_open = [f"已解析 {len(chunks)} 个文本块，但尚未提取到明确的局限性。"]
        possible_followup = self._build_followups(why_open, prefer_chinese=self._summary_policy.is_placeholder_source_title(paper.title))
        related_papers = [paper.id]
        if related_paper is not None:
            related_papers.append(related_paper.id)
        confidence = ConfidenceScore(value=0.5 if why_open else 0.35)
        return OpenQuestionMemory(
            unresolved_question=unresolved_question,
            related_papers=related_papers,
            why_open=why_open,
            possible_followup=possible_followup,
            confidence=confidence,
        )

    def build_source_refs(self, paper_id: str, artifact_id: str, chunks: list[Chunk], context_text: str) -> list[SourceRef]:
        if not chunks:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    section="title",
                    quote=context_text[:240],
                )
            ]
        refs: list[SourceRef] = []
        for chunk in chunks[:2]:
            refs.append(
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.id,
                    quote=chunk.text[:240],
                )
            )
        return refs

    def _build_followups(self, why_open: list[str], *, prefer_chinese: bool = False) -> list[str]:
        text = " ".join(why_open).lower()
        followups: list[str] = []
        if "robust" in text or "shift" in text:
            followups.append("评估方法在分布偏移下的鲁棒性。" if prefer_chinese else "Evaluate robustness under distribution shift.")
        if "scal" in text or "large" in text:
            followups.append("开展更大规模的实验。" if prefer_chinese else "Run larger-scale experiments.")
        if "ablation" in text:
            followups.append("补充主要组件的消融实验。" if prefer_chinese else "Add ablation studies for the main components.")
        if "future work" in text or "not yet" in text:
            followups.append("后续结果发布后，再回到这篇论文核对结论。" if prefer_chinese else "Revisit this paper once follow-up results are published.")
        if not followups:
            followups.append("获得更多证据后，重新阅读原文并更新记忆。" if prefer_chinese else "Revisit the source after gathering more evidence.")
        return followups

    def _limitation_to_question(self, limitation: str, *, prefer_chinese: bool = False) -> str:
        lowered = limitation.lower()
        if "robust" in lowered or "shift" in lowered:
            return "该方法在分布偏移下是否仍然稳定？" if prefer_chinese else "Does the method remain stable under distribution shift?"
        if "scal" in lowered or "large" in lowered:
            return "该方法在更大规模设置下表现如何？" if prefer_chinese else "How does the method behave at larger scale?"
        if "ablation" in lowered:
            return "根据消融结果，哪些组件是必要的？" if prefer_chinese else "Which components are essential according to the ablation story?"
        return f"关于 {limitation.rstrip('.')} 还有哪些结论尚未充分验证？" if prefer_chinese else f"What remains unresolved about {limitation.rstrip('.')}?"

    def _paper_confidence(self, chunks: list[Chunk], key_results: list[str], limitations: list[str]) -> ConfidenceScore:
        value = 0.45
        if chunks:
            value += 0.15
        if key_results:
            value += 0.1
        if limitations:
            value += 0.05
        return ConfidenceScore(value=min(0.9, value))

    def _sentences(self, text: str) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized) if sentence.strip()]
        return sentences or [normalized]

    def _first_sentence(self, sentences: list[str], keywords: tuple[str, ...]) -> str | None:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                return sentence
        return None

    def _collect_sentences(self, sentences: list[str], keywords: tuple[str, ...], limit: int) -> list[str]:
        collected: list[str] = []
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                collected.append(sentence)
            if len(collected) >= limit:
                break
        return collected

    def _infer_relation_type(self, text: str) -> RelationType:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ("contradict", "conflict", "inconsistent", "opposite")):
            return RelationType.CONFLICTS_WITH
        if any(keyword in lowered for keyword in ("compare", "baseline", "bench", "same benchmark")):
            if "same benchmark" in lowered or "benchmark" in lowered:
                return RelationType.USES_SAME_BENCHMARK
            return RelationType.COMPARES_WITH
        if any(keyword in lowered for keyword in ("improve", "better", "outperform", "surpass", "advance")):
            return RelationType.IMPROVES_ON
        if any(keyword in lowered for keyword in ("similar", "variant", "related", "same approach")):
            return RelationType.SIMILAR_TO
        return RelationType.COMPLEMENTS

    def _build_relation_summary(self, source_title: str, target_title: str, relation_type: RelationType) -> str:
        if relation_type is RelationType.IMPROVES_ON:
            return f"{source_title} improves on {target_title}."
        if relation_type is RelationType.SIMILAR_TO:
            return f"{source_title} is similar to {target_title}."
        if relation_type is RelationType.CONFLICTS_WITH:
            return f"{source_title} conflicts with {target_title}."
        if relation_type is RelationType.USES_SAME_BENCHMARK:
            return f"{source_title} uses the same benchmark as {target_title}."
        if relation_type is RelationType.COMPARES_WITH:
            return f"{source_title} compares with {target_title}."
        return f"{source_title} complements {target_title}."

    def _problem_keywords(self) -> tuple[str, ...]:
        return ("problem", "challenge", "task", "goal", "aim", "we study", "we investigate", "we propose")

    def _method_keywords(self) -> tuple[str, ...]:
        return ("method", "approach", "pipeline", "framework", "model", "algorithm", "we use", "we train", "we fine-tune")

    def _result_keywords(self) -> tuple[str, ...]:
        return ("result", "results", "achieve", "improve", "outperform", "accuracy", "performance", "state-of-the-art", "beats")

    def _limitation_keywords(self) -> tuple[str, ...]:
        return ("limitation", "limitations", "future work", "future", "open question", "open questions", "ablation", "not yet", "requires further")

    def _novelty_keywords(self) -> tuple[str, ...]:
        return ("novel", "new", "innovation", "introduce", "propose", "novelty")

    def _relation_keywords(self) -> tuple[str, ...]:
        return ("compare", "comparison", "related", "baseline", "benchmark", "conflict", "contradict", "improve", "similar")
