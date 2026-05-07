from __future__ import annotations

import re
from typing import Literal

from research_agent.adapters.llm.ingest_extraction import StructuredIngestExtractionParseError
from research_agent.domain.models import Chunk, Paper
from research_agent.runtime.ingest_extraction import (
    IngestExtractionCandidate,
    IngestExtractionClient,
    IngestExtractionDecision,
    IngestExtractionRequest,
    IngestExtractionWindow,
)
from research_agent.services.ingest_cleaning import ChunkCleanupReport, CleanedChunkRecord, IngestCleaningHelper
from research_agent.utils import to_json_safe


class IngestExtractionDebugHelper:
    """Build extraction requests, invoke the client, and capture debug payloads."""

    def __init__(self, extraction_client: IngestExtractionClient | None, cleaning_helper: IngestCleaningHelper) -> None:
        self._extraction_client = extraction_client
        self._cleaning = cleaning_helper

    def invoke_extractor(self, request: IngestExtractionRequest) -> tuple[IngestExtractionDecision | None, dict[str, object] | None]:
        if self._extraction_client is None:
            return None, {
                "extractor_stage": "extractor_failed",
                "validation_error": "missing extraction client",
                "raw_response_preview": None,
                "normalized_payload_preview": None,
                "failed_field": None,
            }
        try:
            return self._extraction_client.extract(request), None
        except StructuredIngestExtractionParseError as exc:
            return None, {
                "extractor_stage": exc.extractor_stage,
                "validation_error": exc.validation_error,
                "raw_response_preview": exc.raw_response_preview,
                "normalized_payload_preview": exc.normalized_payload_preview,
                "failed_field": exc.failed_field,
            }
        except Exception:
            return None, {
                "extractor_stage": "extractor_failed",
                "validation_error": "unexpected extractor failure",
                "raw_response_preview": None,
                "normalized_payload_preview": None,
                "failed_field": None,
            }

    def build_extraction_request(
        self,
        *,
        session_id: str,
        paper: Paper,
        chunks: list[Chunk],
        related_paper: Paper | None,
        artifact_id: str,
        window_kind: Literal["broad", "expanded"],
        extraction_stage: Literal["full_text", "batch", "merge"] = "full_text",
        batch_index: int | None = None,
        batch_count: int | None = None,
        batch_label: str | None = None,
        batch_summaries: tuple[dict[str, object], ...] = (),
        cleanup_records: tuple[CleanedChunkRecord, ...] = (),
    ) -> IngestExtractionRequest:
        candidate_passages = self.build_candidate_passages(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            window_kind=window_kind,
            cleanup_records=cleanup_records,
        )
        context_summary = self.build_candidate_context_summary(paper, chunks, candidate_passages)
        return IngestExtractionRequest(
            session_id=session_id,
            paper=paper,
            related_papers=(related_paper,) if related_paper is not None else (),
            window=IngestExtractionWindow(
                kind=window_kind,  # type: ignore[arg-type]
                context_summary=context_summary,
                candidate_passages=candidate_passages,
            ),
            extraction_stage=extraction_stage,
            batch_index=batch_index,
            batch_count=batch_count,
            batch_label=batch_label,
            batch_summaries=batch_summaries,
        )

    def build_candidate_passages(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        window_kind: str,
        cleanup_records: tuple[CleanedChunkRecord, ...] = (),
    ) -> tuple[IngestExtractionCandidate, ...]:
        cleanup_by_chunk_id = {record.chunk_id: record for record in cleanup_records}
        candidates: list[IngestExtractionCandidate] = [
            IngestExtractionCandidate(
                candidate_id="title",
                chunk_id=None,
                page=None,
                section="title",
                content_role="title",
                cleaned_text=paper.title,
                excerpt=paper.title,
                relevance_reason="title always included",
                source_chunk_ids=(),
            ),
        ]
        if paper.abstract:
            candidates.append(
                IngestExtractionCandidate(
                    candidate_id="abstract",
                    chunk_id=None,
                    page=None,
                    section="abstract",
                    content_role="abstract",
                    cleaned_text=paper.abstract,
                    excerpt=paper.abstract,
                    relevance_reason="abstract always included",
                    source_chunk_ids=(),
                )
            )

        if not chunks:
            return tuple(candidates)

        cleaned_chunks = self._cleaning.merge_short_chunk_groups(chunks)
        if not cleaned_chunks:
            return tuple(candidates)

        for index, chunk_group in enumerate(cleaned_chunks):
            first_chunk = chunk_group[0]
            source_chunk_ids = tuple(chunk.id for chunk in chunk_group)
            excerpt = " ".join(chunk.text for chunk in chunk_group if chunk.text.strip()).strip()
            if not excerpt:
                continue
            group_cleaned_texts: list[str] = []
            group_quality_flags: list[str] = []
            for chunk in chunk_group:
                record = cleanup_by_chunk_id.get(chunk.id)
                chunk_text = record.cleaned_text if record is not None else chunk.text
                if chunk_text.strip():
                    group_cleaned_texts.append(chunk_text.strip())
                if record is not None:
                    group_quality_flags.extend(record.quality_flags)
            cleaned_text = " ".join(group_cleaned_texts).strip()[:1200]
            quality_flags = tuple(dict.fromkeys(flag for flag in group_quality_flags))
            candidates.append(
                IngestExtractionCandidate(
                    candidate_id=first_chunk.id if len(chunk_group) == 1 else f"{first_chunk.id}__{chunk_group[-1].id}",
                    chunk_id=first_chunk.id,
                    page=first_chunk.page,
                    section=first_chunk.section,
                    cleaned_text=cleaned_text,
                    content_role=self.classify_chunk_role(first_chunk, excerpt),
                    excerpt=excerpt[:1200],
                    relevance_reason=f"cleaned chunk group {index}; size={len(chunk_group)}",
                    source_chunk_ids=source_chunk_ids,
                    quality_flags=quality_flags,
                    removed_reason=None,
                )
            )
        return tuple(candidates)

    def batch_summaries_for_prompt(
        self,
        batch_results: list[tuple[IngestExtractionRequest, IngestExtractionDecision]],
    ) -> tuple[dict[str, object], ...]:
        summaries: list[dict[str, object]] = []
        for request, decision in batch_results:
            understanding = decision.understanding
            if understanding is not None:
                understanding_payload = {
                    "topic": self.field_payload(understanding.topic),
                    "problem": self.field_payload(understanding.problem),
                    "method": self.field_payload(understanding.method),
                    "novelty_claims": [self.field_payload(item) for item in understanding.novelty_claims],
                    "key_results": [self.field_payload(item) for item in understanding.key_results],
                    "experiment_design": self.field_payload(understanding.experiment_design),
                    "limitations": [self.field_payload(item) for item in understanding.limitations],
                    "open_questions": [self.field_payload(item) for item in understanding.open_questions],
                    "evidence_chunk_ids": understanding.evidence_chunk_ids,
                    "confidence": understanding.confidence,
                }
            else:
                understanding_payload = {
                    "topic": decision.paper_summary.what_it_is_about,
                    "problem": decision.paper_summary.problem_solved,
                    "method": decision.paper.method,
                    "novelty_claims": list(decision.paper_summary.new_ideas),
                    "key_results": list(decision.paper.key_results),
                    "experiment_design": None,
                    "limitations": list(decision.paper_summary.limitations),
                    "open_questions": list(decision.paper_summary.suggestions_or_questions),
                    "evidence_chunk_ids": decision.paper_summary.evidence_candidate_ids,
                    "confidence": decision.paper_summary.confidence,
                }
            summaries.append(
                {
                    "batch_index": request.batch_index,
                    "batch_count": request.batch_count,
                    "batch_label": request.batch_label,
                    "chunk_ids": [candidate.candidate_id for candidate in request.window.candidate_passages if candidate.candidate_id not in {"title", "abstract"}],
                    "needs_more_context": decision.needs_more_context,
                    "context_hints": list(decision.context_hints),
                    "understanding": understanding_payload,
                    "rationale": decision.rationale,
                }
            )
        return tuple(summaries)

    def field_payload(self, field: object) -> dict[str, object] | None:
        if field is None:
            return None
        text = getattr(field, "text", None)
        evidence_chunk_ids = getattr(field, "evidence_chunk_ids", ())
        confidence = getattr(field, "confidence", 0.5)
        evidence_status = getattr(field, "evidence_status", None)
        if isinstance(field, dict):
            text = field.get("text")
            evidence_chunk_ids = field.get("evidence_chunk_ids", ())
            confidence = field.get("confidence", 0.5)
            evidence_status = field.get("evidence_status", evidence_status)
        if not isinstance(text, str) or not text.strip():
            return None
        evidence_ids = tuple(item for item in evidence_chunk_ids if isinstance(item, str))
        if isinstance(evidence_status, str):
            evidence_status = evidence_status.strip().lower()
        if not evidence_ids:
            evidence_status = "weak"
        elif evidence_status not in {"strong", "weak"}:
            evidence_status = "strong"
        return {
            "text": text.strip(),
            "evidence_chunk_ids": evidence_ids,
            "confidence": confidence,
            "evidence_status": evidence_status,
        }

    def debug_input_chunk_ids(self, candidate_passages: tuple[IngestExtractionCandidate, ...]) -> list[str]:
        chunk_ids: list[str] = []
        for candidate in candidate_passages:
            if candidate.source_chunk_ids:
                for chunk_id in candidate.source_chunk_ids:
                    if chunk_id not in chunk_ids:
                        chunk_ids.append(chunk_id)
                continue
            if candidate.chunk_id and candidate.chunk_id not in chunk_ids:
                chunk_ids.append(candidate.chunk_id)
        return chunk_ids

    def debug_text_field(
        self,
        field: str,
        candidate: str | None,
        *,
        evidence_ids: tuple[str, ...],
        paper_title: str | None = None,
    ) -> dict[str, object]:
        raw_text = (candidate or "").strip()
        reject_reason: str | None = None
        accepted = bool(raw_text) and not self._looks_like_summary_noise(raw_text)
        if accepted and paper_title is not None:
            if raw_text == paper_title.strip() or raw_text.lower() == paper_title.strip().lower():
                accepted = False
                reject_reason = "matches_paper_title"
        if not raw_text:
            reject_reason = "missing_model_output"
        elif reject_reason is None and not accepted:
            reject_reason = "missing_or_noisy_model_output"
        return {
            "field": field,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "raw_text": raw_text,
            "evidence_chunk_ids": list(evidence_ids),
        }

    def debug_items_field(
        self,
        field: str,
        candidates: tuple[str, ...],
        *,
        evidence_ids: tuple[str, ...],
        paper_title: str | None = None,
    ) -> dict[str, object]:
        raw_items = [item.strip() for item in candidates if item and item.strip()]
        accepted_items = [item for item in raw_items if not self._looks_like_summary_noise(item)]
        if paper_title is not None:
            accepted_items = [item for item in accepted_items if item != paper_title.strip() and item.lower() != paper_title.strip().lower()]
        reject_reason: str | None = None
        if not raw_items:
            reject_reason = "missing_model_output"
        elif not accepted_items:
            reject_reason = "missing_or_noisy_model_output"
        return {
            "field": field,
            "accepted": bool(accepted_items),
            "reject_reason": reject_reason,
            "raw_items": raw_items,
            "accepted_items": accepted_items,
            "evidence_chunk_ids": list(evidence_ids),
        }

    def rank_candidate_chunks(
        self,
        chunks: list[Chunk],
        keyword_hit_ids: set[str],
    ) -> list[tuple[Chunk, int, list[str], str]]:
        ranked: list[tuple[Chunk, int, list[str], str]] = []
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks):
            score = 0
            reasons: list[str] = []
            section = (chunk.section or "").lower()
            text_lower = chunk.text.lower()
            content_role = self.classify_chunk_role(chunk)
            if content_role == "title":
                score += 40
                reasons.append("title section")
            if content_role == "abstract":
                score += 35
                reasons.append("abstract section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("intro", "introduction", "background")):
                score += 28
                reasons.append("main text section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("method", "approach", "experiment", "evaluation")):
                score += 24
                reasons.append("main text section")
            if content_role == "main" and any(token in section or token in text_lower for token in ("result", "discussion", "conclusion", "analysis")):
                score += 22
                reasons.append("main text section")
            if any(token in section or token in text_lower for token in ("limit", "future work")):
                score += 18
                reasons.append("limitations section")
            if content_role in {"appendix", "table"}:
                score -= 15
                reasons.append("appendix section")
            if content_role == "reference":
                score -= 60
                reasons.append("reference section")
            if chunk.id in keyword_hit_ids:
                score += 20
                reasons.append("keyword hit")

            digit_count = sum(character.isdigit() for character in chunk.text)
            letter_count = sum(character.isalpha() for character in chunk.text)
            if digit_count >= 8 and digit_count > max(letter_count // 4, 1):
                score -= 8
                reasons.append("numeric heavy")
            if "table" in text_lower and digit_count >= 4:
                score -= 6
                reasons.append("table heavy")
            if index < 3:
                score += 6
                reasons.append("early section")
            elif index >= max(total_chunks - 3, 0):
                score += 4
                reasons.append("late section")
            ranked.append((chunk, score, reasons, content_role))
        return sorted(ranked, key=lambda item: (-item[1], item[0].page if item[0].page is not None else 10**9, item[0].id))

    def broad_candidate_limits(self, chunk_count: int) -> tuple[int, int, int]:
        main_limit = min(14, max(6, chunk_count // 3))
        appendix_limit = min(2, max(1, chunk_count // 18))
        reference_limit = 0
        return main_limit, appendix_limit, reference_limit

    def classify_chunk_role(
        self,
        chunk: Chunk,
        text: str | None = None,
    ) -> Literal["title", "abstract", "main", "appendix", "table", "reference", "unknown"]:
        section = (chunk.section or "").lower()
        text_lower = (text if text is not None else chunk.text).lower()
        if any(token in section for token in ("title",)):
            return "title"
        if "abstract" in section or text_lower.startswith("abstract"):
            return "abstract"
        if any(token in section or token in text_lower for token in ("appendix", "appendices", "supplement", "supplementary")):
            return "appendix"
        if any(token in section or token in text_lower for token in ("reference", "bibliography")):
            return "reference"
        if "table" in text_lower or any(token in section or token in text_lower for token in ("intro", "introduction", "background", "method", "approach", "experiment", "evaluation", "result", "discussion", "conclusion", "analysis", "limit", "future work")):
            return "main"
        if len(text_lower.split()) < 18 and sum(character.isdigit() for character in chunk.text) >= 4:
            return "table"
        return "unknown"

    def keyword_hit_ids(self, chunks: list[Chunk]) -> set[str]:
        return {chunk.id for chunk in self.candidate_chunks_by_keywords(chunks)}

    def build_candidate_context_summary(
        self,
        paper: Paper,
        chunks: list[Chunk],
        candidate_passages: tuple[IngestExtractionCandidate, ...],
    ) -> str:
        parts = [paper.title, paper.abstract or ""]
        parts.extend(candidate.excerpt for candidate in candidate_passages[:12])
        if len(candidate_passages) < 3 and chunks:
            parts.extend(chunk.text[:240] for chunk in chunks[:3])
        summary = " ".join(part for part in parts if part).strip()
        return summary[:4000]

    def candidate_chunks_by_keywords(self, chunks: list[Chunk]) -> list[Chunk]:
        keyword_chunks: list[Chunk] = []
        keywords = (
            self._problem_keywords()
            + self._method_keywords()
            + self._result_keywords()
            + self._limitation_keywords()
            + self._novelty_keywords()
            + self._relation_keywords()
        )
        for chunk in chunks:
            lowered = chunk.text.lower()
            if any(keyword in lowered for keyword in keywords):
                keyword_chunks.append(chunk)
        return keyword_chunks

    def build_extraction_debug(
        self,
        *,
        request: IngestExtractionRequest,
        decision: IngestExtractionDecision,
        paper: Paper,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        extraction_mode: str,
        cleanup_report: ChunkCleanupReport | None = None,
    ) -> dict[str, object]:
        input_chunk_ids = self.debug_input_chunk_ids(candidate_passages)
        raw_decision = to_json_safe(
            {
                "understanding": to_json_safe(decision.understanding),
                "paper": to_json_safe(decision.paper),
                "relation": to_json_safe(decision.relation),
                "open_question": to_json_safe(decision.open_question),
                "paper_summary": to_json_safe(decision.paper_summary),
                "needs_more_context": decision.needs_more_context,
                "context_hints": list(decision.context_hints),
                "rationale": decision.rationale,
            }
        )
        field_reviews = [
            self.debug_text_field(
                "paper.problem",
                decision.paper.problem,
                evidence_ids=decision.paper.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_text_field(
                "paper.method",
                decision.paper.method,
                evidence_ids=decision.paper.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_text_field(
                "paper.novelty_claim",
                decision.paper.novelty_claim,
                evidence_ids=decision.paper.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_items_field(
                "paper.key_results",
                decision.paper.key_results,
                evidence_ids=decision.paper.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_items_field(
                "paper.limitations",
                decision.paper.limitations,
                evidence_ids=decision.paper.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_text_field(
                "paper_summary.what_it_is_about",
                decision.paper_summary.what_it_is_about,
                evidence_ids=decision.paper_summary.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_text_field(
                "paper_summary.problem_solved",
                decision.paper_summary.problem_solved,
                evidence_ids=decision.paper_summary.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_items_field(
                "paper_summary.new_ideas",
                decision.paper_summary.new_ideas,
                evidence_ids=decision.paper_summary.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_items_field(
                "paper_summary.limitations",
                decision.paper_summary.limitations,
                evidence_ids=decision.paper_summary.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_items_field(
                "paper_summary.suggestions_or_questions",
                decision.paper_summary.suggestions_or_questions,
                evidence_ids=decision.paper_summary.evidence_candidate_ids,
                paper_title=paper.title,
            ),
            self.debug_text_field(
                "open_question.unresolved_question",
                decision.open_question.unresolved_question,
                evidence_ids=decision.open_question.evidence_candidate_ids,
                paper_title=paper.title,
            ),
        ]
        if request.extraction_stage == "merge":
            field_reviews.append(
                {
                    "field": "merge_context",
                    "accepted": True,
                    "reject_reason": None,
                    "raw_text": decision.rationale,
                    "evidence_chunk_ids": input_chunk_ids,
                }
            )
        payload = {
            "extraction_stage": request.extraction_stage,
            "extraction_mode": extraction_mode,
            "window_kind": request.window.kind,
            "batch_index": request.batch_index,
            "batch_count": request.batch_count,
            "batch_label": request.batch_label,
            "input_chunk_ids": input_chunk_ids,
            "candidate_ids": [candidate.candidate_id for candidate in request.window.candidate_passages],
            "batch_summaries": to_json_safe(request.batch_summaries),
            "raw_decision": raw_decision,
            "field_reviews": field_reviews,
            "needs_more_context": decision.needs_more_context,
            "context_hints": list(decision.context_hints),
            "rationale": decision.rationale,
        }
        if cleanup_report is not None:
            payload["input_cleanup"] = self._cleaning.cleanup_report_payload(cleanup_report)
        return payload

    def _looks_like_summary_noise(self, text: str) -> bool:
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
