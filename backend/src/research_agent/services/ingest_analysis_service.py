"""Model-facing analysis service for parsed ingest source content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from research_agent.domain.enums import RelationType
from research_agent.domain.models import Chunk, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SourceRef
from research_agent.domain.ports import ChunkRepositoryPort, MemoryRepositoryPort, PaperRepositoryPort, SessionRepositoryPort
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services.errors import EntityNotFoundError
from research_agent.services.ingest_cleaning import ChunkCleanupBundle, ChunkCleanupReport, CleanedChunkRecord, IngestCleaningHelper
from research_agent.services.ingest_extraction_debug import IngestExtractionDebugHelper
from research_agent.services.ingest_summary_policy import IngestSummaryPolicy
from research_agent.runtime.ingest_extraction import (
    IngestExtractionCandidate,
    IngestExtractionClient,
    IngestExtractionDecision,
    IngestExtractionRequest,
    IngestOpenQuestionMemoryDraft,
    IngestPaperMemoryDraft,
    IngestPaperSummaryDraft,
    IngestRelationMemoryDraft,
)

_INSUFFICIENT_EVIDENCE_TEXT = "无法基于当前论文内容稳定生成该字段。"


@dataclass(frozen=True, slots=True)
class MemoryAnalysisResult:
    """Structured memory drafts extracted from parsed ingest content."""

    paper_memory: PaperMemory
    relation_memory: RelationMemory | None
    open_question_memory: OpenQuestionMemory
    paper_summary: IngestPaperSummaryDraft
    context_summary: str
    extraction_debug: dict[str, object] | None = None


class IngestAnalysisService:
    """Build structured paper, relation, and open-question memory drafts."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        paper_repository: PaperRepositoryPort,
        chunk_repository: ChunkRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        extraction_client: IngestExtractionClient | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._paper_repository = paper_repository
        self._chunk_repository = chunk_repository
        self._memory_repository = memory_repository
        self._extraction_client = extraction_client
        self._cleaning = IngestCleaningHelper()
        self._debug = IngestExtractionDebugHelper(extraction_client, self._cleaning)
        self._summary_policy = IngestSummaryPolicy()

    def analyze(self, session_id: str, paper_id: str) -> MemoryAnalysisResult:
        """Analyze parsed source content without persisting memories."""

        session = self._require_session(session_id)
        paper = self._require_paper(paper_id)
        chunks = list(self._chunk_repository.list_by_paper_ids([paper_id]))
        cleanup_bundle = self._prepare_clean_chunks(chunks)
        session_documents = list(self._session_repository.list_documents(session.id))
        session_document = next((document for document in session_documents if document.paper_id == paper_id), None)
        if session_document is None:
            raise EntityNotFoundError("SessionDocument", paper_id)

        related_paper = self._select_related_paper(session_id=session_id, current_paper_id=paper_id)
        extraction_result = self._extract_with_client(
            session_id=session_id,
            paper=paper,
            artifact_id=session_document.artifact_id,
            cleanup_bundle=cleanup_bundle,
            related_paper=related_paper,
        )
        if extraction_result is None:
            return self._build_insufficient_evidence_result(
                paper=paper,
                artifact_id=session_document.artifact_id,
                cleanup_bundle=cleanup_bundle,
            )
        return extraction_result

    def _extract_with_client(
        self,
        *,
        session_id: str,
        paper: Paper,
        artifact_id: str,
        cleanup_bundle: ChunkCleanupBundle,
        related_paper: Paper | None,
    ) -> MemoryAnalysisResult | None:
        if self._extraction_client is None:
            return None

        cleaned_chunks = cleanup_bundle.cleaned_chunks
        if not cleaned_chunks:
            return None

        if self._should_use_full_text(cleaned_chunks):
            full_text_request = self._build_extraction_request(
                session_id=session_id,
                paper=paper,
                chunks=cleaned_chunks,
                related_paper=related_paper,
                artifact_id=artifact_id,
                window_kind="broad",
                extraction_stage="full_text",
                cleanup_records=cleanup_bundle.records,
            )
            full_text_decision, full_text_failure = self._invoke_extractor(full_text_request)
            if full_text_failure is not None:
                return self._build_insufficient_evidence_result(
                    paper=paper,
                    artifact_id=artifact_id,
                    cleanup_bundle=cleanup_bundle,
                    extraction_mode="extractor_failed",
                    extraction_stage=str(full_text_failure.get("extractor_stage", "schema_validation")),
                    failure_debug=full_text_failure,
                )
            if full_text_decision is not None and not full_text_decision.needs_more_context:
                return self._result_from_decision(
                    request=full_text_request,
                    decision=full_text_decision,
                    paper=paper,
                    artifact_id=artifact_id,
                    chunks=cleaned_chunks,
                    related_paper=None,
                    candidate_passages=full_text_request.window.candidate_passages,
                    context_summary=full_text_request.window.context_summary,
                    cleanup_report=cleanup_bundle.report,
                )
            if full_text_decision is not None and full_text_decision.needs_more_context:
                expanded_request = self._build_extraction_request(
                    session_id=session_id,
                    paper=paper,
                    chunks=cleaned_chunks,
                    related_paper=related_paper,
                    artifact_id=artifact_id,
                    window_kind="expanded",
                    extraction_stage="full_text",
                    cleanup_records=cleanup_bundle.records,
                )
                expanded_decision, expanded_failure = self._invoke_extractor(expanded_request)
                if expanded_failure is not None:
                    return self._build_insufficient_evidence_result(
                        paper=paper,
                        artifact_id=artifact_id,
                        cleanup_bundle=cleanup_bundle,
                        extraction_mode="extractor_failed",
                        extraction_stage=str(expanded_failure.get("extractor_stage", "schema_validation")),
                        failure_debug=expanded_failure,
                    )
                if expanded_decision is not None and not expanded_decision.needs_more_context:
                    return self._result_from_decision(
                        request=expanded_request,
                        decision=expanded_decision,
                        paper=paper,
                        artifact_id=artifact_id,
                        chunks=cleaned_chunks,
                        related_paper=None,
                        candidate_passages=expanded_request.window.candidate_passages,
                        context_summary=expanded_request.window.context_summary,
                        cleanup_report=cleanup_bundle.report,
                    )

        batches = self._split_clean_chunks_into_batches(cleaned_chunks)
        if not batches:
            return None

        batch_results: list[tuple[IngestExtractionRequest, IngestExtractionDecision]] = []
        for batch_index, batch_chunks in enumerate(batches, start=1):
            batch_request = self._build_extraction_request(
                session_id=session_id,
                paper=paper,
                chunks=batch_chunks,
                related_paper=related_paper,
                artifact_id=artifact_id,
                window_kind="broad",
                extraction_stage="batch",
                batch_index=batch_index,
                batch_count=len(batches),
                batch_label=f"batch-{batch_index}",
                cleanup_records=cleanup_bundle.records,
            )
            batch_decision, batch_failure = self._invoke_extractor(batch_request)
            if batch_failure is not None:
                return self._build_insufficient_evidence_result(
                    paper=paper,
                    artifact_id=artifact_id,
                    cleanup_bundle=cleanup_bundle,
                    extraction_mode="extractor_failed",
                    extraction_stage=str(batch_failure.get("extractor_stage", "schema_validation")),
                    failure_debug=batch_failure,
                )
            if batch_decision is None:
                return None
            batch_results.append((batch_request, batch_decision))

        merge_request = self._build_extraction_request(
            session_id=session_id,
            paper=paper,
            chunks=self._merge_context_chunks(batches),
            related_paper=related_paper,
            artifact_id=artifact_id,
            window_kind="expanded",
            extraction_stage="merge",
            batch_index=len(batch_results),
            batch_count=len(batch_results),
            batch_label="merge",
            batch_summaries=self._batch_summaries_for_prompt(batch_results),
            cleanup_records=cleanup_bundle.records,
        )
        merge_decision, merge_failure = self._invoke_extractor(merge_request)
        if merge_failure is not None:
            return self._build_insufficient_evidence_result(
                paper=paper,
                artifact_id=artifact_id,
                cleanup_bundle=cleanup_bundle,
                extraction_mode="extractor_failed",
                extraction_stage=str(merge_failure.get("extractor_stage", "schema_validation")),
                failure_debug=merge_failure,
            )
        if merge_decision is None:
            return None
        return self._result_from_decision(
            request=merge_request,
            decision=merge_decision,
            paper=paper,
            artifact_id=artifact_id,
            chunks=cleaned_chunks,
            related_paper=None,
            candidate_passages=merge_request.window.candidate_passages,
            context_summary=merge_request.window.context_summary,
            cleanup_report=cleanup_bundle.report,
        )

    def _invoke_extractor(self, request: IngestExtractionRequest) -> tuple[IngestExtractionDecision | None, dict[str, object] | None]:
        return self._debug.invoke_extractor(request)

    def _build_extraction_request(
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
        return self._debug.build_extraction_request(
            session_id=session_id,
            paper=paper,
            chunks=chunks,
            related_paper=related_paper,
            artifact_id=artifact_id,
            window_kind=window_kind,
            extraction_stage=extraction_stage,
            batch_index=batch_index,
            batch_count=batch_count,
            batch_label=batch_label,
            batch_summaries=batch_summaries,
            cleanup_records=cleanup_records,
        )

    def _build_candidate_passages(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        window_kind: str,
        cleanup_records: tuple[CleanedChunkRecord, ...] = (),
    ) -> tuple[IngestExtractionCandidate, ...]:
        return self._debug.build_candidate_passages(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            window_kind=window_kind,
            cleanup_records=cleanup_records,
        )

    def _prepare_clean_chunks(self, chunks: list[Chunk]) -> ChunkCleanupBundle:
        return self._cleaning.prepare_clean_chunks(chunks)

    def _compress_table_like_text(self, text: str) -> str:
        return self._cleaning.compress_table_like_text(text)

    def _looks_like_table_or_noise(self, text: str) -> bool:
        return self._cleaning.looks_like_table_or_noise(text)

    def _looks_like_low_quality_text(self, text: str) -> bool:
        return self._cleaning.looks_like_low_quality_text(text)

    def _looks_like_too_short(self, text: str) -> bool:
        return self._cleaning.looks_like_too_short(text)

    def _cleanup_report_payload(self, report: ChunkCleanupReport) -> dict[str, object]:
        return self._cleaning.cleanup_report_payload(report)

    def _should_use_full_text(self, chunks: list[Chunk]) -> bool:
        return self._cleaning.should_use_full_text(chunks)

    def _split_clean_chunks_into_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        return self._cleaning.split_clean_chunks_into_batches(chunks)

    def _merge_context_chunks(self, batches: list[list[Chunk]]) -> list[Chunk]:
        return self._cleaning.merge_context_chunks(batches)

    def _batch_summaries_for_prompt(
        self,
        batch_results: list[tuple[IngestExtractionRequest, IngestExtractionDecision]],
    ) -> tuple[dict[str, object], ...]:
        return self._debug.batch_summaries_for_prompt(batch_results)

    def _field_payload(self, field: object) -> dict[str, object] | None:
        return self._debug.field_payload(field)

    def _debug_input_chunk_ids(self, candidate_passages: tuple[IngestExtractionCandidate, ...]) -> list[str]:
        return self._debug.debug_input_chunk_ids(candidate_passages)

    def _debug_text_field(
        self,
        field: str,
        candidate: str | None,
        *,
        evidence_ids: tuple[str, ...],
        paper_title: str | None = None,
    ) -> dict[str, object]:
        return self._debug.debug_text_field(
            field,
            candidate,
            evidence_ids=evidence_ids,
            paper_title=paper_title,
        )

    def _debug_items_field(
        self,
        field: str,
        candidates: tuple[str, ...],
        *,
        evidence_ids: tuple[str, ...],
        paper_title: str | None = None,
    ) -> dict[str, object]:
        return self._debug.debug_items_field(
            field,
            candidates,
            evidence_ids=evidence_ids,
            paper_title=paper_title,
        )

    def _rank_candidate_chunks(
        self,
        chunks: list[Chunk],
        keyword_hit_ids: set[str],
    ) -> list[tuple[Chunk, int, list[str], str]]:
        return self._debug.rank_candidate_chunks(chunks, keyword_hit_ids)

    def _broad_candidate_limits(self, chunk_count: int) -> tuple[int, int, int]:
        return self._debug.broad_candidate_limits(chunk_count)

    def _classify_chunk_role(self, chunk: Chunk) -> str:
        return self._debug.classify_chunk_role(chunk)

    def _keyword_hit_ids(self, chunks: list[Chunk]) -> set[str]:
        return self._debug.keyword_hit_ids(chunks)

    def _build_candidate_context_summary(
        self,
        paper: Paper,
        chunks: list[Chunk],
        candidate_passages: tuple[IngestExtractionCandidate, ...],
    ) -> str:
        return self._debug.build_candidate_context_summary(paper, chunks, candidate_passages)

    def _candidate_chunks_by_keywords(self, chunks: list[Chunk]) -> list[Chunk]:
        return self._debug.candidate_chunks_by_keywords(chunks)

    def _result_from_decision(
        self,
        *,
        request: IngestExtractionRequest,
        decision: IngestExtractionDecision,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        related_paper: Paper | None,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        context_summary: str,
        cleanup_report: ChunkCleanupReport | None = None,
    ) -> MemoryAnalysisResult:
        paper_memory = self._paper_memory_from_draft(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            context_summary=context_summary,
            candidate_passages=candidate_passages,
            draft=decision.paper,
        )
        relation_memory = None
        if related_paper is not None and decision.relation is not None:
            relation_memory = self._relation_memory_from_draft(
                paper=paper,
                related_paper=related_paper,
                candidate_passages=candidate_passages,
                draft=decision.relation,
            )

        open_question_memory = self._open_question_memory_from_draft(
            paper=paper,
            related_paper=related_paper,
            chunks=chunks,
            context_summary=context_summary,
            draft=decision.open_question,
        )
        paper_summary = self._paper_summary_from_draft(
            paper=paper,
            paper_memory=paper_memory,
            open_question_memory=open_question_memory,
            candidate_passages=candidate_passages,
            draft=decision.paper_summary,
        )
        return MemoryAnalysisResult(
            paper_memory=paper_memory,
            relation_memory=relation_memory,
            open_question_memory=open_question_memory,
            paper_summary=paper_summary,
            context_summary=context_summary[:240],
            extraction_debug=self._build_extraction_debug(
                request=request,
                decision=decision,
                paper=paper,
                candidate_passages=candidate_passages,
                extraction_mode="full_text" if request.extraction_stage == "full_text" else "hierarchical",
                cleanup_report=cleanup_report,
            ),
        )

    def _build_extraction_debug(
        self,
        *,
        request: IngestExtractionRequest,
        decision: IngestExtractionDecision,
        paper: Paper,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        extraction_mode: str,
        cleanup_report: ChunkCleanupReport | None = None,
    ) -> dict[str, object]:
        return self._debug.build_extraction_debug(
            request=request,
            decision=decision,
            paper=paper,
            candidate_passages=candidate_passages,
            extraction_mode=extraction_mode,
            cleanup_report=cleanup_report,
        )

    def _build_insufficient_evidence_result(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        cleanup_bundle: ChunkCleanupBundle,
        extraction_mode: str = "insufficient_evidence",
        extraction_stage: str = "insufficient_evidence",
        failure_debug: dict[str, object] | None = None,
    ) -> MemoryAnalysisResult:
        chunks = cleanup_bundle.cleaned_chunks
        context_text = self._build_context_text(paper, chunks)
        candidate_passages = self._build_candidate_passages(
            paper=paper,
            artifact_id=artifact_id,
            chunks=chunks,
            window_kind="broad",
            cleanup_records=cleanup_bundle.records,
        )
        source_refs = self._source_refs_for_candidate_ids(
            paper_id=paper.id,
            artifact_id=artifact_id,
            chunks=chunks,
            candidate_passages=candidate_passages,
            candidate_ids=(),
            fallback_text=context_text,
        )
        paper_memory = PaperMemory(
            paper_id=paper.id,
            problem=_INSUFFICIENT_EVIDENCE_TEXT,
            method=_INSUFFICIENT_EVIDENCE_TEXT,
            key_results=[_INSUFFICIENT_EVIDENCE_TEXT],
            limitations=[_INSUFFICIENT_EVIDENCE_TEXT],
            novelty_claim=_INSUFFICIENT_EVIDENCE_TEXT,
            source_refs=source_refs,
            confidence=ConfidenceScore(value=0.1),
        )
        open_question_memory = OpenQuestionMemory(
            unresolved_question=_INSUFFICIENT_EVIDENCE_TEXT,
            related_papers=[paper.id],
            why_open=[_INSUFFICIENT_EVIDENCE_TEXT],
            possible_followup=[_INSUFFICIENT_EVIDENCE_TEXT],
            confidence=ConfidenceScore(value=0.1),
        )
        paper_summary = IngestPaperSummaryDraft(
            what_it_is_about=_INSUFFICIENT_EVIDENCE_TEXT,
            problem_solved=_INSUFFICIENT_EVIDENCE_TEXT,
            new_ideas=(_INSUFFICIENT_EVIDENCE_TEXT,),
            limitations=(_INSUFFICIENT_EVIDENCE_TEXT,),
            suggestions_or_questions=(_INSUFFICIENT_EVIDENCE_TEXT,),
            evidence_candidate_ids=self._preferred_summary_evidence_ids(candidate_passages, ()),
            confidence=0.1,
        )
        debug_payload: dict[str, object] = {
            "extraction_mode": extraction_mode,
            "extraction_stage": extraction_stage,
            "input_chunk_ids": [chunk.id for chunk in chunks],
            "candidate_ids": [candidate.candidate_id for candidate in candidate_passages],
            "raw_decision": None,
            "field_reviews": [
                {
                    "field": "paper.problem",
                    "accepted": False,
                    "reject_reason": extraction_stage,
                    "raw_text": _INSUFFICIENT_EVIDENCE_TEXT,
                    "evidence_chunk_ids": [],
                }
            ],
            "input_cleanup": self._cleaning.cleanup_report_payload(cleanup_bundle.report),
        }
        if failure_debug is not None:
            debug_payload["validation_error"] = failure_debug.get("validation_error")
            debug_payload["raw_response_preview"] = failure_debug.get("raw_response_preview")
            debug_payload["normalized_payload_preview"] = failure_debug.get("normalized_payload_preview")
            debug_payload["failed_field"] = failure_debug.get("failed_field")
            debug_payload["field_reviews"] = [
                {
                    "field": failure_debug.get("failed_field") or "ingest_extractor",
                    "accepted": False,
                    "reject_reason": extraction_stage,
                    "raw_text": failure_debug.get("validation_error"),
                    "evidence_chunk_ids": [],
                }
            ]
        return MemoryAnalysisResult(
            paper_memory=paper_memory,
            relation_memory=None,
            open_question_memory=open_question_memory,
            paper_summary=paper_summary,
            context_summary=context_text[:240],
            extraction_debug=debug_payload,
        )

    def _paper_memory_from_draft(
        self,
        *,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        context_summary: str,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestPaperMemoryDraft,
    ) -> PaperMemory:
        source_refs = self._source_refs_for_candidate_ids(
            paper_id=paper.id,
            artifact_id=artifact_id,
            chunks=chunks,
            candidate_passages=candidate_passages,
            candidate_ids=draft.evidence_candidate_ids,
            fallback_text=context_summary,
        )
        return PaperMemory(
            paper_id=paper.id,
            problem=self._summary_text_or_unavailable(draft.problem, paper_title=paper.title),
            method=self._summary_text_or_unavailable(draft.method, paper_title=paper.title),
            key_results=list(self._summary_items_or_unavailable(draft.key_results)),
            limitations=list(self._summary_items_or_unavailable(draft.limitations)),
            novelty_claim=self._summary_text_or_unavailable(draft.novelty_claim, paper_title=paper.title),
            source_refs=source_refs,
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _relation_memory_from_draft(
        self,
        *,
        paper: Paper,
        related_paper: Paper,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestRelationMemoryDraft,
    ) -> RelationMemory:
        relation_type = self._coerce_relation_type(draft.relation_type)
        return RelationMemory(
            source_paper=paper.id,
            target_paper=related_paper.id,
            relation_type=relation_type,
            summary=draft.summary,
            evidence=self._evidence_for_candidate_ids(candidate_passages, draft.evidence_candidate_ids, draft.summary),
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _open_question_memory_from_draft(
        self,
        *,
        paper: Paper,
        related_paper: Paper | None,
        chunks: list[Chunk],
        context_summary: str,
        draft: IngestOpenQuestionMemoryDraft,
    ) -> OpenQuestionMemory:
        related_papers = [paper.id]
        why_open = list(self._summary_items_or_unavailable(draft.why_open, max_items=3))
        possible_followup = list(self._summary_items_or_unavailable(draft.possible_followup, max_items=3))
        unresolved_question = self._summary_text_or_unavailable(draft.unresolved_question)
        return OpenQuestionMemory(
            unresolved_question=self._summary_text_or_unavailable(draft.unresolved_question, paper_title=paper.title),
            related_papers=related_papers,
            why_open=why_open,
            possible_followup=possible_followup,
            confidence=ConfidenceScore(value=draft.confidence),
        )

    def _paper_summary_from_draft(
        self,
        *,
        paper: Paper,
        paper_memory: PaperMemory,
        open_question_memory: OpenQuestionMemory,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft: IngestPaperSummaryDraft,
    ) -> IngestPaperSummaryDraft:
        evidence_candidate_ids = self._preferred_summary_evidence_ids(candidate_passages, draft.evidence_candidate_ids)
        return IngestPaperSummaryDraft(
            what_it_is_about=self._summary_text_or_unavailable(draft.what_it_is_about, paper_title=paper.title),
            problem_solved=self._summary_text_or_unavailable(draft.problem_solved, paper_title=paper.title),
            new_ideas=self._summary_items_or_unavailable(draft.new_ideas),
            limitations=self._summary_items_or_unavailable(draft.limitations),
            suggestions_or_questions=self._summary_items_or_unavailable(draft.suggestions_or_questions),
            evidence_candidate_ids=evidence_candidate_ids,
            confidence=draft.confidence if draft.confidence else paper_memory.confidence.value,
        )

    def _preferred_summary_evidence_ids(
        self,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        draft_candidate_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_passages}
        preferred_roles = {"title", "abstract", "main"}
        selected = [
            candidate_id
            for candidate_id in draft_candidate_ids
            if candidate_id in candidate_by_id and candidate_by_id[candidate_id].content_role in preferred_roles
        ]
        if selected:
            return tuple(selected[:4])

        preferred = [
            candidate.candidate_id
            for candidate in candidate_passages
            if candidate.content_role in preferred_roles and candidate.candidate_id not in {"title", "abstract"}
        ]
        if preferred:
            return tuple(preferred[:4])

        fallback = [
            candidate.candidate_id
            for candidate in candidate_passages
            if candidate.content_role in {"appendix", "table", "unknown"}
        ]
        return tuple(fallback[:4])

    def _summary_text_or_unavailable(self, text: str | None, *, paper_title: str | None = None) -> str:
        return self._summary_policy.summary_text_or_unavailable(text, paper_title=paper_title)

    def _summary_items_or_unavailable(self, items: tuple[str, ...], *, max_items: int = 2) -> tuple[str, ...]:
        return self._summary_policy.summary_items_or_unavailable(items, max_items=max_items)

    def _looks_like_generic_new_idea(self, text: str) -> bool:
        return self._summary_policy.looks_like_generic_new_idea(text)

    def _looks_like_generic_suggestion(self, text: str) -> bool:
        return self._summary_policy.looks_like_generic_suggestion(text)

    def _sanitize_summary_text(self, text: str | None, *fallback_candidates: str | None) -> str:
        return self._summary_policy.sanitize_summary_text(text, *fallback_candidates)

    def _first_clean_text(self, *candidates: str | None) -> str:
        return self._summary_policy.first_clean_text(*candidates)

    def _sanitize_summary_items(
        self,
        items: tuple[str, ...],
        fallback_items: tuple[str, ...],
        *,
        max_items: int = 2,
    ) -> tuple[str, ...]:
        return self._summary_policy.sanitize_summary_items(items, fallback_items, max_items=max_items)

    def _normalize_summary_item(self, item: str) -> str | None:
        return self._summary_policy.normalize_summary_item(item)

    def _looks_like_summary_noise(self, text: str) -> bool:
        return self._summary_policy.looks_like_summary_noise(text)

    def _is_placeholder_source_title(self, title: str) -> bool:
        return self._summary_policy.is_placeholder_source_title(title)

    def _summary_seed_text(self, *values: str | None) -> str:
        return self._summary_policy.summary_seed_text(*values)

    def _contains_cjk(self, text: str) -> bool:
        return self._summary_policy.contains_cjk(text)

    def _fallback_topic_text(self, seed_text: str, title: str) -> str:
        return self._summary_policy.fallback_topic_text(seed_text, title)

    def _fallback_problem_text(self, seed_text: str, title: str) -> str:
        return self._summary_policy.fallback_problem_text(seed_text, title)

    def _fallback_method_text(self, seed_text: str) -> str:
        return self._summary_policy.fallback_method_text(seed_text)

    def _fallback_novelty_text(self, seed_text: str) -> str:
        return self._summary_policy.fallback_novelty_text(seed_text)

    def _fallback_idea_texts(self, seed_text: str) -> tuple[str, ...]:
        return self._summary_policy.fallback_idea_texts(seed_text)

    def _fallback_key_result_texts(self, seed_text: str) -> tuple[str, ...]:
        return self._summary_policy.fallback_key_result_texts(seed_text)

    def _fallback_limitation_texts(self, seed_text: str, why_open: tuple[str, ...] = ()) -> tuple[str, ...]:
        return self._summary_policy.fallback_limitation_texts(seed_text, why_open)

    def _fallback_why_open_texts(self, seed_text: str, chunk_count: int) -> tuple[str, ...]:
        return self._summary_policy.fallback_why_open_texts(seed_text, chunk_count)

    def _fallback_suggestion_texts(self, seed_text: str, title: str, possible_followup: tuple[str, ...] = ()) -> tuple[str, ...]:
        return self._summary_policy.fallback_suggestion_texts(seed_text, title, possible_followup)

    def _source_refs_for_candidate_ids(
        self,
        *,
        paper_id: str,
        artifact_id: str,
        chunks: list[Chunk],
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        candidate_ids: tuple[str, ...],
        fallback_text: str,
    ) -> list[SourceRef]:
        selected_candidates = [candidate for candidate in candidate_passages if candidate.candidate_id in candidate_ids]
        if selected_candidates:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    page=candidate.page,
                    section=candidate.section,
                    chunk_id=candidate.chunk_id,
                    quote=candidate.excerpt[:240],
                )
                for candidate in selected_candidates
            ]

        selected_chunks = [chunk for chunk in chunks if any(candidate.chunk_id == chunk.id for candidate in candidate_passages)]
        if not selected_chunks and not chunks:
            return [
                SourceRef(
                    paper_id=paper_id,
                    artifact_id=artifact_id,
                    section="summary",
                    quote=fallback_text[:240],
                )
            ]
        if not selected_chunks:
            selected_chunks = chunks[:2]
        return [
            SourceRef(
                paper_id=paper_id,
                artifact_id=artifact_id,
                page=chunk.page,
                section=chunk.section,
                chunk_id=chunk.id,
                quote=chunk.text[:240],
            )
            for chunk in selected_chunks
        ]

    def _evidence_for_candidate_ids(
        self,
        candidate_passages: tuple[IngestExtractionCandidate, ...],
        candidate_ids: tuple[str, ...],
        fallback_text: str,
    ) -> list[str]:
        if not candidate_ids:
            return [fallback_text[:240]]
        selected = [candidate.excerpt for candidate in candidate_passages if candidate.candidate_id in candidate_ids]
        return selected or [fallback_text[:240]]

    def _coerce_relation_type(self, relation_type: str) -> RelationType:
        try:
            return RelationType(relation_type)
        except ValueError:
            return RelationType.COMPLEMENTS

    def _build_paper_memory(
        self,
        paper: Paper,
        artifact_id: str,
        chunks: list[Chunk],
        context_text: str,
    ) -> PaperMemory:
        return self._memory_factory.build_paper_memory(paper, artifact_id, chunks, context_text)

    def _build_relation_memory(self, paper: Paper, related_paper: Paper, context_text: str) -> RelationMemory:
        return self._memory_factory.build_relation_memory(paper, related_paper, context_text)

    def _build_open_question_memory(
        self,
        paper: Paper,
        chunks: list[Chunk],
        context_text: str,
        related_paper: Paper | None,
    ) -> OpenQuestionMemory:
        return self._memory_factory.build_open_question_memory(paper, chunks, context_text, related_paper)

    def _build_source_refs(self, paper_id: str, artifact_id: str, chunks: list[Chunk], context_text: str) -> list[SourceRef]:
        return self._memory_factory.build_source_refs(paper_id, artifact_id, chunks, context_text)

    def _select_related_paper(self, session_id: str, current_paper_id: str) -> Paper | None:
        return None

    def _global_related_paper_ids(self, current_paper_id: str) -> list[str]:
        related_ids: list[str] = []
        for memory in self._memory_repository.list_all_paper_memories():
            if memory.paper_id != current_paper_id:
                related_ids.append(memory.paper_id)
        for memory in self._memory_repository.list_all_relation_memories():
            if memory.source_paper != current_paper_id:
                related_ids.append(memory.source_paper)
            if memory.target_paper != current_paper_id:
                related_ids.append(memory.target_paper)
        return list(dict.fromkeys(related_ids))

    def _build_context_text(self, paper: Paper, chunks: list[Chunk]) -> str:
        chunk_text = " ".join(chunk.text for chunk in chunks if chunk.text.strip())
        parts = [paper.title, paper.abstract or "", chunk_text]
        return " ".join(part for part in parts if part).strip()

    def _require_session(self, session_id: str):
        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        return session

    def _require_paper(self, paper_id: str) -> Paper:
        paper = self._paper_repository.get_by_id(paper_id)
        if paper is None:
            raise EntityNotFoundError("Paper", paper_id)
        return paper
