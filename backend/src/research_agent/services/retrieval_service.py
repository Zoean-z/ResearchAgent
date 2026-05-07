"""Thin retrieval service for mock follow-up query planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
import re

from research_agent.adapters.openviking import (
    NoopOpenVikingMemorySurface,
    OpenVikingMemorySurface,
    OpenVikingRetrievalAdapter,
)
from research_agent.domain.models import Chunk, OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.domain.ports import ChunkRepositoryPort, MemoryRepositoryPort, SessionRepositoryPort
from research_agent.domain.policies import (
    GLOBAL_MEMORY_TOP_K,
    SESSION_MEMORY_TOP_K,
    should_reread_source,
)
from research_agent.services.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class MemoryRetrievalResult:
    """Mock retrieval result for a single retrieval stage."""

    memories: tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]
    coverage_score: float
    matched_query_terms: tuple[str, ...]
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """Combined session/global retrieval result and reread decision."""

    session_memories: MemoryRetrievalResult
    global_memories: MemoryRetrievalResult
    related_paper_ids: tuple[str, ...]
    should_reread_source: bool
    reread_reason: str
    memory_confidence: float


@dataclass(frozen=True, slots=True)
class SourceRereadResult:
    """Source passages selected after the memory-first pass is insufficient."""

    chunks: tuple[Chunk, ...]
    coverage_score: float
    matched_query_terms: tuple[str, ...]
    selection_reasons: tuple[str, ...]


class RetrievalService:
    """Mock retrieval service that preserves the future memory-first ordering."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        chunk_repository: ChunkRepositoryPort | None = None,
        openviking_memory_surface: OpenVikingMemorySurface | None = None,
        openviking_retrieval_adapter: OpenVikingRetrievalAdapter | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._memory_repository = memory_repository
        self._chunk_repository = chunk_repository or _NullChunkRepository()
        self._openviking_retrieval_adapter = openviking_retrieval_adapter or OpenVikingRetrievalAdapter(
            session_repository=session_repository,
            memory_repository=memory_repository,
            memory_surface=openviking_memory_surface or NoopOpenVikingMemorySurface(),
        )

    def retrieve_session_memories(self, session_id: str, query: str, top_k: int = SESSION_MEMORY_TOP_K) -> MemoryRetrievalResult:
        """Return session-related memories using the current session document bindings."""

        session = self._require_session(session_id)
        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session.id)]
        candidates = [
            *self._memory_repository.list_paper_memories_for_papers(paper_ids),
            *self._memory_repository.list_relation_memories_for_papers(paper_ids),
            *self._memory_repository.list_open_question_memories_for_papers(paper_ids),
        ]
        search = self._openviking_retrieval_adapter.search_session_memory(
            session_id=session.id,
            query=query,
            top_k=top_k,
        )
        return self._build_result(candidates, query=query, top_k=top_k, openviking_hits=search.hits)

    def retrieve_global_memories(
        self,
        query: str,
        related_paper_ids: Sequence[str] | None = None,
        top_k: int = GLOBAL_MEMORY_TOP_K,
    ) -> MemoryRetrievalResult:
        """Return global memories with an optional paper-id filter."""

        paper_ids = list(related_paper_ids or [])
        paper_memories = self._memory_repository.list_all_paper_memories()
        relation_memories = self._memory_repository.list_all_relation_memories()
        open_question_memories = self._memory_repository.list_all_open_question_memories()
        if paper_ids:
            paper_id_set = set(paper_ids)
            paper_memories = [memory for memory in paper_memories if memory.paper_id in paper_id_set]
            relation_memories = [
                memory
                for memory in relation_memories
                if memory.source_paper in paper_id_set or memory.target_paper in paper_id_set
            ]
            open_question_memories = [
                memory for memory in open_question_memories if paper_id_set.intersection(memory.related_papers)
            ]
        candidates = [*paper_memories, *relation_memories, *open_question_memories]
        search = self._openviking_retrieval_adapter.search_global_memory(
            query=query,
            related_paper_ids=paper_ids or None,
            top_k=top_k,
        )
        return self._build_result(candidates, query=query, top_k=top_k, openviking_hits=search.hits)

    def build_retrieval_plan(
        self,
        session_id: str,
        query: str,
        top_k: int = SESSION_MEMORY_TOP_K,
    ) -> RetrievalPlan:
        """Build the memory-first retrieval plan used by query execution."""

        session_result = self.retrieve_session_memories(session_id=session_id, query=query, top_k=top_k)
        related_paper_ids = tuple(self._collect_related_paper_ids(session_id))
        global_result = self.retrieve_global_memories(
            query=query,
            related_paper_ids=related_paper_ids or None,
            top_k=top_k,
        )
        combined_memories = (*session_result.memories, *global_result.memories)
        memory_confidence = self._max_confidence(combined_memories)
        has_paper_memory = any(isinstance(memory, PaperMemory) for memory in combined_memories)
        has_evidence_quote = any(self._memory_has_evidence(memory) for memory in combined_memories)
        has_comparison_target = bool(related_paper_ids) or any(isinstance(memory, RelationMemory) for memory in combined_memories)
        reread_required = should_reread_source(
            has_relevant_paper_memory=has_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence,
        )
        reread_reason = self._build_reread_reason(
            has_relevant_paper_memory=has_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence,
            reread_required=reread_required,
        )
        return RetrievalPlan(
            session_memories=session_result,
            global_memories=global_result,
            related_paper_ids=related_paper_ids,
            should_reread_source=reread_required,
            reread_reason=reread_reason,
            memory_confidence=memory_confidence,
        )

    def retrieve_source_passages(
        self,
        session_id: str,
        query: str,
        related_paper_ids: Sequence[str] | None = None,
        top_k: int = GLOBAL_MEMORY_TOP_K,
    ) -> SourceRereadResult:
        """Return source passages from stored chunks when reread is needed."""

        paper_ids = list(related_paper_ids or self._collect_related_paper_ids(session_id))
        chunks = list(self._chunk_repository.list_by_paper_ids(paper_ids))
        scored_chunks = sorted(chunks, key=lambda chunk: self._chunk_sort_key(chunk, query), reverse=True)
        selected = tuple(scored_chunks[:top_k])
        matched_terms = tuple(self._matched_terms_for_text(query, selected, lambda chunk: chunk.text))
        selection_reasons = tuple(self._chunk_selection_reason(chunk, query) for chunk in selected)
        coverage_score = self._coverage_score(selected, top_k)
        return SourceRereadResult(
            chunks=selected,
            coverage_score=coverage_score,
            matched_query_terms=matched_terms,
            selection_reasons=selection_reasons,
        )

    def _require_session(self, session_id: str):
        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        return session

    def _collect_related_paper_ids(self, session_id: str) -> list[str]:
        session = self._require_session(session_id)
        return [document.paper_id for document in self._session_repository.list_documents(session.id)]

    def _build_result(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        *,
        query: str,
        top_k: int,
        openviking_hits=(),
    ) -> MemoryRetrievalResult:
        scored_memories = sorted(memories, key=lambda memory: self._memory_sort_key(memory, query), reverse=True)
        selected = self._select_with_openviking_hits(scored_memories, query=query, top_k=top_k, openviking_hits=openviking_hits)
        matched_terms = tuple(self._matched_terms(query, selected))
        selection_reasons = tuple(self._memory_selection_reason(memory, query) for memory in selected)
        coverage_score = self._coverage_score(selected, top_k)
        return MemoryRetrievalResult(
            memories=selected,
            coverage_score=coverage_score,
            matched_query_terms=matched_terms,
            selection_reasons=selection_reasons,
        )

    def _select_with_openviking_hits(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        *,
        query: str,
        top_k: int,
        openviking_hits,
    ) -> tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]:
        hit_by_id = {
            hit.item_id: hit
            for hit in openviking_hits
            if getattr(hit, "item_id", None) and getattr(hit, "item_kind", "").endswith("memory")
        }
        if not hit_by_id:
            return tuple(memories[:top_k])
        matched = [memory for memory in memories if memory.id in hit_by_id]
        matched.sort(
            key=lambda memory: (
                hit_by_id[memory.id].score,
                self._memory_sort_key(memory, query),
            ),
            reverse=True,
        )
        remaining = [memory for memory in memories if memory.id not in hit_by_id]
        return tuple([*matched[:top_k], *remaining][:top_k])

    def _match_score(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory, query: str) -> int:
        return self._match_score_text(self._memory_text(memory), query)

    def _match_score_text(self, haystack: str, query: str) -> int:
        return sum(1 for term in self._query_terms(query) if term in haystack.lower())

    def _memory_text(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
        if isinstance(memory, PaperMemory):
            source_quotes = " ".join(ref.quote or "" for ref in memory.source_refs)
            return " ".join(
                [memory.problem or "", memory.method or "", " ".join(memory.key_results), " ".join(memory.limitations), memory.novelty_claim or "", source_quotes]
            ).lower()
        if isinstance(memory, RelationMemory):
            return " ".join([memory.summary, " ".join(memory.evidence), memory.source_paper, memory.target_paper]).lower()
        return " ".join([memory.unresolved_question, " ".join(memory.why_open), " ".join(memory.possible_followup)]).lower()

    def _query_terms(self, query: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", query.lower())

    def _matched_terms(
        self,
        query: str,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    ) -> list[str]:
        return self._matched_terms_for_text(query, memories, self._memory_text)

    def _matched_terms_for_text(
        self,
        query: str,
        items: Sequence[Chunk | PaperMemory | RelationMemory | OpenQuestionMemory],
        text_getter: Callable[[Chunk | PaperMemory | RelationMemory | OpenQuestionMemory], str],
    ) -> list[str]:
        terms = self._query_terms(query)
        matched: list[str] = []
        for term in terms:
            if any(term in text_getter(item).lower() for item in items):
                matched.append(term)
        return matched

    def _memory_updated_at(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> datetime:
        return memory.updated_at

    def _memory_sort_key(
        self,
        memory: PaperMemory | RelationMemory | OpenQuestionMemory,
        query: str,
    ) -> tuple[int, int, int, float, datetime]:
        return (
            self._match_score(memory, query),
            self._memory_evidence_score(memory),
            self._memory_type_priority(memory, self._query_terms(query)),
            memory.confidence.value,
            self._memory_updated_at(memory),
        )

    def _memory_evidence_score(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> int:
        if isinstance(memory, PaperMemory):
            if any(ref.quote for ref in memory.source_refs):
                return 2
            return 1 if memory.source_refs else 0
        if isinstance(memory, RelationMemory):
            return 2 if memory.evidence else 0
        return 1 if memory.why_open or memory.possible_followup else 0

    def _memory_type_priority(
        self,
        memory: PaperMemory | RelationMemory | OpenQuestionMemory,
        query_terms: Sequence[str],
    ) -> int:
        query_terms_set = set(query_terms)
        compare_terms = {"compare", "comparison", "baseline", "versus", "vs", "against", "similar", "conflict", "complements"}
        uncertainty_terms = {"open", "question", "future", "followup", "follow", "why", "limitation", "limitations", "uncertain"}
        paper_terms = {"paper", "method", "result", "results", "accuracy", "problem", "novel", "novelty"}
        if isinstance(memory, RelationMemory):
            return 3 if query_terms_set & compare_terms else 2
        if isinstance(memory, OpenQuestionMemory):
            return 3 if query_terms_set & uncertainty_terms else 1
        return 3 if query_terms_set & paper_terms else 2

    def _memory_selection_reason(
        self,
        memory: PaperMemory | RelationMemory | OpenQuestionMemory,
        query: str,
    ) -> str:
        matched_terms = self._matched_terms(query, [memory])
        evidence_score = self._memory_evidence_score(memory)
        confidence = f"{memory.confidence.value:.2f}"
        if isinstance(memory, PaperMemory):
            memory_type = "paper_memory"
        elif isinstance(memory, RelationMemory):
            memory_type = "relation_memory"
        else:
            memory_type = "open_question_memory"
        matched_text = ",".join(matched_terms) if matched_terms else "none"
        return f"type={memory_type}; matched_terms={matched_text}; evidence_score={evidence_score}; confidence={confidence}"

    def _chunk_sort_key(self, chunk: Chunk, query: str) -> tuple[int, int, int, str]:
        return (
            self._match_score_text(chunk.text, query),
            self._chunk_section_priority(chunk.section),
            -(chunk.page or 10**9),
            chunk.id,
        )

    def _chunk_section_priority(self, section: str | None) -> int:
        if section is None:
            return 0
        normalized = section.lower()
        if "abstract" in normalized:
            return 4
        if "intro" in normalized:
            return 3
        if "method" in normalized:
            return 2
        if "result" in normalized or "discussion" in normalized or "conclusion" in normalized:
            return 1
        return 0

    def _chunk_selection_reason(self, chunk: Chunk, query: str) -> str:
        matched_terms = self._matched_terms_for_text(query, [chunk], lambda item: item.text)
        section = chunk.section or "unknown-section"
        page = str(chunk.page) if chunk.page is not None else "unknown-page"
        matched_text = ",".join(matched_terms) if matched_terms else "none"
        return f"matched_terms={matched_text}; section={section}; page={page}"

    def _coverage_score(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        top_k: int,
    ) -> float:
        if not memories:
            return 0.0
        return min(1.0, len(memories) / max(1, top_k))

    def _max_confidence(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    ) -> float:
        if not memories:
            return 0.0
        return max(memory.confidence.value for memory in memories)

    def _memory_has_evidence(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> bool:
        if isinstance(memory, PaperMemory):
            return any(ref.quote for ref in memory.source_refs)
        if isinstance(memory, RelationMemory):
            return any(memory.evidence)
        return any(memory.why_open) or any(memory.possible_followup)

    def _build_reread_reason(
        self,
        *,
        has_relevant_paper_memory: bool,
        has_evidence_quote: bool,
        has_comparison_target: bool,
        memory_confidence: float,
        reread_required: bool,
    ) -> str:
        if not reread_required:
            return "memory_is_sufficient_for_mock_answer"
        reasons: list[str] = []
        if not has_relevant_paper_memory:
            reasons.append("missing_relevant_paper_memory")
        if not has_evidence_quote:
            reasons.append("missing_evidence_quote")
        if not has_comparison_target:
            reasons.append("missing_comparison_target")
        if memory_confidence < 0.6:
            reasons.append("low_memory_confidence")
        return ",".join(reasons) if reasons else "mock_reread_required"


class _NullChunkRepository(ChunkRepositoryPort):
    """Fallback chunk repository used when source reread is not wired."""

    def save_many(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        return chunks

    def list_by_paper_ids(self, paper_ids: Sequence[str]) -> Sequence[Chunk]:
        return []

    def list_by_artifact_id(self, artifact_id: str) -> Sequence[Chunk]:
        return []
