"""Thin context reranking between candidate generation and answer synthesis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable
import re

from research_agent.domain.models import Chunk, OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.domain.policies import CONTEXT_CANDIDATE_TOP_K, CONTEXT_RERANK_TOP_K


def _query_terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _match_score_text(haystack: str, query: str) -> int:
    return sum(1 for term in _query_terms(query) if term in haystack.lower())


def _memory_text(memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
    if isinstance(memory, PaperMemory):
        source_quotes = " ".join(ref.quote or "" for ref in memory.source_refs)
        return " ".join(
            [
                memory.problem or "",
                memory.method or "",
                " ".join(memory.key_results),
                " ".join(memory.limitations),
                memory.novelty_claim or "",
                source_quotes,
            ]
        ).lower()
    if isinstance(memory, RelationMemory):
        return " ".join([memory.summary, " ".join(memory.evidence), memory.source_paper, memory.target_paper]).lower()
    return " ".join([memory.unresolved_question, " ".join(memory.why_open), " ".join(memory.possible_followup)]).lower()


def _memory_evidence_score(memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> int:
    if isinstance(memory, PaperMemory):
        if any(ref.quote for ref in memory.source_refs):
            return 2
        return 1 if memory.source_refs else 0
    if isinstance(memory, RelationMemory):
        return 2 if memory.evidence else 0
    return 1 if memory.why_open or memory.possible_followup else 0


def _memory_type_priority(
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


def _memory_sort_key(
    memory: PaperMemory | RelationMemory | OpenQuestionMemory,
    query: str,
) -> tuple[int, int, int, float, datetime]:
    return (
        _match_score_text(_memory_text(memory), query),
        _memory_evidence_score(memory),
        _memory_type_priority(memory, _query_terms(query)),
        memory.confidence.value,
        memory.updated_at,
    )


def _chunk_section_priority(section: str | None) -> int:
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


def _chunk_sort_key(chunk: Chunk, query: str) -> tuple[int, int, int, str]:
    return (
        _match_score_text(chunk.text, query),
        _chunk_section_priority(chunk.section),
        -(chunk.page or 10**9),
        chunk.id,
    )


@dataclass(frozen=True, slots=True)
class MemoryRerankResult:
    """Result of reranking a bounded memory candidate pool."""

    candidates: tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]
    selected: tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    selection_source: str
    fallback_used: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class ChunkRerankResult:
    """Result of reranking a bounded source chunk candidate pool."""

    candidates: tuple[Chunk, ...]
    selected: tuple[Chunk, ...]
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    selection_source: str
    fallback_used: bool
    rationale: str


@runtime_checkable
class ContextRerankerClient(Protocol):
    """Pluggable reranker boundary for future model-backed selection."""

    def rank_memory_indices(self, query: str, candidates: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory], top_k: int) -> Sequence[int]:
        """Return the ranked candidate indices for memory selection."""

    def rank_chunk_indices(self, query: str, candidates: Sequence[Chunk], top_k: int) -> Sequence[int]:
        """Return the ranked candidate indices for source-chunk selection."""


class HeuristicContextRerankerClient:
    """Deterministic reranker used until a real model-backed reranker is wired."""

    def rank_memory_indices(
        self,
        query: str,
        candidates: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        top_k: int,
    ) -> Sequence[int]:
        scored = sorted(
            enumerate(candidates),
            key=lambda item: _memory_sort_key(item[1], query),
            reverse=True,
        )
        return [index for index, _ in scored[:top_k]]

    def rank_chunk_indices(self, query: str, candidates: Sequence[Chunk], top_k: int) -> Sequence[int]:
        scored = sorted(
            enumerate(candidates),
            key=lambda item: _chunk_sort_key(item[1], query),
            reverse=True,
        )
        return [index for index, _ in scored[:top_k]]


class ContextRerankService:
    """Candidate-bounded reranking with a deterministic fallback path."""

    def __init__(self, reranker_client: ContextRerankerClient | None = None) -> None:
        self._reranker_client = reranker_client or HeuristicContextRerankerClient()

    def rerank_memories(
        self,
        query: str,
        candidates: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        top_k: int = CONTEXT_RERANK_TOP_K,
    ) -> MemoryRerankResult:
        candidate_pool = tuple(candidates[:CONTEXT_CANDIDATE_TOP_K])
        candidate_ids = tuple(memory.id for memory in candidate_pool)
        selected_indices, fallback_used = self._select_indices(
            query=query,
            candidates=candidate_pool,
            top_k=top_k,
            ranker=self._reranker_client.rank_memory_indices,
            fallback_sort_key=lambda memory: _memory_sort_key(memory, query),
        )
        selected = tuple(candidate_pool[index] for index in selected_indices)
        rationale = self._build_rationale("memory", len(candidate_pool), selected, fallback_used)
        return MemoryRerankResult(
            candidates=candidate_pool,
            selected=selected,
            candidate_ids=candidate_ids,
            selected_ids=tuple(memory.id for memory in selected),
            selection_source="rule_fallback" if fallback_used else "model",
            fallback_used=fallback_used,
            rationale=rationale,
        )

    def rerank_chunks(
        self,
        query: str,
        candidates: Sequence[Chunk],
        top_k: int = CONTEXT_RERANK_TOP_K,
    ) -> ChunkRerankResult:
        candidate_pool = tuple(candidates[:CONTEXT_CANDIDATE_TOP_K])
        candidate_ids = tuple(chunk.id for chunk in candidate_pool)
        selected_indices, fallback_used = self._select_indices(
            query=query,
            candidates=candidate_pool,
            top_k=top_k,
            ranker=self._reranker_client.rank_chunk_indices,
            fallback_sort_key=lambda chunk: _chunk_sort_key(chunk, query),
        )
        selected = tuple(candidate_pool[index] for index in selected_indices)
        rationale = self._build_rationale("chunk", len(candidate_pool), selected, fallback_used)
        return ChunkRerankResult(
            candidates=candidate_pool,
            selected=selected,
            candidate_ids=candidate_ids,
            selected_ids=tuple(chunk.id for chunk in selected),
            selection_source="rule_fallback" if fallback_used else "model",
            fallback_used=fallback_used,
            rationale=rationale,
        )

    def _select_indices(
        self,
        *,
        query: str,
        candidates: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory] | Sequence[Chunk],
        top_k: int,
        ranker,
        fallback_sort_key,
    ) -> tuple[tuple[int, ...], bool]:
        if not candidates:
            return (), True
        try:
            proposed_indices = self._validate_indices(ranker(query, candidates, top_k), len(candidates), top_k)
        except Exception:
            proposed_indices = ()
        if len(proposed_indices) < min(top_k, len(candidates)):
            fallback_indices = self._fallback_indices(candidates, fallback_sort_key, top_k)
            return fallback_indices, True
        return proposed_indices, False

    def _validate_indices(self, indices: Sequence[int], candidate_count: int, top_k: int) -> tuple[int, ...]:
        validated: list[int] = []
        seen: set[int] = set()
        for index in indices:
            if index < 0 or index >= candidate_count:
                continue
            if index in seen:
                continue
            validated.append(index)
            seen.add(index)
            if len(validated) >= top_k:
                break
        return tuple(validated)

    def _fallback_indices(
        self,
        candidates: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory] | Sequence[Chunk],
        fallback_sort_key,
        top_k: int,
    ) -> tuple[int, ...]:
        scored = sorted(enumerate(candidates), key=lambda item: fallback_sort_key(item[1]), reverse=True)
        return tuple(index for index, _ in scored[: min(top_k, len(candidates))])

    def _build_rationale(self, kind: str, candidate_count: int, selected: Sequence[object], fallback_used: bool) -> str:
        if not selected:
            return f"no_{kind}_candidates_available"
        if fallback_used:
            return f"rule_fallback_used_for_{kind}_rerank"
        return f"model_reranked_{kind}_candidates_from_{candidate_count}_to_{len(selected)}"

__all__ = [
    "ChunkRerankResult",
    "ContextRerankService",
    "ContextRerankerClient",
    "HeuristicContextRerankerClient",
    "MemoryRerankResult",
]
