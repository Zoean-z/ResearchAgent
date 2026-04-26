"""Explicit OpenViking retrieval and local-memory mapping boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from research_agent.adapters.openviking.surfaces import (
    NoopOpenVikingMemorySurface,
    OpenVikingMemorySurface,
    OpenVikingSearchHit,
)
from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.domain.ports import MemoryRepositoryPort, SessionRepositoryPort
from research_agent.tools.protocol import MemoryDescriptor


MemoryRecord = PaperMemory | RelationMemory | OpenQuestionMemory


@dataclass(frozen=True, slots=True)
class OpenVikingMemorySearchResult:
    """Structured OpenViking retrieval result plus local mapping metadata."""

    scope: str
    hits: tuple[OpenVikingSearchHit, ...]
    memory_descriptors: tuple[MemoryDescriptor, ...]
    matched_local_memory_ids: tuple[str, ...]
    matched_local_count: int


class OpenVikingRetrievalAdapter:
    """Boundary that searches OpenViking and maps results back to local memory ids."""

    def __init__(
        self,
        *,
        session_repository: SessionRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        memory_surface: OpenVikingMemorySurface | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._memory_repository = memory_repository
        self._memory_surface = memory_surface or NoopOpenVikingMemorySurface()

    def search_session_memory(
        self,
        *,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> OpenVikingMemorySearchResult:
        """Search OpenViking session memory and map hits to locally tracked memories."""

        hits = self._memory_surface.search_session_memory(session_id=session_id, query=query, top_k=top_k)
        local_memories = {memory.id: memory for memory in self._list_session_memories(session_id)}
        matched_ids = tuple(hit.item_id for hit in hits if hit.item_id in local_memories)
        return OpenVikingMemorySearchResult(
            scope="session",
            hits=hits,
            memory_descriptors=tuple(
                self._memory_descriptor(local_memories[hit.item_id], hit.score, query)
                for hit in hits
                if hit.item_id in local_memories
            ),
            matched_local_memory_ids=matched_ids,
            matched_local_count=len(matched_ids),
        )

    def search_global_memory(
        self,
        *,
        query: str,
        related_paper_ids: Sequence[str] | None = None,
        top_k: int = 5,
    ) -> OpenVikingMemorySearchResult:
        """Search OpenViking global memory and map hits to locally tracked memories."""

        paper_ids = list(related_paper_ids or [])
        hits = self._memory_surface.search_global_memory(
            query=query,
            related_paper_ids=paper_ids or None,
            top_k=top_k,
        )
        local_memories = {memory.id: memory for memory in self._list_global_memories(paper_ids or None)}
        matched_ids = tuple(hit.item_id for hit in hits if hit.item_id in local_memories)
        return OpenVikingMemorySearchResult(
            scope="global",
            hits=hits,
            memory_descriptors=tuple(
                self._memory_descriptor(local_memories[hit.item_id], hit.score, query)
                for hit in hits
                if hit.item_id in local_memories
            ),
            matched_local_memory_ids=matched_ids,
            matched_local_count=len(matched_ids),
        )

    def _list_session_memories(self, session_id: str) -> tuple[MemoryRecord, ...]:
        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session_id)]
        return self._list_memories_for_papers(paper_ids)

    def _list_global_memories(self, paper_ids: Sequence[str] | None) -> tuple[MemoryRecord, ...]:
        if not paper_ids:
            return (
                *self._memory_repository.list_all_paper_memories(),
                *self._memory_repository.list_all_relation_memories(),
                *self._memory_repository.list_all_open_question_memories(),
            )
        return self._list_memories_for_papers(paper_ids)

    def _list_memories_for_papers(self, paper_ids: Sequence[str]) -> tuple[MemoryRecord, ...]:
        return (
            *self._memory_repository.list_paper_memories_for_papers(paper_ids),
            *self._memory_repository.list_relation_memories_for_papers(paper_ids),
            *self._memory_repository.list_open_question_memories_for_papers(paper_ids),
        )

    def _memory_descriptor(self, memory: MemoryRecord, score: float, query: str) -> MemoryDescriptor:
        if isinstance(memory, PaperMemory):
            memory_type = "paper_memory"
            summary = " | ".join(
                part
                for part in [
                    memory.problem or memory.method or memory.novelty_claim or "paper memory",
                    memory.key_results[0] if memory.key_results else "",
                ]
                if part
            )
        elif isinstance(memory, RelationMemory):
            memory_type = "relation_memory"
            summary = f"{memory.relation_type.value}: {memory.summary}"
        else:
            memory_type = "open_question_memory"
            summary = memory.unresolved_question
        matched_terms = tuple(term for term in query.lower().split() if term and term in summary.lower())
        return MemoryDescriptor(
            memory_id=memory.id,
            memory_type=memory_type,
            summary=summary,
            confidence=score,
            matched_terms=matched_terms,
            selection_reason=f"openviking_scope={memory_type}; score={score:.2f}",
        )


__all__ = ["OpenVikingMemorySearchResult", "OpenVikingRetrievalAdapter"]
