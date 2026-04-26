"""Structured memory repository ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory


@runtime_checkable
class MemoryRepositoryPort(Protocol):
    """Storage boundary for all three memory types."""

    def upsert_paper_memory(self, memory: PaperMemory) -> PaperMemory:
        """Create or replace a paper memory record."""

    def upsert_relation_memory(self, memory: RelationMemory) -> RelationMemory:
        """Create or replace a relation memory record."""

    def upsert_open_question_memory(self, memory: OpenQuestionMemory) -> OpenQuestionMemory:
        """Create or replace an open-question memory record."""

    def list_paper_memories_for_papers(self, paper_ids: Sequence[str]) -> Sequence[PaperMemory]:
        """List paper memories related to paper ids."""

    def list_relation_memories_for_papers(self, paper_ids: Sequence[str]) -> Sequence[RelationMemory]:
        """List relation memories touching paper ids."""

    def list_open_question_memories_for_papers(
        self,
        paper_ids: Sequence[str],
    ) -> Sequence[OpenQuestionMemory]:
        """List open-question memories related to paper ids."""

    def list_all_paper_memories(self) -> Sequence[PaperMemory]:
        """List all paper memories."""

    def list_all_relation_memories(self) -> Sequence[RelationMemory]:
        """List all relation memories."""

    def list_all_open_question_memories(self) -> Sequence[OpenQuestionMemory]:
        """List all open-question memories."""

    def delete_paper_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        """Delete paper memories tied to any of the given papers."""

    def delete_relation_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        """Delete relation memories tied to any of the given papers."""

    def delete_open_question_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        """Delete open-question memories tied to any of the given papers."""

    def delete_paper_memory(self, memory_id: str) -> bool:
        """Delete a single paper memory by id."""

    def delete_relation_memory(self, memory_id: str) -> bool:
        """Delete a single relation memory by id."""

    def delete_open_question_memory(self, memory_id: str) -> bool:
        """Delete a single open-question memory by id."""
