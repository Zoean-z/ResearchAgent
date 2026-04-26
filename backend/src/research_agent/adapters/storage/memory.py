"""In-memory storage for structured research memories."""

from __future__ import annotations

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.domain.ports import MemoryRepositoryPort


class InMemoryMemoryRepository(MemoryRepositoryPort):
    """Simple in-memory storage for all three memory types."""

    def __init__(self) -> None:
        self._paper_memories: dict[str, PaperMemory] = {}
        self._relation_memories: dict[str, RelationMemory] = {}
        self._open_question_memories: dict[str, OpenQuestionMemory] = {}

    def upsert_paper_memory(self, memory: PaperMemory) -> PaperMemory:
        self._paper_memories[memory.id] = memory
        return memory

    def upsert_relation_memory(self, memory: RelationMemory) -> RelationMemory:
        self._relation_memories[memory.id] = memory
        return memory

    def upsert_open_question_memory(self, memory: OpenQuestionMemory) -> OpenQuestionMemory:
        self._open_question_memories[memory.id] = memory
        return memory

    def list_paper_memories_for_papers(self, paper_ids: list[str]) -> list[PaperMemory]:
        paper_id_set = set(paper_ids)
        return [memory for memory in self._paper_memories.values() if memory.paper_id in paper_id_set]

    def list_relation_memories_for_papers(self, paper_ids: list[str]) -> list[RelationMemory]:
        paper_id_set = set(paper_ids)
        return [
            memory
            for memory in self._relation_memories.values()
            if memory.source_paper in paper_id_set or memory.target_paper in paper_id_set
        ]

    def list_open_question_memories_for_papers(self, paper_ids: list[str]) -> list[OpenQuestionMemory]:
        paper_id_set = set(paper_ids)
        return [
            memory
            for memory in self._open_question_memories.values()
            if paper_id_set.intersection(memory.related_papers)
        ]

    def list_all_paper_memories(self) -> list[PaperMemory]:
        return list(self._paper_memories.values())

    def list_all_relation_memories(self) -> list[RelationMemory]:
        return list(self._relation_memories.values())

    def list_all_open_question_memories(self) -> list[OpenQuestionMemory]:
        return list(self._open_question_memories.values())

    def delete_paper_memories_for_papers(self, paper_ids: list[str]) -> int:
        paper_id_set = set(paper_ids)
        removed = [memory_id for memory_id, memory in self._paper_memories.items() if memory.paper_id in paper_id_set]
        for memory_id in removed:
            self._paper_memories.pop(memory_id, None)
        return len(removed)

    def delete_relation_memories_for_papers(self, paper_ids: list[str]) -> int:
        paper_id_set = set(paper_ids)
        removed = [
            memory_id
            for memory_id, memory in self._relation_memories.items()
            if memory.source_paper in paper_id_set or memory.target_paper in paper_id_set
        ]
        for memory_id in removed:
            self._relation_memories.pop(memory_id, None)
        return len(removed)

    def delete_open_question_memories_for_papers(self, paper_ids: list[str]) -> int:
        paper_id_set = set(paper_ids)
        removed = [
            memory_id
            for memory_id, memory in self._open_question_memories.items()
            if paper_id_set.intersection(memory.related_papers)
        ]
        for memory_id in removed:
            self._open_question_memories.pop(memory_id, None)
        return len(removed)

    def delete_paper_memory(self, memory_id: str) -> bool:
        return self._paper_memories.pop(memory_id, None) is not None

    def delete_relation_memory(self, memory_id: str) -> bool:
        return self._relation_memories.pop(memory_id, None) is not None

    def delete_open_question_memory(self, memory_id: str) -> bool:
        return self._open_question_memories.pop(memory_id, None) is not None
