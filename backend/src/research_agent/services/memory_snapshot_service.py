"""Thin read service for session memory snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.domain.ports import MemoryRepositoryPort, SessionRepositoryPort
from research_agent.services.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Grouped session memory snapshot returned by the read service."""

    paper_memories: tuple[PaperMemory, ...]
    relation_memories: tuple[RelationMemory, ...]
    open_question_memories: tuple[OpenQuestionMemory, ...]


class MemorySnapshotService:
    """Read-side memory snapshot composition over repository ports."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        memory_repository: MemoryRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._memory_repository = memory_repository

    def get_snapshot(self, session_id: str) -> MemorySnapshot:
        """Return the grouped memory snapshot for a session."""

        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)

        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session.id)]
        return MemorySnapshot(
            paper_memories=tuple(self._memory_repository.list_paper_memories_for_papers(paper_ids)),
            relation_memories=tuple(self._memory_repository.list_relation_memories_for_papers(paper_ids)),
            open_question_memories=tuple(self._memory_repository.list_open_question_memories_for_papers(paper_ids)),
        )
