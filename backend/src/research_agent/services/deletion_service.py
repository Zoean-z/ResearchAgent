"""Thin deletion service for sessions and memories."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.adapters.openviking import (
    NoopOpenVikingMemorySurface,
    NoopOpenVikingMessageSurface,
    NoopOpenVikingSessionSurface,
    OpenVikingAdapterSurfaceBundle,
)
from research_agent.domain.models import Message, OpenQuestionMemory, PaperMemory, RelationMemory, Session
from research_agent.domain.ports import MemoryRepositoryPort, MessageRepositoryPort, SessionRepositoryPort, TimelineRepositoryPort, TraceRepositoryPort
from research_agent.services.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class DeleteSessionResult:
    """Structured result for a session deletion request."""

    session: Session
    deleted_documents: int
    deleted_messages: int
    deleted_runs: int
    deleted_timeline_events: int
    deleted_memories: int
    mirrored_to_openviking: bool


@dataclass(frozen=True, slots=True)
class DeleteMemoryResult:
    """Structured result for a memory deletion request."""

    session_id: str
    memory_kind: str
    memory_id: str
    deleted: bool
    mirrored_to_openviking: bool


class DeletionService:
    """Host-controlled deletion orchestration for dialogue and memory."""

    def __init__(
        self,
        *,
        session_repository: SessionRepositoryPort,
        message_repository: MessageRepositoryPort,
        memory_repository: MemoryRepositoryPort,
        trace_repository: TraceRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        openviking_bundle: OpenVikingAdapterSurfaceBundle | None = None,
    ) -> None:
        self._session_repository = session_repository
        self._message_repository = message_repository
        self._memory_repository = memory_repository
        self._trace_repository = trace_repository
        self._timeline_repository = timeline_repository
        self._openviking_bundle = openviking_bundle or OpenVikingAdapterSurfaceBundle()

    def delete_session(self, session_id: str) -> DeleteSessionResult:
        """Delete a session's dialogue state without deleting shared memory records."""

        session = self._require_session(session_id)
        messages = list(self._message_repository.list_by_session(session_id))

        deleted_runs = self._trace_repository.delete_runs_for_session(session_id)
        deleted_documents = self._session_repository.delete_documents(session_id)
        deleted_messages = self._message_repository.delete_by_session(session_id)
        deleted_timeline_events = self._timeline_repository.delete_by_session(session_id)
        deleted_session = self._session_repository.delete(session_id)

        self._mirror_message_deletions(session_id, messages)

        if deleted_session is None:
            raise EntityNotFoundError("Session", session_id)
        return DeleteSessionResult(
            session=deleted_session,
            deleted_documents=deleted_documents,
            deleted_messages=deleted_messages,
            deleted_runs=deleted_runs,
            deleted_timeline_events=deleted_timeline_events,
            deleted_memories=0,
            mirrored_to_openviking=self._openviking_active(),
        )

    def delete_memory(self, session_id: str, memory_kind: str, memory_id: str) -> DeleteMemoryResult:
        """Delete a single memory item by id.

        The memory is located by id directly so that global memories
        (belonging to papers outside the current session) can also be deleted.
        """

        self._require_session(session_id)
        if memory_kind == "paper_memory":
            memory = self._find_paper_memory_by_id(memory_id)
            deleted = self._memory_repository.delete_paper_memory(memory_id)
        elif memory_kind == "relation_memory":
            memory = self._find_relation_memory_by_id(memory_id)
            deleted = self._memory_repository.delete_relation_memory(memory_id)
        elif memory_kind == "open_question_memory":
            memory = self._find_open_question_memory_by_id(memory_id)
            deleted = self._memory_repository.delete_open_question_memory(memory_id)
        else:
            raise ValueError(f"Unsupported memory kind: {memory_kind}")

        if not deleted:
            raise EntityNotFoundError(memory_kind, memory_id)

        self._openviking_bundle.memories.delete_memory(memory.id)
        return DeleteMemoryResult(
            session_id=session_id,
            memory_kind=memory_kind,
            memory_id=memory_id,
            deleted=True,
            mirrored_to_openviking=self._openviking_active(),
        )

    def _find_paper_memory(self, session_id: str, memory_id: str) -> PaperMemory:
        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session_id)]
        memory = next((item for item in self._memory_repository.list_paper_memories_for_papers(paper_ids) if item.id == memory_id), None)
        if memory is None:
            raise EntityNotFoundError("PaperMemory", memory_id)
        return memory

    def _find_relation_memory(self, session_id: str, memory_id: str) -> RelationMemory:
        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session_id)]
        memory = next((item for item in self._memory_repository.list_relation_memories_for_papers(paper_ids) if item.id == memory_id), None)
        if memory is None:
            raise EntityNotFoundError("RelationMemory", memory_id)
        return memory

    def _find_open_question_memory(self, session_id: str, memory_id: str) -> OpenQuestionMemory:
        paper_ids = [document.paper_id for document in self._session_repository.list_documents(session_id)]
        memory = next(
            (item for item in self._memory_repository.list_open_question_memories_for_papers(paper_ids) if item.id == memory_id),
            None,
        )
        if memory is None:
            raise EntityNotFoundError("OpenQuestionMemory", memory_id)
        return memory

    def _find_paper_memory_by_id(self, memory_id: str) -> PaperMemory:
        memory = next((item for item in self._memory_repository.list_all_paper_memories() if item.id == memory_id), None)
        if memory is None:
            raise EntityNotFoundError("PaperMemory", memory_id)
        return memory

    def _find_relation_memory_by_id(self, memory_id: str) -> RelationMemory:
        memory = next((item for item in self._memory_repository.list_all_relation_memories() if item.id == memory_id), None)
        if memory is None:
            raise EntityNotFoundError("RelationMemory", memory_id)
        return memory

    def _find_open_question_memory_by_id(self, memory_id: str) -> OpenQuestionMemory:
        memory = next((item for item in self._memory_repository.list_all_open_question_memories() if item.id == memory_id), None)
        if memory is None:
            raise EntityNotFoundError("OpenQuestionMemory", memory_id)
        return memory

    def _mirror_message_deletions(self, session_id: str, messages: list[Message]) -> None:
        for message in messages:
            self._openviking_bundle.messages.delete_message(session_id, message.id)

    def _require_session(self, session_id: str) -> Session:
        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        return session

    def _openviking_active(self) -> bool:
        return not (
            isinstance(self._openviking_bundle.messages, NoopOpenVikingMessageSurface)
            and isinstance(self._openviking_bundle.memories, NoopOpenVikingMemorySurface)
            and isinstance(self._openviking_bundle.sessions, NoopOpenVikingSessionSurface)
        )
