"""Tests for deletion service boundaries."""

from __future__ import annotations

import pytest

from research_agent.adapters.storage import (
    InMemoryMemoryRepository,
    InMemoryMessageRepository,
    InMemorySessionRepository,
    InMemoryTimelineRepository,
    InMemoryTraceRepository,
)
from research_agent.domain.enums import MessageType, RelationType, SourceType, TaskRunStatus
from research_agent.domain.models import Message, OpenQuestionMemory, PaperMemory, RelationMemory, SessionDocument, TimelineEvent, TraceStep, TaskRun
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services import DeletionService, EntityNotFoundError, SessionService


def _build_service_bundle():
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    service = DeletionService(
        session_repository=session_repository,
        message_repository=message_repository,
        memory_repository=memory_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )
    return service, session_repository, message_repository, memory_repository, trace_repository, timeline_repository


def test_delete_session_tombstones_session_and_clears_runtime_state() -> None:
    service, session_repository, message_repository, memory_repository, trace_repository, timeline_repository = _build_service_bundle()
    session = SessionService(session_repository=session_repository).create_session("Delete session")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    message = message_repository.save(
        Message(session_id=session.id, type=MessageType.FOLLOWUP_QUERY, content="What changed?")
    )
    run = trace_repository.save_run(TaskRun(session_id=session.id, message_id=message.id, status=TaskRunStatus.RUNNING))
    trace_repository.save_step(TraceStep(run_id=run.id, action="compose_answer"))
    timeline_repository.save(TimelineEvent(session_id=session.id, run_id=run.id, event_type="step", summary="Tracked"))
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id="paper-1",
            key_results=["Improved accuracy"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    memory_repository.upsert_relation_memory(
        RelationMemory(
            id="relation-memory-1",
            source_paper="paper-1",
            target_paper="paper-2",
            relation_type=RelationType.COMPARES_WITH,
            summary="Compares on the same benchmark.",
            confidence=ConfidenceScore(value=0.7),
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="Does it generalize?",
            related_papers=["paper-1"],
            confidence=ConfidenceScore(value=0.5),
        )
    )

    result = service.delete_session(session.id)

    assert result.deleted_messages == 1
    assert result.deleted_runs == 1
    assert result.deleted_timeline_events == 1
    assert result.deleted_memories == 3
    assert result.mirrored_to_openviking is False
    assert session_repository.get_by_id(session.id).status == "deleted"
    assert message_repository.list_by_session(session.id) == []
    assert trace_repository.get_run(run.id) is None
    assert trace_repository.list_steps(run.id) == []
    assert timeline_repository.list_by_session(session.id) == []
    assert memory_repository.list_paper_memories_for_papers(["paper-1"]) == []
    assert memory_repository.list_relation_memories_for_papers(["paper-1"]) == []
    assert memory_repository.list_open_question_memories_for_papers(["paper-1"]) == []


@pytest.mark.parametrize(
    ("memory_kind", "memory_id"),
    [
        ("paper_memory", "paper-memory-1"),
        ("relation_memory", "relation-memory-1"),
        ("open_question_memory", "open-question-1"),
    ],
)
def test_delete_memory_removes_only_the_target_item(memory_kind: str, memory_id: str) -> None:
    service, session_repository, _, memory_repository, _, _ = _build_service_bundle()
    session = SessionService(session_repository=session_repository).create_session("Delete memory")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id="paper-1",
            key_results=["Improved accuracy"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    memory_repository.upsert_relation_memory(
        RelationMemory(
            id="relation-memory-1",
            source_paper="paper-1",
            target_paper="paper-2",
            relation_type=RelationType.COMPARES_WITH,
            summary="Compares on the same benchmark.",
            confidence=ConfidenceScore(value=0.7),
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="Does it generalize?",
            related_papers=["paper-1"],
            confidence=ConfidenceScore(value=0.5),
        )
    )

    result = service.delete_memory(session.id, memory_kind, memory_id)

    assert result.deleted is True
    assert result.mirrored_to_openviking is False
    paper_memory_ids = {memory.id for memory in memory_repository.list_all_paper_memories()}
    relation_memory_ids = {memory.id for memory in memory_repository.list_all_relation_memories()}
    open_question_memory_ids = {memory.id for memory in memory_repository.list_all_open_question_memories()}

    if memory_kind == "paper_memory":
        assert memory_id not in paper_memory_ids
        assert relation_memory_ids == {"relation-memory-1"}
        assert open_question_memory_ids == {"open-question-1"}
    elif memory_kind == "relation_memory":
        assert memory_id not in relation_memory_ids
        assert paper_memory_ids == {"paper-memory-1"}
        assert open_question_memory_ids == {"open-question-1"}
    else:
        assert memory_id not in open_question_memory_ids
        assert paper_memory_ids == {"paper-memory-1"}
        assert relation_memory_ids == {"relation-memory-1"}


def test_delete_memory_requires_known_session() -> None:
    service, _, _, _, _, _ = _build_service_bundle()

    with pytest.raises(EntityNotFoundError):
        service.delete_memory("missing-session", "paper_memory", "memory-1")
