"""Contract tests that compare InMemory and SQLite storage behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_agent.api.deps import create_repository_bundle
from research_agent.adapters.storage import SQLiteMemoryRepository, SQLiteSessionRepository
from research_agent.domain.enums import ArtifactKind, MessageType, RelationType, SourceType, TaskRunStatus
from research_agent.domain.models import (
    Artifact,
    Chunk,
    Message,
    OpenQuestionMemory,
    Paper,
    PaperMemory,
    RelationMemory,
    Session,
    SessionDocument,
    TaskRun,
    TimelineEvent,
)
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore


def _fixed_timestamp() -> datetime:
    return datetime(2026, 4, 23, 12, 0, tzinfo=UTC)


def _exercise_bundle(bundle) -> dict[str, object]:
    timestamp = _fixed_timestamp()
    session = Session(
        id="session-1",
        title="Contract Session",
        created_at=timestamp,
        updated_at=timestamp,
        status="active",
    )
    message = Message(
        id="message-1",
        session_id=session.id,
        type=MessageType.FOLLOWUP_QUERY,
        content="What changed?",
        created_at=timestamp,
        status="accepted",
    )
    paper = Paper(
        id="paper-1",
        canonical_key=build_canonical_key(arxiv_id="2401.12345"),
        title="Memory-Routed Agents",
        authors=["A. Author"],
        abstract="Abstract",
        year=2024,
        arxiv_id="2401.12345",
        pdf_fingerprint="fingerprint-1",
    )
    related_paper = Paper(
        id="paper-2",
        canonical_key=build_canonical_key(pdf_checksum="checksum-paper-2"),
        title="Related Paper",
        authors=["B. Author"],
    )
    artifact = Artifact(
        id="artifact-1",
        kind=ArtifactKind.ARXIV_PDF,
        uri_or_path="https://arxiv.org/abs/2401.12345",
        checksum="checksum-1",
        page_count=12,
    )
    document = SessionDocument(
        id="document-1",
        session_id=session.id,
        paper_id=paper.id,
        source_type=SourceType.ARXIV,
        artifact_id=artifact.id,
        added_at=timestamp,
    )
    run = TaskRun(
        id="run-1",
        session_id=session.id,
        message_id=message.id,
        status=TaskRunStatus.RUNNING,
        step_count=2,
        started_at=timestamp,
        finished_at=timestamp,
        finish_reason="finished",
    )
    event = TimelineEvent(
        id="event-1",
        session_id=session.id,
        run_id=run.id,
        event_type="run_finished",
        summary="query run completed",
        related_memory_ids=["memory-1"],
        related_paper_ids=[paper.id],
        created_at=timestamp,
    )
    paper_memory = PaperMemory(
        id="paper-memory-1",
        paper_id=paper.id,
        problem="Problem",
        method="Method",
        key_results=["Result"],
        limitations=["Limit"],
        novelty_claim="Novel",
        source_refs=[],
        confidence=ConfidenceScore(value=0.8),
        updated_at=timestamp,
    )
    relation_memory = RelationMemory(
        id="relation-memory-1",
        source_paper=paper.id,
        target_paper=related_paper.id,
        relation_type=RelationType.COMPLEMENTS,
        summary="Complements baseline.",
        evidence=["Evidence"],
        confidence=ConfidenceScore(value=0.7),
        updated_at=timestamp,
    )
    open_question_memory = OpenQuestionMemory(
        id="open-question-1",
        unresolved_question="What happens under shift?",
        related_papers=[paper.id],
        why_open=["No robustness data."],
        possible_followup=["Run robustness tests."],
        confidence=ConfidenceScore(value=0.4),
        updated_at=timestamp,
    )
    chunk = Chunk(
        id="chunk-1",
        paper_id=paper.id,
        artifact_id=artifact.id,
        text="Chunk text",
        page=1,
        section="Intro",
    )

    bundle.sessions.save(session)
    bundle.messages.save(message)
    bundle.papers.save(paper)
    bundle.papers.save(related_paper)
    bundle.artifacts.save(artifact)
    bundle.sessions.save_document(document)
    bundle.memories.upsert_paper_memory(paper_memory)
    bundle.memories.upsert_relation_memory(relation_memory)
    bundle.memories.upsert_open_question_memory(open_question_memory)
    bundle.chunks.save_many([chunk])
    bundle.trace.save_run(run)
    bundle.timeline.save(event)

    return {
        "session": bundle.sessions.get_by_id(session.id),
        "sessions": bundle.sessions.list_all(),
        "message": bundle.messages.get_by_id(message.id),
        "messages": bundle.messages.list_by_session(session.id),
        "paper_by_id": bundle.papers.get_by_id(paper.id),
        "paper_by_key": bundle.papers.get_by_canonical_key(paper.canonical_key),
        "related_paper": bundle.papers.get_by_id(related_paper.id),
        "artifact": bundle.artifacts.get_by_id(artifact.id),
        "document": bundle.sessions.list_documents(session.id),
        "paper_memories": bundle.memories.list_all_paper_memories(),
        "relation_memories": bundle.memories.list_all_relation_memories(),
        "open_question_memories": bundle.memories.list_all_open_question_memories(),
        "chunks": bundle.chunks.list_by_paper_ids([paper.id]),
        "run": bundle.trace.get_run(run.id),
        "runs": bundle.trace.list_runs_by_session(session.id),
        "timeline": bundle.timeline.list_by_session(session.id),
    }


def _exercise_upserts_and_null_round_trips(bundle) -> dict[str, object]:
    timestamp = _fixed_timestamp()
    later_timestamp = datetime(2026, 4, 23, 12, 30, tzinfo=UTC)
    session = Session(
        id="session-2",
        title="Initial Session",
        created_at=timestamp,
        updated_at=timestamp,
        status="active",
    )
    message = Message(
        id="message-2",
        session_id=session.id,
        type=MessageType.INGEST_PDF,
        content="Initial content",
        created_at=timestamp,
        status="accepted",
    )
    paper = Paper(
        id="paper-3",
        canonical_key=build_canonical_key(pdf_checksum="checksum-paper-3"),
        title="Initial Paper",
    )
    artifact = Artifact(
        id="artifact-3",
        kind=ArtifactKind.LOCAL_PDF,
        uri_or_path="/tmp/paper-3.pdf",
        checksum="checksum-3",
    )
    document = SessionDocument(
        id="document-2",
        session_id=session.id,
        paper_id=paper.id,
        source_type=SourceType.PDF,
        artifact_id=artifact.id,
        added_at=timestamp,
    )
    run = TaskRun(
        id="run-2",
        session_id=session.id,
        message_id=message.id,
        status=TaskRunStatus.PENDING,
        step_count=0,
        started_at=timestamp,
        finished_at=None,
        finish_reason=None,
    )
    event = TimelineEvent(
        id="event-2",
        session_id=session.id,
        run_id=None,
        event_type="run_started",
        summary="run started",
        related_memory_ids=[],
        related_paper_ids=[],
        created_at=timestamp,
    )
    paper_memory = PaperMemory(
        id="paper-memory-2",
        paper_id=paper.id,
        key_results=[],
        source_refs=[],
        confidence=ConfidenceScore(value=0.6),
        updated_at=timestamp,
    )
    relation_memory = RelationMemory(
        id="relation-memory-2",
        source_paper=paper.id,
        target_paper="paper-4",
        relation_type=RelationType.SIMILAR_TO,
        summary="Initial relation summary.",
        evidence=[],
        confidence=ConfidenceScore(value=0.6),
        updated_at=timestamp,
    )
    open_question_memory = OpenQuestionMemory(
        id="open-question-2",
        unresolved_question="Initial question?",
        related_papers=[],
        why_open=[],
        possible_followup=[],
        confidence=ConfidenceScore(value=0.3),
        updated_at=timestamp,
    )
    chunk = Chunk(
        id="chunk-2",
        paper_id=paper.id,
        artifact_id=artifact.id,
        text="Initial chunk text",
        page=None,
        section=None,
    )

    bundle.sessions.save(session)
    bundle.messages.save(message)
    bundle.papers.save(paper)
    bundle.papers.save(
        Paper(
            id="paper-4",
            canonical_key=build_canonical_key(pdf_checksum="checksum-paper-4"),
            title="Target Paper",
        )
    )
    bundle.artifacts.save(artifact)
    bundle.sessions.save_document(document)
    bundle.trace.save_run(run)
    bundle.timeline.save(event)
    bundle.memories.upsert_paper_memory(paper_memory)
    bundle.memories.upsert_relation_memory(relation_memory)
    bundle.memories.upsert_open_question_memory(open_question_memory)
    bundle.chunks.save_many([chunk])

    bundle.sessions.save(
        Session(
            id=session.id,
            title="Updated Session",
            created_at=timestamp,
            updated_at=later_timestamp,
            status="active",
        )
    )
    bundle.messages.save(
        Message(
            id=message.id,
            session_id=session.id,
            type=MessageType.FOLLOWUP_QUERY,
            content="Updated content",
            created_at=timestamp,
            status="accepted",
        )
    )
    bundle.papers.save(
        Paper(
            id=paper.id,
            canonical_key=paper.canonical_key,
            title="Updated Paper",
            abstract="Updated abstract",
            year=2026,
            arxiv_id="2401.54321",
            pdf_fingerprint="fingerprint-updated",
        )
    )
    bundle.artifacts.save(
        Artifact(
            id=artifact.id,
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path="/tmp/paper-3-updated.pdf",
            checksum="checksum-3-updated",
            page_count=12,
        )
    )
    bundle.sessions.save_document(
        SessionDocument(
            id=document.id,
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
            added_at=later_timestamp,
        )
    )
    bundle.trace.save_run(
        TaskRun(
            id=run.id,
            session_id=session.id,
            message_id=message.id,
            status=TaskRunStatus.FINISHED,
            step_count=4,
            started_at=timestamp,
            finished_at=later_timestamp,
            finish_reason="done",
        )
    )
    bundle.timeline.save(
        TimelineEvent(
            id=event.id,
            session_id=session.id,
            run_id=run.id,
            event_type="run_finished",
            summary="run completed",
            related_memory_ids=["paper-memory-2"],
            related_paper_ids=[paper.id],
            created_at=later_timestamp,
        )
    )
    bundle.memories.upsert_paper_memory(
        PaperMemory(
            id=paper_memory.id,
            paper_id=paper.id,
            problem="Updated problem",
            method="Updated method",
            key_results=["Updated result"],
            limitations=["Updated limit"],
            novelty_claim="Updated novelty",
            source_refs=[],
            confidence=ConfidenceScore(value=0.9),
            updated_at=later_timestamp,
        )
    )
    bundle.papers.save(
        Paper(
            id="paper-4",
            canonical_key=build_canonical_key(pdf_checksum="checksum-paper-4"),
            title="Target Paper",
        )
    )
    bundle.memories.upsert_relation_memory(
        RelationMemory(
            id=relation_memory.id,
            source_paper=paper.id,
            target_paper="paper-4",
            relation_type=RelationType.COMPLEMENTS,
            summary="Updated relation summary.",
            evidence=["Updated evidence"],
            confidence=ConfidenceScore(value=0.95),
            updated_at=later_timestamp,
        )
    )
    bundle.memories.upsert_open_question_memory(
        OpenQuestionMemory(
            id=open_question_memory.id,
            unresolved_question="Updated question?",
            related_papers=[paper.id],
            why_open=["Updated reason"],
            possible_followup=["Updated followup"],
            confidence=ConfidenceScore(value=0.7),
            updated_at=later_timestamp,
        )
    )
    bundle.chunks.save_many(
        [
            Chunk(
                id=chunk.id,
                paper_id=paper.id,
                artifact_id=artifact.id,
                text="Updated chunk text",
                page=3,
                section="Method",
            )
        ]
    )

    return {
        "session": bundle.sessions.get_by_id(session.id),
        "message": bundle.messages.get_by_id(message.id),
        "paper": bundle.papers.get_by_id(paper.id),
        "artifact": bundle.artifacts.get_by_id(artifact.id),
        "document": bundle.sessions.list_documents(session.id),
        "run": bundle.trace.get_run(run.id),
        "event": bundle.timeline.list_by_session(session.id),
        "paper_memories": bundle.memories.list_all_paper_memories(),
        "relation_memories": bundle.memories.list_all_relation_memories(),
        "open_question_memories": bundle.memories.list_all_open_question_memories(),
        "chunks": bundle.chunks.list_by_paper_ids([paper.id]),
    }


@pytest.mark.parametrize(
    ("backend_name", "sqlite_path"),
    [
        ("memory", None),
        ("sqlite", "sqlite-backend-contract.db"),
    ],
)
def test_core_storage_backends_support_the_same_basic_behavior(
    backend_name: str,
    sqlite_path: str | None,
    tmp_path,
) -> None:
    path = tmp_path / sqlite_path if sqlite_path is not None else None
    bundle = create_repository_bundle(storage_backend=backend_name, sqlite_path=path)
    snapshot = _exercise_bundle(bundle)

    assert snapshot["session"] is not None
    assert snapshot["message"] is not None
    assert snapshot["paper_by_id"] == snapshot["paper_by_key"]
    assert snapshot["related_paper"] is not None
    assert snapshot["artifact"] is not None
    assert snapshot["document"] == [snapshot["document"][0]]
    assert snapshot["paper_memories"] == [snapshot["paper_memories"][0]]
    assert snapshot["relation_memories"] == [snapshot["relation_memories"][0]]
    assert snapshot["open_question_memories"] == [snapshot["open_question_memories"][0]]
    assert snapshot["chunks"] == [snapshot["chunks"][0]]
    assert snapshot["run"] is not None
    assert snapshot["runs"] == [snapshot["runs"][0]]
    assert snapshot["timeline"] == [snapshot["timeline"][0]]


def test_inmemory_and_sqlite_snapshots_match(tmp_path) -> None:
    in_memory_snapshot = _exercise_bundle(create_repository_bundle(storage_backend="memory"))
    sqlite_snapshot = _exercise_bundle(create_repository_bundle(storage_backend="sqlite", sqlite_path=tmp_path / "storage.db"))

    assert in_memory_snapshot == sqlite_snapshot


@pytest.mark.parametrize(
    ("backend_name", "sqlite_path"),
    [
        ("memory", None),
        ("sqlite", "sqlite-upsert-contract.db"),
    ],
)
def test_core_storage_backends_support_upserts_and_null_round_trips(
    backend_name: str,
    sqlite_path: str | None,
    tmp_path,
) -> None:
    path = tmp_path / sqlite_path if sqlite_path is not None else None
    bundle = create_repository_bundle(storage_backend=backend_name, sqlite_path=path)
    snapshot = _exercise_upserts_and_null_round_trips(bundle)

    assert snapshot["session"].title == "Updated Session"
    assert snapshot["message"].content == "Updated content"
    assert snapshot["paper"].abstract == "Updated abstract"
    assert snapshot["paper"].year == 2026
    assert snapshot["paper"].arxiv_id == "2401.54321"
    assert snapshot["paper"].pdf_fingerprint == "fingerprint-updated"
    assert snapshot["artifact"].page_count == 12
    assert snapshot["document"][0].added_at == datetime(2026, 4, 23, 12, 30, tzinfo=UTC)
    assert snapshot["run"].status == TaskRunStatus.FINISHED
    assert snapshot["run"].finished_at == datetime(2026, 4, 23, 12, 30, tzinfo=UTC)
    assert snapshot["run"].finish_reason == "done"
    assert snapshot["event"][0].summary == "run completed"
    assert snapshot["paper_memories"][0].problem == "Updated problem"
    assert snapshot["relation_memories"][0].summary == "Updated relation summary."
    assert snapshot["open_question_memories"][0].unresolved_question == "Updated question?"
    assert snapshot["chunks"][0].text == "Updated chunk text"


def test_repository_bundle_defaults_to_sqlite(tmp_path) -> None:
    bundle = create_repository_bundle(sqlite_path=tmp_path / "default.db")

    assert isinstance(bundle.sessions, SQLiteSessionRepository)
    assert isinstance(bundle.memories, SQLiteMemoryRepository)
