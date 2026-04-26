"""Repository tests for in-memory mock adapters."""

from __future__ import annotations

from research_agent.adapters.storage import (
    InMemoryArtifactRepository,
    InMemoryChunkRepository,
    InMemoryMemoryRepository,
    InMemoryMessageRepository,
    InMemoryPaperRepository,
    InMemorySessionRepository,
    InMemoryTimelineRepository,
    InMemoryTraceRepository,
)
from research_agent.domain.enums import ArtifactKind, RelationType
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
from research_agent.domain.ports import (
    ArtifactRepositoryPort,
    ChunkRepositoryPort,
    MemoryRepositoryPort,
    MessageRepositoryPort,
    PaperRepositoryPort,
    SessionRepositoryPort,
    TimelineRepositoryPort,
    TraceRepositoryPort,
)
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore


def test_inmemory_repositories_satisfy_protocols() -> None:
    assert isinstance(InMemorySessionRepository(), SessionRepositoryPort)
    assert isinstance(InMemoryMessageRepository(), MessageRepositoryPort)
    assert isinstance(InMemoryPaperRepository(), PaperRepositoryPort)
    assert isinstance(InMemoryArtifactRepository(), ArtifactRepositoryPort)
    assert isinstance(InMemoryMemoryRepository(), MemoryRepositoryPort)
    assert isinstance(InMemoryTraceRepository(), TraceRepositoryPort)
    assert isinstance(InMemoryTimelineRepository(), TimelineRepositoryPort)
    assert isinstance(InMemoryChunkRepository(), ChunkRepositoryPort)


def test_session_repository_persists_sessions_and_documents() -> None:
    repository = InMemorySessionRepository()
    session = repository.save(Session(title="Test Session"))
    document = repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type="pdf",
            artifact_id="artifact-1",
        )
    )

    assert repository.get_by_id(session.id) == session
    assert repository.list_all() == [session]
    assert repository.list_documents(session.id) == [document]


def test_memory_repository_filters_by_related_papers() -> None:
    repository = InMemoryMemoryRepository()
    paper_memory = repository.upsert_paper_memory(
        PaperMemory(
            paper_id="paper-1",
            key_results=["result"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    relation_memory = repository.upsert_relation_memory(
        RelationMemory(
            source_paper="paper-1",
            target_paper="paper-2",
            relation_type=RelationType.SIMILAR_TO,
            summary="Similar setup.",
            confidence=ConfidenceScore(value=0.6),
        )
    )
    open_question_memory = repository.upsert_open_question_memory(
        OpenQuestionMemory(
            unresolved_question="What changes under distribution shift?",
            related_papers=["paper-1"],
            why_open=["Missing evaluation."],
            possible_followup=["Run robustness benchmarks."],
            confidence=ConfidenceScore(value=0.4),
        )
    )

    assert repository.list_paper_memories_for_papers(["paper-1"]) == [paper_memory]
    assert repository.list_relation_memories_for_papers(["paper-1"]) == [relation_memory]
    assert repository.list_open_question_memories_for_papers(["paper-1"]) == [open_question_memory]


def test_content_and_runtime_repositories_store_records() -> None:
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()

    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Memory-Routed Agents",
        )
    )
    artifact = artifact_repository.save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path="C:/tmp/paper.pdf",
            checksum="abc123",
            page_count=12,
        )
    )
    chunks = chunk_repository.save_many(
        [Chunk(id="chunk-1", paper_id=paper.id, artifact_id=artifact.id, text="Chunk text")]
    )
    run = trace_repository.save_run(TaskRun(session_id="session-1", message_id="message-1"))
    event = timeline_repository.save(
        TimelineEvent(session_id="session-1", run_id=run.id, event_type="run_started", summary="Run started.")
    )

    assert paper_repository.get_by_canonical_key(paper.canonical_key) == paper
    assert artifact_repository.get_by_id(artifact.id) == artifact
    assert chunk_repository.list_by_paper_ids([paper.id]) == chunks
    assert trace_repository.get_run(run.id) == run
    assert trace_repository.list_runs_by_session("session-1") == [run]
    assert timeline_repository.list_by_session("session-1") == [event]
