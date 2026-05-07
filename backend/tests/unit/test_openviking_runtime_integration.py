"""Tests for OpenViking dual-write, retrieval, and deletion integration."""

from __future__ import annotations

from research_agent.adapters.openviking import (
    OpenVikingMemoryRecord,
    OpenVikingMessageRecord,
    SurfaceBackedOpenVikingMemoryGateway,
    build_inmemory_openviking_surface_bundle,
)
from research_agent.adapters.storage import (
    InMemoryChunkRepository,
    InMemoryMemoryRepository,
    InMemoryMessageRepository,
    InMemoryPaperRepository,
    InMemorySessionRepository,
    InMemoryTimelineRepository,
    InMemoryTraceRepository,
)
from research_agent.domain.enums import MessageType, RelationType, SourceType
from research_agent.domain.models import Message, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SessionDocument
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services import (
    DeletionService,
    MessageIntakeRequest,
    MessageIntakeService,
    RetrievalService,
    SessionService,
    TaskRunService,
)
from research_agent.services.ingest_analysis_service import MemoryAnalysisResult
from research_agent.runtime.ingest_extraction import IngestPaperSummaryDraft


def test_inmemory_openviking_bundle_supports_message_and_memory_search() -> None:
    bundle = build_inmemory_openviking_surface_bundle()
    bundle.sessions.ensure_session("session-1", title="Session 1")
    bundle.messages.mirror_message(
        OpenVikingMessageRecord(
            session_id="session-1",
            message_id="message-1",
            role="user",
            content="What changed after the new accuracy result?",
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="memory-1",
            memory_kind="paper_memory",
            session_id="session-1",
            paper_id="paper-1",
            payload={"key_results": ["accuracy improved over the baseline"]},
        )
    )

    snapshot = bundle.sessions.commit_session("session-1")
    hits = bundle.memories.search_session_memory("session-1", "accuracy baseline", top_k=5)

    assert snapshot.message_count == 1
    assert snapshot.memory_count == 1
    assert bundle.messages.list_messages("session-1")[0].content.startswith("What changed")
    assert hits[0].item_id == "memory-1"


def test_message_intake_service_dual_writes_user_messages_to_openviking() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("Dual Write")
    session_repository.save(session)
    service = MessageIntakeService(
        task_run_service=TaskRunService(
            session_repository=session_repository,
            message_repository=message_repository,
            trace_repository=trace_repository,
        ),
        openviking_bundle=bundle,
    )

    accepted = service.submit(session.id, MessageIntakeRequest(text="How does it compare?"))

    mirrored_messages = bundle.messages.list_messages(session.id)
    snapshot = bundle.sessions.ensure_session(session.id)

    assert accepted.message.type is MessageType.FOLLOWUP_QUERY
    assert len(mirrored_messages) == 1
    assert mirrored_messages[0].message_id == accepted.message.id
    assert mirrored_messages[0].metadata["message_type"] == MessageType.FOLLOWUP_QUERY.value
    assert snapshot.message_count == 1


def test_retrieval_service_prefers_openviking_hits_over_local_sort_when_present() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("Retrieval")
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
            key_results=["accuracy improved over the baseline"],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="What explains the accuracy delta?",
            related_papers=["paper-1"],
            confidence=ConfidenceScore(value=0.4),
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="open-question-1",
            memory_kind="open_question_memory",
            session_id=session.id,
            paper_id="paper-1",
            payload={"unresolved_question": "accuracy accuracy delta remains open"},
        )
    )
    service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        openviking_memory_surface=bundle.memories,
    )

    result = service.retrieve_session_memories(session.id, "accuracy delta", top_k=1)

    assert len(result.memories) == 1
    assert result.memories[0].id == "open-question-1"


def test_surface_backed_gateway_mirrors_three_memory_types() -> None:
    bundle = build_inmemory_openviking_surface_bundle()
    gateway = SurfaceBackedOpenVikingMemoryGateway(
        memory_surface=bundle.memories,
        session_surface=bundle.sessions,
    )
    paper = Paper(
        id="paper-1",
        canonical_key=build_canonical_key(arxiv_id="2401.12345"),
        title="Mirrored Paper",
    )
    analysis = MemoryAnalysisResult(
        paper_memory=PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            key_results=["stronger accuracy"],
            confidence=ConfidenceScore(value=0.8),
        ),
        relation_memory=RelationMemory(
            id="relation-memory-1",
            source_paper=paper.id,
            target_paper="paper-2",
            relation_type=RelationType.COMPARES_WITH,
            summary="Compares with prior work.",
            confidence=ConfidenceScore(value=0.6),
        ),
        open_question_memory=OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="Does it scale?",
            related_papers=[paper.id],
            confidence=ConfidenceScore(value=0.5),
        ),
        paper_summary=IngestPaperSummaryDraft(
            what_it_is_about="The paper proposes a mirrored memory flow.",
            problem_solved="It reduces duplicate reasoning across context stores.",
            new_ideas=("A mirrored ingest memory flow.",),
            limitations=("It still needs live-server validation.",),
            suggestions_or_questions=("Test the mirrored flow against a live server.",),
            evidence_candidate_ids=("paper-memory-1",),
            confidence=0.6,
        ),
        context_summary="summary",
    )

    gateway.mirror_ingest_result(session_id="session-1", paper=paper, analysis=analysis)

    session_hits = bundle.memories.search_session_memory("session-1", "scale accuracy compare", top_k=10)
    hit_ids = {hit.item_id for hit in session_hits}

    assert {"paper-memory-1", "relation-memory-1", "open-question-1"}.issubset(hit_ids)


def test_deletion_service_propagates_to_openviking_bundle() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("Delete")
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
        Message(
            session_id=session.id,
            type=MessageType.FOLLOWUP_QUERY,
            content="delete me",
        )
    )
    bundle.messages.mirror_message(
        OpenVikingMessageRecord(
            session_id=session.id,
            message_id=message.id,
            role="user",
            content=message.content,
        )
    )
    memory = memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id="paper-1",
            key_results=["accuracy"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id=memory.id,
            memory_kind="paper_memory",
            session_id=session.id,
            paper_id="paper-1",
            payload={"key_results": ["accuracy"]},
        )
    )
    service = DeletionService(
        session_repository=session_repository,
        message_repository=message_repository,
        memory_repository=memory_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        openviking_bundle=bundle,
    )

    result = service.delete_session(session.id)

    assert result.mirrored_to_openviking is True
    assert bundle.messages.list_messages(session.id) == ()
    assert len(bundle.memories.search_session_memory(session.id, "accuracy", top_k=5)) == 1
