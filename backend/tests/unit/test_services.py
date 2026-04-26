"""Unit tests for the thin application services."""

from __future__ import annotations

import pytest

from research_agent.adapters.llm import ModelBackedQueryToolPlannerClient, StaticStructuredPlannerTransport
from research_agent.adapters.llm.ingest_extraction import (
    ModelBackedIngestExtractionClient,
    StructuredIngestExtractionChoice,
    StructuredIngestExtractionPrompt,
    StructuredIngestOpenQuestionDraft,
    StructuredIngestPaperDraft,
    StructuredIngestPaperSummaryDraft,
    StructuredIngestRelationDraft,
)
from research_agent.adapters.openviking import build_inmemory_openviking_surface_bundle
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
from research_agent.domain.enums import ArtifactKind, MessageType, RelationType, SourceType, TaskRunStatus
from research_agent.domain.models import (
    Artifact,
    Chunk,
    Message,
    OpenQuestionMemory,
    Paper,
    PaperMemory,
    RelationMemory,
    SessionDocument,
    TimelineEvent,
    SourceRef,
    TraceStep,
)
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services import (
    AcceptedTaskRun,
    EntityNotFoundError,
    ContextRerankService,
    IngestExecutionService,
    IngestAnalysisService,
    IngestMaterializationService,
    MemoryExtractionService,
    MemorySnapshotService,
    MessageIntakeRequest,
    MessageIntakeService,
    MessageQueryService,
    QueryExecutionService,
    RetrievalService,
    SessionService,
    TaskRunService,
    TimelineQueryService,
    TraceQueryService,
)
from research_agent.services.ingest_materialization_service import IngestMaterializationService
from research_agent.runtime import QueryRuntimeService, TaskRuntimeService
from research_agent.runtime.ingest_extraction import IngestExtractionCandidate
from research_agent.runtime.agent_protocol import AgentActionType, AgentStopReason, AgentTurnDecision, AgentTurnRequest
from research_agent.tools import HeuristicQueryToolPlannerClient, InternalToolRegistry, QueryToolExecutor
from research_agent.tools import StaticFinalAnswerQueryAgentClient


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf_bytes(text: str) -> bytes:
    content_stream = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET\n".encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n" + content_stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(payload)
        if not payload.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"endobj\n")
    xref_start = len(output)
    output.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(b"trailer << /Root 1 0 R /Size 6 >>\nstartxref\n")
    output.extend(f"{xref_start}\n".encode("ascii"))
    output.extend(b"%%EOF\n")
    return bytes(output)


@pytest.fixture(autouse=True)
def _stub_arxiv_download(monkeypatch) -> None:
    monkeypatch.setattr(
        IngestMaterializationService,
        "_download_arxiv_pdf",
        lambda self, pdf_url, source_value: _build_minimal_pdf_bytes("ArXiv text that should be extracted."),
    )


def test_session_service_creates_and_reads_sessions() -> None:
    repository = InMemorySessionRepository()
    service = SessionService(session_repository=repository)

    created = service.create_session("Service Session")

    assert created.title == "Service Session"
    assert service.list_sessions() == [created]
    assert service.get_session(created.id) == created


def test_message_intake_service_classifies_all_supported_inputs() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    session = SessionService(session_repository=session_repository).create_session("Intake")
    session_repository.save(session)
    service = MessageIntakeService(
        task_run_service=TaskRunService(
            session_repository=session_repository,
            message_repository=message_repository,
            trace_repository=trace_repository,
        )
    )

    query_result = service.submit(session.id, MessageIntakeRequest(text="What changed?"))
    arxiv_result = service.submit(session.id, MessageIntakeRequest(arxiv_url="https://arxiv.org/abs/2401.12345"))
    pdf_result = service.submit(session.id, MessageIntakeRequest(file_path="C:/papers/example.pdf"))

    assert query_result.message_type.value == MessageType.FOLLOWUP_QUERY.value
    assert arxiv_result.message_type.value == MessageType.INGEST_ARXIV.value
    assert pdf_result.message_type.value == MessageType.INGEST_PDF.value
    assert [message.type.value for message in message_repository.list_by_session(session.id)] == [
        MessageType.FOLLOWUP_QUERY.value,
        MessageType.INGEST_ARXIV.value,
        MessageType.INGEST_PDF.value,
    ]


def test_internal_tool_registry_lists_the_first_batch_of_tools() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Registry paper",
        )
    )
    session = SessionService(session_repository=session_repository).create_session("Tools")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            key_results=["Improved accuracy"],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=paper.id,
                artifact_id="artifact-1",
                text="The method improves accuracy over the baseline.",
                page=1,
                section="Abstract",
            )
        ]
    )
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    context_rerank_service = ContextRerankService()
    memory_extraction_service = MemoryExtractionService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )
    registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=context_rerank_service,
        memory_extraction_service=memory_extraction_service,
    )

    tool_names = {tool.name for tool in registry.list_tools()}

    assert tool_names == {
        "register_paper",
        "extract_memories",
        "search_openviking_memory",
        "search_session_memory",
        "search_global_memory",
        "search_source_chunks",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    }
    assert registry.invoke("search_session_memory", session_id=session.id, query="accuracy", top_k=5).memories
    assert registry.invoke("read_source_passages", session_id=session.id, query="accuracy", related_paper_ids=[paper.id], top_k=3).selected


def test_message_query_service_requires_session() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    service = MessageQueryService(
        session_repository=session_repository,
        message_repository=message_repository,
    )

    with pytest.raises(EntityNotFoundError):
        service.list_messages("missing-session")

    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Messages"))
    message = message_repository.save(
        Message(
            session_id=session.id,
            type=MessageType.FOLLOWUP_QUERY,
            content="What changed after reading memory?",
        )
    )

    assert service.list_messages(session.id) == [message]


def test_timeline_query_service_requires_session() -> None:
    session_repository = InMemorySessionRepository()
    timeline_repository = InMemoryTimelineRepository()
    service = TimelineQueryService(
        session_repository=session_repository,
        timeline_repository=timeline_repository,
    )

    with pytest.raises(EntityNotFoundError):
        service.list_timeline("missing-session")

    session = SessionService(session_repository=session_repository).create_session("Timeline")
    session_repository.save(session)
    event = timeline_repository.save(
        TimelineEvent(
            session_id=session.id,
            run_id="run-1",
            event_type="memory_updated",
            summary="Added paper memory.",
        )
    )

    assert service.list_timeline(session.id) == [event]


def test_memory_snapshot_service_groups_memories_by_session_documents() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    service = MemorySnapshotService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    session = SessionService(session_repository=session_repository).create_session("Snapshot")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )

    paper_memory = memory_repository.upsert_paper_memory(
        PaperMemory(
            paper_id="paper-1",
            key_results=["Improved benchmark score"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    relation_memory = memory_repository.upsert_relation_memory(
        RelationMemory(
            source_paper="paper-1",
            target_paper="paper-2",
            relation_type=RelationType.IMPROVES_ON,
            summary="Improves on the baseline.",
            evidence=["Higher accuracy on the same dataset."],
            confidence=ConfidenceScore(value=0.7),
        )
    )
    open_question_memory = memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            unresolved_question="Does performance hold under shift?",
            related_papers=["paper-1"],
            why_open=["No robustness evaluation."],
            possible_followup=["Run cross-domain testing."],
            confidence=ConfidenceScore(value=0.5),
        )
    )

    snapshot = service.get_snapshot(session.id)

    assert snapshot.paper_memories == (paper_memory,)
    assert snapshot.relation_memories == (relation_memory,)
    assert snapshot.open_question_memories == (open_question_memory,)


def test_memory_snapshot_service_requires_session() -> None:
    service = MemorySnapshotService(
        session_repository=InMemorySessionRepository(),
        memory_repository=InMemoryMemoryRepository(),
    )

    with pytest.raises(EntityNotFoundError):
        service.get_snapshot("missing-session")


def test_retrieval_service_builds_session_first_plan() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
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
            paper_id="paper-1",
            key_results=["Better score"],
            source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="evidence")],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    memory_repository.upsert_relation_memory(
        RelationMemory(
            source_paper="paper-1",
            target_paper="paper-2",
            relation_type=RelationType.COMPARES_WITH,
            summary="Compares on the same benchmark.",
            evidence=["same benchmark"],
            confidence=ConfidenceScore(value=0.7),
        )
    )

    service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    plan = service.build_retrieval_plan(session.id, "better score benchmark")

    assert plan.session_memories.memories
    assert plan.global_memories.memories
    assert plan.should_reread_source is False
    assert plan.reread_reason == "memory_is_sufficient_for_mock_answer"


def test_task_run_service_accepts_requests_and_persists_message_and_run() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    session = SessionService(session_repository=session_repository).create_session("Accept")
    session_repository.save(session)
    service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )

    accepted: AcceptedTaskRun = service.accept_followup_query(session.id, "How does this compare to baseline?")

    assert accepted.message.session_id == session.id
    assert accepted.message.type is MessageType.FOLLOWUP_QUERY
    assert accepted.message.status == "accepted"
    assert accepted.task_run.session_id == session.id
    assert accepted.task_run.message_id == accepted.message.id
    assert accepted.task_run.status.value == "pending"
    assert accepted.task_run.status is TaskRunStatus.PENDING


def test_task_run_service_get_run_is_session_scoped() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    session_service = SessionService(session_repository=session_repository)
    first_session = session_repository.save(session_service.create_session("First"))
    second_session = session_repository.save(session_service.create_session("Second"))
    service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = service.accept_arxiv_ingest(first_session.id, "https://arxiv.org/abs/2401.12345")

    assert service.get_run(first_session.id, accepted.task_run.id) == accepted.task_run

    with pytest.raises(EntityNotFoundError):
        service.get_run(second_session.id, accepted.task_run.id)


def test_query_execution_service_writes_mock_chain_for_running_run() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_repository.save(session_service.create_session("Execute"))
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
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.RUNNING
    assert "Mock answer for:" not in result.answer
    assert "\u8bb0\u5fc6" in result.answer
    assert result.used_memory_citations[0].memory_id == "paper-memory-1"
    assert result.used_memory_citations[0].selection_reason.startswith("type=paper_memory")
    assert "rerank_strategy=model" in result.used_memory_citations[0].selection_reason
    assert "当前记忆" in result.answer
    assert trace_repository.list_steps(accepted.task_run.id)[0].action == "retrieve_session_memories"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "compose_mock_answer",
    ]
    assert [event.summary for event in timeline_repository.list_by_session(session.id)] == [
        "checked session memory: paper_memory:paper-memory-1",
        "checked global memory: paper_memory:paper-memory-1",
        "reranked context candidates: paper_memory:paper-memory-1",
        "decided whether to reread",
        "query run completed",
    ]


def test_query_execution_service_persists_and_mirrors_assistant_answer() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    openviking_bundle = build_inmemory_openviking_surface_bundle()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Assistant Mirror"))
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
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        openviking_bundle=openviking_bundle,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    messages = message_repository.list_by_session(session.id)
    assert "Mock answer for:" not in result.answer
    assert "\u8bb0\u5fc6" in result.answer
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].content == result.answer
    mirrored = openviking_bundle.messages.list_messages(session.id)
    assert len(mirrored) == 1
    assert mirrored[0].role == "assistant"
    assert mirrored[0].content == result.answer
    assert mirrored[0].metadata["run_id"] == accepted.task_run.id


def test_query_execution_service_rereads_source_chunks_when_memory_is_insufficient() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_repository.save(session_service.create_session("Reread"))
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id="paper-1",
                artifact_id="artifact-1",
                text="The method improves accuracy over the baseline.",
                page=1,
                section="Abstract",
            )
        ]
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.RUNNING
    assert result.should_reread_source is True
    assert result.source_reread_chunks[0].chunk_id == "chunk-1"
    assert "matched_terms=" in result.source_reread_chunks[0].selection_reason
    assert "rerank_strategy=model" in result.source_reread_chunks[0].selection_reason
    assert "\u539f\u6587\u56de\u8bfb\u5230\u7684\u5173\u952e\u7247\u6bb5" in result.answer
    assert result.memory_selection_source == "rule_fallback"
    assert result.source_selection_source == "model"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "reread_source_passages",
        "compose_mock_answer",
    ]
    assert [event.summary for event in timeline_repository.list_by_session(session.id)] == [
        "checked session memory (no memories)",
        "checked global memory (no memories)",
        "reranked context candidates (no memories)",
        "decided whether to reread",
        "reread source passages: chunk-1",
        "query run completed",
    ]


def test_query_execution_service_allows_direct_final_answer_when_the_agent_chooses_to_finish_early() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Model Final Answer After Rerank")
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
            key_results=["Higher accuracy"],
            confidence=ConfidenceScore(value=0.2),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=InternalToolRegistry(
            paper_repository=paper_repository,
            retrieval_service=retrieval_service,
            context_rerank_service=ContextRerankService(),
            memory_extraction_service=MemoryExtractionService(
                session_repository=session_repository,
                paper_repository=paper_repository,
                chunk_repository=chunk_repository,
                memory_repository=memory_repository,
            ),
        ),
        query_tool_executor=QueryToolExecutor(
            InternalToolRegistry(
                paper_repository=paper_repository,
                retrieval_service=retrieval_service,
                context_rerank_service=ContextRerankService(),
                memory_extraction_service=MemoryExtractionService(
                    session_repository=session_repository,
                    paper_repository=paper_repository,
                    chunk_repository=chunk_repository,
                    memory_repository=memory_repository,
                ),
            )
        ),
        query_agent_client=StaticFinalAnswerQueryAgentClient("Agent-generated final answer."),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "Agent-generated final answer."
    assert result.should_reread_source is False
    assert result.source_reread_chunks == ()
    assert result.used_memory_citations == ()
    assert result.tool_calls[-1].action_type == "final_answer"
    assert len(result.tool_calls) == 1
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "compose_mock_answer",
    ]
    assert [event.summary for event in timeline_repository.list_by_session(session.id)] == [
        "checked session memory (no memories)",
        "checked global memory (no memories)",
        "reranked context candidates (no memories)",
        "decided whether to reread",
        "query run completed",
    ]


def test_query_execution_service_uses_host_controlled_tool_loop_when_executor_is_available() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_repository.save(session_service.create_session("Tool Loop"))
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Tool Loop Paper",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id=paper.id, artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    tool_registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert [call.tool_name for call in result.tool_calls] == [
        "search_session_memory",
        "search_global_memory",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    ]
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert steps[0].input_payload["planner_decision"]["selected_tool"] == "search_session_memory"
    assert steps[1].input_payload["planner_decision"]["selected_tool"] == "search_global_memory"
    assert steps[2].input_payload["planner_decision"]["selected_tool"] == "rerank_candidates"
    assert steps[3].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[4].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[5].input_payload["planner_decision"]["selected_tool"] == "compose_answer"


def test_query_execution_service_tool_loop_routes_to_source_reread_when_needed() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_repository.save(session_service.create_session("Tool Loop Reread"))
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.99999"),
            title="Tool Loop Reread Paper",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=paper.id,
                artifact_id="artifact-1",
                text="The method improves accuracy over the baseline.",
                page=1,
                section="Abstract",
            )
        ]
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    tool_registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert [call.tool_name for call in result.tool_calls] == [
        "search_session_memory",
        "search_global_memory",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    ]
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert steps[3].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[4].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"


def test_query_execution_service_accepts_model_backed_planner_and_records_planner_source() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Model Planner")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Model Planner Paper",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id=paper.id, artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    tool_registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
    )
    planner = ModelBackedQueryToolPlannerClient(
        transport=StaticStructuredPlannerTransport(
            tool_name="search_session_memory",
            rationale="model_prefers_session_memory_first",
        ),
        fallback=HeuristicQueryToolPlannerClient(),
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_tool_planner=planner,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.tool_calls[0].agent_name == "model_adapter"
    assert result.tool_calls[0].fallback_used is False
    assert trace_repository.list_steps(accepted.task_run.id)[0].input_payload["planner_decision"]["agent_name"] == "model_adapter"
    assert trace_repository.list_steps(accepted.task_run.id)[0].input_payload["planner_decision"]["fallback_used"] is False


def test_query_execution_service_accepts_agent_final_answer_path() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Agent Final Answer")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Agent Final Answer Paper",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id=paper.id, artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    tool_registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=StaticFinalAnswerQueryAgentClient("Agent-generated final answer."),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "Agent-generated final answer."
    assert result.tool_calls[-1].action_type == "final_answer"
    assert result.tool_calls[-1].tool_name is None
    compose_step = trace_repository.list_steps(accepted.task_run.id)[-1]
    assert compose_step.action == "compose_mock_answer"
    assert compose_step.input_payload["planner_decision"]["action_type"] == "final_answer"
    assert compose_step.input_payload["planner_decision"]["final_answer_used"] is True
    assert compose_step.result_payload["answer_preview"] == "Agent-generated final answer."


def test_query_execution_service_uses_frozen_agent_turn_protocol() -> None:
    class TurnOnlyQueryAgentClient:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "turn_only_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if request.final_answer_allowed and "compose_answer" in request.allowed_actions:
                return AgentTurnDecision(
                    action_type=AgentActionType.FINAL_ANSWER,
                    final_answer="Turn-generated final answer.",
                    rationale="turn_agent_finishes_when_compose_answer_is_available",
                    stop_reason=AgentStopReason.FINAL_ANSWER_READY,
                )
            if not request.allowed_actions:
                return None
            return AgentTurnDecision(
                action_type=AgentActionType.TOOL_CALL,
                tool_name=request.allowed_actions[0],
                rationale=f"turn_agent_selects_{request.allowed_actions[0]}",
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Frozen Turn Protocol")
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
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    tool_registry = InternalToolRegistry(
        paper_repository=InMemoryPaperRepository(),
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=InMemoryPaperRepository(),
            chunk_repository=InMemoryChunkRepository(),
            memory_repository=memory_repository,
        ),
    )
    turn_client = TurnOnlyQueryAgentClient()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=turn_client,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "Turn-generated final answer."
    assert turn_client.requests
    assert all(isinstance(request, AgentTurnRequest) for request in turn_client.requests)
    assert len(turn_client.requests) == 1
    assert turn_client.requests[-1].observations == ()
    assert turn_client.requests[-1].final_answer_allowed is True
    assert turn_client.requests[-1].allowed_actions == (
        "search_session_memory",
        "search_global_memory",
        "search_openviking_memory",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    )
    assert result.tool_calls[-1].action_type == "final_answer"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "compose_mock_answer",
    ]
    assert [event.summary for event in timeline_repository.list_by_session(session.id)] == [
        "checked session memory (no memories)",
        "checked global memory (no memories)",
        "reranked context candidates (no memories)",
        "decided whether to reread",
        "query run completed",
    ]


def test_query_execution_service_forces_compose_after_repeated_low_yield_turns() -> None:
    class LowYieldFirstQueryAgentClient:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "low_yield_first_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            for tool_name in (
                "search_session_memory",
                "search_global_memory",
                "read_source_passages",
                "search_openviking_memory",
                "rerank_candidates",
            ):
                if tool_name in request.allowed_actions:
                    return AgentTurnDecision(
                        action_type=AgentActionType.TOOL_CALL,
                        tool_name=tool_name,
                        rationale=f"probe_{tool_name}_before_answering",
                    )
            return None

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Anti Stall")
    session_repository.save(session)
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Compare missing evidence")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=InMemoryMemoryRepository(),
        chunk_repository=InMemoryChunkRepository(),
    )
    tool_registry = InternalToolRegistry(
        paper_repository=InMemoryPaperRepository(),
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=InMemoryPaperRepository(),
            chunk_repository=InMemoryChunkRepository(),
            memory_repository=InMemoryMemoryRepository(),
        ),
    )
    turn_client = LowYieldFirstQueryAgentClient()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=turn_client,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert "Mock answer for:" not in result.answer
    assert "\u8bb0\u5fc6" in result.answer
    assert [call.tool_name for call in result.tool_calls] == [
        "search_session_memory",
        "search_global_memory",
        "read_source_passages",
        "compose_answer",
    ]
    assert result.tool_calls[-1].agent_name == "host_runtime"
    assert result.tool_calls[-1].fallback_used is True
    assert "host_forces_compose_answer_after_3_low_yield_turns" in result.tool_calls[-1].rationale
    assert len(turn_client.requests) == 3
    compose_step = trace_repository.list_steps(accepted.task_run.id)[-1]
    assert compose_step.action == "compose_mock_answer"
    assert compose_step.input_payload["planner_decision"]["selected_tool"] == "compose_answer"
    assert compose_step.input_payload["planner_decision"]["agent_name"] == "host_runtime"
    assert compose_step.input_payload["planner_decision"]["fallback_used"] is True


def test_query_execution_service_answers_simple_greeting_without_tools() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Greeting")
    session_repository.save(session)
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "你好。")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=RetrievalService(
            session_repository=session_repository,
            memory_repository=InMemoryMemoryRepository(),
            chunk_repository=InMemoryChunkRepository(),
        ),
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "\u4f60\u597d\uff0c\u6211\u5728\u3002"
    assert result.should_reread_source is False
    assert result.reread_reason == "direct_conversational_turn"
    assert result.tool_calls[0].action_type == "final_answer"
    assert result.tool_calls[0].agent_name == "host_conversational_preflight"
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert [step.action for step in steps] == ["direct_final_answer"]
    assert steps[0].input_payload["retrieval_skipped"] is True
    messages = message_repository.list_by_session(session.id)
    assert messages[-1].role == "assistant"
    assert messages[-1].content == "\u4f60\u597d\uff0c\u6211\u5728\u3002"


def test_query_execution_service_forces_compose_after_duplicate_global_memory_signature() -> None:
    class DuplicateGlobalSearchQueryAgentClient:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "duplicate_global_search_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            for tool_name in (
                "search_session_memory",
                "search_global_memory",
                "search_openviking_memory",
                "rerank_candidates",
                "read_source_passages",
            ):
                if tool_name in request.allowed_actions:
                    return AgentTurnDecision(
                        action_type=AgentActionType.TOOL_CALL,
                        tool_name=tool_name,
                        rationale=f"probe_{tool_name}",
                    )
            return None

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Duplicate Signature")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.77777"),
            title="Duplicate Signature Paper",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id=paper.id,
            problem="Accuracy improvement",
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id=paper.id, artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=InMemoryChunkRepository(),
    )
    tool_registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=InMemoryChunkRepository(),
            memory_repository=memory_repository,
        ),
    )
    turn_client = DuplicateGlobalSearchQueryAgentClient()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=turn_client,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert [call.tool_name for call in result.tool_calls] == [
        "search_session_memory",
        "search_global_memory",
        "search_openviking_memory",
        "compose_answer",
    ]
    assert result.tool_calls[-1].agent_name == "host_runtime"
    assert result.tool_calls[-1].fallback_used is True
    assert "host_forces_compose_answer_after_duplicate_signature:memory_search|global|did it improve accuracy?|paper-1" == result.tool_calls[-1].rationale
    assert len(turn_client.requests) == 3
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "compose_mock_answer",
    ]
    compose_step = trace_repository.list_steps(accepted.task_run.id)[-1]
    assert compose_step.input_payload["planner_decision"]["selected_tool"] == "compose_answer"
    assert compose_step.input_payload["planner_decision"]["agent_name"] == "host_runtime"
    assert compose_step.input_payload["planner_decision"]["fallback_used"] is True


def test_query_runtime_service_runs_mock_chain_to_completion_and_writes_narratives() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_repository.save(session_service.create_session("Execute Runtime"))
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
            key_results=["Higher accuracy"],
            source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "Did it improve accuracy?")
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )
    runtime_service = QueryRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=execution_service,
        trace_repository=trace_repository,
    )

    result = runtime_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.FINISHED
    assert result.task_run.finish_reason == "mock_query_completed"
    assert result.task_run.finished_at is not None
    assert result.task_run.step_count == 5
    assert len(trace_repository.list_narratives(accepted.task_run.id)) == 5
    assert [narrative.reason_text for narrative in trace_repository.list_narratives(accepted.task_run.id)][0].startswith(
        "Session memory is checked first"
    )
    assert "paper_memory:paper-memory-1" in trace_repository.list_narratives(accepted.task_run.id)[0].impact_text
    assert "reranked context candidates" in trace_repository.list_narratives(accepted.task_run.id)[2].reason_text.lower()
    assert "model" in trace_repository.list_narratives(accepted.task_run.id)[2].impact_text.lower()


def test_ingest_materialization_service_extracts_arxiv_pdf_chunks() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    session = SessionService(session_repository=session_repository).create_session("Materialize")
    session_repository.save(session)
    service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
    )

    result = service.materialize_arxiv_source(session.id, "https://arxiv.org/abs/2401.12345")

    assert result.operation == "created"
    assert result.artifact.kind is ArtifactKind.ARXIV_PDF
    assert result.artifact.page_count == 1
    assert result.paper.canonical_key.value.startswith("paper:arxiv:")
    assert result.paper.title == "Imported arXiv paper 2401.12345"
    assert result.chunk_count == 1
    assert result.session_document.paper_id == result.paper.id
    assert result.session_document.artifact_id == result.artifact.id
    assert session_repository.list_documents(session.id)[0] == result.session_document
    assert paper_repository.get_by_id(result.paper.id) == result.paper
    assert artifact_repository.get_by_id(result.artifact.id) == result.artifact
    assert chunk_repository.list_by_paper_ids([result.paper.id])[0].text == "ArXiv text that should be extracted."


def test_ingest_materialization_service_normalizes_arxiv_prefix() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    session = SessionService(session_repository=session_repository).create_session("Materialize Prefix")
    session_repository.save(session)
    service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
    )

    result = service.materialize_arxiv_source(session.id, "arXiv:2401.12345")

    assert result.paper.canonical_key.value == "paper:arxiv:2401.12345"
    assert result.artifact.kind is ArtifactKind.ARXIV_PDF
    assert result.chunk_count == 1


def test_ingest_materialization_service_extracts_local_pdf_chunks(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    session = SessionService(session_repository=session_repository).create_session("Materialize PDF")
    session_repository.save(session)
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(_build_minimal_pdf_bytes("Local PDF text that should be extracted."))
    service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
    )

    result = service.materialize_pdf_source(session.id, str(pdf_path))

    assert result.operation == "created"
    assert result.artifact.kind is ArtifactKind.LOCAL_PDF
    assert result.artifact.page_count == 1
    assert result.paper.title == "Imported local PDF example"
    assert result.chunk_count == 1
    assert chunk_repository.list_by_paper_ids([result.paper.id])[0].text == "Local PDF text that should be extracted."


def test_ingest_materialization_service_cleans_pdf_page_text() -> None:
    service = IngestMaterializationService(
        session_repository=InMemorySessionRepository(),
        paper_repository=InMemoryPaperRepository(),
        artifact_repository=InMemoryArtifactRepository(),
    )

    noisy_text = (
        "Positional Failures in Long-Context LLMs:\n"
        "A Blind Spot in Reasoning Benchmarks\n"
        "Anonymous ACL submission\n"
        "Abstract001\n"
        "Position-controlled evaluation of long-context002\n"
        "LLMs exists for retrieval tasks (Needle-in-a-003\n"
        "Haystack, RULER), but mainstream reasoning004\n"
        "benchmarks leave the position of target infor-005\n"
        "mation uncharacterized and uncontrolled."
    )

    cleaned = service._clean_pdf_page_text(noisy_text)

    assert "001" not in cleaned
    assert "002" not in cleaned
    assert "003" not in cleaned
    assert "004" not in cleaned
    assert "005" not in cleaned
    assert "Needle-in-a-Haystack" in cleaned
    assert "position of target information uncharacterized and uncontrolled" in cleaned
    assert "Abstract" in cleaned


def test_ingest_analysis_service_prefers_main_text_over_appendix_but_keeps_appendix_candidates() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Main text priority")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-main-appendix",
            canonical_key=build_canonical_key(pdf_checksum="main-appendix"),
            title="Main Text Priority",
            abstract="A short abstract about the method and results.",
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-main-appendix",
        )
    )
    chunks = chunk_repository.save_many(
        [
            Chunk(
                id="chunk-main",
                paper_id=paper.id,
                artifact_id="artifact-main-appendix",
                text="We propose a new method that improves accuracy over the baseline.",
                page=2,
                section="page-2",
            ),
            Chunk(
                id="chunk-appendix",
                paper_id=paper.id,
                artifact_id="artifact-main-appendix",
                text="Appendix A Table 21 summarizes the detailed benchmark results and ablations.",
                page=14,
                section="page-14",
            ),
            Chunk(
                id="chunk-conclusion",
                paper_id=paper.id,
                artifact_id="artifact-main-appendix",
                text="Future work remains on robustness under distribution shift.",
                page=9,
                section="page-9",
            ),
        ]
    )

    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )

    candidate_passages = service._build_candidate_passages(
        paper=paper,
        artifact_id="artifact-main-appendix",
        chunks=list(chunks),
        window_kind="broad",
    )

    candidate_ids = [candidate.candidate_id for candidate in candidate_passages]
    candidate_roles = {candidate.candidate_id: candidate.content_role for candidate in candidate_passages}
    main_index = candidate_ids.index("chunk-main")
    appendix_index = candidate_ids.index("chunk-appendix")

    assert "chunk-appendix" in candidate_ids
    assert main_index < appendix_index
    assert candidate_roles["chunk-main"] == "main"
    assert candidate_roles["chunk-appendix"] == "appendix"


def test_memory_extraction_service_writes_three_memory_types_for_parsed_pdf(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Memory Extraction")
    session_repository.save(session)
    current_pdf = tmp_path / "current.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(pdf_checksum="current-checksum"),
            title="Current Paper",
        )
    )
    related_paper = paper_repository.save(
        Paper(
            id="paper-2",
            canonical_key=build_canonical_key(pdf_checksum="related-checksum"),
            title="Related Paper",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="current-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=related_paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-2",
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method. It improves accuracy over the baseline. Future work remains on robustness.",
                page=1,
                section="page-1",
            )
        ]
    )
    service = MemoryExtractionService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )

    result = service.extract_and_store_memories(session.id, current_paper.id)

    assert result.paper_operation == "created"
    assert result.paper_memory.paper_id == current_paper.id
    assert result.relation_memory is not None
    assert result.relation_memory.target_paper == related_paper.id
    assert result.open_question_memory.related_papers == [current_paper.id, related_paper.id]
    assert len(memory_repository.list_paper_memories_for_papers([current_paper.id])) == 1
    assert len(memory_repository.list_relation_memories_for_papers([current_paper.id])) == 1
    assert len(memory_repository.list_open_question_memories_for_papers([current_paper.id])) == 1


def test_ingest_analysis_service_uses_model_backed_candidates_and_rereads_when_needed(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Ingest Analysis Model")
    session_repository.save(session)
    current_pdf = tmp_path / "current-model.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-model-1",
            canonical_key=build_canonical_key(pdf_checksum="model-checksum"),
            title="Model Current Paper",
            abstract="A compact abstract for model-backed ingest analysis.",
        )
    )
    related_paper = paper_repository.save(
        Paper(
            id="paper-model-2",
            canonical_key=build_canonical_key(pdf_checksum="model-related-checksum"),
            title="Model Related Paper",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-model-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="model-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=related_paper.id,
            source_type=SourceType.PDF,
            artifact_id="artifact-model-2",
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-model-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method.",
                page=1,
                section="Abstract",
            ),
            Chunk(
                id="chunk-model-2",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="It improves accuracy over the baseline.",
                page=1,
                section="Results",
            ),
            Chunk(
                id="chunk-model-3",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="Future work remains on robustness.",
                page=1,
                section="Limitations",
            ),
        ]
    )

    class RecordingTransport:
        def __init__(self) -> None:
            self.prompts: list[StructuredIngestExtractionPrompt] = []

        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return StructuredIngestExtractionChoice(
                    paper=StructuredIngestPaperDraft(
                        problem="Initial model pass needs more evidence.",
                        method="Initial model pass needs more evidence.",
                        key_results=("Initial evidence is incomplete.",),
                        limitations=("Initial evidence is incomplete.",),
                        novelty_claim="Initial evidence is incomplete.",
                        evidence_candidate_ids=("chunk-model-1",),
                        confidence=0.35,
                    ),
                    relation=None,
                    open_question=StructuredIngestOpenQuestionDraft(
                        unresolved_question="Initial evidence is incomplete.",
                        why_open=("Initial evidence is incomplete.",),
                        possible_followup=("Initial evidence is incomplete.",),
                        evidence_candidate_ids=("chunk-model-1",),
                        confidence=0.35,
                    ),
                    paper_summary=StructuredIngestPaperSummaryDraft(
                        what_it_is_about="Initial model pass needs more evidence.",
                        problem_solved="Initial model pass needs more evidence.",
                        new_ideas=("Initial evidence is incomplete.",),
                        limitations=("Initial evidence is incomplete.",),
                        suggestions_or_questions=("Initial evidence is incomplete.",),
                        evidence_candidate_ids=("chunk-model-1",),
                        confidence=0.35,
                    ),
                    needs_more_context=True,
                    context_hints=("include results and limitations",),
                    rationale="Need more context before committing to memory drafts.",
                )
            return StructuredIngestExtractionChoice(
                paper=StructuredIngestPaperDraft(
                    problem="A new method for improving accuracy over the baseline.",
                    method="A new method that is evaluated on accuracy.",
                    key_results=("It improves accuracy over the baseline.",),
                    limitations=("Future work remains on robustness.",),
                    novelty_claim="We propose a new method.",
                    evidence_candidate_ids=("chunk-model-1", "chunk-model-2", "chunk-model-3"),
                    confidence=0.9,
                ),
                relation=StructuredIngestRelationDraft(
                    relation_type=RelationType.IMPROVES_ON.value,
                    summary="The paper improves on a related baseline.",
                    evidence_candidate_ids=("chunk-model-2",),
                    confidence=0.8,
                ),
                open_question=StructuredIngestOpenQuestionDraft(
                    unresolved_question="Does the method remain stable under distribution shift?",
                    why_open=("Future work remains on robustness.",),
                    possible_followup=("Evaluate robustness under distribution shift.",),
                    evidence_candidate_ids=("chunk-model-3",),
                    confidence=0.7,
                ),
                paper_summary=StructuredIngestPaperSummaryDraft(
                    what_it_is_about="A new method for improving accuracy over the baseline.",
                    problem_solved="A new method for improving accuracy over the baseline.",
                    new_ideas=("A new method that is evaluated on accuracy.", "We propose a new method."),
                    limitations=("Future work remains on robustness.",),
                    suggestions_or_questions=("Evaluate robustness under distribution shift.",),
                    evidence_candidate_ids=("chunk-model-1", "chunk-model-2", "chunk-model-3"),
                    confidence=0.85,
                ),
                needs_more_context=False,
                context_hints=(),
                rationale="Broad and expanded candidate windows contain enough evidence.",
            )

    transport = RecordingTransport()
    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=ModelBackedIngestExtractionClient(transport=transport),
    )

    result = service.analyze(session.id, current_paper.id)

    assert [prompt.window_kind for prompt in transport.prompts] == ["broad", "expanded"]
    assert len(transport.prompts[0].candidate_passages) >= 3
    assert len(transport.prompts[1].candidate_passages) >= len(transport.prompts[0].candidate_passages)
    assert result.paper_memory.problem == "A new method for improving accuracy over the baseline."
    assert result.paper_memory.method == "A new method that is evaluated on accuracy."
    assert result.paper_memory.key_results == ["It improves accuracy over the baseline."]
    assert result.paper_memory.limitations == ["Future work remains on robustness."]
    assert result.relation_memory is not None
    assert result.relation_memory.relation_type is RelationType.IMPROVES_ON
    assert result.relation_memory.summary == "The paper improves on a related baseline."
    assert result.open_question_memory.unresolved_question == "Does the method remain stable under distribution shift?"
    assert result.open_question_memory.possible_followup == ["Evaluate robustness under distribution shift."]
    assert result.paper_summary.what_it_is_about == "A new method for improving accuracy over the baseline."
    assert "robustness" in " ".join(result.paper_summary.limitations).lower()


def test_ingest_analysis_service_falls_back_to_rules_when_model_extraction_fails(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Ingest Analysis Fallback")
    session_repository.save(session)
    current_pdf = tmp_path / "current-fallback.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-fallback-1",
            canonical_key=build_canonical_key(pdf_checksum="fallback-checksum"),
            title="Fallback Current Paper",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-fallback-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="fallback-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-fallback-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method. It improves accuracy over the baseline. Future work remains on robustness.",
                page=1,
                section="Body",
            )
        ]
    )

    class FailingTransport:
        def __init__(self) -> None:
            self.prompts: list[StructuredIngestExtractionPrompt] = []

        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            self.prompts.append(prompt)
            raise RuntimeError("model unavailable")

    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=ModelBackedIngestExtractionClient(transport=FailingTransport()),
    )

    result = service.analyze(session.id, current_paper.id)

    assert result.paper_memory.paper_id == current_paper.id
    assert result.paper_memory.problem is not None
    assert "propose" in result.paper_memory.problem.lower()
    assert "improves accuracy" in " ".join(result.paper_memory.key_results).lower()
    assert result.open_question_memory.related_papers == [current_paper.id]
    assert result.relation_memory is None
    assert result.paper_summary.what_it_is_about


def test_ingest_analysis_service_sanitizes_noisy_model_summary_output(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Ingest Analysis Summary Cleanup")
    session_repository.save(session)
    current_pdf = tmp_path / "current-summary-cleanup.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-summary-1",
            canonical_key=build_canonical_key(pdf_checksum="summary-cleanup-checksum"),
            title="Summary Cleanup Paper",
            abstract="A compact abstract for summary cleanup.",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-summary-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="summary-cleanup-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-summary-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method. It improves accuracy over the baseline. Future work remains on robustness.",
                page=1,
                section="Body",
            )
        ]
    )

    class NoisyTransport:
        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            return StructuredIngestExtractionChoice(
                paper=StructuredIngestPaperDraft(
                    problem="A new method for improving accuracy over the baseline.",
                    method="A new method that is evaluated on accuracy.",
                    key_results=("It improves accuracy over the baseline.",),
                    limitations=("Future work remains on robustness.",),
                    novelty_claim="We propose a new method.",
                    evidence_candidate_ids=("chunk-summary-1",),
                    confidence=0.9,
                ),
                relation=None,
                open_question=StructuredIngestOpenQuestionDraft(
                    unresolved_question="Does the method remain stable under distribution shift?",
                    why_open=("Future work remains on robustness.",),
                    possible_followup=("Evaluate robustness under distribution shift.",),
                    evidence_candidate_ids=("chunk-summary-1",),
                    confidence=0.7,
                ),
                paper_summary=StructuredIngestPaperSummaryDraft(
                    what_it_is_about="In Proceedings of the Theory and Practice of Software, 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems",
                    problem_solved="In Proceedings of the Theory and Practice of Software, 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems",
                    new_ideas=(
                        "In Proceedings of the Theory and Practice of Software, 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems",
                        "Unlike DeepSeek-V3.2, which discarded thinking traces upon each new user turn, DeepSeek-V4 series retain the complete reasoning history across all rounds, including across user message boundaries.",
                    ),
                    limitations=("Future work remains on robustness.",),
                    suggestions_or_questions=("Revisit the source after removing reference noise.",),
                    evidence_candidate_ids=("chunk-summary-1",),
                    confidence=0.9,
                ),
                needs_more_context=False,
                context_hints=(),
                rationale="Noisy draft for sanitization test.",
            )

    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=ModelBackedIngestExtractionClient(transport=NoisyTransport()),
    )

    result = service.analyze(session.id, current_paper.id)

    assert result.paper_summary.what_it_is_about == "Summary Cleanup Paper"
    assert result.paper_summary.problem_solved == "A new method that is evaluated on accuracy."
    assert "proceedings of" not in " ".join(result.paper_summary.new_ideas).lower()
    assert result.paper_summary.suggestions_or_questions == ("Revisit the source after removing reference noise.",)


def test_ingest_analysis_service_uses_chinese_fallbacks_for_imported_placeholder_titles(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Imported Placeholder Fallback")
    session_repository.save(session)
    current_pdf = tmp_path / "imported-placeholder.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "Across 11 long-context benchmarks, position-aware evaluation appears well established for retrieval but remains largely unaudited in mainstream reasoning evaluation, leaving model-dependent positional vulnerabilities uncharacterized. Run larger-scale experiments."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-imported-placeholder-1",
            canonical_key=build_canonical_key(pdf_checksum="imported-placeholder-checksum"),
            title="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
            abstract="Across 11 long-context benchmarks, position-aware evaluation appears well established for retrieval but remains largely unaudited in mainstream reasoning evaluation.",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-imported-placeholder-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="imported-placeholder-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-imported-placeholder-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="Across 11 long-context benchmarks, position-aware evaluation appears well established for retrieval but remains largely unaudited in mainstream reasoning evaluation, leaving model-dependent positional vulnerabilities uncharacterized.",
                page=1,
                section="Introduction",
            ),
            Chunk(
                id="chunk-imported-placeholder-2",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="Run larger-scale experiments to better characterize positional vulnerabilities.",
                page=2,
                section="Limitations",
            ),
        ]
    )

    class PlaceholderTransport:
        def __init__(self) -> None:
            self.prompts: list[StructuredIngestExtractionPrompt] = []

        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            self.prompts.append(prompt)
            return StructuredIngestExtractionChoice(
                paper=StructuredIngestPaperDraft(
                    problem="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    method="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    key_results=("Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",),
                    limitations=("Run larger-scale experiments.",),
                    novelty_claim="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    evidence_candidate_ids=("chunk-imported-placeholder-1",),
                    confidence=0.9,
                ),
                relation=None,
                open_question=StructuredIngestOpenQuestionDraft(
                    unresolved_question="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    why_open=("Run larger-scale experiments.",),
                    possible_followup=("Run larger-scale experiments.",),
                    evidence_candidate_ids=("chunk-imported-placeholder-2",),
                    confidence=0.7,
                ),
                paper_summary=StructuredIngestPaperSummaryDraft(
                    what_it_is_about="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    problem_solved="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
                    new_ideas=("Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",),
                    limitations=("Run larger-scale experiments.",),
                    suggestions_or_questions=("Run larger-scale experiments.",),
                    evidence_candidate_ids=("chunk-imported-placeholder-1", "chunk-imported-placeholder-2"),
                    confidence=0.85,
                ),
                needs_more_context=False,
                context_hints=(),
                rationale="Placeholder imported title fallback test.",
            )

    transport = PlaceholderTransport()
    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=ModelBackedIngestExtractionClient(transport=transport),
    )

    result = service.analyze(session.id, current_paper.id)

    assert len(transport.prompts) == 1
    assert len(transport.prompts[0].candidate_passages) == 4
    assert {candidate["candidate_id"] for candidate in transport.prompts[0].candidate_passages} == {
        "title",
        "abstract",
        "chunk-imported-placeholder-1",
        "chunk-imported-placeholder-2",
    }
    assert result.paper_memory.problem.startswith("本文")
    assert result.paper_memory.problem != current_paper.title
    assert result.paper_summary.what_it_is_about.startswith("本文")
    assert result.paper_summary.problem_solved.startswith("本文")
    assert result.paper_summary.what_it_is_about != current_paper.title
    assert result.paper_summary.problem_solved != current_paper.title
    assert any("\u4e00" <= char <= "\u9fff" for char in result.paper_summary.suggestions_or_questions[0])
    assert "Run larger-scale experiments." not in " ".join(result.paper_summary.suggestions_or_questions)


def test_ingest_analysis_service_prefers_main_text_for_summary_evidence_ids(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Ingest Summary Evidence")
    session_repository.save(session)
    current_pdf = tmp_path / "current-summary-evidence.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-summary-evidence-1",
            canonical_key=build_canonical_key(pdf_checksum="summary-evidence-checksum"),
            title="Summary Evidence Paper",
            abstract="A compact abstract for summary evidence.",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-summary-evidence-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="summary-evidence-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=current_paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-main-summary",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method that improves accuracy over the baseline.",
                page=1,
                section="Results",
            ),
            Chunk(
                id="chunk-appendix-summary",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="Appendix A Table 21 summarizes the detailed benchmark results and ablations.",
                page=12,
                section="Appendix",
            ),
        ]
    )

    class AppendixHeavyTransport:
        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            return StructuredIngestExtractionChoice(
                paper=StructuredIngestPaperDraft(
                    problem="A new method for improving accuracy over the baseline.",
                    method="A new method that is evaluated on accuracy.",
                    key_results=("It improves accuracy over the baseline.",),
                    limitations=("Future work remains on robustness.",),
                    novelty_claim="We propose a new method.",
                    evidence_candidate_ids=("chunk-appendix-summary",),
                    confidence=0.9,
                ),
                relation=None,
                open_question=StructuredIngestOpenQuestionDraft(
                    unresolved_question="Does the method remain stable under distribution shift?",
                    why_open=("Future work remains on robustness.",),
                    possible_followup=("Evaluate robustness under distribution shift.",),
                    evidence_candidate_ids=("chunk-appendix-summary",),
                    confidence=0.7,
                ),
                paper_summary=StructuredIngestPaperSummaryDraft(
                    what_it_is_about="A new method for improving accuracy over the baseline.",
                    problem_solved="A new method for improving accuracy over the baseline.",
                    new_ideas=("A new method that is evaluated on accuracy.",),
                    limitations=("Future work remains on robustness.",),
                    suggestions_or_questions=("Evaluate robustness under distribution shift.",),
                    evidence_candidate_ids=("chunk-appendix-summary",),
                    confidence=0.85,
                ),
                needs_more_context=False,
                context_hints=(),
                rationale="Main text is enough for summary evidence.",
            )

    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=AppendixHeavyTransport(),
    )

    result = service.analyze(session.id, current_paper.id)

    assert result.paper_summary.evidence_candidate_ids == ("chunk-main-summary",)


def test_task_runtime_service_runs_mock_ingest_chain_to_completion() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Ingest Runtime"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_arxiv_ingest(session.id, "https://arxiv.org/abs/2401.12345")
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
    )
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )
    materialization_service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=InMemoryArtifactRepository(),
        chunk_repository=chunk_repository,
    )
    memory_extraction_service = MemoryExtractionService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )
    ingest_execution_service = IngestExecutionService(
        message_repository=message_repository,
        materialization_service=materialization_service,
        memory_extraction_service=memory_extraction_service,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
    )
    runtime_service = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=query_execution_service,
        ingest_execution_service=ingest_execution_service,
        trace_repository=trace_repository,
    )

    result = runtime_service.execute_ingest_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.FINISHED
    assert result.task_run.finish_reason == "mock_ingest_completed"
    assert result.task_run.finished_at is not None
    assert result.task_run.step_count == 7
    assert result.source_type is SourceType.ARXIV
    assert result.materialization.paper.id == result.materialization.session_document.paper_id
    assert result.materialization.artifact.id == result.materialization.session_document.artifact_id
    assert "已解析 arXiv PDF" in result.ingest_summary
    assert "论文主题" in result.ingest_summary
    assert result.memory_extraction.paper_memory.paper_id == result.materialization.paper.id
    assert result.paper_summary.what_it_is_about
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "inspect_ingest_request",
        "extract_arxiv_pdf_text",
        "persist_arxiv_chunks",
        "compose_ingest_summary",
        "extract_paper_memory",
        "derive_relation_memory",
        "capture_open_questions",
    ]
    assert [event.summary for event in timeline_repository.list_by_session(session.id)] == [
        "已检查导入请求",
        "已抽取 arXiv PDF 文本",
        "已保存 arXiv 分块",
        "已生成arXiv PDF摘要",
        "已抽取论文记忆",
        "已生成关系记忆",
        "已记录开放问题",
        "导入运行已完成",
    ]
    assert len(trace_repository.list_narratives(accepted.task_run.id)) == 7
    assert len(memory_repository.list_paper_memories_for_papers([result.materialization.paper.id])) == 1
    assert len(memory_repository.list_open_question_memories_for_papers([result.materialization.paper.id])) == 1
    messages = message_repository.list_by_session(session.id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].content == result.ingest_summary


def test_ingest_execution_service_persists_and_mirrors_assistant_summary() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    openviking_bundle = build_inmemory_openviking_surface_bundle()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Ingest Mirror"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_arxiv_ingest(session.id, "https://arxiv.org/abs/2401.12345")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    service = IngestExecutionService(
        message_repository=message_repository,
        materialization_service=IngestMaterializationService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            artifact_repository=InMemoryArtifactRepository(),
            chunk_repository=chunk_repository,
        ),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        openviking_bundle=openviking_bundle,
    )

    result = service.execute_ingest_run(session.id, accepted.task_run.id)

    messages = message_repository.list_by_session(session.id)
    assert len(messages) == 2
    assert messages[1].role == "assistant"
    assert messages[1].type is MessageType.INGEST_ARXIV
    assert messages[1].content == result.ingest_summary
    mirrored = openviking_bundle.messages.list_messages(session.id)
    assert len(mirrored) == 1
    assert mirrored[0].role == "assistant"
    assert mirrored[0].content == result.ingest_summary
    assert mirrored[0].metadata["run_id"] == accepted.task_run.id


def test_trace_query_service_returns_steps_and_narratives_for_run() -> None:
    session_repository = InMemorySessionRepository()
    trace_repository = InMemoryTraceRepository()
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=InMemoryMessageRepository(),
        trace_repository=trace_repository,
    )
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Trace"))
    accepted = task_run_service.accept_followup_query(session.id, "Explain the result.")
    trace_repository.save_step(
        TraceStep(
            run_id=accepted.task_run.id,
            action="retrieve_session_memories",
            result_payload={"memory_ids": []},
        )
    )

    service = TraceQueryService(
        session_repository=session_repository,
        trace_repository=trace_repository,
    )
    trace = service.get_trace(session.id, accepted.task_run.id)

    assert len(trace.steps) == 1
    assert trace.steps[0].action == "retrieve_session_memories"
    assert trace.narratives == ()


def test_timeline_query_service_can_filter_events_by_run() -> None:
    session_repository = InMemorySessionRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Timeline")
    session_repository.save(session)
    run_a = "run-a"
    run_b = "run-b"
    timeline_repository.save(TimelineEvent(session_id=session.id, run_id=run_a, event_type="step_completed", summary="A"))
    timeline_repository.save(TimelineEvent(session_id=session.id, run_id=run_b, event_type="step_completed", summary="B"))
    service = TimelineQueryService(session_repository=session_repository, timeline_repository=timeline_repository)

    events = service.list_events_for_run(session.id, run_a)

    assert len(events) == 1
    assert events[0].summary == "A"
