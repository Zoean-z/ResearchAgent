"""Unit tests for the thin application services."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
import ssl
from urllib.error import URLError
from uuid import uuid4

import pytest
from pydantic import BaseModel

from research_agent.adapters.llm import (
    DeepSeekHttpResponse,
    DeepSeekStructuredQueryAgentTransport,
    ModelBackedQueryAgentClient,
    ModelBackedQueryToolPlannerClient,
    StaticStructuredPlannerTransport,
)
from research_agent.adapters.llm.ingest_extraction import (
    ModelBackedIngestExtractionClient,
    StructuredIngestExtractionChoice,
    StructuredIngestExtractionParseError,
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
    ArxivHttpResponse,
    ArxivImportToolService,
    ArxivSearchService,
    EntityNotFoundError,
    ContextRerankService,
    IngestExecutionService,
    IngestAnalysisService,
    IngestMaterializationService,
    MemoryBundleService,
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
from research_agent.services.query_execution_service import QueryExecutionError, QueryFailureDetail
from research_agent.services.ingest_materialization_service import IngestMaterializationService
from research_agent.runtime import QueryRuntimeService, RuntimeEventBroker, TaskRuntimeService
from research_agent.runtime.ingest_extraction import IngestExtractionCandidate
from research_agent.runtime.streaming import RuntimeStreamEvent
from research_agent.runtime.agent_protocol import AgentActionType, AgentObservation, AgentStopReason, AgentTurnDecision, AgentTurnRequest
from research_agent.tools import HeuristicQueryToolPlannerClient, InternalToolRegistry, QueryToolExecutor
from research_agent.tools import StaticFinalAnswerQueryAgentClient
from research_agent.services.task_run_streaming_service import TaskRunStreamingService
from research_agent.api.schemas.query_execution import QueryExecutionResponse
from research_agent.services.query_execution_service import QueryExecutionResult
from research_agent.services.retrieval_service import MemoryRetrievalResult, RetrievalPlan


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


_ARXIV_SEARCH_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2026-01-02T00:00:00Z</updated>
    <published>2026-01-01T00:00:00Z</published>
    <title>Memory-Routed Research Agents</title>
    <summary>Search-first discovery for paper agents.</summary>
    <author><name>Alice</name></author>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1.pdf" rel="related" type="application/pdf" />
    <category term="cs.AI" />
  </entry>
</feed>
"""


class _JsonSafeEnum(str, Enum):
    READY = "ready"


class _JsonSafeModel(BaseModel):
    value: int


def _deepseek_response(payload: dict[str, object], status_code: int = 200) -> DeepSeekHttpResponse:
    return DeepSeekHttpResponse(status_code=status_code, body=json.dumps(payload).encode("utf-8"))


@pytest.fixture(autouse=True)
def _stub_arxiv_download(monkeypatch) -> None:
    monkeypatch.setattr(
        IngestMaterializationService,
        "_download_arxiv_pdf",
        lambda self, pdf_url, source_value: _build_minimal_pdf_bytes("ArXiv text that should be extracted."),
    )


class _SequentialQueryAgentClient:
    def __init__(self, answer_text: str = "Model-generated final answer.", agent_name: str = "sequential_query_agent") -> None:
        self._answer_text = answer_text
        self._agent_name = agent_name
        self.requests: list[AgentTurnRequest] = []

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        self.requests.append(request)
        for tool_name in (
            "search_session_memory",
            "search_global_memory",
            "search_openviking_memory",
            "rerank_candidates",
            "read_source_passages",
            "compose_answer",
        ):
            if tool_name in request.allowed_actions:
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name=tool_name,
                    rationale=f"sequential_agent_selects_{tool_name}",
                )
        if request.final_answer_allowed:
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer=self._answer_text,
                rationale="sequential_agent_finishes_with_final_answer",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )
        return None


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
        "import_arxiv_paper",
        "search_arxiv",
        "search_openviking_memory",
        "search_session_memory",
        "search_global_memory",
        "search_source_chunks",
        "list_recent_messages",
        "get_conversation_context",
        "list_session_papers",
        "get_paper_memory_bundle",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    }
    assert registry.invoke("search_session_memory", session_id=session.id, query="accuracy", top_k=5).memories
    assert registry.invoke("read_source_passages", session_id=session.id, query="accuracy", related_paper_ids=[paper.id], top_k=3).selected


@pytest.mark.parametrize(
    ("arxiv_id_or_url", "expected_abs_url", "expected_arxiv_id"),
    [
        ("2401.12345", "https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345", "https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345.pdf", "https://arxiv.org/abs/2401.12345", "2401.12345"),
    ],
)
def test_arxiv_import_tool_service_reuses_existing_accept_and_runtime_chain(
    monkeypatch,
    arxiv_id_or_url: str,
    expected_abs_url: str,
    expected_arxiv_id: str,
) -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    openviking_bundle = build_inmemory_openviking_surface_bundle()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Tool Import"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    message_intake_service = MessageIntakeService(
        task_run_service=task_run_service,
        openviking_bundle=openviking_bundle,
    )
    ingest_execution_service = IngestExecutionService(
        message_repository=message_repository,
        materialization_service=IngestMaterializationService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            artifact_repository=artifact_repository,
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
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=RetrievalService(
            session_repository=session_repository,
            memory_repository=memory_repository,
            chunk_repository=chunk_repository,
        ),
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        query_agent_client=StaticFinalAnswerQueryAgentClient("unused"),
    )
    task_runtime_service = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=query_execution_service,
        ingest_execution_service=ingest_execution_service,
        trace_repository=trace_repository,
    )
    service = ArxivImportToolService(
        message_intake_service=message_intake_service,
        task_runtime_service=task_runtime_service,
    )

    result = service.import_arxiv_paper(
        session_id=session.id,
        arxiv_id_or_url=arxiv_id_or_url,
    )

    assert result.submitted.message.type is MessageType.INGEST_ARXIV
    assert result.submitted.message.content == expected_abs_url
    assert result.execution.task_run.status is TaskRunStatus.FINISHED
    assert result.execution.source_type is SourceType.ARXIV
    assert result.execution.materialization.paper.arxiv_id == expected_arxiv_id
    assert result.execution.materialization.artifact.uri_or_path == expected_abs_url
    assert result.execution.chunk_count == 1


def test_arxiv_import_tool_service_rejects_invalid_input_before_submit() -> None:
    class StubMessageIntakeService:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def submit_arxiv_ingest(self, session_id: str, arxiv_url: str):
            self.calls.append((session_id, arxiv_url))
            raise AssertionError("submit_arxiv_ingest should not be called for invalid input")

    class StubTaskRuntimeService:
        def execute_ingest_run(self, session_id: str, run_id: str):
            raise AssertionError("execute_ingest_run should not be called for invalid input")

    intake_service = StubMessageIntakeService()
    service = ArxivImportToolService(
        message_intake_service=intake_service,  # type: ignore[arg-type]
        task_runtime_service=StubTaskRuntimeService(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        service.import_arxiv_paper(
            session_id="session-1",
            arxiv_id_or_url="2401.12",
        )

    assert intake_service.calls == []


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


def test_memory_bundle_service_groups_memories_by_paper_source() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session_service = SessionService(session_repository=session_repository)
    session = session_service.create_session("Bundles")
    session_repository.save(session)

    paper_one = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(pdf_checksum="checksum-1"),
            title="Paper One",
            authors=["Alice"],
            abstract="Abstract one.",
            pdf_fingerprint="checksum-1",
        )
    )
    paper_two = paper_repository.save(
        Paper(
            id="paper-2",
            canonical_key=build_canonical_key(pdf_checksum="checksum-2"),
            title="Paper Two",
            authors=["Bob"],
            abstract="Abstract two.",
            pdf_fingerprint="checksum-2",
        )
    )
    artifact_one = artifact_repository.save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=r"C:\\papers\\paper-one.pdf",
            checksum="checksum-1",
            page_count=2,
        )
    )
    artifact_two = artifact_repository.save(
        Artifact(
            id="artifact-2",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=r"C:\\papers\\paper-two.pdf",
            checksum="checksum-2",
            page_count=2,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper_one.id,
            source_type=SourceType.PDF,
            artifact_id=artifact_one.id,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper_two.id,
            source_type=SourceType.PDF,
            artifact_id=artifact_two.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=paper_one.id,
                artifact_id=artifact_one.id,
                text="Paper one chunk text.",
                page=1,
                section="Introduction",
            ),
            Chunk(
                id="chunk-2",
                paper_id=paper_two.id,
                artifact_id=artifact_two.id,
                text="Paper two chunk text.",
                page=1,
                section="Introduction",
            ),
        ]
    )

    paper_memory = memory_repository.upsert_paper_memory(
        PaperMemory(
            paper_id=paper_one.id,
            problem="Problem one",
            method="Method one",
            key_results=["Result one"],
            limitations=["Limit one"],
            novelty_claim="Novelty one",
            source_refs=[SourceRef(paper_id=paper_one.id, artifact_id=artifact_one.id, chunk_id="chunk-1", quote="Paper one chunk text.")],
            confidence=ConfidenceScore(value=0.8),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    relation_memory = memory_repository.upsert_relation_memory(
        RelationMemory(
            source_paper=paper_one.id,
            target_paper=paper_two.id,
            relation_type=RelationType.IMPROVES_ON,
            summary="Paper one improves on paper two.",
            evidence=["Paper one is stronger."],
            confidence=ConfidenceScore(value=0.7),
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    )
    open_question_memory = memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            unresolved_question="Does it generalize?",
            related_papers=[paper_one.id, paper_two.id],
            why_open=["No cross-domain test."],
            possible_followup=["Run domain shift evaluation."],
            confidence=ConfidenceScore(value=0.5),
            updated_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    )

    service = MemoryBundleService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )

    bundle = service.get_bundle(session.id, source_chunk_limit=1)

    assert [group.paper.paper_id for group in bundle.papers] == [paper_two.id, paper_one.id]

    paper_one_group = next(group for group in bundle.papers if group.paper.paper_id == paper_one.id)
    paper_two_group = next(group for group in bundle.papers if group.paper.paper_id == paper_two.id)

    assert paper_one_group.paper.title == "Paper One"
    assert paper_one_group.paper.file_name == "paper-one.pdf"
    assert paper_one_group.paper.memory_count == 3
    assert paper_one_group.paper.created_at is not None
    assert paper_one_group.source_chunk_count == 1
    assert paper_one_group.source_chunks[0].chunk_id == "chunk-1"
    assert [item.memory_type for item in paper_one_group.paper_memories] == ["paper_memory"]
    assert [item.memory_type for item in paper_one_group.open_question_memories] == ["open_question_memory"]
    assert [item.relation_direction for item in paper_one_group.relation_memories] == ["source"]

    assert paper_two_group.paper.title == "Paper Two"
    assert paper_two_group.paper.file_name == "paper-two.pdf"
    assert paper_two_group.paper.memory_count == 2
    assert [item.memory_type for item in paper_two_group.open_question_memories] == ["open_question_memory"]
    assert [item.relation_direction for item in paper_two_group.relation_memories] == ["target"]

    assert bundle.unscoped_memories == ()
    assert paper_one_group.paper_memories[0].id == paper_memory.id
    assert paper_one_group.relation_memories[0].id == relation_memory.id
    assert paper_one_group.open_question_memories[0].id == open_question_memory.id


def test_memory_bundle_service_groups_global_memories_by_paper_source() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session_one = SessionService(session_repository=session_repository).create_session("Global One")
    session_two = SessionService(session_repository=session_repository).create_session("Global Two")
    session_repository.save(session_one)
    session_repository.save(session_two)

    paper_one = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(pdf_checksum="checksum-1"),
            title="Paper One",
            authors=["Alice"],
            abstract="Abstract one.",
            pdf_fingerprint="checksum-1",
        )
    )
    paper_two = paper_repository.save(
        Paper(
            id="paper-2",
            canonical_key=build_canonical_key(pdf_checksum="checksum-2"),
            title="Paper Two",
            authors=["Bob"],
            abstract="Abstract two.",
            pdf_fingerprint="checksum-2",
        )
    )
    artifact_one = artifact_repository.save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=r"C:\\papers\\paper-one.pdf",
            checksum="checksum-1",
            page_count=2,
        )
    )
    artifact_two = artifact_repository.save(
        Artifact(
            id="artifact-2",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=r"C:\\papers\\paper-two.pdf",
            checksum="checksum-2",
            page_count=2,
        )
    )
    document_one = session_repository.save_document(
        SessionDocument(
            session_id=session_one.id,
            paper_id=paper_one.id,
            source_type=SourceType.PDF,
            artifact_id=artifact_one.id,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session_two.id,
            paper_id=paper_two.id,
            source_type=SourceType.PDF,
            artifact_id=artifact_two.id,
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=paper_one.id,
                artifact_id=artifact_one.id,
                text="Paper one chunk text.",
                page=1,
                section="Introduction",
            ),
            Chunk(
                id="chunk-2",
                paper_id=paper_two.id,
                artifact_id=artifact_two.id,
                text="Paper two chunk text.",
                page=1,
                section="Introduction",
            ),
        ]
    )

    memory_repository.upsert_paper_memory(
        PaperMemory(
            paper_id=paper_one.id,
            problem="Problem one",
            method="Method one",
            key_results=["Result one"],
            limitations=["Limit one"],
            novelty_claim="Novelty one",
            source_refs=[SourceRef(paper_id=paper_one.id, artifact_id=artifact_one.id, chunk_id="chunk-1", quote="Paper one chunk text.")],
            confidence=ConfidenceScore(value=0.8),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    memory_repository.upsert_relation_memory(
        RelationMemory(
            source_paper=paper_one.id,
            target_paper=paper_two.id,
            relation_type=RelationType.IMPROVES_ON,
            summary="Paper one improves on paper two.",
            evidence=["Paper one is stronger."],
            confidence=ConfidenceScore(value=0.7),
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            unresolved_question="Does it generalize?",
            related_papers=[paper_one.id, paper_two.id],
            why_open=["No cross-domain test."],
            possible_followup=["Run domain shift evaluation."],
            confidence=ConfidenceScore(value=0.5),
            updated_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )
    )

    service = MemoryBundleService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
    )

    bundle = service.get_global_bundle(source_chunk_limit=1)

    assert [group.paper.paper_id for group in bundle.papers] == [paper_two.id, paper_one.id]

    paper_one_group = next(group for group in bundle.papers if group.paper.paper_id == paper_one.id)
    paper_two_group = next(group for group in bundle.papers if group.paper.paper_id == paper_two.id)

    assert paper_one_group.paper.title == "Paper One"
    assert paper_one_group.paper.file_name == "paper-one.pdf"
    assert paper_one_group.paper.created_at == document_one.added_at
    assert paper_one_group.paper.memory_count == 3
    assert paper_one_group.source_chunk_count == 1
    assert paper_one_group.source_chunks[0].chunk_id == "chunk-1"
    assert [item.memory_type for item in paper_one_group.paper_memories] == ["paper_memory"]
    assert [item.memory_type for item in paper_one_group.open_question_memories] == ["open_question_memory"]
    assert [item.relation_direction for item in paper_one_group.relation_memories] == ["source"]

    assert paper_two_group.paper.title == "Paper Two"
    assert paper_two_group.paper.file_name == "paper-two.pdf"
    assert paper_two_group.paper.memory_count == 2
    assert [item.memory_type for item in paper_two_group.open_question_memories] == ["open_question_memory"]
    assert [item.relation_direction for item in paper_two_group.relation_memories] == ["target"]

    assert bundle.unscoped_memories == ()


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
        query_agent_client=StaticFinalAnswerQueryAgentClient("Model-generated final answer."),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.RUNNING
    assert result.answer == "Model-generated final answer."
    assert result.used_memory_citations[0].memory_id == "paper-memory-1"
    assert result.used_memory_citations[0].selection_reason.startswith("type=paper_memory")
    assert "rerank_strategy=model" in result.used_memory_citations[0].selection_reason
    assert trace_repository.list_steps(accepted.task_run.id)[0].action == "retrieve_session_memories"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "final_answer",
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
        query_agent_client=StaticFinalAnswerQueryAgentClient("Model-generated final answer."),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    messages = message_repository.list_by_session(session.id)
    assert result.answer == "Model-generated final answer."
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
        query_agent_client=StaticFinalAnswerQueryAgentClient("Model-generated final answer."),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.task_run.status is TaskRunStatus.RUNNING
    assert result.should_reread_source is True
    assert result.source_reread_chunks[0].chunk_id == "chunk-1"
    assert "matched_terms=" in result.source_reread_chunks[0].selection_reason
    assert "rerank_strategy=model" in result.source_reread_chunks[0].selection_reason
    assert result.answer == "Model-generated final answer."
    assert result.memory_selection_source == "rule_fallback"
    assert result.source_selection_source == "model"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "reread_source_passages",
        "final_answer",
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
        "final_answer",
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
        query_agent_client=_SequentialQueryAgentClient(agent_name="model_adapter"),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.tool_calls[0].tool_name == "search_session_memory"
    assert result.tool_calls[1].tool_name == "search_global_memory"
    assert result.tool_calls[-1].action_type == "final_answer"
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert steps[0].input_payload["planner_decision"]["selected_tool"] == "search_session_memory"
    assert steps[1].input_payload["planner_decision"]["selected_tool"] == "search_global_memory"
    assert steps[2].input_payload["planner_decision"]["selected_tool"] == "rerank_candidates"
    assert steps[3].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[4].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[5].action == "final_answer"


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
        query_agent_client=_SequentialQueryAgentClient(agent_name="model_adapter"),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.tool_calls[0].tool_name == "search_session_memory"
    assert result.tool_calls[1].tool_name == "search_global_memory"
    assert result.tool_calls[-1].action_type == "final_answer"
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert steps[3].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"
    assert steps[4].input_payload["planner_decision"]["selected_tool"] == "read_source_passages"


def test_query_execution_service_accepts_model_backed_agent_name_and_records_model_source() -> None:
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
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=_SequentialQueryAgentClient(agent_name="model_adapter"),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.tool_calls[0].agent_name == "model_adapter"
    assert result.tool_calls[0].fallback_used is False
    assert trace_repository.list_steps(accepted.task_run.id)[0].input_payload["planner_decision"]["agent_name"] == "model_adapter"
    assert trace_repository.list_steps(accepted.task_run.id)[0].input_payload["planner_decision"]["fallback_used"] is False


def test_query_execution_service_returns_list_session_papers_observation_to_model() -> None:
    class ListPapersThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "list_papers_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "session_papers" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="list_session_papers",
                    tool_parameters={"limit": 10},
                    rationale="model_requests_current_session_papers",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型根据 list_session_papers 的 observation 回答。",
                rationale="model_answers_from_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("List Papers"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.11111"), title="Listed Paper"))
    artifact_repository.save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path="C:/papers/listed.pdf",
            checksum="listed-checksum",
        )
    )
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    memory_repository.upsert_paper_memory(PaperMemory(id="paper-memory-1", paper_id=paper.id, confidence=ConfidenceScore(value=0.8)))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "当前有哪些论文？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    agent = ListPapersThenFinalAgent()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型根据 list_session_papers 的 observation 回答。"
    assert result.tool_calls[0].tool_name == "list_session_papers"
    assert result.tool_calls[0].tool_parameters == {"limit": 10}
    assert len(agent.requests) == 3
    observation = agent.requests[1].observations[0]
    assert observation.kind == "session_papers"
    assert observation.payload["papers"][0]["paper_id"] == paper.id
    assert observation.payload["papers"][0]["file_name"] == "listed.pdf"
    assert trace_repository.list_steps(accepted.task_run.id)[0].action == "list_session_papers"


def test_query_execution_service_returns_arxiv_import_observation_to_model(monkeypatch) -> None:
    class ImportThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "import_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "arxiv_import" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="import_arxiv_paper",
                    tool_parameters={"arxiv_id_or_url": "2401.12345"},
                    rationale="model_imports_arxiv_before_answering",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型已通过 import_arxiv_paper 导入论文。",
                rationale="model_answers_from_import_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    openviking_bundle = build_inmemory_openviking_surface_bundle()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Import In Query"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "请导入这篇 arXiv 论文并告诉我是否导入成功。")
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    ingest_execution_service = IngestExecutionService(
        message_repository=message_repository,
        materialization_service=IngestMaterializationService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            artifact_repository=artifact_repository,
            chunk_repository=chunk_repository,
            tool_registry=tool_registry,
        ),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        openviking_bundle=openviking_bundle,
    )
    agent = ImportThenFinalAgent()
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
        openviking_bundle=openviking_bundle,
    )
    task_runtime_service = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=query_execution_service,
        ingest_execution_service=ingest_execution_service,
        trace_repository=trace_repository,
    )
    message_intake_service = MessageIntakeService(
        task_run_service=task_run_service,
        openviking_bundle=openviking_bundle,
    )
    tool_registry.set_arxiv_import_service(
        ArxivImportToolService(
            message_intake_service=message_intake_service,
            task_runtime_service=task_runtime_service,
        )
    )

    result = query_execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型已通过 import_arxiv_paper 导入论文。"
    assert result.tool_calls[0].tool_name == "import_arxiv_paper"
    trace_steps = trace_repository.list_steps(accepted.task_run.id)
    assert trace_steps[0].action == "import_arxiv_paper"
    imported_runs = [run for run in task_run_service.list_runs(session.id) if run.id != accepted.task_run.id]
    assert len(imported_runs) == 1
    observation = agent.requests[1].observations[0]
    assert observation.kind == "arxiv_import"
    assert observation.payload["paper_id"]
    assert observation.payload["arxiv_id"] == "2401.12345"


def test_query_execution_service_returns_no_result_observation_when_arxiv_import_download_fails(monkeypatch) -> None:
    class ImportThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "import_failure_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "arxiv_import" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="import_arxiv_paper",
                    tool_parameters={"arxiv_id_or_url": "2401.12345"},
                    rationale="model_attempts_import_before_answering",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型确认这次没有可导入的 arXiv 结果。",
                rationale="model_answers_from_failed_import_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    monkeypatch.setattr(
        IngestMaterializationService,
        "_download_arxiv_pdf",
        lambda self, pdf_url, source_value: (_ for _ in ()).throw(URLError("down")),
    )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    openviking_bundle = build_inmemory_openviking_surface_bundle()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Import Failure In Query"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "请导入这篇 arXiv 论文，如果失败就告诉我没有结果。")
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    ingest_execution_service = IngestExecutionService(
        message_repository=message_repository,
        materialization_service=IngestMaterializationService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            artifact_repository=artifact_repository,
            chunk_repository=chunk_repository,
            tool_registry=tool_registry,
        ),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        openviking_bundle=openviking_bundle,
    )
    agent = ImportThenFinalAgent()
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
        openviking_bundle=openviking_bundle,
    )
    task_runtime_service = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=query_execution_service,
        ingest_execution_service=ingest_execution_service,
        trace_repository=trace_repository,
    )
    message_intake_service = MessageIntakeService(
        task_run_service=task_run_service,
        openviking_bundle=openviking_bundle,
    )
    tool_registry.set_arxiv_import_service(
        ArxivImportToolService(
            message_intake_service=message_intake_service,
            task_runtime_service=task_runtime_service,
        )
    )

    result = query_execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型确认这次没有可导入的 arXiv 结果。"
    observation = agent.requests[1].observations[0]
    assert observation.kind == "arxiv_import"
    assert observation.payload["success"] is False
    assert observation.payload["paper_id"] is None
    assert observation.payload["error"]["code"] == "no_results"
    assert observation.payload["error"]["upstream_error_code"] == "tool_execution_failed"
    trace_steps = trace_repository.list_steps(accepted.task_run.id)
    assert trace_steps[0].action == "import_arxiv_paper"
    assert trace_steps[0].result_payload["success"] is False


def test_query_execution_service_returns_arxiv_search_observation_to_model() -> None:
    class SearchThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "search_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "arxiv_search" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="search_arxiv",
                    tool_parameters={"query": "memory-routed research agents", "max_results": 5},
                    rationale="model_searches_arxiv_before_importing",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型已检索到候选 arXiv 论文。",
                rationale="model_answers_from_arxiv_search_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Search In Query"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "帮我找一些 arXiv 论文。")
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    tool_registry.set_arxiv_search_service(
        ArxivSearchService(
            http_get=lambda url, timeout: ArxivHttpResponse(
                status_code=200,
                body=_ARXIV_SEARCH_FEED.encode("utf-8"),
            )
        )
    )
    agent = SearchThenFinalAgent()
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = query_execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型已检索到候选 arXiv 论文。"
    assert result.tool_calls[0].tool_name == "search_arxiv"
    trace_steps = trace_repository.list_steps(accepted.task_run.id)
    assert trace_steps[0].action == "search_arxiv"
    observation = agent.requests[1].observations[0]
    assert observation.kind == "arxiv_search"
    assert observation.payload["success"] is True
    assert observation.payload["papers"][0]["arxiv_id"] == "2401.12345v1"
    assert observation.payload["papers"][0]["abs_url"] == "https://arxiv.org/abs/2401.12345v1"


def test_query_execution_service_returns_no_result_observation_when_arxiv_search_fails() -> None:
    class SearchThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "search_failure_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "arxiv_search" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="search_arxiv",
                    tool_parameters={"query": "memory-routed research agents", "max_results": 5},
                    rationale="model_searches_arxiv_before_answering",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型确认当前没有可用的 arXiv 搜索结果。",
                rationale="model_answers_from_failed_arxiv_search_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Search Failure In Query"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "帮我找一些 arXiv 论文，如果搜不到就告诉我没有结果。")
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    tool_registry.set_arxiv_search_service(
        ArxivSearchService(
            http_get=lambda url, timeout: (_ for _ in ()).throw(URLError("down"))
        )
    )
    agent = SearchThenFinalAgent()
    query_execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = query_execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型确认当前没有可用的 arXiv 搜索结果。"
    observation = agent.requests[1].observations[0]
    assert observation.kind == "arxiv_search"
    assert observation.payload["success"] is False
    assert observation.payload["count"] == 0
    assert observation.payload["papers"] == []
    assert observation.payload["error"]["code"] == "no_results"
    assert observation.payload["error"]["upstream_error_code"] == "network_error"
    trace_steps = trace_repository.list_steps(accepted.task_run.id)
    assert trace_steps[0].action == "search_arxiv"
    assert trace_steps[0].result_payload["success"] is False


def test_query_execution_service_returns_paper_memory_bundle_observation_to_model() -> None:
    class BundleThenFinalAgent:
        def __init__(self, paper_id: str) -> None:
            self._paper_id = paper_id
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "paper_bundle_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "paper_memory_bundle" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="get_paper_memory_bundle",
                    tool_parameters={"paper_id": self._paper_id, "source_chunk_limit": 2},
                    rationale="model_requests_one_paper_bundle",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型根据 paper memory bundle 回答。",
                rationale="model_answers_from_bundle_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Bundle Papers"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.22222"), title="Bundle Paper"))
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    memory_repository.upsert_paper_memory(PaperMemory(id="paper-memory-1", paper_id=paper.id, problem="Bundle problem", confidence=ConfidenceScore(value=0.8)))
    chunk_repository.save_many((Chunk(id="chunk-1", paper_id=paper.id, artifact_id="artifact-1", text="Bundle evidence."),))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "这篇论文的记忆是什么？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    agent = BundleThenFinalAgent(paper.id)
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "模型根据 paper memory bundle 回答。"
    assert result.tool_calls[0].tool_name == "get_paper_memory_bundle"
    assert result.tool_calls[0].tool_parameters["paper_id"] == paper.id
    assert len(agent.requests) == 2
    observation = agent.requests[1].observations[0]
    assert observation.kind == "paper_memory_bundle"
    bundle = observation.payload["bundle"]
    assert bundle["paper"]["paper_id"] == paper.id
    assert bundle["paper_memory"]["id"] == "paper-memory-1"
    assert bundle["evidence_source_chunks"][0]["chunk_id"] == "chunk-1"


def test_query_execution_service_returns_source_chunk_search_observation_to_model() -> None:
    class SourceChunkSearchThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "source_chunk_search_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            if not any(observation.kind == "source_chunk_search" for observation in request.observations):
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="search_source_chunks",
                    tool_parameters={"query": "mechanism", "top_k": 1},
                    rationale="model_requests_original_source_chunks",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="模型根据 source chunks 回答。",
                rationale="model_answers_from_source_chunk_observation",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Source Chunks"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.44444"), title="Source Chunk Paper"))
    artifact_repository = InMemoryArtifactRepository()
    artifact_repository.save(
        Artifact(
            id="artifact-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path="C:/papers/source-chunks.pdf",
            checksum="source-chunks-checksum",
        )
    )
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-1",
                paper_id=paper.id,
                artifact_id="artifact-1",
                text="The method improves accuracy through a new retrieval mechanism.",
                page=1,
                section="Abstract",
            )
        ]
    )
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "这篇论文的原文证据是什么？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    agent = SourceChunkSearchThenFinalAgent()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)
    response = QueryExecutionResponse.from_result(result)

    assert result.answer == "模型根据 source chunks 回答。"
    assert result.tool_calls[0].tool_name == "search_source_chunks"
    assert len(agent.requests) == 3
    observation = agent.requests[1].observations[0]
    assert observation.kind == "source_chunk_search"
    assert observation.payload["chunk_ids"] == ["chunk-1"]
    assert response.debug.decisions[0].tool_name == "search_source_chunks"
    assert response.debug.decisions[0].turn_index == 0
    assert response.debug.decisions[0].final_answer_present is False
    assert response.debug.observations_summary[0].kind == "source_chunk_search"
    assert response.debug.observations_summary[0].payload["chunk_ids"] == ["chunk-1"]
    assert trace_repository.list_steps(accepted.task_run.id)[0].action == "search_source_chunks"


def test_query_execution_service_serializes_cre_v2_bundle_flow_without_json_errors(tmp_path) -> None:
    captured_requests: list[dict[str, object]] = []

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        request_body = json.loads(body.decode("utf-8"))
        captured_requests.append(request_body)
        turn_index = len(captured_requests) - 1
        if turn_index == 0:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "action_type": "tool_call",
                                        "tool_name": "list_session_papers",
                                        "arguments": {"limit": 5},
                                        "rationale": "Current session papers are needed first.",
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )
        if turn_index == 1:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "action_type": "tool_call",
                                        "tool_name": "get_paper_memory_bundle",
                                        "arguments": {
                                            "paper_id": "paper-cre-v2",
                                            "source_chunk_limit": 3,
                                        },
                                        "rationale": "The selected paper bundle should answer the question.",
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
                }
            )
        if turn_index == 2:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "CRE_v2 的新想法是：论文提出了一个更清晰的长上下文位置敏感评估视角。"
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 14, "completion_tokens": 8, "total_tokens": 22},
                }
            )
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "CRE_v2 的新想法是：论文提出了一个更清晰的长上下文位置敏感评估视角。",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 16, "completion_tokens": 10, "total_tokens": 26},
            }
        )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("CRE_v2 Flow"))
    paper = paper_repository.save(
        Paper(
            id="paper-cre-v2",
            canonical_key=build_canonical_key(pdf_checksum="cre-v2-checksum"),
            title="Imported local PDF 583efe20-2783-42ba-bdaa-1a2016e46787-CRE_v2",
            abstract="A paper about long-context evaluation and positional robustness.",
        )
    )
    artifact_path = tmp_path / "CRE_v2.pdf"
    artifact_path.write_bytes(_build_minimal_pdf_bytes("CRE_v2 regression test PDF"))
    artifact = artifact_repository.save(
        Artifact(
            id="artifact-cre-v2",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(artifact_path),
            checksum="cre-v2-checksum",
            page_count=1,
        )
    )
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact.id,
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-cre-v2",
            paper_id=paper.id,
            problem="Position-aware evaluation remains incomplete.",
            method="The paper studies long-context benchmarking.",
            key_results=["The evaluation reveals positional sensitivity."],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    memory_repository.upsert_relation_memory(
        RelationMemory(
            id="relation-memory-cre-v2",
            source_paper=paper.id,
            target_paper="paper-related-2",
            relation_type=RelationType.COMPARES_WITH,
            summary="Compares positional evaluation across benchmarks.",
            evidence=["It compares benchmark behavior under positional shifts."],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-cre-v2",
            unresolved_question="Does the evaluation hold under broader context shifts?",
            related_papers=[paper.id],
            why_open=["The benchmark coverage is still narrow."],
            possible_followup=["Run a broader context-shift evaluation."],
            confidence=ConfidenceScore(value=0.7),
        )
    )
    chunk_repository.save_many(
        [
            Chunk(
                id="chunk-cre-v2-1",
                paper_id=paper.id,
                artifact_id=artifact.id,
                text="The paper argues that positional behavior should be evaluated more carefully under long contexts.",
                page=1,
                section="Introduction",
            )
        ]
    )
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "CRE_v2.pdf 的新想法是什么")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
        artifact_repository=artifact_repository,
    )
    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
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
        query_agent_client=ModelBackedQueryAgentClient(transport=transport, fallback=None, agent_name="deepseek_cre_v2_agent"),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "CRE_v2 的新想法是：论文提出了一个更清晰的长上下文位置敏感评估视角。"
    assert [tool_call.tool_name for tool_call in result.tool_calls[:2]] == ["list_session_papers", "get_paper_memory_bundle"]
    assert len(result.tool_calls) == 2
    assert len(captured_requests) == 3

    second_prompt = json.loads(captured_requests[1]["messages"][1]["content"])
    assert second_prompt["observations"][0]["kind"] == "session_papers"
    assert second_prompt["observations"][0]["payload"]["papers"][0]["file_name"] == "CRE_v2.pdf"
    assert second_prompt["observations"][0]["payload"]["papers"][0]["paper_id"] == paper.id

    finalization_request = captured_requests[2]
    assert "response_format" not in finalization_request
    finalization_prompt = json.loads(finalization_request["messages"][1]["content"])
    assert "state_summary" not in finalization_prompt
    assert "completed_actions" not in finalization_prompt
    assert "observations" not in finalization_prompt
    assert "evidence_view" in finalization_prompt
    bundle_observation = next(item for item in finalization_prompt["evidence_view"] if item["kind"] == "paper_memory_bundle")
    bundle = bundle_observation["payload"]["bundle"]
    assert bundle["paper"]["paper_id"] == paper.id
    assert bundle["paper"]["file_name"] == "CRE_v2.pdf"
    assert bundle["paper_memory"] == {
        "id": "paper-memory-cre-v2",
        "problem": "Position-aware evaluation remains incomplete.",
        "method": "The paper studies long-context benchmarking.",
        "novelty_claim": None,
        "key_results": ["The evaluation reveals positional sensitivity."],
        "limitations": [],
    }
    assert bundle["open_questions"] == [
        {
            "id": "open-question-cre-v2",
            "unresolved_question": "Does the evaluation hold under broader context shifts?",
            "related_papers": [paper.id],
            "why_open": ["The benchmark coverage is still narrow."],
            "possible_followup": ["Run a broader context-shift evaluation."],
        }
    ]
    assert bundle["relations"] == [
        {
            "id": "relation-memory-cre-v2",
            "source_paper": paper.id,
            "target_paper": "paper-related-2",
            "relation_type": RelationType.COMPARES_WITH.value,
            "summary": "Compares positional evaluation across benchmarks.",
        }
    ]
    assert bundle["evidence_source_chunks"][0]["chunk_id"] == "chunk-cre-v2-1"
    assert bundle["evidence_source_chunks"][0]["excerpt"] == "The paper argues that positional behavior should be evaluated more carefully under long contexts."
    assert finalization_prompt["query"] == "CRE_v2.pdf 的新想法是什么"
    assert "paper_memory_bundle" in json.dumps(finalization_prompt["evidence_view"], ensure_ascii=False)


def test_query_execution_response_safely_serializes_observation_payloads() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Response Safety")
    session_repository.save(session)
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "Safety check")
    when = datetime(2026, 4, 27, 15, 30, tzinfo=timezone.utc)
    result = QueryExecutionResult(
        task_run=accepted.task_run,
        answer="safe",
        retrieval_plan=RetrievalPlan(
            session_memories=MemoryRetrievalResult(memories=(), coverage_score=0.0, matched_query_terms=(), selection_reasons=()),
            global_memories=MemoryRetrievalResult(memories=(), coverage_score=0.0, matched_query_terms=(), selection_reasons=()),
            related_paper_ids=(),
            should_reread_source=False,
            reread_reason="",
            memory_confidence=0.0,
        ),
        should_reread_source=False,
        reread_reason="",
        memory_selection_source="rule_fallback",
        memory_selection_fallback_used=False,
        used_memory_citations=(),
        matched_query_terms=(),
        source_selection_source=None,
        source_selection_fallback_used=False,
        source_reread_chunks=(),
        observations=(
            AgentObservation(
                kind="json_safe",
                summary="payload contains datetime and enum",
                payload={
                    "when": when,
                    "enum": _JsonSafeEnum.READY,
                    "model": _JsonSafeModel(value=7),
                },
            ),
        ),
        tool_calls=(),
    )

    response = QueryExecutionResponse.from_result(result)

    assert response.debug.observations_summary[0].payload == {
        "when": when.isoformat(),
        "enum": "ready",
        "model": {"value": 7},
    }


def test_query_execution_service_empty_session_memory_does_not_mean_no_session_papers() -> None:
    class SearchThenListThenFinalAgent:
        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "search_then_list_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            observed_kinds = {observation.kind for observation in request.observations}
            if "memory_search" not in observed_kinds:
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="search_session_memory",
                    rationale="model_checks_session_memory_first",
                )
            if "session_papers" not in observed_kinds:
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="list_session_papers",
                    rationale="model_checks_session_documents_after_empty_memory",
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="检索为空，但当前 session 仍有论文。",
                rationale="model_answers_after_distinguishing_memory_from_documents",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("No Memory Has Paper"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.33333"), title="Paper Without Memory"))
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "有哪些论文？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    agent = SearchThenListThenFinalAgent()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=agent,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "检索为空，但当前 session 仍有论文。"
    assert result.tool_calls[0].tool_name == "search_session_memory"
    assert result.tool_calls[1].tool_name == "list_session_papers"
    assert agent.requests[1].observations[0].payload["memory_ids"] == []
    assert agent.requests[2].observations[1].payload["papers"][0]["paper_id"] == paper.id


def test_query_execution_service_tool_failure_preserves_structured_error_fields() -> None:
    class MissingPaperIdAgent:
        def __init__(self) -> None:
            self._agent_name = "missing_paper_id_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            return AgentTurnDecision(
                action_type=AgentActionType.TOOL_CALL,
                tool_name="get_paper_memory_bundle",
                rationale="model_selected_bundle_without_required_business_parameter",
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Structured Error"))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "读取某篇论文记忆")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=MissingPaperIdAgent(),
    )

    with pytest.raises(QueryExecutionError) as error:
        execution_service.execute_query_run(session.id, accepted.task_run.id)

    detail = error.value.to_dict()
    assert detail["error_code"] == "invalid_parameter"
    assert detail["failed_stage"] == "tool_execution"
    assert detail["run_id"] == accepted.task_run.id
    assert detail["tool_name"] == "get_paper_memory_bundle"
    assert "paper_id" in detail["validation_error"]


def test_query_execution_service_labels_empty_model_response_as_structured_model_error() -> None:
    class EmptyResponseAgent:
        def __init__(self) -> None:
            self._agent_name = "empty_response_agent"
            self.fallback_reason = "RuntimeError: DeepSeek query-agent response contained empty content."
            self.fallback_used = False
            self.failure_detail = {
                "failure_stage_detail": "empty_content",
                "status_code": 200,
                "repair_attempted": True,
                "raw_response_preview": "{\"choices\":[]}",
                "content_preview": None,
            }

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            return None

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = SessionService(session_repository=session_repository).create_session("Empty Response")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "当前有哪些论文信息？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository)
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
        tool_registry=tool_registry,
        query_tool_executor=QueryToolExecutor(tool_registry),
        query_agent_client=EmptyResponseAgent(),
    )

    with pytest.raises(QueryExecutionError) as error:
        execution_service.execute_query_run(session.id, accepted.task_run.id)

    detail = error.value.to_dict()
    assert detail["error_code"] == "model_empty_response"
    assert detail["failed_stage"] == "model_decision"
    assert "empty content" in detail["error_message"]
    assert detail["run_id"] == accepted.task_run.id
    assert detail["failure_stage_detail"] == "empty_content"
    assert detail["status_code"] == 200
    assert detail["repair_attempted"] is True
    assert detail["raw_response_preview"] == "{\"choices\":[]}"
    failure_steps = trace_repository.list_steps(accepted.task_run.id)
    assert failure_steps[-1].action == "model_decision_failed"
    assert failure_steps[-1].status == "failed"
    assert failure_steps[-1].result_payload["failure_stage_detail"] == "empty_content"


def test_task_run_streaming_service_run_failed_event_includes_structured_error() -> None:
    class FailingRuntime:
        def __init__(self, task_run_service: TaskRunService, broker: RuntimeEventBroker) -> None:
            self._task_run_service = task_run_service
            self._broker = broker

        def execute_running_task_run(self, *, session_id: str, run_id: str) -> None:
            raise QueryExecutionError(
                QueryFailureDetail(
                    error_code="model_decision_failed",
                    failed_stage="model_decision",
                    error_message="adapter returned no valid decision",
                    run_id=run_id,
                    fallback_reason="validation_failed",
                    validation_error="missing final_answer",
                )
            )

        def fail_running_task_run(self, session_id: str, run_id: str, reason: str, error: dict[str, object] | None = None):
            failed_run = self._task_run_service.fail_run(session_id, run_id, reason)
            self._broker.publish_run_failed(failed_run, reason, error)
            return failed_run

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    broker = RuntimeEventBroker()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Stream Failure"))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "触发失败")
    streaming_service = TaskRunStreamingService(
        task_run_service=task_run_service,
        task_runtime_service=FailingRuntime(task_run_service, broker),
        runtime_event_broker=broker,
    )

    streaming_service.start_query_run(session.id, accepted.task_run.id)
    subscription = streaming_service.subscribe(session.id, accepted.task_run.id)
    events = list(subscription.iter_events(timeout_seconds=2.0))
    failed_event = [event for event in events if event is not None and event.event_type == "run_failed"][0]

    assert failed_event.payload["error"]["error_code"] == "model_decision_failed"
    assert failed_event.payload["error"]["failed_stage"] == "model_decision"
    assert failed_event.payload["error"]["run_id"] == accepted.task_run.id
    assert failed_event.payload["error"]["fallback_reason"] == "validation_failed"


def test_task_runtime_service_marks_query_run_failed_on_query_execution_error() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    broker = RuntimeEventBroker()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Runtime Failure"))
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "触发失败")

    class FailingQueryExecutionService:
        def execute_query_run(self, *, session_id: str, run_id: str):
            raise QueryExecutionError(
                QueryFailureDetail(
                    error_code="model_decision_failed",
                    failed_stage="model_decision",
                    error_message="adapter returned no valid decision",
                    run_id=run_id,
                    fallback_reason="validation_failed",
                    validation_error="missing final_answer",
                    failure_stage_detail="normalize_choice",
                    raw_response_preview="{\"choices\":[]}",
                )
            )

    class DummyIngestExecutionService:
        def execute_ingest_run(self, *, session_id: str, run_id: str):
            raise AssertionError("ingest path should not run in this test")

    runtime = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=FailingQueryExecutionService(),
        ingest_execution_service=DummyIngestExecutionService(),
        trace_repository=trace_repository,
        runtime_event_broker=broker,
    )

    with pytest.raises(QueryExecutionError):
        runtime.execute_query_run(session.id, accepted.task_run.id)

    failed_run = task_run_service.get_run(session.id, accepted.task_run.id)
    assert failed_run.status is TaskRunStatus.FAILED
    assert failed_run.finish_reason == "model_decision_failed"
    subscription = broker.subscribe(accepted.task_run.id, replay=True)
    events = [event for event in subscription.iter_events(timeout_seconds=1.0) if event is not None]
    failed_event = [event for event in events if event.event_type == "run_failed"][0]
    assert failed_event.payload["reason"] == "model_decision_failed"
    assert failed_event.payload["error"]["failure_stage_detail"] == "normalize_choice"


def test_query_execution_service_saves_json_safe_trace_steps_and_streams_them() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    broker = RuntimeEventBroker()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Trace Safety"))
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    accepted = task_run_service.accept_followup_query(session.id, "Trace safety")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=None,
        context_rerank_service=None,
        session_repository=session_repository,
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        runtime_event_broker=broker,
    )
    subscription = broker.subscribe(accepted.task_run.id)
    when = datetime(2026, 4, 27, 15, 45, tzinfo=timezone.utc)

    service._save_tool_trace_step(  # noqa: SLF001
        session_id=session.id,
        run_id=accepted.task_run.id,
        action="list_session_papers",
        input_payload={"when": when, "enum": _JsonSafeEnum.READY},
        result_payload={"payload": {"when": when, "enum": _JsonSafeEnum.READY, "model": _JsonSafeModel(value=5)}},
    )

    event = next(item for item in subscription.iter_events(timeout_seconds=0.1) if item is not None)

    assert trace_repository.list_steps(accepted.task_run.id)[0].result_payload == {
        "payload": {
            "when": when.isoformat(),
            "enum": "ready",
            "model": {"value": 5},
        }
    }
    assert json.dumps(event.to_dict(), ensure_ascii=False)
    assert event.to_dict()["payload"]["trace_step"]["result_payload"]["payload"]["when"] == when.isoformat()


def test_task_runtime_service_failure_keeps_completed_step_count() -> None:
    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    trace_repository = InMemoryTraceRepository()
    task_run_service = TaskRunService(session_repository=session_repository, message_repository=message_repository, trace_repository=trace_repository)
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Failure Step Count"))
    accepted = task_run_service.accept_followup_query(session.id, "当前有哪些论文信息？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    trace_repository.save_step(
        TraceStep(
            run_id=accepted.task_run.id,
            action="list_session_papers",
            input_payload={"limit": 20},
            result_payload={"papers": [], "total_count": 0},
        )
    )
    runtime = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=object(),
        ingest_execution_service=object(),
        trace_repository=trace_repository,
    )

    failed_run = runtime.fail_running_task_run(session.id, accepted.task_run.id, "boom", {"error_code": "model_empty_response"})

    assert failed_run.status is TaskRunStatus.FAILED
    assert failed_run.step_count == 1


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
    assert compose_step.action == "final_answer"
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
            if request.final_answer_allowed:
                return AgentTurnDecision(
                    action_type=AgentActionType.FINAL_ANSWER,
                    final_answer="Turn-generated final answer.",
                    rationale="turn_agent_finishes_when_final_answer_is_allowed",
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
    assert len(turn_client.requests) == 2
    assert turn_client.requests[-1].observations == ()
    assert turn_client.requests[-1].final_answer_allowed is True
    assert turn_client.requests[-1].allowed_actions == ()
    assert result.tool_calls[-1].action_type == "final_answer"
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "final_answer",
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
            if request.final_answer_allowed:
                return AgentTurnDecision(
                    action_type=AgentActionType.FINAL_ANSWER,
                    final_answer="Model-generated final answer.",
                    rationale="low_yield_agent_finishes_with_final_answer",
                    stop_reason=AgentStopReason.FINAL_ANSWER_READY,
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

    assert result.answer == "Model-generated final answer."
    assert result.tool_calls[-1].action_type == "final_answer"
    assert result.tool_calls[-1].agent_name == "low_yield_first_agent"
    assert result.tool_calls[-1].fallback_used is False
    assert len(turn_client.requests) >= 1
    compose_step = trace_repository.list_steps(accepted.task_run.id)[-1]
    assert compose_step.action == "final_answer"
    assert compose_step.input_payload["planner_decision"]["action_type"] == "final_answer"
    assert compose_step.input_payload["planner_decision"]["agent_name"] == "low_yield_first_agent"
    assert compose_step.input_payload["planner_decision"]["fallback_used"] is False


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
        query_agent_client=_SequentialQueryAgentClient(answer_text="你好，我在。", agent_name="greeting_agent"),
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    assert result.answer == "\u4f60\u597d\uff0c\u6211\u5728\u3002"
    assert result.tool_calls == ()
    steps = trace_repository.list_steps(accepted.task_run.id)
    assert [step.action for step in steps] == ["retrieve_session_memories", "retrieve_global_memories", "rerank_context_candidates", "decide_reread_source", "reread_source_passages", "final_answer"]
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
            if request.final_answer_allowed:
                return AgentTurnDecision(
                    action_type=AgentActionType.FINAL_ANSWER,
                    final_answer="Model-generated final answer.",
                    rationale="duplicate_signature_agent_finishes_with_final_answer",
                    stop_reason=AgentStopReason.FINAL_ANSWER_READY,
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

    assert result.answer == "Model-generated final answer."
    assert result.tool_calls[0].tool_name == "search_session_memory"
    assert result.tool_calls[1].tool_name == "search_global_memory"
    assert result.tool_calls[-1].action_type == "final_answer"
    assert result.tool_calls[-1].agent_name == "duplicate_global_search_agent"
    assert result.tool_calls[-1].fallback_used is False
    assert len(turn_client.requests) >= 3
    assert [step.action for step in trace_repository.list_steps(accepted.task_run.id)] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "reread_source_passages",
        "final_answer",
    ]
    compose_step = trace_repository.list_steps(accepted.task_run.id)[-1]
    assert compose_step.input_payload["planner_decision"]["action_type"] == "final_answer"


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
        query_agent_client=StaticFinalAnswerQueryAgentClient("Model-generated final answer."),
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


def test_ingest_materialization_service_retries_arxiv_download_with_certifi_context(monkeypatch) -> None:
    import importlib
    import research_agent.services.ingest_materialization_service as ingest_materialization_module

    reloaded_module = importlib.reload(ingest_materialization_module)
    service = reloaded_module.IngestMaterializationService(
        session_repository=InMemorySessionRepository(),
        paper_repository=InMemoryPaperRepository(),
        artifact_repository=InMemoryArtifactRepository(),
    )
    expected_pdf = _build_minimal_pdf_bytes("ArXiv text that should be extracted.")
    calls: list[ssl.SSLContext | None] = []

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self._payload

    def fake_urlopen(request, timeout=30, context=None):
        calls.append(context)
        if context is None:
            raise URLError(ssl.SSLCertVerificationError("certificate verify failed"))
        return _Response(expected_pdf)

    monkeypatch.setattr(reloaded_module, "urlopen", fake_urlopen)

    payload = service._download_arxiv_pdf("https://arxiv.org/pdf/2401.12345.pdf", "https://arxiv.org/abs/2401.12345")

    assert payload == expected_pdf
    assert len(calls) == 2
    assert calls[0] is None
    assert isinstance(calls[1], ssl.SSLContext)


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


def test_ingest_materialization_service_is_idempotent_for_repeated_local_pdf_imports_same_session(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    session = SessionService(session_repository=session_repository).create_session("Repeated Materialize PDF")
    session_repository.save(session)
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(_build_minimal_pdf_bytes("Local PDF text that should be extracted."))
    service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
    )

    first = service.materialize_pdf_source(session.id, str(pdf_path))
    second = service.materialize_pdf_source(session.id, str(pdf_path))
    third = service.materialize_pdf_source(session.id, str(pdf_path))

    assert first.paper.id == second.paper.id == third.paper.id
    assert first.artifact.id == second.artifact.id == third.artifact.id
    assert first.session_document.id == second.session_document.id == third.session_document.id
    assert len(session_repository.list_documents(session.id)) == 1
    assert len(chunk_repository.list_by_artifact_id(first.artifact.id)) == 1
    assert len(chunk_repository.list_by_paper_ids([first.paper.id])) == 1
    assert first.chunk_count == second.chunk_count == third.chunk_count == 1


def test_ingest_materialization_service_reuses_artifact_across_sessions_for_same_pdf(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    chunk_repository = InMemoryChunkRepository()
    session_one = SessionService(session_repository=session_repository).create_session("Session One")
    session_two = SessionService(session_repository=session_repository).create_session("Session Two")
    session_repository.save(session_one)
    session_repository.save(session_two)
    pdf_path = tmp_path / "shared.pdf"
    pdf_path.write_bytes(_build_minimal_pdf_bytes("Shared PDF text that should be extracted."))
    service = IngestMaterializationService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        artifact_repository=artifact_repository,
        chunk_repository=chunk_repository,
    )

    first = service.materialize_pdf_source(session_one.id, str(pdf_path))
    second = service.materialize_pdf_source(session_two.id, str(pdf_path))

    assert first.paper.id == second.paper.id
    assert first.artifact.id == second.artifact.id
    assert len(session_repository.list_documents(session_one.id)) == 1
    assert len(session_repository.list_documents(session_two.id)) == 1
    assert len(chunk_repository.list_by_artifact_id(first.artifact.id)) == 1
    assert len(chunk_repository.list_by_paper_ids([first.paper.id])) == 1


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


def test_ingest_analysis_service_cleans_model_input_before_prompting() -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Model input cleaning")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-cleaning",
            canonical_key=build_canonical_key(pdf_checksum="cleaning-checksum"),
            title="Cleaning Test Paper",
            abstract="A compact abstract for cleaning verification.",
        )
    )
    artifact_id = "artifact-cleaning"
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id=paper.id,
            source_type=SourceType.PDF,
            artifact_id=artifact_id,
        )
    )
    chunks = chunk_repository.save_many(
        [
            Chunk(
                id="chunk-clean-1",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\nWe propose a new method that improves accuracy over the baseline.\nACME Research 2026",
                page=1,
                section="Introduction",
            ),
            Chunk(
                id="chunk-clean-2",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\nWe propose a new method that improves accuracy over the baseline.\nACME Research 2026",
                page=2,
                section="Introduction",
            ),
            Chunk(
                id="chunk-clean-3",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\nReferences\n[1] A. Author. Prior work. ACM 2020.\nACME Research 2026",
                page=3,
                section="References",
            ),
            Chunk(
                id="chunk-clean-4",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\n1 2 3 4 5 6 7 8 9 10 | 11 | 12\nACME Research 2026",
                page=4,
                section="Results",
            ),
            Chunk(
                id="chunk-clean-5",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\n@@@ ### 1234 5678 §§§\nACME Research 2026",
                page=5,
                section="Appendix",
            ),
            Chunk(
                id="chunk-clean-6",
                paper_id=paper.id,
                artifact_id=artifact_id,
                text="ACME Research 2026\nWe preserve provenance and remove invisible characters.\u200b\u0000\nACME Research 2026",
                page=6,
                section="Discussion",
            ),
        ]
    )

    class RecordingTransport:
        def __init__(self) -> None:
            self.prompts: list[StructuredIngestExtractionPrompt] = []

        def extract(self, prompt: StructuredIngestExtractionPrompt) -> StructuredIngestExtractionChoice:
            self.prompts.append(prompt)
            return StructuredIngestExtractionChoice(
                understanding={
                    "topic": {"text": "The paper presents a cleaning-aware input pipeline.", "evidence_chunk_ids": ["chunk-clean-1"], "confidence": 0.9},
                    "problem": {"text": "The paper reduces prompt noise before model extraction.", "evidence_chunk_ids": ["chunk-clean-1"], "confidence": 0.9},
                    "method": {"text": "It normalizes text, removes headers, references, and duplicates.", "evidence_chunk_ids": ["chunk-clean-6"], "confidence": 0.9},
                    "novelty_claims": [
                        {"text": "The cleaning pipeline preserves provenance while reducing noise.", "evidence_chunk_ids": ["chunk-clean-6"], "confidence": 0.9}
                    ],
                    "key_results": [
                        {"text": "Repeated chunks are collapsed before prompting.", "evidence_chunk_ids": ["chunk-clean-2"], "confidence": 0.9}
                    ],
                    "experiment_design": {"text": "The paper evaluates cleaning on repeated, noisy, and reference-like chunks.", "evidence_chunk_ids": ["chunk-clean-3"], "confidence": 0.9},
                    "limitations": [
                        {"text": "Table-like content is compressed instead of fully expanded.", "evidence_chunk_ids": ["chunk-clean-4"], "confidence": 0.9}
                    ],
                    "open_questions": [
                        {"text": "How much more noise can be removed without harming recall?", "evidence_chunk_ids": ["chunk-clean-5"], "confidence": 0.9}
                    ],
                    "evidence_chunk_ids": ["chunk-clean-1", "chunk-clean-4", "chunk-clean-5", "chunk-clean-6"],
                    "confidence": 0.9,
                },
                paper=StructuredIngestPaperDraft(
                    problem="The paper reduces prompt noise before model extraction.",
                    method="It normalizes text, removes headers, references, and duplicates.",
                    key_results=("Repeated chunks are collapsed before prompting.",),
                    limitations=("Table-like content is compressed instead of fully expanded.",),
                    novelty_claim="The cleaning pipeline preserves provenance while reducing noise.",
                    evidence_candidate_ids=("chunk-clean-1", "chunk-clean-4", "chunk-clean-6"),
                    confidence=0.9,
                ),
                relation=None,
                open_question=StructuredIngestOpenQuestionDraft(
                    unresolved_question="How much more noise can be removed without harming recall?",
                    why_open=("Table-like content is compressed instead of fully expanded.",),
                    possible_followup=("Measure recall after stronger cleaning heuristics.",),
                    evidence_candidate_ids=("chunk-clean-5",),
                    confidence=0.8,
                ),
                paper_summary=StructuredIngestPaperSummaryDraft(
                    what_it_is_about="The paper presents a cleaning-aware input pipeline.",
                    problem_solved="The paper reduces prompt noise before model extraction.",
                    new_ideas=("The cleaning pipeline preserves provenance while reducing noise.",),
                    limitations=("Table-like content is compressed instead of fully expanded.",),
                    suggestions_or_questions=("Measure recall after stronger cleaning heuristics.",),
                    evidence_candidate_ids=("chunk-clean-1", "chunk-clean-4", "chunk-clean-6"),
                    confidence=0.9,
                ),
                needs_more_context=False,
                context_hints=(),
                rationale="Cleaning test.",
            )

    transport = RecordingTransport()
    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=ModelBackedIngestExtractionClient(transport=transport),
    )

    result = service.analyze(session.id, paper.id)
    prompt = transport.prompts[0]
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in prompt.candidate_passages}
    cleanup = result.extraction_debug["input_cleanup"]

    assert cleanup["chunks_before"] == 6
    assert cleanup["chunks_after"] == 4
    assert cleanup["removed_duplicate_count"] == 1
    assert cleanup["references_removed_count"] == 1
    assert cleanup["low_quality_count"] >= 1
    assert cleanup["total_chars_after"] < cleanup["total_chars_before"]
    assert any(record["chunk_id"] == "chunk-clean-3" and record["removed_reason"] == "references_section" for record in cleanup["chunks"])
    assert any(record["chunk_id"] == "chunk-clean-2" and record["removed_reason"] == "duplicate_text" for record in cleanup["chunks"])
    assert any(record["chunk_id"] == "chunk-clean-4" and "table_or_noise" in record["quality_flags"] for record in cleanup["chunks"])
    assert any(record["chunk_id"] == "chunk-clean-6" and "\u200b" not in record["cleaned_text"] and "\x00" not in record["cleaned_text"] for record in cleanup["chunks"])
    assert "chunk-clean-2" not in candidate_by_id
    assert "chunk-clean-3" not in candidate_by_id
    assert candidate_by_id["chunk-clean-4"]["cleaned_text"].startswith("[table-like content compressed:")
    assert "quality_flags" in candidate_by_id["chunk-clean-4"]
    assert "removed_reason" in candidate_by_id["chunk-clean-4"]
    assert result.paper_summary.what_it_is_about == "The paper presents a cleaning-aware input pipeline."


def test_ingest_analysis_service_table_like_chunks_keep_compact_excerpt() -> None:
    service = IngestAnalysisService(
        session_repository=InMemorySessionRepository(),
        paper_repository=InMemoryPaperRepository(),
        chunk_repository=InMemoryChunkRepository(),
        memory_repository=InMemoryMemoryRepository(),
    )

    compressed = service._compress_table_like_text(
        "Table 1. Accuracy on benchmark A: 0.91 | benchmark B: 0.87 | benchmark C: 0.92 | benchmark D: 0.89"
    )

    assert compressed.startswith("[table-like content compressed:")
    assert "Table 1. Accuracy" in compressed


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
    assert result.relation_memory is None
    assert result.open_question_memory.related_papers == [current_paper.id]
    assert len(memory_repository.list_paper_memories_for_papers([current_paper.id])) == 1
    assert len(memory_repository.list_relation_memories_for_papers([current_paper.id])) == 0
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
    assert result.relation_memory is None
    assert result.open_question_memory.related_papers == [current_paper.id]
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

    unavailable = "无法基于当前论文内容稳定生成该字段。"
    assert result.paper_memory.paper_id == current_paper.id
    assert result.paper_memory.problem == unavailable
    assert result.paper_memory.method == unavailable
    assert result.paper_memory.key_results == [unavailable]
    assert result.paper_memory.limitations == [unavailable]
    assert result.open_question_memory.related_papers == [current_paper.id]
    assert result.relation_memory is None
    assert result.paper_summary.what_it_is_about == unavailable
    assert result.paper_summary.problem_solved == unavailable
    assert result.paper_summary.new_ideas == (unavailable,)
    assert result.paper_summary.suggestions_or_questions == (unavailable,)


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

    unavailable = "无法基于当前论文内容稳定生成该字段。"
    assert result.paper_summary.what_it_is_about == unavailable
    assert result.paper_summary.problem_solved == unavailable
    assert result.paper_summary.new_ideas == (
        "Unlike DeepSeek-V3.2, which discarded thinking traces upon each new user turn, DeepSeek-V4 series retain the complete reasoning history across all rounds, including across user message boundaries.",
    )
    assert result.paper_summary.suggestions_or_questions == ("Revisit the source after removing reference noise.",)


def test_ingest_analysis_service_records_schema_validation_failure(tmp_path) -> None:
    session_repository = InMemorySessionRepository()
    paper_repository = InMemoryPaperRepository()
    chunk_repository = InMemoryChunkRepository()
    memory_repository = InMemoryMemoryRepository()
    session = SessionService(session_repository=session_repository).create_session("Schema Failure")
    session_repository.save(session)
    current_pdf = tmp_path / "schema-failure.pdf"
    current_pdf.write_bytes(
        _build_minimal_pdf_bytes(
            "We propose a new method. It improves accuracy over the baseline. Future work remains on robustness."
        )
    )
    current_paper = paper_repository.save(
        Paper(
            id="paper-schema-failure-1",
            canonical_key=build_canonical_key(pdf_checksum="schema-failure-checksum"),
            title="Schema Failure Paper",
        )
    )
    artifact = InMemoryArtifactRepository().save(
        Artifact(
            id="artifact-schema-failure-1",
            kind=ArtifactKind.LOCAL_PDF,
            uri_or_path=str(current_pdf),
            checksum="schema-failure-checksum",
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
                id="chunk-schema-failure-1",
                paper_id=current_paper.id,
                artifact_id=artifact.id,
                text="We propose a new method.",
                page=1,
                section="Abstract",
            )
        ]
    )

    class FailingClient:
        def extract(self, request: StructuredIngestExtractionPrompt) -> object:
            raise StructuredIngestExtractionParseError(
                extractor_stage="schema_validation",
                raw_response_preview='{"choices":[...]}',
                normalized_payload_preview='{"understanding":{"topic":{"value":"bad"}}}',
                validation_error="ValidationError: field type mismatch",
                failed_field="understanding.novelty_claims",
            )

    service = IngestAnalysisService(
        session_repository=session_repository,
        paper_repository=paper_repository,
        chunk_repository=chunk_repository,
        memory_repository=memory_repository,
        extraction_client=FailingClient(),
    )

    result = service.analyze(session.id, current_paper.id)

    assert result.extraction_debug is not None
    assert result.extraction_debug["extraction_mode"] == "extractor_failed"
    assert result.extraction_debug["extraction_stage"] == "schema_validation"
    assert result.extraction_debug["failed_field"] == "understanding.novelty_claims"
    assert result.extraction_debug["validation_error"] == "ValidationError: field type mismatch"
    assert result.extraction_debug["raw_response_preview"] == '{"choices":[...]}'
    assert result.extraction_debug["normalized_payload_preview"] == '{"understanding":{"topic":{"value":"bad"}}}'
    assert result.paper_summary.what_it_is_about == "无法基于当前论文内容稳定生成该字段。"


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
                    new_ideas=("本文采用检索与评估结合的方式展开分析。",),
                    limitations=("Run larger-scale experiments.",),
                    suggestions_or_questions=("继续回读原文，并补充更多证据。",),
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
    unavailable = "无法基于当前论文内容稳定生成该字段。"
    assert result.paper_memory.problem == unavailable
    assert result.paper_summary.what_it_is_about == unavailable
    assert result.paper_summary.problem_solved == unavailable
    assert result.paper_summary.new_ideas == (unavailable,)
    assert result.paper_summary.suggestions_or_questions == (unavailable,)


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
    assert result.extraction_debug is not None
    assert result.extraction_debug["extraction_mode"] == "full_text"
    assert "chunk-main-summary" in result.extraction_debug["input_chunk_ids"]


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


def test_memory_content_shapes_query_answer() -> None:
    """Demonstrate that stored memory content influences the model's answer."""

    class MemoryAwareAgent:
        """Agent that inspects memory observations and uses them in its answer."""

        def __init__(self) -> None:
            self.requests: list[AgentTurnRequest] = []
            self._agent_name = "memory_aware_agent"

        def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
            self.requests.append(request)
            observed_kinds = {obs.kind for obs in request.observations}
            # Step 1: search session memory
            if "memory_search" not in observed_kinds:
                return AgentTurnDecision(
                    action_type=AgentActionType.TOOL_CALL,
                    tool_name="search_session_memory",
                    rationale="check_session_memory",
                )
            # Step 2: answer based on memory content
            memory_obs = next(obs for obs in request.observations if obs.kind == "memory_search")
            memories = memory_obs.payload.get("memories", [])
            if memories:
                summary = memories[0].get("summary", "")
                return AgentTurnDecision(
                    action_type=AgentActionType.FINAL_ANSWER,
                    final_answer=f"根据记忆，{summary}",
                    rationale="answer_from_memory",
                    stop_reason=AgentStopReason.FINAL_ANSWER_READY,
                )
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer="没有找到相关记忆。",
                rationale="no_memory",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )

    session_repository = InMemorySessionRepository()
    message_repository = InMemoryMessageRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    trace_repository = InMemoryTraceRepository()
    timeline_repository = InMemoryTimelineRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Memory Test"))
    session_repository.save_document(
        SessionDocument(session_id=session.id, paper_id="paper-1", source_type=SourceType.PDF, artifact_id="artifact-1")
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="pm-1",
            paper_id="paper-1",
            problem="Transformer self-attention has O(n^2) complexity",
            method="Linear attention with random feature maps reduces to O(n)",
            key_results=["10x faster inference on long sequences", "Comparable accuracy on GLUE benchmark"],
            limitations=["Slight degradation on copying tasks"],
            novelty_claim="First practical linear attention for production LLMs",
            source_refs=[],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    task_run_service = TaskRunService(
        session_repository=session_repository,
        message_repository=message_repository,
        trace_repository=trace_repository,
    )
    accepted = task_run_service.accept_followup_query(session.id, "这篇论文的方法是什么？")
    task_run_service.mark_running(session.id, accepted.task_run.id)
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    agent = MemoryAwareAgent()
    execution_service = QueryExecutionService(
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        trace_repository=trace_repository,
        timeline_repository=timeline_repository,
        query_agent_client=agent,
    )

    result = execution_service.execute_query_run(session.id, accepted.task_run.id)

    # The answer should contain content from the PaperMemory
    assert "Linear attention" in result.answer
    assert "O(n)" in result.answer
    # The agent should have received memory observations
    assert len(agent.requests) >= 2
    memory_obs = [obs for obs in agent.requests[1].observations if obs.kind == "memory_search"]
    assert len(memory_obs) == 1
    assert len(memory_obs[0].payload["memories"]) > 0
    assert memory_obs[0].payload["memories"][0]["confidence"] == 0.9
