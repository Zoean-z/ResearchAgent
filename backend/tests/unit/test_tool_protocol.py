"""Unit tests for the frozen query tool protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.adapters.llm import ModelBackedQueryToolPlannerClient, StaticStructuredPlannerTransport
from research_agent.adapters.openviking import (
    OpenVikingMemoryRecord,
    OpenVikingRetrievalAdapter,
    build_inmemory_openviking_surface_bundle,
)
from research_agent.adapters.storage import (
    InMemoryChunkRepository,
    InMemoryArtifactRepository,
    InMemoryMemoryRepository,
    InMemoryPaperRepository,
    InMemorySessionRepository,
)
from research_agent.domain.enums import RelationType, SourceType
from research_agent.domain.models import Chunk, OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SessionDocument
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services import ContextRerankService, MemoryExtractionService, RetrievalService, SessionService
from research_agent.tools import (
    HeuristicQueryToolPlannerClient,
    InternalToolRegistry,
    PlannerBackedQueryAgentClient,
    QueryAgentState,
    QueryToolExecutor,
    QueryToolPlannerState,
    StaticFinalAnswerQueryAgentClient,
)
from research_agent.tools.protocol import (
    ChunkDescriptor,
    ComposeAnswerInput,
    ComposeAnswerOutput,
    GetPaperMemoryBundleInput,
    GetPaperMemoryBundleOutput,
    ListSessionPapersInput,
    ListSessionPapersOutput,
    MemoryDescriptor,
    OpenVikingHitDescriptor,
    QueryToolName,
    QUERY_TOOL_DEFINITIONS,
    QUERY_TOOL_BY_NAME,
    ReadSourcePassagesInput,
    ReadSourcePassagesOutput,
    RerankCandidatesInput,
    RerankCandidatesOutput,
    SearchGlobalMemoryInput,
    SearchGlobalMemoryOutput,
    SearchOpenVikingMemoryInput,
    SearchOpenVikingMemoryOutput,
    SearchSessionMemoryInput,
    SearchSessionMemoryOutput,
    SearchSourceChunksInput,
    SearchSourceChunksOutput,
    ToolDefinition,
    ToolError,
    ToolErrorCode,
    ToolRequest,
    ToolResponse,
    get_query_tool_definition,
    is_query_tool,
    validate_tool_request,
)


# ---------------------------------------------------------------------------
# QueryToolName enum
# ---------------------------------------------------------------------------


def test_query_tool_names_are_stable() -> None:
    assert QueryToolName.SEARCH_SESSION_MEMORY.value == "search_session_memory"
    assert QueryToolName.SEARCH_GLOBAL_MEMORY.value == "search_global_memory"
    assert QueryToolName.SEARCH_OPENVIKING_MEMORY.value == "search_openviking_memory"
    assert QueryToolName.SEARCH_SOURCE_CHUNKS.value == "search_source_chunks"
    assert QueryToolName.LIST_SESSION_PAPERS.value == "list_session_papers"
    assert QueryToolName.GET_PAPER_MEMORY_BUNDLE.value == "get_paper_memory_bundle"
    assert QueryToolName.RERANK_CANDIDATES.value == "rerank_candidates"
    assert QueryToolName.READ_SOURCE_PASSAGES.value == "read_source_passages"
    assert QueryToolName.COMPOSE_ANSWER.value == "compose_answer"


def test_query_tool_name_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        QueryToolName("unknown_tool")


def test_query_tool_name_rejects_ingest_tools() -> None:
    with pytest.raises(ValueError):
        QueryToolName("register_paper")

    with pytest.raises(ValueError):
        QueryToolName("extract_memories")


# ---------------------------------------------------------------------------
# ToolErrorCode enum
# ---------------------------------------------------------------------------


def test_tool_error_codes_are_stable() -> None:
    assert ToolErrorCode.INVALID_PARAMETER.value == "invalid_parameter"
    assert ToolErrorCode.ENTITY_NOT_FOUND.value == "entity_not_found"
    assert ToolErrorCode.TOOL_NOT_FOUND.value == "tool_not_found"
    assert ToolErrorCode.TOOL_EXECUTION_FAILED.value == "tool_execution_failed"
    assert ToolErrorCode.CANDIDATE_KIND_UNSUPPORTED.value == "candidate_kind_unsupported"
    assert ToolErrorCode.EMPTY_CANDIDATES.value == "empty_candidates"


# ---------------------------------------------------------------------------
# MemoryDescriptor
# ---------------------------------------------------------------------------


def test_memory_descriptor_requires_memory_type() -> None:
    with pytest.raises(ValidationError):
        MemoryDescriptor(
            memory_id="mem-1",
            memory_type="invalid_type",  # type: ignore[arg-type]
            summary="test",
            confidence=0.8,
        )


def test_memory_descriptor_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        MemoryDescriptor(
            memory_id="mem-1",
            memory_type="paper_memory",
            summary="test",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        MemoryDescriptor(
            memory_id="mem-2",
            memory_type="relation_memory",
            summary="test",
            confidence=-0.3,
        )


def test_memory_descriptor_accepts_valid_values() -> None:
    desc = MemoryDescriptor(
        memory_id="mem-1",
        memory_type="open_question_memory",
        summary="Why does this model fail on small datasets?",
        confidence=0.75,
        matched_terms=("model", "fail"),
        selection_reason="type=open_question_memory; matched_terms=model,fail",
    )
    assert desc.memory_id == "mem-1"
    assert desc.memory_type == "open_question_memory"
    assert desc.confidence == 0.75
    assert desc.matched_terms == ("model", "fail")


# ---------------------------------------------------------------------------
# ChunkDescriptor
# ---------------------------------------------------------------------------


def test_chunk_descriptor_defaults() -> None:
    desc = ChunkDescriptor(
        chunk_id="chunk-1",
        paper_id="paper-1",
        excerpt="This paper proposes a novel approach...",
    )
    assert desc.page is None
    assert desc.section is None
    assert desc.matched_terms == ()
    assert desc.selection_reason == ""


def test_chunk_descriptor_full() -> None:
    desc = ChunkDescriptor(
        chunk_id="chunk-2",
        paper_id="paper-2",
        excerpt="Our experiments show a 12% improvement...",
        page=3,
        section="results",
        matched_terms=("experiments", "improvement"),
        selection_reason="matched_terms=experiments,improvement; section=results; page=3",
    )
    assert desc.page == 3
    assert desc.section == "results"
    assert len(desc.matched_terms) == 2


# ---------------------------------------------------------------------------
# Input model validation
# ---------------------------------------------------------------------------


class TestSearchSessionMemoryInput:
    def test_valid_minimal(self) -> None:
        inp = SearchSessionMemoryInput(session_id="s1", query="What is the main result?")
        assert inp.session_id == "s1"
        assert inp.top_k == 5

    def test_invalid_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            SearchSessionMemoryInput(session_id="s1", query="")

    def test_top_k_clamped(self) -> None:
        with pytest.raises(ValidationError):
            SearchSessionMemoryInput(session_id="s1", query="q", top_k=0)
        with pytest.raises(ValidationError):
            SearchSessionMemoryInput(session_id="s1", query="q", top_k=21)


class TestSearchGlobalMemoryInput:
    def test_valid_minimal(self) -> None:
        inp = SearchGlobalMemoryInput(query="transformer architecture")
        assert inp.related_paper_ids is None
        assert inp.top_k == 5

    def test_with_paper_filter(self) -> None:
        inp = SearchGlobalMemoryInput(
            query="attention mechanism",
            related_paper_ids=["paper-1", "paper-2"],
            top_k=10,
        )
        assert len(inp.related_paper_ids) == 2


class TestSearchOpenVikingMemoryInput:
    def test_valid_session_scope(self) -> None:
        inp = SearchOpenVikingMemoryInput(scope="session", session_id="s1", query="accuracy")
        assert inp.scope == "session"
        assert inp.session_id == "s1"
        assert inp.top_k == 5

    def test_valid_global_scope(self) -> None:
        inp = SearchOpenVikingMemoryInput(scope="global", query="accuracy")
        assert inp.scope == "global"
        assert inp.session_id is None

    def test_rejects_session_scope_without_session_id(self) -> None:
        with pytest.raises(ValidationError):
            SearchOpenVikingMemoryInput(scope="session", query="accuracy")


class TestRerankCandidatesInput:
    def test_valid_memory_kind(self) -> None:
        inp = RerankCandidatesInput(
            candidate_kind="memory",
            query="test query",
            candidates=[
                MemoryDescriptor(memory_id="m1", memory_type="paper_memory", summary="a", confidence=0.8),
                MemoryDescriptor(memory_id="m2", memory_type="paper_memory", summary="b", confidence=0.7),
            ],
        )
        assert inp.candidate_kind == "memory"
        assert inp.top_k == 3

    def test_valid_chunk_kind(self) -> None:
        inp = RerankCandidatesInput(
            candidate_kind="chunk",
            query="test",
            candidates=[
                ChunkDescriptor(chunk_id="c1", paper_id="p1", excerpt="alpha"),
                ChunkDescriptor(chunk_id="c2", paper_id="p1", excerpt="beta"),
            ],
        )
        assert inp.candidate_kind == "chunk"

    def test_rejects_invalid_kind(self) -> None:
        with pytest.raises(ValidationError):
            RerankCandidatesInput(
                candidate_kind="invalid",  # type: ignore[arg-type]
                query="test",
                candidates=[MemoryDescriptor(memory_id="m1", memory_type="paper_memory", summary="x", confidence=0.9)],
            )

    def test_rejects_empty_candidates(self) -> None:
        with pytest.raises(ValidationError):
            RerankCandidatesInput(
                candidate_kind="memory",
                query="test",
                candidates=[],
            )

    def test_rejects_mismatched_candidate_types(self) -> None:
        with pytest.raises(ValidationError):
            RerankCandidatesInput(
                candidate_kind="memory",
                query="test",
                candidates=[ChunkDescriptor(chunk_id="c1", paper_id="p1", excerpt="x")],
            )


class TestComposeAnswerInput:
    def test_valid_minimal(self) -> None:
        inp = ComposeAnswerInput(query="What is the novelty?")
        assert inp.query == "What is the novelty?"
        assert inp.memory_context == []
        assert inp.source_context == []

    def test_with_context(self) -> None:
        mem = MemoryDescriptor(
            memory_id="m1",
            memory_type="paper_memory",
            summary="A novel training method",
            confidence=0.9,
        )
        inp = ComposeAnswerInput(
            query="What is the novelty?",
            memory_context=[mem],
            session_memory_count=3,
            global_memory_count=1,
            memory_selection_source="model",
        )
        assert len(inp.memory_context) == 1
        assert inp.session_memory_count == 3
        assert inp.memory_selection_source == "model"


# ---------------------------------------------------------------------------
# Output model construction
# ---------------------------------------------------------------------------


def test_search_session_memory_output() -> None:
    mem = MemoryDescriptor(
        memory_id="m1",
        memory_type="paper_memory",
        summary="Paper about transformers",
        confidence=0.85,
    )
    out = SearchSessionMemoryOutput(
        memories=(mem,),
        coverage_score=0.2,
        matched_query_terms=("transformer",),
        selection_reasons=("matched transformer keyword",),
    )
    assert len(out.memories) == 1
    assert out.coverage_score == 0.2


def test_search_openviking_memory_output() -> None:
    hit = OpenVikingHitDescriptor(
        item_kind="paper_memory",
        item_id="m1",
        session_id="s1",
        score=0.9,
        summary="Paper about transformers",
    )
    mem = MemoryDescriptor(
        memory_id="m1",
        memory_type="paper_memory",
        summary="Paper about transformers",
        confidence=0.9,
    )
    out = SearchOpenVikingMemoryOutput(
        scope="session",
        hits=(hit,),
        memories=(mem,),
        coverage_score=0.4,
        matched_query_terms=("transformer",),
        selection_reasons=("matched transformer keyword",),
        matched_local_memory_ids=("m1",),
        matched_local_count=1,
    )
    assert out.scope == "session"
    assert len(out.hits) == 1
    assert len(out.memories) == 1
    assert out.matched_local_memory_ids == ("m1",)


def test_search_source_chunks_output() -> None:
    chunk = ChunkDescriptor(
        chunk_id="c1",
        paper_id="p1",
        excerpt="The key insight is...",
    )
    out = SearchSourceChunksOutput(
        chunks=(chunk,),
        coverage_score=0.5,
        matched_query_terms=("insight",),
        selection_reasons=("matched in abstract",),
    )
    assert len(out.chunks) == 1
    assert out.coverage_score == 0.5


def test_list_session_papers_schema_does_not_require_session_id() -> None:
    params = ListSessionPapersInput()
    assert params.limit == 20
    assert ListSessionPapersOutput(papers=(), total_count=0).total_count == 0


def test_get_paper_memory_bundle_schema_requires_business_paper_id_only() -> None:
    params = GetPaperMemoryBundleInput(paper_id="paper-1")
    assert params.paper_id == "paper-1"
    assert params.source_chunk_limit == 5
    output = GetPaperMemoryBundleOutput(
        bundle={
            "paper": {"paper_id": "paper-1", "title": "Paper"},
            "empty_fields": ("paper_memory",),
        }
    )
    assert output.bundle.empty_fields == ("paper_memory",)


def test_rerank_candidates_output() -> None:
    out = RerankCandidatesOutput(
        selected_ids=("m3", "m1"),
        selection_source="model",
        fallback_used=False,
        rationale="model_reranked_memory_candidates_from_5_to_2",
    )
    assert out.selected_ids == ("m3", "m1")
    assert not out.fallback_used


def test_compose_answer_output() -> None:
    mem = MemoryDescriptor(
        memory_id="m1",
        memory_type="paper_memory",
        summary="test",
        confidence=0.8,
    )
    out = ComposeAnswerOutput(
        evidence_package="evidence package for model answer",
        citations=(mem,),
        source_citations=(),
        memory_influence="Answer was grounded entirely in memory; no source reread was needed.",
    )
    assert "evidence package" in out.evidence_package
    assert len(out.citations) == 1
    assert len(out.source_citations) == 0


# ---------------------------------------------------------------------------
# ToolRequest / ToolResponse / ToolError
# ---------------------------------------------------------------------------


def test_tool_request_serialization() -> None:
    request = ToolRequest(
        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
        parameters={"session_id": "s1", "query": "What is attention?", "top_k": 5},
    )
    data = request.model_dump()
    assert data["tool_name"] == "search_session_memory"
    assert data["parameters"]["session_id"] == "s1"


def test_tool_request_rejects_invalid_tool_name() -> None:
    with pytest.raises(ValidationError):
        ToolRequest(tool_name="invalid_tool", parameters={})  # type: ignore[arg-type]


def test_tool_response_serialization() -> None:
    response = ToolResponse(
        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
        result={"memories": [], "coverage_score": 0.0},
    )
    data = response.model_dump()
    assert data["tool_name"] == "search_session_memory"
    assert data["result"]["coverage_score"] == 0.0


def test_tool_error_construction() -> None:
    error = ToolError(
        tool_name="search_session_memory",
        error_code=ToolErrorCode.ENTITY_NOT_FOUND,
        message="Session 'xyz' does not exist.",
        details={"session_id": "xyz"},
    )
    assert error.error_code == ToolErrorCode.ENTITY_NOT_FOUND
    assert error.details == {"session_id": "xyz"}


def test_tool_error_minimal() -> None:
    error = ToolError(
        tool_name="compose_answer",
        error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
        message="Composition failed unexpectedly.",
    )
    assert error.details is None


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


def test_tool_definition_binds_models() -> None:
    definition = get_query_tool_definition(QueryToolName.SEARCH_SESSION_MEMORY)
    assert definition is not None
    assert definition.name == QueryToolName.SEARCH_SESSION_MEMORY
    assert definition.input_model is SearchSessionMemoryInput
    assert definition.output_model is SearchSessionMemoryOutput


def test_tool_definition_binds_all_seven_tools() -> None:
    for definition in QUERY_TOOL_DEFINITIONS:
        assert isinstance(definition, ToolDefinition)
        assert isinstance(definition.name, QueryToolName)
        assert definition.description
        assert definition.input_model is not None
        assert definition.output_model is not None


# ---------------------------------------------------------------------------
# QUERY_TOOL_BY_NAME completeness
# ---------------------------------------------------------------------------


def test_query_tool_by_name_has_all_query_tools() -> None:
    assert len(QUERY_TOOL_BY_NAME) == len(QueryToolName)
    for name in QueryToolName:
        assert name in QUERY_TOOL_BY_NAME
        assert QUERY_TOOL_BY_NAME[name].name == name


# ---------------------------------------------------------------------------
# get_query_tool_definition
# ---------------------------------------------------------------------------


def test_get_query_tool_definition_by_enum() -> None:
    definition = get_query_tool_definition(QueryToolName.COMPOSE_ANSWER)
    assert definition is not None
    assert definition.name == QueryToolName.COMPOSE_ANSWER


def test_get_query_tool_definition_by_string() -> None:
    definition = get_query_tool_definition("rerank_candidates")
    assert definition is not None
    assert definition.name == QueryToolName.RERANK_CANDIDATES


def test_get_query_tool_definition_unknown_string() -> None:
    assert get_query_tool_definition("nonexistent") is None


def test_get_query_tool_definition_ingest_tool() -> None:
    assert get_query_tool_definition("register_paper") is None
    assert get_query_tool_definition("extract_memories") is None


def test_get_query_tool_definition_openviking_tool() -> None:
    definition = get_query_tool_definition("search_openviking_memory")
    assert definition is not None
    assert definition.name is QueryToolName.SEARCH_OPENVIKING_MEMORY


# ---------------------------------------------------------------------------
# is_query_tool
# ---------------------------------------------------------------------------


def test_is_query_tool_valid() -> None:
    assert is_query_tool("search_session_memory") is True
    assert is_query_tool("search_openviking_memory") is True
    assert is_query_tool("compose_answer") is True


def test_is_query_tool_invalid() -> None:
    assert is_query_tool("register_paper") is False
    assert is_query_tool("random_tool") is False
    assert is_query_tool("") is False


# ---------------------------------------------------------------------------
# validate_tool_request
# ---------------------------------------------------------------------------


def test_validate_valid_tool_request() -> None:
    request = ToolRequest(
        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
        parameters={"session_id": "s1", "query": "test"},
    )
    assert validate_tool_request(request) is None


def test_validate_tool_not_found() -> None:
    request = ToolRequest(
        tool_name=QueryToolName.COMPOSE_ANSWER,  # valid enum
        parameters={},
    )
    # We need to test with an unknown tool name string
    error = validate_tool_request(
        ToolRequest.model_construct(
            tool_name="nonexistent_tool",  # type: ignore[arg-type]
            parameters={},
        )
    )
    assert error is not None
    assert error.error_code == ToolErrorCode.TOOL_NOT_FOUND


def test_validate_invalid_parameters() -> None:
    request = ToolRequest(
        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
        parameters={"session_id": "s1"},  # missing required "query"
    )
    error = validate_tool_request(request)
    assert error is not None
    assert error.error_code == ToolErrorCode.INVALID_PARAMETER
    assert error.tool_name == "search_session_memory"


def test_validate_rerank_candidates_invalid_kind() -> None:
    request = ToolRequest(
        tool_name=QueryToolName.RERANK_CANDIDATES,
        parameters={
            "candidate_kind": "invalid_kind",
            "query": "test",
            "candidates": [MemoryDescriptor(memory_id="x", memory_type="paper_memory", summary="x", confidence=0.6).model_dump()],
        },
    )
    error = validate_tool_request(request)
    assert error is not None
    assert error.error_code == ToolErrorCode.INVALID_PARAMETER


# ---------------------------------------------------------------------------
# Round-trip: all 6 tools can validate
# ---------------------------------------------------------------------------


def test_all_tools_accept_valid_minimal_parameters() -> None:
    valid_parameters: dict[QueryToolName, dict] = {
        QueryToolName.SEARCH_SESSION_MEMORY: {"session_id": "s1", "query": "q"},
        QueryToolName.SEARCH_GLOBAL_MEMORY: {"query": "q"},
        QueryToolName.SEARCH_OPENVIKING_MEMORY: {"scope": "session", "session_id": "s1", "query": "q"},
        QueryToolName.SEARCH_SOURCE_CHUNKS: {"session_id": "s1", "query": "q"},
        QueryToolName.LIST_SESSION_PAPERS: {},
        QueryToolName.GET_PAPER_MEMORY_BUNDLE: {"paper_id": "paper-1"},
        QueryToolName.RERANK_CANDIDATES: {
            "candidate_kind": "memory",
            "query": "q",
            "candidates": [
                MemoryDescriptor(memory_id="a", memory_type="paper_memory", summary="a", confidence=0.8).model_dump(),
                MemoryDescriptor(memory_id="b", memory_type="paper_memory", summary="b", confidence=0.7).model_dump(),
            ],
        },
        QueryToolName.READ_SOURCE_PASSAGES: {"session_id": "s1", "query": "q"},
        QueryToolName.COMPOSE_ANSWER: {"query": "q"},
    }
    for tool_name, params in valid_parameters.items():
        request = ToolRequest(tool_name=tool_name, parameters=params)
        error = validate_tool_request(request)
        assert error is None, f"Tool {tool_name.value} should validate but got: {error}"


def test_query_tool_executor_search_session_memory_returns_tool_response() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    session = SessionService(session_repository=session_repository).create_session("Executor")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Executor paper",
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
            key_results=["Improved accuracy"],
            confidence=ConfidenceScore(value=0.9),
        )
    )
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    registry = InternalToolRegistry(
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
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(
        ToolRequest(
            tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
            parameters={"session_id": session.id, "query": "accuracy", "top_k": 5},
        )
    )

    assert isinstance(outcome, ToolResponse)
    assert outcome.tool_name is QueryToolName.SEARCH_SESSION_MEMORY
    assert outcome.result["memories"][0]["memory_id"] == "paper-memory-1"


def test_query_tool_executor_list_session_papers_uses_runtime_context() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    artifact_repository = InMemoryArtifactRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Papers"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.12345"), title="Session Paper"))
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    memory_repository.upsert_paper_memory(PaperMemory(id="paper-memory-1", paper_id=paper.id, confidence=ConfidenceScore(value=0.8)))
    registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository),
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
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(
        ToolRequest(tool_name=QueryToolName.LIST_SESSION_PAPERS, parameters={"limit": 5}),
        runtime_context={"session_id": session.id},
    )

    assert isinstance(outcome, ToolResponse)
    assert outcome.result["papers"][0]["paper_id"] == "paper-1"
    assert outcome.result["papers"][0]["memory_count"] == 1


def test_query_tool_executor_get_paper_memory_bundle_returns_memory_and_evidence() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    session = session_repository.save(SessionService(session_repository=session_repository).create_session("Bundle"))
    paper = paper_repository.save(Paper(id="paper-1", canonical_key=build_canonical_key(arxiv_id="2401.12345"), title="Bundle Paper"))
    session_repository.save_document(SessionDocument(session_id=session.id, paper_id=paper.id, source_type=SourceType.PDF, artifact_id="artifact-1"))
    memory_repository.upsert_paper_memory(PaperMemory(id="paper-memory-1", paper_id=paper.id, problem="Problem", confidence=ConfidenceScore(value=0.8)))
    memory_repository.upsert_open_question_memory(OpenQuestionMemory(id="open-1", unresolved_question="What remains?", related_papers=[paper.id]))
    memory_repository.upsert_relation_memory(
        RelationMemory(
            id="relation-1",
            source_paper=paper.id,
            target_paper="paper-2",
            relation_type=RelationType.COMPLEMENTS,
            summary="Complements prior work.",
        )
    )
    chunk_repository.save_many((Chunk(id="chunk-1", paper_id=paper.id, artifact_id="artifact-1", text="Evidence chunk text."),))
    registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=RetrievalService(session_repository=session_repository, memory_repository=memory_repository, chunk_repository=chunk_repository),
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
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(ToolRequest(tool_name=QueryToolName.GET_PAPER_MEMORY_BUNDLE, parameters={"paper_id": paper.id}))

    assert isinstance(outcome, ToolResponse)
    bundle = outcome.result["bundle"]
    assert bundle["paper"]["paper_id"] == paper.id
    assert bundle["paper_memory"]["id"] == "paper-memory-1"
    assert bundle["open_questions"][0]["id"] == "open-1"
    assert bundle["relations"][0]["id"] == "relation-1"
    assert bundle["evidence_source_chunks"][0]["chunk_id"] == "chunk-1"


def test_query_tool_executor_search_openviking_memory_returns_tool_response() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("Executor OV")
    session_repository.save(session)
    paper = paper_repository.save(
        Paper(
            id="paper-1",
            canonical_key=build_canonical_key(arxiv_id="2401.12345"),
            title="Executor paper",
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
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="Why is the delta still open?",
            related_papers=[paper.id],
            confidence=ConfidenceScore(value=0.6),
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="open-question-1",
            memory_kind="open_question_memory",
            session_id=session.id,
            paper_id=paper.id,
            payload={"unresolved_question": "Why is the delta still open?"},
        )
    )
    retrieval_service = RetrievalService(
        session_repository=session_repository,
        memory_repository=memory_repository,
        chunk_repository=chunk_repository,
    )
    openviking_adapter = OpenVikingRetrievalAdapter(
        session_repository=session_repository,
        memory_repository=memory_repository,
        memory_surface=bundle.memories,
    )
    registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=retrieval_service,
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
        openviking_retrieval_adapter=openviking_adapter,
    )
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(
        ToolRequest(
            tool_name=QueryToolName.SEARCH_OPENVIKING_MEMORY,
            parameters={"scope": "session", "session_id": session.id, "query": "delta", "top_k": 5},
        )
    )

    assert isinstance(outcome, ToolResponse)
    assert outcome.tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY
    assert outcome.result["matched_local_memory_ids"] == ("open-question-1",)


def test_query_tool_executor_invalid_request_returns_tool_error() -> None:
    registry = InternalToolRegistry(
        paper_repository=InMemoryPaperRepository(),
        retrieval_service=RetrievalService(
            session_repository=InMemorySessionRepository(),
            memory_repository=InMemoryMemoryRepository(),
            chunk_repository=InMemoryChunkRepository(),
        ),
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=InMemorySessionRepository(),
            paper_repository=InMemoryPaperRepository(),
            chunk_repository=InMemoryChunkRepository(),
            memory_repository=InMemoryMemoryRepository(),
        ),
    )
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(
        ToolRequest(
            tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
            parameters={"session_id": "missing-session"},
        )
    )

    assert isinstance(outcome, ToolError)
    assert outcome.error_code is ToolErrorCode.INVALID_PARAMETER


def test_query_tool_executor_rerank_candidates_uses_protocol_descriptors() -> None:
    registry = InternalToolRegistry(
        paper_repository=InMemoryPaperRepository(),
        retrieval_service=RetrievalService(
            session_repository=InMemorySessionRepository(),
            memory_repository=InMemoryMemoryRepository(),
            chunk_repository=InMemoryChunkRepository(),
        ),
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=InMemorySessionRepository(),
            paper_repository=InMemoryPaperRepository(),
            chunk_repository=InMemoryChunkRepository(),
            memory_repository=InMemoryMemoryRepository(),
        ),
    )
    executor = QueryToolExecutor(registry)

    outcome = executor.execute(
        ToolRequest(
            tool_name=QueryToolName.RERANK_CANDIDATES,
            parameters={
                "candidate_kind": "memory",
                "query": "accuracy",
                "candidates": [
                    MemoryDescriptor(
                        memory_id="m1",
                        memory_type="paper_memory",
                        summary="Higher accuracy",
                        confidence=0.8,
                        matched_terms=("accuracy",),
                    ).model_dump(),
                    MemoryDescriptor(
                        memory_id="m2",
                        memory_type="paper_memory",
                        summary="Weaker match",
                        confidence=0.4,
                        matched_terms=(),
                    ).model_dump(),
                ],
                "top_k": 1,
            },
        )
    )

    assert isinstance(outcome, ToolResponse)
    assert outcome.tool_name is QueryToolName.RERANK_CANDIDATES
    assert tuple(outcome.result["selected_ids"]) == ("m1",)


def test_heuristic_query_tool_planner_chooses_expected_next_tool() -> None:
    planner = HeuristicQueryToolPlannerClient()

    first = planner.choose_next_tool(
        query="Did it improve accuracy?",
        state=QueryToolPlannerState(),
        allowed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
    )
    assert first is not None
    assert first.tool_name is QueryToolName.SEARCH_SESSION_MEMORY

    after_rerank = planner.choose_next_tool(
        query="Did it improve accuracy?",
        state=QueryToolPlannerState(
            completed_tools=(
                QueryToolName.SEARCH_SESSION_MEMORY,
                QueryToolName.SEARCH_GLOBAL_MEMORY,
                QueryToolName.RERANK_CANDIDATES,
            ),
            should_reread_source=True,
        ),
        allowed_tools=(QueryToolName.READ_SOURCE_PASSAGES,),
    )
    assert after_rerank is not None
    assert after_rerank.tool_name is QueryToolName.READ_SOURCE_PASSAGES
    assert "reread" in after_rerank.rationale


def test_heuristic_query_tool_planner_returns_none_for_disallowed_tools() -> None:
    planner = HeuristicQueryToolPlannerClient()

    decision = planner.choose_next_tool(
        query="Did it improve accuracy?",
        state=QueryToolPlannerState(),
        allowed_tools=(QueryToolName.SEARCH_SOURCE_CHUNKS,),
    )

    assert decision is not None
    assert decision.tool_name is QueryToolName.SEARCH_SOURCE_CHUNKS


def test_model_backed_query_tool_planner_accepts_valid_model_choice() -> None:
    planner = ModelBackedQueryToolPlannerClient(
        transport=StaticStructuredPlannerTransport(
            tool_name="search_global_memory",
            rationale="model_selected_global_memory_after_session_memory",
        ),
        fallback=HeuristicQueryToolPlannerClient(),
    )

    decision = planner.choose_next_tool(
        query="Did it improve accuracy?",
        state=QueryToolPlannerState(
            completed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
        ),
        allowed_tools=(QueryToolName.SEARCH_GLOBAL_MEMORY,),
    )

    assert decision is not None
    assert decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY
    assert decision.planner_name == "model_adapter"
    assert decision.fallback_used is False
    assert decision.rationale == "model_selected_global_memory_after_session_memory"


def test_model_backed_query_tool_planner_falls_back_when_model_choice_is_invalid() -> None:
    planner = ModelBackedQueryToolPlannerClient(
        transport=StaticStructuredPlannerTransport(
            tool_name="compose_answer",
            rationale="invalid_model_choice",
        ),
        fallback=HeuristicQueryToolPlannerClient(),
    )

    decision = planner.choose_next_tool(
        query="Did it improve accuracy?",
        state=QueryToolPlannerState(),
        allowed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
    )

    assert decision is not None
    assert decision.tool_name is QueryToolName.SEARCH_SESSION_MEMORY
    assert decision.planner_name == "model_adapter"
    assert decision.fallback_used is True
    assert "fallback_after_model_adapter_error" in decision.rationale


def test_planner_backed_query_agent_returns_tool_call_decision() -> None:
    agent = PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient())

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(),
        allowed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
        final_answer_allowed=False,
    )

    assert decision is not None
    assert decision.action_type == "tool_call"
    assert decision.tool_name is QueryToolName.SEARCH_SESSION_MEMORY
    assert decision.agent_name == "planner_backed_agent"


def test_static_final_answer_query_agent_returns_final_answer_when_allowed() -> None:
    agent = StaticFinalAnswerQueryAgentClient("Agent final answer")

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(
            completed_tools=(
                QueryToolName.SEARCH_SESSION_MEMORY,
                QueryToolName.SEARCH_GLOBAL_MEMORY,
                QueryToolName.RERANK_CANDIDATES,
            ),
            selected_memory_ids=("memory-1",),
            should_reread_source=False,
        ),
        allowed_tools=(QueryToolName.COMPOSE_ANSWER,),
        final_answer_allowed=True,
    )

    assert decision is not None
    assert decision.action_type == "final_answer"
    assert decision.final_answer == "Agent final answer"
