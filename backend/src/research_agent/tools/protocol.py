"""Frozen query-only tool protocol with stable request/response and error semantics.

This module defines the Phase 1 query tool contract. Every query tool that a model
or runtime loop may invoke is declared here with its name, description, structured
input model, and structured output model. The protocol is intentionally decoupled
from domain models so it can evolve independently.

Query tool subset (frozen for Phase 1):
  - search_arxiv
  - import_arxiv_paper
  - search_session_memory
  - search_global_memory
  - search_source_chunks
  - list_recent_messages
  - get_conversation_context
  - rerank_candidates
  - read_source_passages
  - compose_answer
  - list_session_papers
  - get_paper_memory_bundle
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from research_agent.tools.arxiv_reference import normalize_arxiv_id_or_url


# ---------------------------------------------------------------------------
# Tool identity
# ---------------------------------------------------------------------------


class QueryToolName(StrEnum):
    """Frozen set of tools available for query execution.

    Only these tools may be invoked during a follow-up query run.
    The runtime must reject any tool not in this enumeration.
    """

    IMPORT_ARXIV_PAPER = "import_arxiv_paper"
    SEARCH_ARXIV = "search_arxiv"
    SEARCH_SESSION_MEMORY = "search_session_memory"
    SEARCH_GLOBAL_MEMORY = "search_global_memory"
    SEARCH_OPENVIKING_MEMORY = "search_openviking_memory"
    SEARCH_SOURCE_CHUNKS = "search_source_chunks"
    LIST_RECENT_MESSAGES = "list_recent_messages"
    GET_CONVERSATION_CONTEXT = "get_conversation_context"
    LIST_SESSION_PAPERS = "list_session_papers"
    GET_PAPER_MEMORY_BUNDLE = "get_paper_memory_bundle"
    RERANK_CANDIDATES = "rerank_candidates"
    READ_SOURCE_PASSAGES = "read_source_passages"
    COMPOSE_ANSWER = "compose_answer"


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class ToolErrorCode(StrEnum):
    """Standard error codes returned by tool invocations."""

    INVALID_PARAMETER = "invalid_parameter"
    """A required parameter is missing or has an invalid value."""

    ENTITY_NOT_FOUND = "entity_not_found"
    """A referenced entity (session, paper, etc.) does not exist."""

    TOOL_NOT_FOUND = "tool_not_found"
    """The requested tool name is not in the frozen query tool set."""

    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    """The tool raised an unexpected error during execution."""

    CANDIDATE_KIND_UNSUPPORTED = "candidate_kind_unsupported"
    """The rerank candidate_kind is neither 'memory' nor 'chunk'."""

    EMPTY_CANDIDATES = "empty_candidates"
    """Rerank or retrieval was called with an empty candidate pool."""


class ToolError(BaseModel):
    """Structured error returned when a tool invocation fails.

    Every tool failure must return this model so callers can branch
    on error_code instead of parsing free-text messages.
    """

    tool_name: str = Field(description="The tool that produced this error")
    error_code: ToolErrorCode = Field(description="Machine-readable error category")
    message: str = Field(description="Human-readable error description")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional context such as the invalid parameter name"
    )


# ---------------------------------------------------------------------------
# Shared descriptors (lightweight, serializable views)
# ---------------------------------------------------------------------------


class MemoryDescriptor(BaseModel):
    """Lightweight view of a memory record returned by retrieval tools.

    This is intentionally a flat, serializable subset. Callers that need
    the full domain object must resolve it from the repository layer.
    """

    memory_id: str = Field(description="Unique memory identifier")
    memory_type: Literal["paper_memory", "relation_memory", "open_question_memory"] = Field(
        description="Which of the three structured memory kinds this is"
    )
    summary: str = Field(description="Short human-readable summary for selection decisions")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score at retrieval time")
    matched_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that matched this memory"
    )
    selection_reason: str = Field(
        default="", description="Why this memory was selected over others"
    )


class OpenVikingHitDescriptor(BaseModel):
    """Lightweight view of an OpenViking hit returned by explicit retrieval."""

    item_kind: str = Field(description="Kind of item matched in OpenViking")
    item_id: str = Field(description="Stable item identifier")
    session_id: str | None = Field(default=None, description="Session that owns the hit if available")
    score: float = Field(ge=0.0, le=1.0, description="OpenViking retrieval score")
    summary: str = Field(description="Short summary for selection decisions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Opaque hit metadata")


class ChunkDescriptor(BaseModel):
    """Lightweight view of a source chunk returned by passage retrieval tools."""

    chunk_id: str = Field(description="Unique chunk identifier")
    paper_id: str = Field(description="Paper this chunk belongs to")
    excerpt: str = Field(description="First ~220 characters of the chunk text")
    page: int | None = Field(default=None, description="Page number if known")
    section: str | None = Field(default=None, description="Section label if known")
    matched_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that matched this chunk"
    )
    selection_reason: str = Field(
        default="", description="Why this chunk was selected over others"
    )


class SessionPaperDescriptor(BaseModel):
    """Paper/document summary visible to the query agent for the current session."""

    paper_id: str
    title: str
    file_name: str | None = None
    created_at: str
    memory_count: int = Field(ge=0)
    summary_status: Literal["available", "missing"]


class PaperInfoDescriptor(BaseModel):
    """Stable paper metadata returned in a memory bundle."""

    paper_id: str
    title: str
    authors: tuple[str, ...] = Field(default_factory=tuple)
    abstract: str | None = None
    year: int | None = None
    arxiv_id: str | None = None
    file_name: str | None = None
    created_at: str | None = None


class PaperMemoryBundleDescriptor(BaseModel):
    """Aggregated memory and evidence view for a single paper."""

    paper: PaperInfoDescriptor
    paper_memory: dict[str, Any] | None = None
    open_questions: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    relations: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    evidence_source_chunks: tuple[ChunkDescriptor, ...] = Field(default_factory=tuple)
    empty_fields: tuple[str, ...] = Field(default_factory=tuple)


class ConversationEvidenceRefDescriptor(BaseModel):
    """Compact evidence reference visible in recent conversation context."""

    ref_type: Literal["memory", "chunk"] = Field(description="Which evidence family produced this reference")
    ref_id: str = Field(description="Stable evidence id")
    paper_id: str | None = Field(default=None, description="Paper this evidence points to if available")
    summary: str = Field(description="Short human-readable evidence summary")
    quote: str | None = Field(default=None, description="Short quoted snippet if available")
    page: int | None = Field(default=None, description="Page number if known")
    section: str | None = Field(default=None, description="Section label if known")
    memory_type: str | None = Field(default=None, description="Memory type when ref_type is memory")


class RecentConversationMessageDescriptor(BaseModel):
    """Compact recent message visible to the query agent."""

    message_id: str = Field(description="Stable message id")
    role: Literal["user", "assistant"] = Field(description="Message role within the conversation")
    content: str = Field(description="Compact message content")
    created_at: str = Field(description="ISO timestamp")
    paper_id: str | None = Field(default=None, description="Associated paper id if available")
    run_id: str | None = Field(default=None, description="Associated query run id if available")
    source_refs: tuple[ConversationEvidenceRefDescriptor, ...] = Field(
        default_factory=tuple,
        description="Evidence references attached to the message when available",
    )


class RecentConversationContextDescriptor(BaseModel):
    """Compact session conversation context injected into each query turn."""

    recent_user_messages: tuple[RecentConversationMessageDescriptor, ...] = Field(default_factory=tuple)
    recent_assistant_answers: tuple[RecentConversationMessageDescriptor, ...] = Field(default_factory=tuple)
    active_paper_id: str | None = Field(default=None)
    active_paper_file_name: str | None = Field(default=None)
    active_topic: str | None = Field(default=None)
    last_answer_summary: str | None = Field(default=None)
    last_evidence_refs: tuple[ConversationEvidenceRefDescriptor, ...] = Field(default_factory=tuple)
    recent_message_count: int = Field(default=0, ge=0)
    recent_turn_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Per-tool input models
# ---------------------------------------------------------------------------


class SearchSessionMemoryInput(BaseModel):
    """Parameters for searching memories scoped to the current session."""

    session_id: str = Field(description="The session whose document bindings scope the search")
    query: str = Field(min_length=1, description="Natural-language query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum memories to return")


class SearchGlobalMemoryInput(BaseModel):
    """Parameters for searching globally stored memories."""

    query: str = Field(min_length=1, description="Natural-language query text")
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter; when set only memories linked to these papers are considered"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum memories to return")


class SearchOpenVikingMemoryInput(BaseModel):
    """Parameters for searching OpenViking-backed memory explicitly."""

    scope: Literal["session", "global"] = Field(description="Which OpenViking scope to search")
    query: str = Field(min_length=1, description="Natural-language query text")
    session_id: str | None = Field(default=None, description="Required when scope is session")
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter for global lookups"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum hits to return")

    @model_validator(mode="after")
    def _validate_scope_fields(self) -> "SearchOpenVikingMemoryInput":
        if self.scope == "session" and not self.session_id:
            raise ValueError("session_id is required when scope is 'session'")
        return self


class SearchSourceChunksInput(BaseModel):
    """Parameters for retrieving stored source chunks before a reread."""

    session_id: str = Field(description="Session used to resolve document bindings")
    query: str = Field(min_length=1, description="Natural-language query text")
    paper_id: str | None = Field(
        default=None, description="Optional single paper-id to search within; overrides related_paper_ids when set"
    )
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter; defaults to session-document paper ids"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum chunks to return")


class SearchArxivInput(BaseModel):
    """Parameters for searching lightweight arXiv metadata only."""

    query: str = Field(min_length=1, description="Natural-language research query for arXiv discovery")
    max_results: int = Field(default=10, ge=1, description="Requested result count; the runtime clamps it to at most 50")
    category: str | None = Field(default=None, description="Optional arXiv category such as cs.LG or cs.CL")
    sort_by: str = Field(default="relevance", description="arXiv API sortBy value: relevance, lastUpdatedDate, or submittedDate")
    sort_order: str = Field(default="descending", description="arXiv API sortOrder value: ascending or descending")


class ImportArxivPaperInput(BaseModel):
    """Parameters for importing one arXiv paper into the current session."""

    arxiv_id_or_url: str = Field(
        min_length=1,
        validation_alias=AliasChoices("arxiv_id_or_url", "arxiv_url"),
        description=(
            "Accepts an arXiv id like 2401.12345 or 2401.12345v2, or an arXiv abs/pdf URL. "
            "The runtime normalizes it to a canonical https://arxiv.org/abs/{id} URL before import."
        ),
    )

    @field_validator("arxiv_id_or_url")
    @classmethod
    def _normalize_arxiv_reference(cls, value: str) -> str:
        return normalize_arxiv_id_or_url(value)


class RerankCandidatesInput(BaseModel):
    """Parameters for reranking a bounded pool of memory or chunk candidates."""

    candidate_kind: Literal["memory", "chunk"] = Field(
        description="Whether the candidates are memories or source chunks"
    )
    query: str = Field(min_length=1, description="Natural-language query text used for scoring")
    candidates: list[MemoryDescriptor | ChunkDescriptor] = Field(
        min_length=1, description="Bounded candidate descriptors to rerank"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top candidates to select")

    @model_validator(mode="after")
    def _validate_candidates_match_kind(self) -> "RerankCandidatesInput":
        if self.candidate_kind == "memory":
            if any(not isinstance(candidate, MemoryDescriptor) for candidate in self.candidates):
                raise ValueError("memory rerank requires only MemoryDescriptor candidates")
        else:
            if any(not isinstance(candidate, ChunkDescriptor) for candidate in self.candidates):
                raise ValueError("chunk rerank requires only ChunkDescriptor candidates")
        return self


class ReadSourcePassagesInput(BaseModel):
    """Parameters for the combined retrieve-and-rerank source passage tool."""

    session_id: str = Field(description="Session used to resolve document bindings and chunk storage")
    query: str = Field(min_length=1, description="Natural-language query text")
    paper_id: str | None = Field(
        default=None, description="Optional single paper-id to search within; overrides related_paper_ids when set"
    )
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top passages to select")


class ListSessionPapersInput(BaseModel):
    """Parameters for listing papers bound to the current runtime session."""

    limit: int = Field(default=20, ge=1, le=100, description="Maximum papers/documents to return")


class ListRecentMessagesInput(BaseModel):
    """Parameters for listing recent conversation messages for the current session."""

    limit: int = Field(default=8, ge=1, le=20, description="Maximum recent query messages to return")


class GetConversationContextInput(BaseModel):
    """Parameters for retrieving the compact recent conversation context."""

    limit: int = Field(default=8, ge=1, le=20, description="Maximum recent query messages to inspect")


class GetPaperMemoryBundleInput(BaseModel):
    """Parameters for retrieving all query-visible memory for one paper."""

    paper_id: str = Field(min_length=1, description="Paper id to inspect")
    source_chunk_limit: int = Field(default=5, ge=0, le=20, description="Maximum source chunk summaries to include")


class ComposeAnswerInput(BaseModel):
    """Parameters for composing the final answer from retrieved context."""

    query: str = Field(min_length=1, description="The original user query")
    memory_context: list[MemoryDescriptor] = Field(
        default_factory=list, description="Memories selected for answering"
    )
    source_context: list[ChunkDescriptor] = Field(
        default_factory=list, description="Source chunks selected for answering (empty if no reread)"
    )
    session_memory_count: int = Field(default=0, ge=0, description="Total session memories retrieved")
    global_memory_count: int = Field(default=0, ge=0, description="Total global memories retrieved")
    memory_selection_source: str = Field(
        default="rule_fallback", description="Which reranker selected the memory candidates"
    )
    should_reread_source: bool = Field(
        default=False, description="Whether source reread was required"
    )
    reread_reason: str = Field(
        default="", description="Why reread was or was not required"
    )


# ---------------------------------------------------------------------------
# Per-tool output models
# ---------------------------------------------------------------------------


class SearchSessionMemoryOutput(BaseModel):
    """Result of a session-scoped memory search."""

    memories: tuple[MemoryDescriptor, ...] = Field(
        default_factory=tuple, description="Selected memory records in ranked order"
    )
    coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of top_k slots filled"
    )
    matched_query_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that had at least one hit across all memories"
    )
    selection_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Per-memory selection rationale"
    )


class SearchGlobalMemoryOutput(BaseModel):
    """Result of a global memory search."""

    memories: tuple[MemoryDescriptor, ...] = Field(
        default_factory=tuple, description="Selected memory records in ranked order"
    )
    coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of top_k slots filled"
    )
    matched_query_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that had at least one hit"
    )
    selection_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Per-memory selection rationale"
    )


class SearchOpenVikingMemoryOutput(BaseModel):
    """Result of an explicit OpenViking-backed memory search."""

    scope: Literal["session", "global"] = Field(description="OpenViking scope used for retrieval")
    hits: tuple[OpenVikingHitDescriptor, ...] = Field(
        default_factory=tuple, description="Raw OpenViking hits in ranked order"
    )
    memories: tuple[MemoryDescriptor, ...] = Field(
        default_factory=tuple, description="Local memory records matched by the OpenViking hits"
    )
    coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of top_k slots filled by local matches"
    )
    matched_query_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that had at least one hit"
    )
    selection_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Per-hit selection rationale"
    )
    matched_local_memory_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Local memory ids matched by the OpenViking hits"
    )
    matched_local_count: int = Field(
        default=0, ge=0, description="Number of local memory ids matched by the OpenViking hits"
    )


class SearchSourceChunksOutput(BaseModel):
    """Result of a source chunk retrieval."""

    chunks: tuple[ChunkDescriptor, ...] = Field(
        default_factory=tuple, description="Selected chunk records in ranked order"
    )
    coverage_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of top_k slots filled"
    )
    matched_query_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that had at least one hit"
    )
    selection_reasons: tuple[str, ...] = Field(
        default_factory=tuple, description="Per-chunk selection rationale"
    )


class ArxivPaperDescriptor(BaseModel):
    """Lightweight arXiv paper metadata returned by search_arxiv."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    published: str
    updated: str
    categories: list[str] = Field(default_factory=list)
    abs_url: str
    pdf_url: str


class SearchArxivOutput(BaseModel):
    """Result of searching arXiv without importing or downloading papers."""

    success: bool
    query: str
    count: int = Field(default=0, ge=0)
    papers: list[ArxivPaperDescriptor] = Field(default_factory=list)
    error: dict[str, Any] | None = Field(
        default=None,
        description="Structured error with code and message when success is false",
    )


class ImportedPaperSummaryDescriptor(BaseModel):
    """Compact paper summary returned after an arXiv import finishes."""

    what_it_is_about: str
    problem_solved: str
    new_ideas: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    suggestions_or_questions: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(ge=0.0, le=1.0)


class ImportArxivPaperOutput(BaseModel):
    """Result of importing one arXiv paper into the current session."""

    run_id: str
    message_id: str
    paper_id: str
    title: str
    arxiv_id: str | None = None
    artifact_id: str
    session_document_id: str
    source_type: Literal["arxiv"]
    operation: str
    chunk_count: int = Field(ge=0)
    ingest_summary: str
    paper_summary: ImportedPaperSummaryDescriptor


class ListSessionPapersOutput(BaseModel):
    """Result of listing papers/documents for the current session."""

    papers: tuple[SessionPaperDescriptor, ...] = Field(default_factory=tuple)
    total_count: int = Field(default=0, ge=0)


class ListRecentMessagesOutput(BaseModel):
    """Result of listing recent messages for the current session."""

    messages: tuple[RecentConversationMessageDescriptor, ...] = Field(default_factory=tuple)
    total_count: int = Field(default=0, ge=0)
    window_count: int = Field(default=0, ge=0)


class GetConversationContextOutput(BaseModel):
    """Result of retrieving the compact recent conversation context."""

    context: RecentConversationContextDescriptor


class GetPaperMemoryBundleOutput(BaseModel):
    """Result of aggregating memory and evidence for a paper."""

    bundle: PaperMemoryBundleDescriptor


class RerankCandidatesOutput(BaseModel):
    """Result of reranking a bounded candidate pool."""

    selected_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Candidate IDs selected after reranking"
    )
    selection_source: str = Field(
        default="rule_fallback", description="Which reranker produced this selection"
    )
    fallback_used: bool = Field(
        default=False, description="True when the primary reranker failed and the fallback was used"
    )
    rationale: str = Field(
        default="", description="Human-readable explanation of the rerank decision"
    )


class ReadSourcePassagesOutput(BaseModel):
    """Result of the combined retrieve-and-rerank source passage tool."""

    chunks: tuple[ChunkDescriptor, ...] = Field(
        default_factory=tuple, description="Final selected passages in ranked order"
    )
    selection_source: str = Field(
        default="rule_fallback", description="Which reranker produced the final selection"
    )
    fallback_used: bool = Field(
        default=False, description="True when the primary reranker failed and the fallback was used"
    )
    rationale: str = Field(
        default="", description="Human-readable explanation of the selection"
    )
    matched_query_terms: tuple[str, ...] = Field(
        default_factory=tuple, description="Query terms that matched the selected passages"
    )


class ComposeAnswerOutput(BaseModel):
    """Evidence package prepared for the model to generate the final answer."""

    evidence_package: str = Field(
        description="Compact evidence package for the model to use when drafting the final answer"
    )
    citations: tuple[MemoryDescriptor, ...] = Field(
        default_factory=tuple, description="Memories cited in the evidence package"
    )
    source_citations: tuple[ChunkDescriptor, ...] = Field(
        default_factory=tuple, description="Source chunks cited in the evidence package"
    )
    memory_influence: str = Field(
        default="", description="How memory shaped the evidence package vs. source reread"
    )


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Complete definition of a query tool in the frozen protocol.

    Each tool binds a name, description, and the Pydantic models that
    define its input and output contract. This is the unit that a model
    or runtime loop inspects before deciding which tool to call.
    """

    name: QueryToolName
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]


# ---------------------------------------------------------------------------
# Request / response envelopes
# ---------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """Envelope for invoking a tool.

    The runtime validates that tool_name is in the frozen query set
    and that parameters conform to the tool's input model before
    forwarding to the tool implementation.
    """

    tool_name: QueryToolName = Field(description="Which tool to invoke")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Tool-specific keyword arguments"
    )


class ToolResponse(BaseModel):
    """Envelope for a successful tool result."""

    tool_name: QueryToolName = Field(description="The tool that produced this result")
    result: dict[str, Any] = Field(description="Tool-specific output payload")


ToolOutcome = Annotated[
    ToolResponse | ToolError,
    Field(discriminator="tool_name"),
]
"""Union of successful and failed tool outcomes.

Callers must check for ToolError before interpreting result.
"""


# ---------------------------------------------------------------------------
# Frozen query tool set
# ---------------------------------------------------------------------------


QUERY_TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name=QueryToolName.SEARCH_ARXIV,
        description="Search arXiv for papers matching a research query. Returns lightweight metadata including title, authors, abstract, arXiv id, abs URL, and PDF URL. Use this to discover candidate papers. This tool does not import, download, or parse papers. To add a selected paper to the current research session, call import_arxiv_paper with its arxiv_id or URL.",
        input_model=SearchArxivInput,
        output_model=SearchArxivOutput,
    ),
    ToolDefinition(
        name=QueryToolName.IMPORT_ARXIV_PAPER,
        description="Import one arXiv paper into the current session through the existing ingest run flow. "
        "Use this when the user explicitly asks to import an arXiv paper or provides an arXiv link that should be added to the session before answering.",
        input_model=ImportArxivPaperInput,
        output_model=ImportArxivPaperOutput,
    ),
    ToolDefinition(
        name=QueryToolName.SEARCH_SESSION_MEMORY,
        description="Search memories scoped to the current session by document bindings. "
        "Always called first in the memory-first retrieval order.",
        input_model=SearchSessionMemoryInput,
        output_model=SearchSessionMemoryOutput,
    ),
    ToolDefinition(
        name=QueryToolName.SEARCH_GLOBAL_MEMORY,
        description="Search globally stored memories with an optional paper-id filter. "
        "Called after session memory to widen recall before deciding whether to reread source passages.",
        input_model=SearchGlobalMemoryInput,
        output_model=SearchGlobalMemoryOutput,
    ),
    ToolDefinition(
        name=QueryToolName.SEARCH_OPENVIKING_MEMORY,
        description="Search OpenViking-backed memory explicitly and map hits back to local memory records.",
        input_model=SearchOpenVikingMemoryInput,
        output_model=SearchOpenVikingMemoryOutput,
    ),
    ToolDefinition(
        name=QueryToolName.SEARCH_SOURCE_CHUNKS,
        description="Search stored source chunks for papers bound to a session. "
        "Use this for original passages, quoted sentences, specific claims, mechanisms, limitations, or evidence snippets. "
        "Pass paper_id to search within a specific paper (use the paper_id from list_session_papers results).",
        input_model=SearchSourceChunksInput,
        output_model=SearchSourceChunksOutput,
    ),
    ToolDefinition(
        name=QueryToolName.LIST_RECENT_MESSAGES,
        description="List the most recent follow-up conversation messages for the current session. "
        "The host injects session_id; use this when you need more conversation history than the compact turn context already provides.",
        input_model=ListRecentMessagesInput,
        output_model=ListRecentMessagesOutput,
    ),
    ToolDefinition(
        name=QueryToolName.GET_CONVERSATION_CONTEXT,
        description="Return the compact recent conversation context for the current session, including recent user and assistant messages, the active paper, the active topic, and recent evidence references. "
        "The host injects session_id; use this when you need a compact history summary before deciding on the next answer or tool.",
        input_model=GetConversationContextInput,
        output_model=GetConversationContextOutput,
    ),
    ToolDefinition(
        name=QueryToolName.LIST_SESSION_PAPERS,
        description="List papers/documents imported into the current runtime session. "
        "Use this when the user asks what papers are currently in the session, which documents were imported, or what is available to inspect. "
        "The host injects session_id; the model may only provide business parameters such as limit.",
        input_model=ListSessionPapersInput,
        output_model=ListSessionPapersOutput,
    ),
    ToolDefinition(
        name=QueryToolName.GET_PAPER_MEMORY_BUNDLE,
        description="Return the memory bundle for one paper_id: paper info, paper memory, open questions, relations, and source chunk summaries. "
        "Use this when answering around one paper and you need the paper-specific memory bundle.",
        input_model=GetPaperMemoryBundleInput,
        output_model=GetPaperMemoryBundleOutput,
    ),
    ToolDefinition(
        name=QueryToolName.RERANK_CANDIDATES,
        description="Rerank a bounded pool of memory or chunk candidates using the active "
        "reranker with deterministic fallback. Accepts candidate IDs and returns the top-k "
        "selection with rationale.",
        input_model=RerankCandidatesInput,
        output_model=RerankCandidatesOutput,
    ),
    ToolDefinition(
        name=QueryToolName.READ_SOURCE_PASSAGES,
        description="Combined retrieve-and-rerank for source passages. Searches stored chunks, "
        "then reranks the candidate pool to produce the final passage selection. "
        "Pass paper_id to search within a specific paper (use the paper_id from list_session_papers results).",
        input_model=ReadSourcePassagesInput,
        output_model=ReadSourcePassagesOutput,
    ),
    ToolDefinition(
        name=QueryToolName.COMPOSE_ANSWER,
        description="Package selected evidence from memories and optional source reread chunks for the model to use when drafting the final answer.",
        input_model=ComposeAnswerInput,
        output_model=ComposeAnswerOutput,
    ),
)

QUERY_TOOL_BY_NAME: dict[QueryToolName, ToolDefinition] = {
    definition.name: definition for definition in QUERY_TOOL_DEFINITIONS
}


def get_query_tool_definition(name: QueryToolName | str) -> ToolDefinition | None:
    """Look up a frozen query tool definition by name.

    Returns None when the tool is not in the query subset (e.g. it is
    an ingest-only tool or does not exist).
    """
    if isinstance(name, str):
        try:
            name = QueryToolName(name)
        except ValueError:
            return None
    return QUERY_TOOL_BY_NAME.get(name)


def is_query_tool(name: str) -> bool:
    """Return True when *name* belongs to the frozen query tool set."""
    try:
        QueryToolName(name)
    except ValueError:
        return False
    return True


def validate_tool_request(request: ToolRequest) -> ToolError | None:
    """Validate a ToolRequest against the frozen protocol.

    Returns a ToolError when:
      - The tool name is not in the query tool set
      - Required parameters are missing or invalid

    Returns None when the request is valid.
    """
    definition = get_query_tool_definition(request.tool_name)
    if definition is None:
        return ToolError(
            tool_name=request.tool_name.value
            if isinstance(request.tool_name, QueryToolName)
            else str(request.tool_name),
            error_code=ToolErrorCode.TOOL_NOT_FOUND,
            message=f"Tool '{request.tool_name}' is not in the frozen query tool set.",
        )
    try:
        definition.input_model.model_validate(request.parameters)
    except Exception as exc:
        return ToolError(
            tool_name=request.tool_name.value,
            error_code=ToolErrorCode.INVALID_PARAMETER,
            message=f"Invalid parameters for tool '{request.tool_name.value}': {exc}",
            details={"validation_error": str(exc)},
        )
    return None


__all__ = [
    "ArxivPaperDescriptor",
    "ChunkDescriptor",
    "ComposeAnswerInput",
    "ComposeAnswerOutput",
    "ConversationEvidenceRefDescriptor",
    "GetConversationContextInput",
    "GetConversationContextOutput",
    "ImportedPaperSummaryDescriptor",
    "ImportArxivPaperInput",
    "ImportArxivPaperOutput",
    "ListRecentMessagesInput",
    "ListRecentMessagesOutput",
    "MemoryDescriptor",
    "GetPaperMemoryBundleInput",
    "GetPaperMemoryBundleOutput",
    "ListSessionPapersInput",
    "ListSessionPapersOutput",
    "RecentConversationContextDescriptor",
    "RecentConversationMessageDescriptor",
    "OpenVikingHitDescriptor",
    "PaperInfoDescriptor",
    "PaperMemoryBundleDescriptor",
    "QueryToolName",
    "QUERY_TOOL_DEFINITIONS",
    "QUERY_TOOL_BY_NAME",
    "SearchArxivInput",
    "SearchArxivOutput",
    "ReadSourcePassagesInput",
    "ReadSourcePassagesOutput",
    "RerankCandidatesInput",
    "RerankCandidatesOutput",
    "SearchGlobalMemoryInput",
    "SearchGlobalMemoryOutput",
    "SearchOpenVikingMemoryInput",
    "SearchOpenVikingMemoryOutput",
    "SearchSessionMemoryInput",
    "SearchSessionMemoryOutput",
    "SearchSourceChunksInput",
    "SearchSourceChunksOutput",
    "SessionPaperDescriptor",
    "ToolDefinition",
    "ToolError",
    "ToolErrorCode",
    "ToolRequest",
    "ToolResponse",
    "ToolOutcome",
    "get_query_tool_definition",
    "is_query_tool",
    "validate_tool_request",
]
