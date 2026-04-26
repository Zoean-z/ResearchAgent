"""Frozen query-only tool protocol with stable request/response and error semantics.

This module defines the Phase 1 query tool contract. Every query tool that a model
or runtime loop may invoke is declared here with its name, description, structured
input model, and structured output model. The protocol is intentionally decoupled
from domain models so it can evolve independently.

Query tool subset (frozen for Phase 1):
  - search_session_memory
  - search_global_memory
  - search_source_chunks
  - rerank_candidates
  - read_source_passages
  - compose_answer
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Tool identity
# ---------------------------------------------------------------------------


class QueryToolName(StrEnum):
    """Frozen set of tools available for query execution.

    Only these tools may be invoked during a follow-up query run.
    The runtime must reject any tool not in this enumeration.
    """

    SEARCH_SESSION_MEMORY = "search_session_memory"
    SEARCH_GLOBAL_MEMORY = "search_global_memory"
    SEARCH_OPENVIKING_MEMORY = "search_openviking_memory"
    SEARCH_SOURCE_CHUNKS = "search_source_chunks"
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
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter; defaults to session-document paper ids"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Maximum chunks to return")


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
    related_paper_ids: list[str] | None = Field(
        default=None, description="Optional paper-id filter"
    )
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top passages to select")


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
    """Result of composing the final answer."""

    answer: str = Field(description="The composed answer text")
    citations: tuple[MemoryDescriptor, ...] = Field(
        default_factory=tuple, description="Memories cited in the answer"
    )
    source_citations: tuple[ChunkDescriptor, ...] = Field(
        default_factory=tuple, description="Source chunks cited in the answer"
    )
    memory_influence: str = Field(
        default="", description="How memory shaped the answer vs. source reread"
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
        description="Retrieve stored source chunks for papers bound to a session. "
        "Only called when memory-first retrieval is insufficient.",
        input_model=SearchSourceChunksInput,
        output_model=SearchSourceChunksOutput,
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
        "then reranks the candidate pool to produce the final passage selection.",
        input_model=ReadSourcePassagesInput,
        output_model=ReadSourcePassagesOutput,
    ),
    ToolDefinition(
        name=QueryToolName.COMPOSE_ANSWER,
        description="Compose the final answer from selected memories and optional source "
        "reread chunks. Returns the answer text with citations and a memory-influence summary.",
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
    "ChunkDescriptor",
    "ComposeAnswerInput",
    "ComposeAnswerOutput",
    "MemoryDescriptor",
    "OpenVikingHitDescriptor",
    "QueryToolName",
    "QUERY_TOOL_DEFINITIONS",
    "QUERY_TOOL_BY_NAME",
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
