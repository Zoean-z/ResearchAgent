"""Query execution API schemas."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict

from research_agent.api.schemas.task_runs import TaskRunResponse
from research_agent.utils import to_json_safe


class MemoryCitationResponse(BaseModel):
    """Serialized pointer to a retrieved memory record."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    memory_type: str
    summary: str
    selection_reason: str


class SourceRereadChunkResponse(BaseModel):
    """Serialized pointer to a reread source chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    paper_id: str
    page: int | None
    section: str | None
    excerpt: str
    selection_reason: str


class QueryDecisionDebugResponse(BaseModel):
    """Serialized per-turn query decision for debugging."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int
    action_type: str
    tool_name: str | None
    tool_parameters: dict[str, Any]
    rationale: str
    final_answer_present: bool
    validation_error: str | None = None
    fallback_reason: str | None = None


class QueryObservationDebugResponse(BaseModel):
    """Serialized observation visible to the query loop."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    summary: str
    payload: dict[str, Any] | None = None


class QueryExecutionDebugResponse(BaseModel):
    """Serialized query-loop debug payload."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[QueryDecisionDebugResponse]
    tool_calls: list[QueryDecisionDebugResponse]
    observations_summary: list[QueryObservationDebugResponse]


class QueryExecutionResponse(BaseModel):
    """Serialized mock query execution result."""

    model_config = ConfigDict(extra="forbid")

    task_run: TaskRunResponse
    answer: str
    should_reread_source: bool
    reread_reason: str
    memory_selection_source: str
    memory_selection_fallback_used: bool
    session_memory_count: int
    global_memory_count: int
    session_coverage_score: float
    global_coverage_score: float
    matched_query_terms: list[str]
    used_memory_citations: list[MemoryCitationResponse]
    source_selection_source: str | None
    source_selection_fallback_used: bool
    source_reread_chunk_count: int
    source_reread_chunks: list[SourceRereadChunkResponse]
    debug: QueryExecutionDebugResponse

    @classmethod
    def from_result(cls, result) -> "QueryExecutionResponse":
        debug_decisions = [
            QueryDecisionDebugResponse(
                turn_index=tool_call.turn_index,
                action_type=tool_call.action_type,
                tool_name=tool_call.tool_name,
                tool_parameters=to_json_safe(tool_call.tool_parameters),
                rationale=tool_call.rationale,
                final_answer_present=bool(tool_call.final_answer),
                validation_error=tool_call.validation_error,
                fallback_reason=tool_call.fallback_reason,
            )
            for tool_call in result.tool_calls
        ]
        return cls(
            task_run=TaskRunResponse.from_domain(result.task_run),
            answer=result.answer,
            should_reread_source=result.should_reread_source,
            reread_reason=result.reread_reason,
            memory_selection_source=result.memory_selection_source,
            memory_selection_fallback_used=result.memory_selection_fallback_used,
            session_memory_count=len(result.retrieval_plan.session_memories.memories),
            global_memory_count=len(result.retrieval_plan.global_memories.memories),
            session_coverage_score=result.retrieval_plan.session_memories.coverage_score,
            global_coverage_score=result.retrieval_plan.global_memories.coverage_score,
            matched_query_terms=list(result.matched_query_terms),
            used_memory_citations=[MemoryCitationResponse.model_validate(asdict(citation)) for citation in result.used_memory_citations],
            source_selection_source=result.source_selection_source,
            source_selection_fallback_used=result.source_selection_fallback_used,
            source_reread_chunk_count=len(result.source_reread_chunks),
            source_reread_chunks=[SourceRereadChunkResponse.model_validate(asdict(chunk)) for chunk in result.source_reread_chunks],
            debug=QueryExecutionDebugResponse(
                decisions=debug_decisions,
                tool_calls=[
                    QueryDecisionDebugResponse.model_validate(to_json_safe(decision.model_dump(mode="python")))
                    for decision in debug_decisions
                ],
                observations_summary=[
                    QueryObservationDebugResponse.model_validate(to_json_safe(observation.model_dump(mode="python")))
                    for observation in result.observations
                ],
            ),
        )


class QueryExecutionErrorResponse(BaseModel):
    """Structured query failure detail returned by execute/start surfaces."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    failed_stage: str
    error_message: str
    run_id: str
    tool_name: str | None = None
    fallback_reason: str | None = None
    validation_error: str | None = None
    failure_stage_detail: str | None = None
    status_code: int | None = None
    repair_attempted: bool | None = None
    raw_response_preview: str | None = None
    content_preview: str | None = None
