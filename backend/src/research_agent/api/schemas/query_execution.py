"""Query execution API schemas."""

from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel, ConfigDict

from research_agent.api.schemas.task_runs import TaskRunResponse


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

    @classmethod
    def from_result(cls, result) -> "QueryExecutionResponse":
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
        )
