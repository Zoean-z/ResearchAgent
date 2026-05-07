"""Executor that enforces the frozen query tool protocol at runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from research_agent.domain.models import Chunk, OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.services.retrieval_service import MemoryRetrievalResult, SourceRereadResult
from research_agent.tools.protocol import (
    ChunkDescriptor,
    ComposeAnswerOutput,
    GetConversationContextOutput,
    GetPaperMemoryBundleOutput,
    ListSessionPapersOutput,
    ListRecentMessagesOutput,
    MemoryDescriptor,
    OpenVikingHitDescriptor,
    QueryToolName,
    ReadSourcePassagesOutput,
    RerankCandidatesInput,
    RerankCandidatesOutput,
    SearchGlobalMemoryOutput,
    SearchOpenVikingMemoryOutput,
    SearchSessionMemoryOutput,
    SearchSourceChunksOutput,
    ToolError,
    ToolErrorCode,
    ToolRequest,
    ToolResponse,
    get_query_tool_definition,
    validate_tool_request,
)
from research_agent.tools.registry import InternalToolRegistry


MemoryRecord = PaperMemory | RelationMemory | OpenQuestionMemory


@dataclass(frozen=True, slots=True)
class ToolExecutionEnvelope:
    """Validated protocol outcome plus the raw result used internally by the runtime."""

    outcome: ToolResponse | ToolError
    raw_result: Any | None = None


class QueryToolExecutor:
    """Execute the frozen query-only tool subset through protocol envelopes."""

    def __init__(self, registry: InternalToolRegistry) -> None:
        self._registry = registry

    def execute(self, request: ToolRequest, *, runtime_context: dict[str, Any] | None = None) -> ToolResponse | ToolError:
        """Execute a validated query tool request and return only the protocol outcome."""

        return self.execute_with_raw(request, runtime_context=runtime_context).outcome

    def execute_with_raw(self, request: ToolRequest, *, runtime_context: dict[str, Any] | None = None) -> ToolExecutionEnvelope:
        """Execute a query tool request and keep the raw result for host-side logic."""

        validation_error = validate_tool_request(request)
        if validation_error is not None:
            return ToolExecutionEnvelope(outcome=validation_error)

        try:
            params = get_query_tool_definition(request.tool_name).input_model.model_validate(request.parameters)
            if request.tool_name is QueryToolName.SEARCH_SESSION_MEMORY:
                raw_result = self._registry.search_session_memory(
                    session_id=params.session_id,
                    query=params.query,
                    top_k=params.top_k,
                )
                output = self._memory_output(SearchSessionMemoryOutput, raw_result)
            elif request.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY:
                raw_result = self._registry.search_global_memory(
                    query=params.query,
                    related_paper_ids=params.related_paper_ids,
                    top_k=params.top_k,
                )
                output = self._memory_output(SearchGlobalMemoryOutput, raw_result)
            elif request.tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY:
                raw_result = self._registry.search_openviking_memory(
                    scope=params.scope,
                    session_id=params.session_id,
                    query=params.query,
                    related_paper_ids=params.related_paper_ids,
                    top_k=params.top_k,
                )
                output = self._openviking_output(raw_result)
            elif request.tool_name is QueryToolName.SEARCH_SOURCE_CHUNKS:
                raw_result = self._registry.search_source_chunks(
                    session_id=params.session_id,
                    query=params.query,
                    related_paper_ids=params.related_paper_ids,
                    top_k=params.top_k,
                    paper_id=params.paper_id,
                )
                output = self._chunk_search_output(raw_result)
            elif request.tool_name is QueryToolName.LIST_RECENT_MESSAGES:
                session_id = self._runtime_session_id(runtime_context)
                current_message_id = self._runtime_message_id(runtime_context)
                output = self._registry.list_recent_messages(
                    session_id=session_id,
                    limit=params.limit,
                    exclude_message_id=current_message_id,
                )
                raw_result = output
            elif request.tool_name is QueryToolName.GET_CONVERSATION_CONTEXT:
                session_id = self._runtime_session_id(runtime_context)
                current_message_id = self._runtime_message_id(runtime_context)
                output = self._registry.get_conversation_context(
                    session_id=session_id,
                    limit=params.limit,
                    exclude_message_id=current_message_id,
                )
                raw_result = output
            elif request.tool_name is QueryToolName.RERANK_CANDIDATES:
                raw_result, output = self._execute_rerank(params)
            elif request.tool_name is QueryToolName.READ_SOURCE_PASSAGES:
                raw_result = self._registry.read_source_passages(
                    session_id=params.session_id,
                    query=params.query,
                    related_paper_ids=params.related_paper_ids,
                    top_k=params.top_k,
                    paper_id=params.paper_id,
                )
                output = self._read_source_output(raw_result, params.query)
            elif request.tool_name is QueryToolName.LIST_SESSION_PAPERS:
                session_id = self._runtime_session_id(runtime_context)
                output = self._registry.list_session_papers(session_id=session_id, limit=params.limit)
                raw_result = output
            elif request.tool_name is QueryToolName.GET_PAPER_MEMORY_BUNDLE:
                output = self._registry.get_paper_memory_bundle(
                    paper_id=params.paper_id,
                    source_chunk_limit=params.source_chunk_limit,
                )
                raw_result = output
            elif request.tool_name is QueryToolName.COMPOSE_ANSWER:
                evidence_package = self._registry.compose_answer(
                    query=params.query,
                    session_memory_count=params.session_memory_count,
                    global_memory_count=params.global_memory_count,
                    memory_selection_source=params.memory_selection_source,
                    memory_selection_fallback_used=False,
                    should_reread_source=params.should_reread_source,
                    reread_reason=params.reread_reason,
                    used_memory_citations=params.memory_context,
                    source_reread_chunks=params.source_context,
                    source_selection_source="model" if params.source_context else None,
                )
                output = ComposeAnswerOutput(
                    evidence_package=evidence_package,
                    citations=tuple(params.memory_context),
                    source_citations=tuple(params.source_context),
                    memory_influence=self._memory_influence_text(
                        memory_context=params.memory_context,
                        source_context=params.source_context,
                        should_reread_source=params.should_reread_source,
                    ),
                )
                raw_result = output
            else:  # pragma: no cover - guarded by protocol enum
                return ToolExecutionEnvelope(
                    outcome=ToolError(
                        tool_name=str(request.tool_name),
                        error_code=ToolErrorCode.TOOL_NOT_FOUND,
                        message=f"Unsupported query tool: {request.tool_name}",
                    )
                )
        except ValueError as exc:
            return ToolExecutionEnvelope(
                outcome=ToolError(
                    tool_name=request.tool_name.value,
                    error_code=ToolErrorCode.INVALID_PARAMETER,
                    message=str(exc),
                    details={"validation_error": str(exc)},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            return ToolExecutionEnvelope(
                outcome=ToolError(
                    tool_name=request.tool_name.value,
                    error_code=ToolErrorCode.TOOL_EXECUTION_FAILED,
                    message=f"Tool execution failed: {exc}",
                )
            )

        return ToolExecutionEnvelope(
            outcome=ToolResponse(
                tool_name=request.tool_name,
                result=output.model_dump(mode="python"),
            ),
            raw_result=raw_result,
        )

    def _runtime_session_id(self, runtime_context: dict[str, Any] | None) -> str:
        session_id = (runtime_context or {}).get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("runtime_context.session_id is required for this tool")
        return session_id

    def _runtime_message_id(self, runtime_context: dict[str, Any] | None) -> str | None:
        message_id = (runtime_context or {}).get("message_id")
        if message_id is None:
            return None
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("runtime_context.message_id must be a non-empty string when provided")
        return message_id

    def _memory_output(
        self,
        output_model: type[SearchSessionMemoryOutput | SearchGlobalMemoryOutput],
        result: MemoryRetrievalResult,
    ) -> SearchSessionMemoryOutput | SearchGlobalMemoryOutput:
        return output_model(
            memories=tuple(self._memory_descriptor(memory, result.matched_query_terms) for memory in result.memories),
            coverage_score=result.coverage_score,
            matched_query_terms=result.matched_query_terms,
            selection_reasons=result.selection_reasons,
        )

    def _chunk_search_output(self, result: SourceRereadResult) -> SearchSourceChunksOutput:
        return SearchSourceChunksOutput(
            chunks=tuple(self._chunk_descriptor(chunk, result.matched_query_terms) for chunk in result.chunks),
            coverage_score=result.coverage_score,
            matched_query_terms=result.matched_query_terms,
            selection_reasons=result.selection_reasons,
        )

    def _openviking_output(self, result) -> SearchOpenVikingMemoryOutput:
        matched_terms = tuple(self._matched_terms_for_memories(result.memory_descriptors))
        return SearchOpenVikingMemoryOutput(
            scope=result.scope,
            hits=tuple(
                OpenVikingHitDescriptor(
                    item_kind=hit.item_kind,
                    item_id=hit.item_id,
                    session_id=hit.session_id,
                    score=hit.score,
                    summary=hit.summary,
                    metadata=hit.metadata,
                )
                for hit in result.hits
            ),
            memories=tuple(result.memory_descriptors),
            coverage_score=min(1.0, len(result.memory_descriptors) / max(1, len(result.hits))),
            matched_query_terms=matched_terms,
            selection_reasons=tuple(memory.selection_reason for memory in result.memory_descriptors),
            matched_local_memory_ids=result.matched_local_memory_ids,
            matched_local_count=result.matched_local_count,
        )

    def _execute_rerank(self, params: RerankCandidatesInput) -> tuple[Any, RerankCandidatesOutput]:
        if not params.candidates:
            raise ValueError(ToolErrorCode.EMPTY_CANDIDATES.value)
        if params.candidate_kind == "memory":
            ranked = sorted(
                params.candidates,
                key=lambda candidate: (
                    len(candidate.matched_terms),
                    candidate.confidence,
                    candidate.memory_id,
                ),
                reverse=True,
            )
            selected = tuple(candidate.memory_id for candidate in ranked[: params.top_k])
        elif params.candidate_kind == "chunk":
            ranked = sorted(
                params.candidates,
                key=lambda candidate: (
                    len(candidate.matched_terms),
                    -(candidate.page or 10**9),
                    candidate.chunk_id,
                ),
                reverse=True,
            )
            selected = tuple(candidate.chunk_id for candidate in ranked[: params.top_k])
        else:
            raise ValueError(ToolErrorCode.CANDIDATE_KIND_UNSUPPORTED.value)
        return ranked, RerankCandidatesOutput(
            selected_ids=selected,
            selection_source="model",
            fallback_used=False,
            rationale=f"executor_reranked_{params.candidate_kind}_candidates_from_{len(params.candidates)}_to_{len(selected)}",
        )

    def _read_source_output(self, result, query: str) -> ReadSourcePassagesOutput:
        matched_terms = tuple(self._matched_terms_for_chunks(query, result.selected))
        return ReadSourcePassagesOutput(
            chunks=tuple(self._chunk_descriptor(chunk, matched_terms, selection_source=result.selection_source) for chunk in result.selected),
            selection_source=result.selection_source,
            fallback_used=result.fallback_used,
            rationale=result.rationale,
            matched_query_terms=matched_terms,
        )

    def _memory_descriptor(
        self,
        memory: MemoryRecord,
        matched_terms: Sequence[str],
    ) -> MemoryDescriptor:
        if isinstance(memory, PaperMemory):
            memory_type = "paper_memory"
            summary = " | ".join(part for part in [memory.problem or memory.method or memory.novelty_claim or "paper memory", memory.key_results[0] if memory.key_results else ""] if part)
            evidence_score = 2 if any(ref.quote for ref in memory.source_refs) else 1 if memory.source_refs else 0
        elif isinstance(memory, RelationMemory):
            memory_type = "relation_memory"
            summary = f"{memory.relation_type.value}: {memory.summary}"
            evidence_score = 2 if memory.evidence else 0
        else:
            memory_type = "open_question_memory"
            summary = memory.unresolved_question
            evidence_score = 1 if memory.why_open or memory.possible_followup else 0
        return MemoryDescriptor(
            memory_id=memory.id,
            memory_type=memory_type,
            summary=summary,
            confidence=memory.confidence.value,
            matched_terms=tuple(term for term in matched_terms if term in self._memory_text(memory)),
            selection_reason=f"type={memory_type}; evidence_score={evidence_score}; confidence={memory.confidence.value:.2f}",
        )

    def _chunk_descriptor(
        self,
        chunk: Chunk,
        matched_terms: Sequence[str],
        selection_source: str | None = None,
    ) -> ChunkDescriptor:
        excerpt = chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."
        suffix = f"; rerank_strategy={selection_source}" if selection_source else ""
        return ChunkDescriptor(
            chunk_id=chunk.id,
            paper_id=chunk.paper_id,
            excerpt=excerpt,
            page=chunk.page,
            section=chunk.section,
            matched_terms=tuple(term for term in matched_terms if term in chunk.text.lower()),
            selection_reason=(
                f"matched_terms={','.join(term for term in matched_terms if term in chunk.text.lower()) or 'none'}; "
                f"section={chunk.section or 'unknown-section'}; page={chunk.page if chunk.page is not None else 'unknown-page'}{suffix}"
            ),
        )

    def _memory_text(self, memory: MemoryRecord) -> str:
        if isinstance(memory, PaperMemory):
            return " ".join(
                [
                    memory.problem or "",
                    memory.method or "",
                    " ".join(memory.key_results),
                    " ".join(memory.limitations),
                    memory.novelty_claim or "",
                    " ".join(ref.quote or "" for ref in memory.source_refs),
                ]
            ).lower()
        if isinstance(memory, RelationMemory):
            return " ".join([memory.summary, " ".join(memory.evidence), memory.source_paper, memory.target_paper]).lower()
        return " ".join([memory.unresolved_question, " ".join(memory.why_open), " ".join(memory.possible_followup)]).lower()

    def _matched_terms_for_chunks(self, query: str, chunks: Sequence[Chunk]) -> list[str]:
        terms = [part for part in query.lower().split() if part]
        return [term for term in terms if any(term in chunk.text.lower() for chunk in chunks)]

    def _matched_terms_for_memories(self, memories: Sequence[MemoryDescriptor]) -> list[str]:
        matched: list[str] = []
        for memory in memories:
            for term in memory.matched_terms:
                if term not in matched:
                    matched.append(term)
        return matched

    def _memory_influence_text(
        self,
        *,
        memory_context: Sequence[MemoryDescriptor],
        source_context: Sequence[ChunkDescriptor],
        should_reread_source: bool,
    ) -> str:
        if source_context:
            return "Answer was shaped by retrieved memory first, then supplemented by source reread."
        if memory_context and not should_reread_source:
            return "Answer was grounded in memory without source reread."
        return "Answer required no supporting memory and no source reread."


__all__ = ["QueryToolExecutor", "ToolExecutionEnvelope"]
