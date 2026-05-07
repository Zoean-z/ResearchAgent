"""Thin execution service for mock follow-up query runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from research_agent.adapters.openviking import OpenVikingAdapterSurfaceBundle, OpenVikingMessageRecord
from research_agent.domain.enums import MessageType, TaskRunStatus
from research_agent.domain.models import Message, TaskRun
from research_agent.domain.policies import CONTEXT_CANDIDATE_TOP_K, CONTEXT_RERANK_TOP_K, build_reread_reason, should_reread_source
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort, TimelineRepositoryPort, TraceRepositoryPort
from research_agent.runtime.agent_protocol import AgentObservation
from research_agent.runtime.streaming import RuntimeEventBroker
from research_agent.runtime.query_orchestration import QueryOrchestrationRunner
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import ChunkDescriptor, MemoryDescriptor, QueryToolName, ToolError, ToolRequest
from research_agent.tools.query_executor import QueryToolExecutor
from research_agent.tools.query_agent import PlannerBackedQueryAgentClient
from research_agent.tools.query_planner import HeuristicQueryToolPlannerClient, QueryToolPlannerClient
from research_agent.tools.registry import InternalToolRegistry
from research_agent.utils import to_json_safe
from research_agent.services.context_rerank_service import ChunkRerankResult, ContextRerankService, MemoryRerankResult
from research_agent.services.errors import EntityNotFoundError, InvalidTaskRunStateError
from research_agent.services.retrieval_service import MemoryRetrievalResult, RetrievalPlan, RetrievalService

from research_agent.services.query_execution_models import (
    PlannedToolCall,
    QueryExecutionError,
    QueryFailureDetail,
    RetrievedMemoryCitation,
    SourceRereadCitation,
)
from research_agent.services.query_citation_builder import (
    build_memory_citations,
    build_source_reread_citations,
    combined_matched_terms,
    matched_terms_for_memories,
    matched_terms_for_text,
    memory_has_evidence,
    max_confidence,
    unique_memories,
)
from research_agent.services.query_observation_builder import (
    memory_rerank_observation,
    memory_search_observation,
    openviking_search_observation,
    reread_decision_observation,
    source_reread_observation,
    turn_observation,
)
from research_agent.services.query_answer_composer import compose_mock_answer_preview
from research_agent.services.query_trace_writer import QueryTraceWriter


RECENT_CONVERSATION_WINDOW = 5


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    """Mock query execution output."""

    task_run: TaskRun
    answer: str
    retrieval_plan: RetrievalPlan
    should_reread_source: bool
    reread_reason: str
    memory_selection_source: str
    memory_selection_fallback_used: bool
    used_memory_citations: tuple[RetrievedMemoryCitation, ...]
    matched_query_terms: tuple[str, ...]
    source_selection_source: str | None
    source_selection_fallback_used: bool
    source_reread_chunks: tuple[SourceRereadCitation, ...]
    observations: tuple[AgentObservation, ...] = ()
    tool_calls: tuple[PlannedToolCall, ...] = ()


class QueryExecutionService:
    """Execute a pending follow-up query run using the thin retrieval service."""

    def __init__(
        self,
        message_repository: MessageRepositoryPort,
        trace_repository: TraceRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        retrieval_service: RetrievalService | None = None,
        context_rerank_service: ContextRerankService | None = None,
        session_repository: SessionRepositoryPort | None = None,
        tool_registry: InternalToolRegistry | None = None,
        query_tool_executor: QueryToolExecutor | None = None,
        query_tool_planner: QueryToolPlannerClient | None = None,
        query_agent_client: QueryTurnClient | None = None,
        openviking_bundle: OpenVikingAdapterSurfaceBundle | None = None,
        runtime_event_broker: RuntimeEventBroker | None = None,
    ) -> None:
        self._message_repository = message_repository
        self._retrieval_service = retrieval_service
        self._context_rerank_service = context_rerank_service
        self._trace_repository = trace_repository
        self._timeline_repository = timeline_repository
        self._session_repository = session_repository
        self._tool_registry = tool_registry
        self._query_tool_executor = query_tool_executor
        self._query_tool_planner = query_tool_planner or HeuristicQueryToolPlannerClient()
        self._openviking_bundle = openviking_bundle or OpenVikingAdapterSurfaceBundle()
        self._query_orchestration = QueryOrchestrationRunner(
            query_agent_client or PlannerBackedQueryAgentClient(self._query_tool_planner)
        )
        self._runtime_event_broker = runtime_event_broker
        self._trace_writer = QueryTraceWriter(trace_repository, timeline_repository, runtime_event_broker)

    def execute_query_run(self, session_id: str, run_id: str) -> QueryExecutionResult:
        """Run the mock query chain and persist trace/timeline placeholders."""

        task_run = self._trace_repository.get_run(run_id)
        if task_run is None:
            raise EntityNotFoundError("TaskRun", run_id)
        if task_run.session_id != session_id:
            raise EntityNotFoundError("TaskRun", run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        message = self._message_repository.get_by_id(task_run.message_id)
        if message is None:
            raise EntityNotFoundError("Message", task_run.message_id)
        if message.type is not MessageType.FOLLOWUP_QUERY:
            raise InvalidTaskRunStateError(run_id, MessageType.FOLLOWUP_QUERY.value, message.type.value)

        if self._query_tool_executor is not None and self._session_repository is not None:
            return self._execute_query_run_with_tools(session_id=session_id, run_id=run_id, task_run=task_run, query=message.content)
        return self._execute_query_run_with_services(session_id=session_id, run_id=run_id, task_run=task_run, query=message.content)

    def _execute_query_run_with_tools(
        self,
        *,
        session_id: str,
        run_id: str,
        task_run: TaskRun,
        query: str,
    ) -> QueryExecutionResult:
        related_paper_ids = tuple(self._related_paper_ids(session_id))
        recent_conversation_context = self._recent_conversation_context(
            session_id=session_id,
            current_message_id=task_run.message_id,
        )
        tool_state = QueryTurnState()
        tool_calls: list[PlannedToolCall] = []
        planner_metadata: dict[str, dict[str, object]] = {}
        session_memories = self._empty_memory_retrieval_result()
        global_memories = self._empty_memory_retrieval_result()
        memory_descriptor_map: dict[str, dict] = {}
        selected_chunk_descriptors: list[dict] = []
        observations: list[AgentObservation] = []
        plan = RetrievalPlan(
            session_memories=session_memories,
            global_memories=global_memories,
            related_paper_ids=related_paper_ids,
            should_reread_source=False,
            reread_reason="",
            memory_confidence=0.0,
        )
        memory_selection = MemoryRerankResult(
            candidates=(),
            selected=(),
            candidate_ids=(),
            selected_ids=(),
            selection_source="rule_fallback",
            fallback_used=True,
            rationale="no_memory_candidates_available",
        )
        source_selection: ChunkRerankResult | None = None
        while True:
            next_turn = self._choose_next_query_action(
                session_id=session_id,
                query=query,
                state=tool_state,
                run_id=run_id,
                observations=tuple(observations),
                recent_conversation_context=recent_conversation_context,
            )
            decision, allowed_tools, final_answer_allowed = (
                next_turn.decision,
                next_turn.allowed_tools,
                next_turn.final_answer_allowed,
            )
            turn_index = len(tool_calls)
            tool_calls.append(self._planned_tool_call(decision, allowed_tools, turn_index=turn_index))

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_SESSION_MEMORY:
                planner_metadata["retrieve_session_memories"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                session_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
                        parameters={"session_id": session_id, "query": query, "top_k": CONTEXT_CANDIDATE_TOP_K},
                    )
                )
                session_output = self._unwrap_tool_output(session_execution.outcome, "search_session_memory", run_id=run_id)
                session_memories = session_execution.raw_result
                observations.append(
                    memory_search_observation(
                        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
                        scope="session",
                        memories=session_memories.memories,
                        coverage_score=session_memories.coverage_score,
                        matched_query_terms=session_memories.matched_query_terms,
                        selection_reasons=session_memories.selection_reasons,
                        decision_impact="Use these session memories before widening recall or rereading source passages.",
                    )
                )
                for descriptor in session_output.result["memories"]:
                    memory_descriptor_map[descriptor["memory_id"]] = descriptor
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.SEARCH_SESSION_MEMORY),
                    session_memories=tuple(MemoryDescriptor.model_validate(item) for item in session_output.result["memories"]),
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY:
                planner_metadata["retrieve_global_memories"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                global_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.SEARCH_GLOBAL_MEMORY,
                        parameters={
                            "query": query,
                            "related_paper_ids": list(related_paper_ids) or None,
                            "top_k": CONTEXT_CANDIDATE_TOP_K,
                        },
                    )
                )
                global_output = self._unwrap_tool_output(global_execution.outcome, "search_global_memory", run_id=run_id)
                global_memories = global_execution.raw_result
                observations.append(
                    memory_search_observation(
                        tool_name=QueryToolName.SEARCH_GLOBAL_MEMORY,
                        scope="global",
                        memories=global_memories.memories,
                        coverage_score=global_memories.coverage_score,
                        matched_query_terms=global_memories.matched_query_terms,
                        selection_reasons=global_memories.selection_reasons,
                        decision_impact="Use these global memories to widen recall before deciding whether source reread is needed.",
                    )
                )
                for descriptor in global_output.result["memories"]:
                    memory_descriptor_map[descriptor["memory_id"]] = descriptor
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.SEARCH_GLOBAL_MEMORY),
                    global_memories=tuple(MemoryDescriptor.model_validate(item) for item in global_output.result["memories"]),
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY:
                scope = self._openviking_search_scope(tool_state)
                planner_metadata["retrieve_openviking_memories"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                openviking_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.SEARCH_OPENVIKING_MEMORY,
                        parameters={
                            "scope": scope,
                            "session_id": session_id if scope == "session" else None,
                            "query": query,
                            "related_paper_ids": list(related_paper_ids) or None,
                            "top_k": CONTEXT_CANDIDATE_TOP_K,
                        },
                    )
                )
                openviking_output = self._unwrap_tool_output(openviking_execution.outcome, "search_openviking_memory", run_id=run_id)
                openviking_memories = tuple(MemoryDescriptor.model_validate(item) for item in openviking_output.result["memories"])
                openviking_result = MemoryRetrievalResult(
                    memories=tuple(memory for memory in openviking_execution.raw_result.memory_descriptors),
                    coverage_score=float(openviking_output.result["coverage_score"]),
                    matched_query_terms=tuple(openviking_output.result["matched_query_terms"]),
                    selection_reasons=tuple(openviking_output.result["selection_reasons"]),
                )
                observations.append(
                    openviking_search_observation(
                        scope=scope,
                        result=openviking_execution.raw_result,
                        decision_impact="Use these OpenViking-mapped memories before falling back to source reread.",
                    )
                )
                if scope == "session":
                    session_memories = openviking_result
                    for descriptor in openviking_output.result["memories"]:
                        memory_descriptor_map[descriptor["memory_id"]] = descriptor
                    tool_state = replace(
                        tool_state,
                        completed_tools=(
                            *tool_state.completed_tools,
                            QueryToolName.SEARCH_SESSION_MEMORY,
                            QueryToolName.SEARCH_OPENVIKING_MEMORY,
                        ),
                        session_memories=openviking_memories,
                    )
                else:
                    global_memories = openviking_result
                    for descriptor in openviking_output.result["memories"]:
                        memory_descriptor_map[descriptor["memory_id"]] = descriptor
                    tool_state = replace(
                        tool_state,
                        completed_tools=(
                            *tool_state.completed_tools,
                            QueryToolName.SEARCH_GLOBAL_MEMORY,
                            QueryToolName.SEARCH_OPENVIKING_MEMORY,
                        ),
                        global_memories=openviking_memories,
                    )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.LIST_SESSION_PAPERS:
                planner_metadata["list_session_papers"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                requested_limit = self._decision_parameter(decision, "limit", 20)
                list_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.LIST_SESSION_PAPERS,
                        parameters={"limit": requested_limit},
                    ),
                    runtime_context={"session_id": session_id},
                )
                list_output = self._unwrap_tool_output(list_execution.outcome, "list_session_papers", run_id=run_id)
                observation = turn_observation(
                    kind="session_papers",
                    summary=f"Current session has {list_output.result['total_count']} imported papers/documents.",
                    payload={
                        "tool_name": QueryToolName.LIST_SESSION_PAPERS.value,
                        "papers": list_output.result["papers"],
                        "decision_impact": "Use these papers when the user asks what is in the current session.",
                    },
                )
                observations.append(observation)
                self._trace_writer.save_tool_trace_step(
                    session_id=session_id,
                    run_id=run_id,
                    action="list_session_papers",
                    input_payload=self._with_planner_payload({"limit": requested_limit}, planner_metadata, "list_session_papers"),
                    result_payload=list_output.result,
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.LIST_SESSION_PAPERS),
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_SOURCE_CHUNKS:
                planner_metadata["search_source_chunks"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                model_paper_id = self._decision_parameter(decision, "paper_id", None)
                source_search_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.SEARCH_SOURCE_CHUNKS,
                        parameters={
                            "session_id": session_id,
                            "query": query,
                            "paper_id": model_paper_id,
                            "related_paper_ids": list(related_paper_ids) or None,
                            "top_k": CONTEXT_CANDIDATE_TOP_K,
                        },
                    )
                )
                source_search_output = self._unwrap_tool_output(source_search_execution.outcome, "search_source_chunks", run_id=run_id)
                observations.append(
                    turn_observation(
                        kind="source_chunk_search",
                        summary=f"Source chunk search returned {len(source_search_output.result['chunks'])} candidate chunks.",
                        payload={
                            "tool_name": QueryToolName.SEARCH_SOURCE_CHUNKS.value,
                            "chunk_ids": [chunk["chunk_id"] for chunk in source_search_output.result["chunks"]],
                            "coverage_score": source_search_output.result["coverage_score"],
                            "matched_query_terms": source_search_output.result["matched_query_terms"],
                            "selection_reasons": source_search_output.result["selection_reasons"],
                            "decision_impact": (
                                "Use these chunks when the user asks for original passages, citations, claims, mechanisms, limits, or evidence."
                            ),
                        },
                    )
                )
                self._trace_writer.save_tool_trace_step(
                    session_id=session_id,
                    run_id=run_id,
                    action="search_source_chunks",
                    input_payload=self._with_planner_payload({"query": query, "top_k": CONTEXT_CANDIDATE_TOP_K}, planner_metadata, "search_source_chunks"),
                    result_payload=source_search_output.result,
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.SEARCH_SOURCE_CHUNKS),
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.GET_PAPER_MEMORY_BUNDLE:
                planner_metadata["get_paper_memory_bundle"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                paper_id = self._decision_parameter(decision, "paper_id", None)
                source_chunk_limit = self._decision_parameter(decision, "source_chunk_limit", 5)
                bundle_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.GET_PAPER_MEMORY_BUNDLE,
                        parameters={"paper_id": paper_id, "source_chunk_limit": source_chunk_limit},
                    )
                )
                bundle_output = self._unwrap_tool_output(bundle_execution.outcome, "get_paper_memory_bundle", run_id=run_id)
                bundle = bundle_output.result["bundle"]
                observations.append(
                    turn_observation(
                        kind="paper_memory_bundle",
                        summary=f"Memory bundle loaded for paper {bundle['paper']['paper_id']}; empty_fields={bundle['empty_fields']}.",
                        payload={
                            "tool_name": QueryToolName.GET_PAPER_MEMORY_BUNDLE.value,
                            "bundle": bundle,
                            "decision_impact": "Use this paper-specific bundle to answer questions about the selected paper.",
                        },
                    )
                )
                self._trace_writer.save_tool_trace_step(
                    session_id=session_id,
                    run_id=run_id,
                    action="get_paper_memory_bundle",
                    input_payload=self._with_planner_payload(
                        {"paper_id": paper_id, "source_chunk_limit": source_chunk_limit},
                        planner_metadata,
                        "get_paper_memory_bundle",
                    ),
                    result_payload=bundle_output.result,
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.GET_PAPER_MEMORY_BUNDLE),
                )
                return self._finalize_query_run(
                    session_id=session_id,
                    run_id=run_id,
                    query=query,
                    state=tool_state,
                    plan=plan,
                    memory_selection=memory_selection,
                    source_selection=source_selection,
                    observations=tuple(observations),
                    tool_calls=tuple(tool_calls),
                    planner_metadata=planner_metadata,
                    task_run=task_run,
                    recent_conversation_context=recent_conversation_context,
                )

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.RERANK_CANDIDATES:
                planner_metadata["rerank_context_candidates"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                plan = RetrievalPlan(
                    session_memories=session_memories,
                    global_memories=global_memories,
                    related_paper_ids=related_paper_ids,
                    should_reread_source=False,
                    reread_reason="",
                    memory_confidence=max_confidence([*session_memories.memories, *global_memories.memories]),
                )
                combined_memories = unique_memories([*session_memories.memories, *global_memories.memories])
                combined_descriptors = [*tool_state.session_memories, *tool_state.global_memories]
                if combined_memories:
                    memory_rerank_execution = self._query_tool_executor.execute_with_raw(
                        ToolRequest(
                            tool_name=QueryToolName.RERANK_CANDIDATES,
                            parameters={
                                "candidate_kind": "memory",
                                "query": query,
                                "candidates": [descriptor.model_dump(mode="python") for descriptor in combined_descriptors],
                                "top_k": CONTEXT_RERANK_TOP_K,
                            },
                        )
                    )
                    memory_rerank_output = self._unwrap_tool_output(memory_rerank_execution.outcome, "rerank_candidates", run_id=run_id)
                    selected_memory_ids = set(memory_rerank_output.result["selected_ids"])
                    selected_memories = tuple(memory for memory in combined_memories if memory.id in selected_memory_ids)
                    memory_selection = MemoryRerankResult(
                        candidates=combined_memories,
                        selected=selected_memories,
                        candidate_ids=tuple(memory.id for memory in combined_memories),
                        selected_ids=tuple(memory.id for memory in selected_memories),
                        selection_source=memory_rerank_output.result["selection_source"],
                        fallback_used=memory_rerank_output.result["fallback_used"],
                        rationale=memory_rerank_output.result["rationale"],
                    )
                else:
                    memory_selection = MemoryRerankResult(
                        candidates=(),
                        selected=(),
                        candidate_ids=(),
                        selected_ids=(),
                        selection_source="rule_fallback",
                        fallback_used=True,
                        rationale="no_memory_candidates_available",
                    )
                should_reread, reread_reason = self._evaluate_reread_decision(
                    selected_memories=memory_selection.selected,
                    related_paper_ids=plan.related_paper_ids,
                )
                plan = replace(
                    plan,
                    should_reread_source=should_reread,
                    reread_reason=reread_reason,
                )
                observations.append(
                    memory_rerank_observation(
                        memory_selection=memory_selection,
                        should_reread_source=should_reread,
                        reread_reason=reread_reason,
                    )
                )
                observations.append(
                    reread_decision_observation(
                        should_reread_source=should_reread,
                        reread_reason=reread_reason,
                        selected_memory_ids=memory_selection.selected_ids,
                        related_paper_ids=plan.related_paper_ids,
                    )
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.RERANK_CANDIDATES),
                    selected_memory_ids=memory_selection.selected_ids,
                    should_reread_source=should_reread,
                )
                next_turn = self._choose_next_query_action(
                    session_id=session_id,
                    query=query,
                    state=tool_state,
                    run_id=run_id,
                    observations=tuple(observations),
                    recent_conversation_context=recent_conversation_context,
                )
                if next_turn is not None:
                    planner_metadata["decide_reread_source"] = self._planner_payload(
                        next_turn.decision,
                        next_turn.allowed_tools,
                        turn_index=len(tool_calls),
                    )
                    decision = next_turn.decision
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.READ_SOURCE_PASSAGES:
                planner_metadata["reread_source_passages"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                model_paper_id = self._decision_parameter(decision, "paper_id", None)
                source_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.READ_SOURCE_PASSAGES,
                        parameters={
                            "session_id": session_id,
                            "query": query,
                            "paper_id": model_paper_id,
                            "related_paper_ids": list(plan.related_paper_ids),
                            "top_k": CONTEXT_RERANK_TOP_K,
                        },
                    )
                )
                source_output = self._unwrap_tool_output(source_execution.outcome, "read_source_passages", run_id=run_id)
                selected_chunk_descriptors = list(source_output.result["chunks"])
                raw_source_selection = source_execution.raw_result
                selected_chunk_ids = {chunk["chunk_id"] for chunk in source_output.result["chunks"]}
                selected_chunks = tuple(chunk for chunk in raw_source_selection.selected if chunk.id in selected_chunk_ids)
                source_selection = ChunkRerankResult(
                    candidates=raw_source_selection.candidates,
                    selected=selected_chunks,
                    candidate_ids=raw_source_selection.candidate_ids,
                    selected_ids=tuple(chunk.id for chunk in selected_chunks),
                    selection_source=source_output.result["selection_source"],
                    fallback_used=source_output.result["fallback_used"],
                    rationale=source_output.result["rationale"],
                )
                observations.append(
                    source_reread_observation(
                        source_selection=source_selection,
                        source_reread_chunks=tuple(
                            SourceRereadCitation(
                                chunk_id=chunk.id,
                                paper_id=chunk.paper_id,
                                page=chunk.page,
                                section=chunk.section,
                                excerpt=chunk.text.strip().replace("\n", " ")[:220],
                                selection_reason="",
                            )
                            for chunk in selected_chunks
                        ),
                    )
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.READ_SOURCE_PASSAGES),
                    selected_chunks=tuple(ChunkDescriptor.model_validate(item) for item in source_output.result["chunks"]),
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.COMPOSE_ANSWER:
                planner_metadata["compose_answer_context"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                used_memory_citations = build_memory_citations(
                    memory_selection.selected,
                    query,
                    memory_selection.selection_source,
                )
                source_reread_chunks = build_source_reread_citations(
                    source_selection.selected if source_selection is not None else (),
                    query,
                    source_selection.selection_source if source_selection is not None else None,
                )

                answer_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.COMPOSE_ANSWER,
                        parameters={
                            "query": query,
                            "memory_context": [memory_descriptor_map[citation.memory_id] for citation in used_memory_citations],
                            "source_context": selected_chunk_descriptors,
                            "session_memory_count": len(session_memories.memories),
                            "global_memory_count": len(global_memories.memories),
                            "memory_selection_source": memory_selection.selection_source,
                            "should_reread_source": plan.should_reread_source,
                            "reread_reason": plan.reread_reason,
                        },
                    )
                )
                answer_output = self._unwrap_tool_output(answer_execution.outcome, "compose_answer", run_id=run_id)
                observations.append(
                    turn_observation(
                        kind="answer_composition",
                        summary=(
                            f"Answer synthesized from {len(used_memory_citations)} memories and "
                            f"{len(source_reread_chunks)} source chunks."
                        ),
                        payload={
                            "tool_name": QueryToolName.COMPOSE_ANSWER.value,
                            "memory_ids": [citation.memory_id for citation in used_memory_citations],
                            "chunk_ids": [citation.chunk_id for citation in source_reread_chunks],
                            "memory_selection_source": memory_selection.selection_source,
                            "source_selection_source": source_selection.selection_source if source_selection is not None else None,
                            "evidence_package": answer_output.result["evidence_package"],
                            "decision_impact": "Return to the model for the final answer.",
                        },
                    )
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.COMPOSE_ANSWER),
                )
                continue

            if decision.action_type == "final_answer" and final_answer_allowed:
                planner_metadata["final_answer"] = self._planner_payload(decision, allowed_tools, turn_index=turn_index)
                return self._finalize_query_run(
                    session_id=session_id,
                    run_id=run_id,
                    query=query,
                    state=tool_state,
                    plan=plan,
                    memory_selection=memory_selection,
                    source_selection=source_selection,
                    observations=tuple(observations),
                    tool_calls=tuple(tool_calls),
                    planner_metadata=planner_metadata,
                    task_run=task_run,
                    recent_conversation_context=recent_conversation_context,
                )

        raise QueryExecutionError(
            QueryFailureDetail(
                error_code="model_finalization_empty_response",
                failed_stage="finalization",
                error_message="query loop ended without a model finalization answer",
                run_id=run_id,
                validation_error="loop_ended_without_final_answer",
            )
        )

    def _unwrap_tool_output(self, outcome, tool_name: str, run_id: str | None = None):
        if isinstance(outcome, ToolError):
            raise QueryExecutionError(
                QueryFailureDetail(
                    error_code=outcome.error_code.value,
                    failed_stage="tool_execution",
                    error_message=outcome.message,
                    run_id=run_id or "",
                    tool_name=tool_name,
                    validation_error=(outcome.details or {}).get("validation_error") if outcome.details else None,
                )
            )
        return outcome

    def _decision_parameter(self, decision: QueryTurnDecision, name: str, default):
        parameters = decision.tool_parameters or {}
        return parameters.get(name, default)

    def _choose_next_query_action(
        self,
        *,
        session_id: str,
        query: str,
        state: QueryTurnState,
        run_id: str,
        observations: tuple[AgentObservation, ...],
        recent_conversation_context: dict[str, object] | None,
    ):
        try:
            return self._query_orchestration.choose_next_action_for_query_loop(
                query=query,
                state=state,
                run_id=run_id,
                observations=observations,
                recent_conversation_context=recent_conversation_context,
            )
        except InvalidTaskRunStateError as exc:
            fallback_reason = self._query_orchestration.query_agent_fallback_reason
            failure_detail = self._query_orchestration.query_agent_failure_detail
            error_code = self._model_decision_error_code(fallback_reason, str(exc))
            allowed_tools, final_answer_allowed = self._query_orchestration.allowed_actions_for_query_loop(state)
            detail = QueryFailureDetail(
                error_code=error_code,
                failed_stage="model_decision",
                error_message=str(exc),
                run_id=run_id,
                fallback_reason=fallback_reason,
                validation_error=str(exc),
                **self._query_failure_detail_kwargs(failure_detail),
            )
            self._trace_writer.save_failure_trace_step(
                session_id=session_id,
                run_id=run_id,
                action="model_decision_failed",
                input_payload={
                    "query": query,
                    "completed_tools": [tool.value for tool in state.completed_tools],
                    "allowed_tools": [tool.value for tool in allowed_tools],
                    "final_answer_allowed": final_answer_allowed,
                    "observation_count": len(observations),
                    "has_recent_conversation_context": recent_conversation_context is not None,
                },
                error_payload=detail.to_dict(),
            )
            raise QueryExecutionError(detail) from exc

    def _generate_final_answer_text(
        self,
        *,
        session_id: str,
        run_id: str,
        query: str,
        state: QueryTurnState,
        observations: tuple[AgentObservation, ...],
        recent_conversation_context: dict[str, object] | None,
    ) -> str:
        try:
            final_answer_text = self._query_orchestration.generate_final_answer_for_query_loop(
                query=query,
                state=state,
                observations=observations,
                recent_conversation_context=recent_conversation_context,
            )
        except Exception as exc:
            fallback_reason = self._query_orchestration.query_agent_fallback_reason
            failure_detail = self._query_orchestration.query_agent_failure_detail
            error_code = self._model_finalization_error_code(fallback_reason, str(exc))
            detail = QueryFailureDetail(
                error_code=error_code,
                failed_stage="finalization",
                error_message=str(exc),
                run_id=run_id,
                fallback_reason=fallback_reason,
                validation_error=str(exc),
                **self._query_failure_detail_kwargs(failure_detail),
            )
            self._trace_writer.save_failure_trace_step(
                session_id=session_id,
                run_id=run_id,
                action="finalization_failed",
                input_payload={
                    "query": query,
                    "completed_tools": [tool.value for tool in state.completed_tools],
                    "observation_count": len(observations),
                    "has_recent_conversation_context": recent_conversation_context is not None,
                },
                error_payload=detail.to_dict(),
            )
            raise QueryExecutionError(detail) from exc

        if final_answer_text is None or not final_answer_text.strip():
            raise QueryExecutionError(
                QueryFailureDetail(
                    error_code="model_finalization_empty_response",
                    failed_stage="finalization",
                    error_message="DeepSeek query-finalization response contained empty content.",
                    run_id=run_id,
                    validation_error="empty finalization answer",
                )
            )
        return final_answer_text.strip()

    def _finalize_query_run(
        self,
        *,
        session_id: str,
        run_id: str,
        query: str,
        state: QueryTurnState,
        plan: RetrievalPlan,
        memory_selection: MemoryRerankResult,
        source_selection: ChunkRerankResult | None,
        observations: tuple[AgentObservation, ...],
        tool_calls: tuple[PlannedToolCall, ...],
        planner_metadata: dict[str, dict[str, object]],
        task_run,
        recent_conversation_context: dict[str, object] | None,
    ) -> QueryExecutionResult:
        used_memory_citations = build_memory_citations(
            memory_selection.selected,
            query,
            memory_selection.selection_source,
        )
        source_reread_chunks = build_source_reread_citations(
            source_selection.selected if source_selection is not None else (),
            query,
            source_selection.selection_source if source_selection is not None else None,
        )
        matched_query_terms = combined_matched_terms(
            query,
            memory_selection.selected,
            source_selection.selected if source_selection is not None else (),
        )
        final_answer_text = self._generate_final_answer_text(
            session_id=session_id,
            run_id=run_id,
            query=query,
            state=state,
            observations=observations,
            recent_conversation_context=recent_conversation_context,
        )

        self._trace_writer.write_trace_and_timeline(
            session_id=session_id,
            run_id=run_id,
            query=query,
            plan=plan,
            memory_selection=memory_selection,
            should_reread_source=plan.should_reread_source,
            reread_reason=plan.reread_reason,
            used_memory_citations=used_memory_citations,
            source_selection=source_selection,
            source_reread_chunks=source_reread_chunks,
            planner_metadata=planner_metadata,
            final_answer_text=final_answer_text,
        )
        assistant_message = self._persist_assistant_answer(
            session_id=session_id,
            run_id=run_id,
            answer=final_answer_text,
            query=query,
            used_memory_citations=used_memory_citations,
            source_reread_chunks=source_reread_chunks,
        )
        observations_list = list(observations)
        observations_list.append(
            turn_observation(
                kind="answer_composition",
                summary=(
                    f"Final answer returned directly after {len(used_memory_citations)} memories and "
                    f"{len(source_reread_chunks)} source chunks."
                ),
                payload={
                    "tool_name": "final_answer",
                    "memory_ids": [citation.memory_id for citation in used_memory_citations],
                    "chunk_ids": [citation.chunk_id for citation in source_reread_chunks],
                    "memory_selection_source": memory_selection.selection_source,
                    "decision_impact": "The host can finish the run because the agent returned a final answer.",
                },
            )
        )

        return QueryExecutionResult(
            task_run=task_run,
            answer=assistant_message.content,
            retrieval_plan=plan,
            should_reread_source=plan.should_reread_source,
            reread_reason=plan.reread_reason,
            memory_selection_source=memory_selection.selection_source,
            memory_selection_fallback_used=memory_selection.fallback_used,
            used_memory_citations=used_memory_citations,
            matched_query_terms=matched_query_terms,
            source_selection_source=source_selection.selection_source if source_selection is not None else None,
            source_selection_fallback_used=source_selection.fallback_used if source_selection is not None else False,
            source_reread_chunks=source_reread_chunks,
            observations=tuple(observations_list),
            tool_calls=tool_calls,
        )

    def _model_decision_error_code(self, fallback_reason: str | None, validation_error: str) -> str:
        reason_text = " ".join(part for part in (fallback_reason, validation_error) if part).lower()
        if "empty content" in reason_text:
            return "model_empty_response"
        return "model_decision_failed"

    def _model_finalization_error_code(self, fallback_reason: str | None, validation_error: str) -> str:
        reason_text = " ".join(part for part in (fallback_reason, validation_error) if part).lower()
        if "empty content" in reason_text:
            return "model_finalization_empty_response"
        return "model_finalization_failed"

    def _query_failure_detail_kwargs(self, failure_detail: dict[str, object] | None) -> dict[str, object]:
        if not isinstance(failure_detail, dict):
            return {}
        allowed_fields = {
            "failure_stage_detail",
            "status_code",
            "repair_attempted",
            "raw_response_preview",
            "content_preview",
        }
        return {
            key: value
            for key, value in failure_detail.items()
            if key in allowed_fields and value is not None
        }

    def _planned_tool_call(
        self,
        decision: QueryTurnDecision,
        allowed_tools: Sequence[QueryToolName],
        *,
        turn_index: int,
    ) -> PlannedToolCall:
        return PlannedToolCall(
            turn_index=turn_index,
            action_type=decision.action_type,
            tool_name=decision.tool_name.value if decision.tool_name is not None else None,
            tool_parameters=decision.tool_parameters or {},
            final_answer=decision.final_answer,
            allowed_tools=tuple(tool.value for tool in allowed_tools),
            rationale=decision.rationale,
            agent_name=decision.agent_name,
            fallback_used=decision.fallback_used,
            validation_error=None,
            fallback_reason=decision.fallback_reason,
        )

    def _planner_payload(
        self,
        decision: QueryTurnDecision,
        allowed_tools: Sequence[QueryToolName],
        *,
        turn_index: int | None = None,
    ) -> dict[str, object]:
        payload = {
            "action_type": decision.action_type,
            "selected_tool": decision.tool_name.value if decision.tool_name is not None else None,
            "tool_parameters": decision.tool_parameters or {},
            "allowed_tools": [tool.value for tool in allowed_tools],
            "rationale": decision.rationale,
            "agent_name": decision.agent_name,
            "fallback_used": decision.fallback_used,
            "final_answer_used": decision.action_type == "final_answer",
        }
        if turn_index is not None:
            payload["turn_index"] = turn_index
        if decision.fallback_reason:
            payload["fallback_reason"] = decision.fallback_reason
        return payload

    def _with_planner_payload(
        self,
        payload: dict[str, object],
        planner_metadata: dict[str, dict[str, object]] | None,
        action: str,
    ) -> dict[str, object]:
        if planner_metadata is None or action not in planner_metadata:
            return payload
        return {
            **payload,
            "planner_decision": planner_metadata[action],
        }

    def _save_tool_trace_step(
        self,
        *,
        session_id: str,
        run_id: str,
        action: str,
        input_payload: dict[str, object],
        result_payload: dict[str, object],
    ) -> None:
        self._trace_writer.save_tool_trace_step(
            session_id=session_id,
            run_id=run_id,
            action=action,
            input_payload=input_payload,
            result_payload=result_payload,
        )

    def _empty_memory_retrieval_result(self) -> MemoryRetrievalResult:
        return MemoryRetrievalResult(
            memories=(),
            coverage_score=0.0,
            matched_query_terms=(),
            selection_reasons=(),
        )

    def _related_paper_ids(self, session_id: str) -> list[str]:
        if self._session_repository is None:
            return []
        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        return [document.paper_id for document in self._session_repository.list_documents(session.id)]

    def _select_source_chunks(
        self,
        *,
        session_id: str,
        query: str,
        plan: RetrievalPlan,
        should_reread_source: bool,
    ) -> ChunkRerankResult | None:
        if not should_reread_source:
            return None
        source_reread_result = self._retrieval_service.retrieve_source_passages(
            session_id=session_id,
            query=query,
            related_paper_ids=plan.related_paper_ids,
            top_k=CONTEXT_CANDIDATE_TOP_K,
        )
        return self._context_rerank_service.rerank_chunks(
            query,
            source_reread_result.chunks,
            top_k=CONTEXT_RERANK_TOP_K,
        )

    def _evaluate_reread_decision(
        self,
        *,
        selected_memories: Sequence,
        related_paper_ids: Sequence[str],
    ) -> tuple[bool, str]:
        from research_agent.domain.models import PaperMemory, RelationMemory
        has_relevant_paper_memory = any(isinstance(memory, PaperMemory) for memory in selected_memories)
        has_evidence_quote = any(memory_has_evidence(memory) for memory in selected_memories)
        has_comparison_target = bool(related_paper_ids) or any(isinstance(memory, RelationMemory) for memory in selected_memories)
        memory_confidence_val = max_confidence(selected_memories)
        reread_required = should_reread_source(
            has_relevant_paper_memory=has_relevant_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence_val,
        )
        return reread_required, build_reread_reason(
            has_relevant_paper_memory=has_relevant_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence_val,
            reread_required=reread_required,
        )

    def _openviking_search_scope(self, state: QueryTurnState) -> str:
        if QueryToolName.SEARCH_SESSION_MEMORY not in state.completed_tools:
            return "session"
        return "global"

    def _recent_conversation_context(
        self,
        *,
        session_id: str,
        current_message_id: str | None,
        limit: int = RECENT_CONVERSATION_WINDOW,
    ) -> dict[str, object] | None:
        if self._tool_registry is None:
            return None
        context_tool = getattr(self._tool_registry, "get_conversation_context", None)
        if not callable(context_tool):
            return None
        context = context_tool(
            session_id=session_id,
            limit=limit,
            exclude_message_id=current_message_id,
        )
        return to_json_safe(context.model_dump(mode="python"))

    def _persist_assistant_answer(
        self,
        *,
        session_id: str,
        run_id: str,
        answer: str,
        query: str,
        used_memory_citations: Sequence[RetrievedMemoryCitation],
        source_reread_chunks: Sequence[SourceRereadCitation],
    ) -> Message:
        assistant_message = self._message_repository.save(
            Message(
                session_id=session_id,
                role="assistant",
                type=MessageType.FOLLOWUP_QUERY,
                content=answer,
                status="completed",
            )
        )
        self._openviking_bundle.sessions.ensure_session(session_id)
        self._openviking_bundle.messages.mirror_message(
            OpenVikingMessageRecord(
                session_id=session_id,
                message_id=assistant_message.id,
                role="assistant",
                content=assistant_message.content,
                metadata={
                    "message_type": assistant_message.type.value,
                    "status": assistant_message.status,
                    "run_id": run_id,
                    "query": query,
                    "used_memory_ids": [citation.memory_id for citation in used_memory_citations],
                    "used_chunk_ids": [citation.chunk_id for citation in source_reread_chunks],
                },
            )
        )
        self._openviking_bundle.sessions.commit_session(session_id)
        if self._runtime_event_broker is not None:
            self._runtime_event_broker.publish_assistant_message(run_id, assistant_message)
        return assistant_message

    def _execute_query_run_with_services(
        self,
        *,
        session_id: str,
        run_id: str,
        task_run: TaskRun,
        query: str,
    ) -> QueryExecutionResult:
        recent_conversation_context = self._recent_conversation_context(
            session_id=session_id,
            current_message_id=task_run.message_id,
        )
        plan = self._retrieval_service.build_retrieval_plan(
            session_id=session_id,
            query=query,
            top_k=CONTEXT_CANDIDATE_TOP_K,
        )
        memory_candidates = unique_memories([*plan.session_memories.memories, *plan.global_memories.memories])
        memory_selection = self._context_rerank_service.rerank_memories(
            query,
            memory_candidates,
            top_k=CONTEXT_RERANK_TOP_K,
        )
        should_reread, reread_reason = self._evaluate_reread_decision(
            selected_memories=memory_selection.selected,
            related_paper_ids=plan.related_paper_ids,
        )
        source_selection = self._select_source_chunks(
            session_id=session_id,
            query=query,
            plan=plan,
            should_reread_source=should_reread,
        )
        used_memory_citations = build_memory_citations(
            memory_selection.selected,
            query,
            memory_selection.selection_source,
        )
        source_reread_chunks = build_source_reread_citations(
            source_selection.selected if source_selection is not None else (),
            query,
            source_selection.selection_source if source_selection is not None else None,
        )
        observations: list[AgentObservation] = [
            memory_search_observation(
                tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
                scope="session",
                memories=plan.session_memories.memories,
                coverage_score=plan.session_memories.coverage_score,
                matched_query_terms=plan.session_memories.matched_query_terms,
                selection_reasons=plan.session_memories.selection_reasons,
                decision_impact="Use these session memories before widening recall or rereading source passages.",
            ),
            memory_search_observation(
                tool_name=QueryToolName.SEARCH_GLOBAL_MEMORY,
                scope="global",
                memories=plan.global_memories.memories,
                coverage_score=plan.global_memories.coverage_score,
                matched_query_terms=plan.global_memories.matched_query_terms,
                selection_reasons=plan.global_memories.selection_reasons,
                decision_impact="Use these global memories to widen recall before deciding whether source reread is needed.",
            ),
            memory_rerank_observation(
                memory_selection=memory_selection,
                should_reread_source=should_reread,
                reread_reason=reread_reason,
            ),
            reread_decision_observation(
                should_reread_source=should_reread,
                reread_reason=reread_reason,
                selected_memory_ids=memory_selection.selected_ids,
                related_paper_ids=plan.related_paper_ids,
            ),
        ]
        if source_selection is not None:
            observations.append(
                source_reread_observation(
                    source_selection=source_selection,
                    source_reread_chunks=source_reread_chunks,
                )
            )
        matched_query_terms = combined_matched_terms(
            query,
            memory_selection.selected,
            source_selection.selected if source_selection is not None else (),
        )

        self._trace_writer.write_trace_and_timeline(
            session_id=session_id,
            run_id=run_id,
            query=query,
            plan=plan,
            memory_selection=memory_selection,
            should_reread_source=should_reread,
            reread_reason=reread_reason,
            used_memory_citations=used_memory_citations,
            source_selection=source_selection,
            source_reread_chunks=source_reread_chunks,
        )

        turn_state = QueryTurnState(
            completed_tools=(
                QueryToolName.SEARCH_SESSION_MEMORY,
                QueryToolName.SEARCH_GLOBAL_MEMORY,
                QueryToolName.RERANK_CANDIDATES,
                *(
                    (QueryToolName.READ_SOURCE_PASSAGES,)
                    if source_selection is not None
                    else ()
                ),
            ),
            session_memories=tuple(plan.session_memories.memories),
            global_memories=tuple(plan.global_memories.memories),
            selected_memory_ids=memory_selection.selected_ids,
            should_reread_source=should_reread,
            selected_chunks=tuple(source_selection.selected if source_selection is not None else ()),
        )
        answer = self._generate_final_answer_text(
            session_id=session_id,
            run_id=run_id,
            query=query,
            state=turn_state,
            observations=tuple(observations),
            recent_conversation_context=recent_conversation_context,
        )
        assistant_message = self._persist_assistant_answer(
            session_id=session_id,
            run_id=run_id,
            answer=answer,
            query=query,
            used_memory_citations=used_memory_citations,
            source_reread_chunks=source_reread_chunks,
        )
        return QueryExecutionResult(
            task_run=task_run,
            answer=assistant_message.content,
            retrieval_plan=plan,
            should_reread_source=should_reread,
            reread_reason=reread_reason,
            memory_selection_source=memory_selection.selection_source,
            memory_selection_fallback_used=memory_selection.fallback_used,
            used_memory_citations=used_memory_citations,
            matched_query_terms=matched_query_terms,
            source_selection_source=source_selection.selection_source if source_selection is not None else None,
            source_selection_fallback_used=source_selection.fallback_used if source_selection is not None else False,
            source_reread_chunks=source_reread_chunks,
            observations=tuple(observations),
        )


# Backward-compatible re-exports
__all__ = [
    "QueryExecutionResult",
    "QueryExecutionService",
    "QueryExecutionError",
    "QueryFailureDetail",
    "RetrievedMemoryCitation",
    "SourceRereadCitation",
    "PlannedToolCall",
]
