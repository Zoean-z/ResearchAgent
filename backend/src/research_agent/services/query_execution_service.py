"""Thin execution service for mock follow-up query runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import re

from research_agent.adapters.openviking import OpenVikingAdapterSurfaceBundle, OpenVikingMessageRecord
from research_agent.domain.enums import MessageType, TaskRunStatus
from research_agent.domain.models import Chunk, Message, OpenQuestionMemory, PaperMemory, RelationMemory, TaskRun, TimelineEvent, TraceStep
from research_agent.domain.policies import CONTEXT_CANDIDATE_TOP_K, CONTEXT_RERANK_TOP_K, build_reread_reason, should_reread_source
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort, TimelineRepositoryPort, TraceRepositoryPort
from research_agent.runtime.agent_protocol import AgentObservation
from research_agent.runtime.streaming import RuntimeEventBroker
from research_agent.runtime.query_orchestration import QueryOrchestrationRunner
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import ChunkDescriptor, MemoryDescriptor, QueryToolName, ToolError, ToolRequest
from research_agent.tools.query_executor import QueryToolExecutor
from research_agent.tools.query_agent import (
    PlannerBackedQueryAgentClient,
)
from research_agent.tools.query_planner import (
    HeuristicQueryToolPlannerClient,
    QueryToolPlannerClient,
)
from research_agent.tools.registry import InternalToolRegistry
from research_agent.services.context_rerank_service import ChunkRerankResult, ContextRerankService, MemoryRerankResult
from research_agent.services.errors import EntityNotFoundError, InvalidTaskRunStateError
from research_agent.services.retrieval_service import MemoryRetrievalResult, RetrievalPlan, RetrievalService, SourceRereadResult

LOW_YIELD_TURN_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class RetrievedMemoryCitation:
    """Human-readable pointer to a retrieved memory record."""

    memory_id: str
    memory_type: str
    summary: str
    selection_reason: str


@dataclass(frozen=True, slots=True)
class SourceRereadCitation:
    """Human-readable pointer to a reread source chunk."""

    chunk_id: str
    paper_id: str
    page: int | None
    section: str | None
    excerpt: str
    selection_reason: str


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    """Single host-controlled tool-planning decision captured during query execution."""

    action_type: str
    tool_name: str | None
    allowed_tools: tuple[str, ...]
    rationale: str
    agent_name: str
    fallback_used: bool
    fallback_reason: str | None = None


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

        conversational_answer = self._conversational_preflight_answer(message.content)
        if conversational_answer is not None:
            return self._execute_conversational_preflight(
                session_id=session_id,
                run_id=run_id,
                task_run=task_run,
                query=message.content,
                answer=conversational_answer,
            )

        if self._query_tool_executor is not None and self._session_repository is not None:
            return self._execute_query_run_with_tools(session_id=session_id, run_id=run_id, task_run=task_run, query=message.content)
        return self._execute_query_run_with_services(session_id=session_id, run_id=run_id, task_run=task_run, query=message.content)

    def _execute_conversational_preflight(
        self,
        *,
        session_id: str,
        run_id: str,
        task_run: TaskRun,
        query: str,
        answer: str,
    ) -> QueryExecutionResult:
        plan = RetrievalPlan(
            session_memories=self._empty_memory_retrieval_result(),
            global_memories=self._empty_memory_retrieval_result(),
            related_paper_ids=(),
            should_reread_source=False,
            reread_reason="direct_conversational_turn",
            memory_confidence=0.0,
        )
        memory_selection = MemoryRerankResult(
            candidates=(),
            selected=(),
            candidate_ids=(),
            selected_ids=(),
            selection_source="conversational_preflight",
            fallback_used=False,
            rationale="query_is_ordinary_conversation_no_retrieval_needed",
        )
        trace_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="direct_final_answer",
                input_payload={
                    "query": query,
                    "preflight": "conversational",
                    "retrieval_skipped": True,
                    "reason": "ordinary_conversation_does_not_need_memory_or_source_tools",
                },
                result_payload={
                    "answer_preview": answer,
                    "memory_citations": [],
                    "source_reread_chunks": [],
                },
            )
        )
        self._publish_step_event(session_id, run_id, trace_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary="answered directly without retrieval",
            )
        )
        assistant_message = self._persist_assistant_answer(
            session_id=session_id,
            run_id=run_id,
            answer=answer,
            query=query,
            used_memory_citations=(),
            source_reread_chunks=(),
        )
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="run_finished",
                summary="query run completed",
            )
        )
        return QueryExecutionResult(
            task_run=task_run,
            answer=assistant_message.content,
            retrieval_plan=plan,
            should_reread_source=False,
            reread_reason=plan.reread_reason,
            memory_selection_source=memory_selection.selection_source,
            memory_selection_fallback_used=memory_selection.fallback_used,
            used_memory_citations=(),
            matched_query_terms=(),
            source_selection_source=None,
            source_selection_fallback_used=False,
            source_reread_chunks=(),
            tool_calls=(
                PlannedToolCall(
                    action_type="final_answer",
                    tool_name=None,
                    allowed_tools=(),
                    rationale=memory_selection.rationale,
                    agent_name="host_conversational_preflight",
                    fallback_used=False,
                ),
            ),
        )

    def _conversational_preflight_answer(self, query: str) -> str | None:
        normalized = self._normalize_conversational_query(query)
        greetings = {"\u4f60\u597d", "\u60a8\u597d", "\u55e8", "\u54c8\u55b7", "hello", "hi", "hey"}
        thanks = {"\u8c22\u8c22", "\u591a\u8c22", "thanks", "thankyou"}
        acknowledgements = {"\u597d\u7684", "ok", "okay", "\u6536\u5230", "\u660e\u767d", "\u4e86\u89e3"}
        if normalized in greetings:
            return "\u4f60\u597d\uff0c\u6211\u5728\u3002"
        if normalized in thanks:
            return "\u4e0d\u5ba2\u6c14\u3002"
        if normalized in acknowledgements:
            return "\u597d\u7684\u3002"
        return None

    def _normalize_conversational_query(self, query: str) -> str:
        normalized = query.strip().lower()
        normalized = re.sub(r"[\s,\uff0c\u3002\uff01\uff1f\u3001:\uff1a\uff1b;]+", "", normalized)
        return normalized

    def _execute_query_run_with_tools(

        self,
        *,
        session_id: str,
        run_id: str,
        task_run: TaskRun,
        query: str,
    ) -> QueryExecutionResult:
        related_paper_ids = tuple(self._related_paper_ids(session_id))
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
        pending_decision: tuple[QueryTurnDecision, tuple[QueryToolName, ...], bool] | None = None
        consecutive_low_yield_turns = 0
        seen_tool_signatures: set[tuple[str, ...]] = set()

        while True:
            if pending_decision is not None:
                decision, allowed_tools, final_answer_allowed = pending_decision
                pending_decision = None
            else:
                next_turn = self._query_orchestration.choose_next_action_for_query_loop(
                    query=query,
                    state=tool_state,
                    run_id=run_id,
                    observations=tuple(observations),
                )
                decision, allowed_tools, final_answer_allowed = (
                    next_turn.decision,
                    next_turn.allowed_tools,
                    next_turn.final_answer_allowed,
                )
            tool_calls.append(self._planned_tool_call(decision, allowed_tools))

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_SESSION_MEMORY:
                duplicate_decision = self._host_duplicate_signature_decision(
                    state=tool_state,
                    seen_signatures=seen_tool_signatures,
                    signature=self._tool_call_signature(
                        kind="memory_search",
                        scope="session",
                        query=query,
                    ),
                )
                if duplicate_decision is not None:
                    pending_decision = duplicate_decision
                    continue
                planner_metadata["retrieve_session_memories"] = self._planner_payload(decision, allowed_tools)
                session_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
                        parameters={"session_id": session_id, "query": query, "top_k": CONTEXT_CANDIDATE_TOP_K},
                    )
                )
                session_output = self._unwrap_tool_output(session_execution.outcome, "search_session_memory")
                session_memories = session_execution.raw_result
                observations.append(
                    self._memory_search_observation(
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
                seen_tool_signatures.add(
                    self._tool_call_signature(
                        kind="memory_search",
                        scope="session",
                        query=query,
                    )
                )
                consecutive_low_yield_turns = self._next_low_yield_streak(
                    consecutive_low_yield_turns,
                    made_progress=bool(session_memories.memories),
                )
                pending_decision = self._host_forced_compose_decision(
                    state=tool_state,
                    consecutive_low_yield_turns=consecutive_low_yield_turns,
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY:
                duplicate_decision = self._host_duplicate_signature_decision(
                    state=tool_state,
                    seen_signatures=seen_tool_signatures,
                    signature=self._tool_call_signature(
                        kind="memory_search",
                        scope="global",
                        query=query,
                        related_paper_ids=related_paper_ids,
                    ),
                )
                if duplicate_decision is not None:
                    pending_decision = duplicate_decision
                    continue
                planner_metadata["retrieve_global_memories"] = self._planner_payload(decision, allowed_tools)
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
                global_output = self._unwrap_tool_output(global_execution.outcome, "search_global_memory")
                global_memories = global_execution.raw_result
                observations.append(
                    self._memory_search_observation(
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
                seen_tool_signatures.add(
                    self._tool_call_signature(
                        kind="memory_search",
                        scope="global",
                        query=query,
                        related_paper_ids=related_paper_ids,
                    )
                )
                consecutive_low_yield_turns = self._next_low_yield_streak(
                    consecutive_low_yield_turns,
                    made_progress=bool(global_memories.memories),
                )
                pending_decision = self._host_forced_compose_decision(
                    state=tool_state,
                    consecutive_low_yield_turns=consecutive_low_yield_turns,
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY:
                scope = self._openviking_search_scope(tool_state)
                duplicate_decision = self._host_duplicate_signature_decision(
                    state=tool_state,
                    seen_signatures=seen_tool_signatures,
                    signature=self._tool_call_signature(
                        kind="memory_search",
                        scope=scope,
                        query=query,
                        related_paper_ids=related_paper_ids if scope == "global" else (),
                    ),
                )
                if duplicate_decision is not None:
                    pending_decision = duplicate_decision
                    continue
                planner_metadata["retrieve_openviking_memories"] = self._planner_payload(decision, allowed_tools)
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
                openviking_output = self._unwrap_tool_output(openviking_execution.outcome, "search_openviking_memory")
                openviking_memories = tuple(MemoryDescriptor.model_validate(item) for item in openviking_output.result["memories"])
                openviking_result = MemoryRetrievalResult(
                    memories=tuple(memory for memory in openviking_execution.raw_result.memory_descriptors),
                    coverage_score=float(openviking_output.result["coverage_score"]),
                    matched_query_terms=tuple(openviking_output.result["matched_query_terms"]),
                    selection_reasons=tuple(openviking_output.result["selection_reasons"]),
                )
                observations.append(
                    self._openviking_search_observation(
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
                seen_tool_signatures.add(
                    self._tool_call_signature(
                        kind="memory_search",
                        scope=scope,
                        query=query,
                        related_paper_ids=related_paper_ids if scope == "global" else (),
                    )
                )
                consecutive_low_yield_turns = self._next_low_yield_streak(
                    consecutive_low_yield_turns,
                    made_progress=bool(openviking_result.memories),
                )
                pending_decision = self._host_forced_compose_decision(
                    state=tool_state,
                    consecutive_low_yield_turns=consecutive_low_yield_turns,
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.RERANK_CANDIDATES:
                planner_metadata["rerank_context_candidates"] = self._planner_payload(decision, allowed_tools)
                plan = RetrievalPlan(
                    session_memories=session_memories,
                    global_memories=global_memories,
                    related_paper_ids=related_paper_ids,
                    should_reread_source=False,
                    reread_reason="",
                    memory_confidence=self._max_confidence([*session_memories.memories, *global_memories.memories]),
                )
                combined_memories = self._unique_memories([*session_memories.memories, *global_memories.memories])
                combined_descriptors = [*tool_state.session_memories, *tool_state.global_memories]
                duplicate_decision = self._host_duplicate_signature_decision(
                    state=tool_state,
                    seen_signatures=seen_tool_signatures,
                    signature=self._tool_call_signature(
                        kind="memory_rerank",
                        candidate_ids=tuple(descriptor.memory_id for descriptor in combined_descriptors),
                    ),
                )
                if duplicate_decision is not None:
                    pending_decision = duplicate_decision
                    planner_metadata["decide_reread_source"] = self._planner_payload(
                        duplicate_decision[0],
                        duplicate_decision[1],
                    )
                    continue
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
                    memory_rerank_output = self._unwrap_tool_output(memory_rerank_execution.outcome, "rerank_candidates")
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
                    self._memory_rerank_observation(
                        memory_selection=memory_selection,
                        should_reread_source=should_reread,
                        reread_reason=reread_reason,
                    )
                )
                observations.append(
                    self._reread_decision_observation(
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
                seen_tool_signatures.add(
                    self._tool_call_signature(
                        kind="memory_rerank",
                        candidate_ids=tuple(descriptor.memory_id for descriptor in combined_descriptors),
                    )
                )
                pending_decision = self._host_forced_compose_decision(
                    state=tool_state,
                    consecutive_low_yield_turns=consecutive_low_yield_turns,
                )
                if pending_decision is not None:
                    planner_metadata["decide_reread_source"] = self._planner_payload(
                        pending_decision[0],
                        pending_decision[1],
                    )
                    continue
                next_turn = self._query_orchestration.choose_next_action_for_query_loop(
                    query=query,
                    state=tool_state,
                    run_id=run_id,
                    observations=tuple(observations),
                )
                if next_turn is not None:
                    planner_metadata["decide_reread_source"] = self._planner_payload(
                        next_turn.decision,
                        next_turn.allowed_tools,
                    )
                    pending_decision = (
                        next_turn.decision,
                        next_turn.allowed_tools,
                        next_turn.final_answer_allowed,
                    )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.READ_SOURCE_PASSAGES:
                duplicate_decision = self._host_duplicate_signature_decision(
                    state=tool_state,
                    seen_signatures=seen_tool_signatures,
                    signature=self._tool_call_signature(
                        kind="source_reread",
                        query=query,
                        related_paper_ids=plan.related_paper_ids,
                    ),
                )
                if duplicate_decision is not None:
                    pending_decision = duplicate_decision
                    continue
                planner_metadata["reread_source_passages"] = self._planner_payload(decision, allowed_tools)
                source_execution = self._query_tool_executor.execute_with_raw(
                    ToolRequest(
                        tool_name=QueryToolName.READ_SOURCE_PASSAGES,
                        parameters={
                            "session_id": session_id,
                            "query": query,
                            "related_paper_ids": list(plan.related_paper_ids),
                            "top_k": CONTEXT_RERANK_TOP_K,
                        },
                    )
                )
                source_output = self._unwrap_tool_output(source_execution.outcome, "read_source_passages")
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
                    self._source_reread_observation(
                        source_selection=source_selection,
                        source_reread_chunks=tuple(
                            self._source_reread_citation(chunk, query, source_selection.selection_source)
                            for chunk in selected_chunks
                        ),
                    )
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.READ_SOURCE_PASSAGES),
                    selected_chunks=tuple(ChunkDescriptor.model_validate(item) for item in source_output.result["chunks"]),
                )
                seen_tool_signatures.add(
                    self._tool_call_signature(
                        kind="source_reread",
                        query=query,
                        related_paper_ids=plan.related_paper_ids,
                    )
                )
                consecutive_low_yield_turns = self._next_low_yield_streak(
                    consecutive_low_yield_turns,
                    made_progress=bool(source_selection.selected_ids),
                )
                pending_decision = self._host_forced_compose_decision(
                    state=tool_state,
                    consecutive_low_yield_turns=consecutive_low_yield_turns,
                )
                continue

            if decision.action_type == "tool_call" and decision.tool_name is QueryToolName.COMPOSE_ANSWER:
                planner_metadata["compose_mock_answer"] = self._planner_payload(decision, allowed_tools)
                used_memory_citations = self._build_memory_citations(
                    memory_selection.selected,
                    query,
                    memory_selection.selection_source,
                )
                source_reread_chunks = self._build_source_reread_citations(
                    source_selection.selected if source_selection is not None else (),
                    query,
                    source_selection.selection_source if source_selection is not None else None,
                )
                matched_query_terms = tuple(
                    dict.fromkeys(
                        [
                            *self._matched_terms(query, memory_selection.selected),
                            *self._matched_terms_for_text(
                                query,
                                source_selection.selected if source_selection is not None else (),
                                lambda chunk: chunk.text,
                            ),
                        ]
                    )
                )

                self._write_trace_and_timeline(
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
                answer_output = self._unwrap_tool_output(answer_execution.outcome, "compose_answer")
                observations.append(
                    self._turn_observation(
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
                            "decision_impact": "The answer is ready to return without another tool call.",
                        },
                    )
                )
                assistant_message = self._persist_assistant_answer(
                    session_id=session_id,
                    run_id=run_id,
                    answer=answer_output.result["answer"],
                    query=query,
                    used_memory_citations=used_memory_citations,
                    source_reread_chunks=source_reread_chunks,
                )
                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.COMPOSE_ANSWER),
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
                    tool_calls=tuple(tool_calls),
                )

            if decision.action_type == "final_answer" and final_answer_allowed:
                planner_metadata["compose_mock_answer"] = self._planner_payload(decision, allowed_tools)
                used_memory_citations = self._build_memory_citations(
                    memory_selection.selected,
                    query,
                    memory_selection.selection_source,
                )
                source_reread_chunks = self._build_source_reread_citations(
                    source_selection.selected if source_selection is not None else (),
                    query,
                    source_selection.selection_source if source_selection is not None else None,
                )
                matched_query_terms = tuple(
                    dict.fromkeys(
                        [
                            *self._matched_terms(query, memory_selection.selected),
                            *self._matched_terms_for_text(
                                query,
                                source_selection.selected if source_selection is not None else (),
                                lambda chunk: chunk.text,
                            ),
                        ]
                    )
                )

                self._write_trace_and_timeline(
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
                    final_answer_text=decision.final_answer,
                )
                assistant_message = self._persist_assistant_answer(
                    session_id=session_id,
                    run_id=run_id,
                    answer=decision.final_answer or "",
                    query=query,
                    used_memory_citations=used_memory_citations,
                    source_reread_chunks=source_reread_chunks,
                )
                observations.append(
                    self._turn_observation(
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

                tool_state = replace(
                    tool_state,
                    completed_tools=(*tool_state.completed_tools, QueryToolName.COMPOSE_ANSWER),
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
                    tool_calls=tuple(tool_calls),
                )

        raise InvalidTaskRunStateError(run_id, "query_tool_loop_completion", "loop_ended_without_compose")

    def _unwrap_tool_output(self, outcome, tool_name: str):
        if isinstance(outcome, ToolError):
            raise InvalidTaskRunStateError(tool_name, "successful_tool_response", outcome.error_code.value)
        return outcome

    def _empty_memory_retrieval_result(self) -> MemoryRetrievalResult:
        return MemoryRetrievalResult(
            memories=(),
            coverage_score=0.0,
            matched_query_terms=(),
            selection_reasons=(),
        )

    def _planned_tool_call(
        self,
        decision: QueryTurnDecision,
        allowed_tools: Sequence[QueryToolName],
    ) -> PlannedToolCall:
        return PlannedToolCall(
            action_type=decision.action_type,
            tool_name=decision.tool_name.value if decision.tool_name is not None else None,
            allowed_tools=tuple(tool.value for tool in allowed_tools),
            rationale=decision.rationale,
            agent_name=decision.agent_name,
            fallback_used=decision.fallback_used,
            fallback_reason=decision.fallback_reason,
        )

    def _planner_payload(
        self,
        decision: QueryTurnDecision,
        allowed_tools: Sequence[QueryToolName],
    ) -> dict[str, object]:
        payload = {
            "action_type": decision.action_type,
            "selected_tool": decision.tool_name.value if decision.tool_name is not None else None,
            "allowed_tools": [tool.value for tool in allowed_tools],
            "rationale": decision.rationale,
            "agent_name": decision.agent_name,
            "fallback_used": decision.fallback_used,
            "final_answer_used": decision.action_type == "final_answer",
        }
        if decision.fallback_reason:
            payload["fallback_reason"] = decision.fallback_reason
        return payload

    def _next_low_yield_streak(self, current_streak: int, *, made_progress: bool) -> int:
        if made_progress:
            return 0
        return current_streak + 1

    def _tool_call_signature(
        self,
        *,
        kind: str,
        query: str | None = None,
        scope: str | None = None,
        related_paper_ids: Sequence[str] = (),
        candidate_ids: Sequence[str] = (),
    ) -> tuple[str, ...]:
        parts = [kind]
        if scope is not None:
            parts.append(scope)
        if query is not None:
            parts.append(self._normalize_query_text(query))
        if related_paper_ids:
            parts.append(",".join(sorted(related_paper_ids)))
        if candidate_ids:
            parts.append(",".join(sorted(candidate_ids)))
        return tuple(parts)

    def _normalize_query_text(self, query: str) -> str:
        return " ".join(query.lower().split())

    def _host_duplicate_signature_decision(
        self,
        *,
        state: QueryTurnState,
        seen_signatures: set[tuple[str, ...]],
        signature: tuple[str, ...],
    ) -> tuple[QueryTurnDecision, tuple[QueryToolName, ...], bool] | None:
        if signature not in seen_signatures:
            return None
        allowed_tools, final_answer_allowed = self._query_orchestration.allowed_actions_for_query_loop(state)
        if QueryToolName.COMPOSE_ANSWER not in allowed_tools:
            return None
        return (
            QueryTurnDecision(
                action_type="tool_call",
                tool_name=QueryToolName.COMPOSE_ANSWER,
                rationale=f"host_forces_compose_answer_after_duplicate_signature:{'|'.join(signature)}",
                agent_name="host_runtime",
                fallback_used=True,
            ),
            allowed_tools,
            final_answer_allowed,
        )

    def _host_forced_compose_decision(
        self,
        *,
        state: QueryTurnState,
        consecutive_low_yield_turns: int,
    ) -> tuple[QueryTurnDecision, tuple[QueryToolName, ...], bool] | None:
        if consecutive_low_yield_turns < LOW_YIELD_TURN_THRESHOLD:
            return None
        allowed_tools, final_answer_allowed = self._query_orchestration.allowed_actions_for_query_loop(state)
        if QueryToolName.COMPOSE_ANSWER not in allowed_tools:
            return None
        return (
            QueryTurnDecision(
                action_type="tool_call",
                tool_name=QueryToolName.COMPOSE_ANSWER,
                rationale=f"host_forces_compose_answer_after_{consecutive_low_yield_turns}_low_yield_turns",
                agent_name="host_runtime",
                fallback_used=True,
            ),
            allowed_tools,
            final_answer_allowed,
        )

    def _openviking_search_scope(self, state: QueryTurnState) -> str:
        if QueryToolName.SEARCH_SESSION_MEMORY not in state.completed_tools:
            return "session"
        return "global"

    def _execute_query_run_with_services(
        self,
        *,
        session_id: str,
        run_id: str,
        task_run: TaskRun,
        query: str,
    ) -> QueryExecutionResult:
        plan = self._retrieval_service.build_retrieval_plan(
            session_id=session_id,
            query=query,
            top_k=CONTEXT_CANDIDATE_TOP_K,
        )
        memory_candidates = self._unique_memories([*plan.session_memories.memories, *plan.global_memories.memories])
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
        used_memory_citations = self._build_memory_citations(
            memory_selection.selected,
            query,
            memory_selection.selection_source,
        )
        source_reread_chunks = self._build_source_reread_citations(
            source_selection.selected if source_selection is not None else (),
            query,
            source_selection.selection_source if source_selection is not None else None,
        )
        observations: list[AgentObservation] = [
            self._memory_search_observation(
                tool_name=QueryToolName.SEARCH_SESSION_MEMORY,
                scope="session",
                memories=plan.session_memories.memories,
                coverage_score=plan.session_memories.coverage_score,
                matched_query_terms=plan.session_memories.matched_query_terms,
                selection_reasons=plan.session_memories.selection_reasons,
                decision_impact="Use these session memories before widening recall or rereading source passages.",
            ),
            self._memory_search_observation(
                tool_name=QueryToolName.SEARCH_GLOBAL_MEMORY,
                scope="global",
                memories=plan.global_memories.memories,
                coverage_score=plan.global_memories.coverage_score,
                matched_query_terms=plan.global_memories.matched_query_terms,
                selection_reasons=plan.global_memories.selection_reasons,
                decision_impact="Use these global memories to widen recall before deciding whether source reread is needed.",
            ),
            self._memory_rerank_observation(
                memory_selection=memory_selection,
                should_reread_source=should_reread,
                reread_reason=reread_reason,
            ),
            self._reread_decision_observation(
                should_reread_source=should_reread,
                reread_reason=reread_reason,
                selected_memory_ids=memory_selection.selected_ids,
                related_paper_ids=plan.related_paper_ids,
            ),
        ]
        if source_selection is not None:
            observations.append(
                self._source_reread_observation(
                    source_selection=source_selection,
                    source_reread_chunks=source_reread_chunks,
                )
            )
        matched_query_terms = tuple(
            dict.fromkeys(
                [
                    *self._matched_terms(query, memory_selection.selected),
                    *self._matched_terms_for_text(
                        query,
                        source_selection.selected if source_selection is not None else (),
                        lambda chunk: chunk.text,
                    ),
                ]
            )
        )

        self._write_trace_and_timeline(
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
        final_turn = self._query_orchestration.choose_next_action_for_query_loop(
            query=query,
            state=turn_state,
            run_id=run_id,
            observations=tuple(observations),
        )
        answer = (
            final_turn.decision.final_answer
            if final_turn.decision.action_type == "final_answer" and final_turn.decision.final_answer
            else self._compose_mock_answer(
                query,
                plan,
                memory_selection,
                should_reread,
                reread_reason,
                used_memory_citations,
                source_reread_chunks,
                source_selection.selection_source if source_selection is not None else None,
            )
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
        )

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

    def _write_trace_and_timeline(
        self,
        *,
        session_id: str,
        run_id: str,
        query: str,
        plan: RetrievalPlan,
        memory_selection: MemoryRerankResult,
        should_reread_source: bool,
        reread_reason: str,
        used_memory_citations: tuple[RetrievedMemoryCitation, ...],
        source_selection: ChunkRerankResult | None,
        source_reread_chunks: tuple[SourceRereadCitation, ...],
        planner_metadata: dict[str, dict[str, object]] | None = None,
        final_answer_text: str | None = None,
    ) -> None:
        session_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="retrieve_session_memories",
                input_payload=self._with_planner_payload(
                    {"session_id": session_id, "top_k": len(plan.session_memories.memories)},
                    planner_metadata,
                    "retrieve_session_memories",
                ),
                result_payload={
                    "memory_ids": [memory.id for memory in plan.session_memories.memories],
                    "memory_citations": [
                        asdict(citation)
                        for citation in self._citations_for(plan.session_memories.memories, query, selection_source="retrieval")
                    ],
                    "coverage_score": plan.session_memories.coverage_score,
                    "matched_query_terms": list(plan.session_memories.matched_query_terms),
                    "selection_reasons": list(plan.session_memories.selection_reasons),
                },
            )
        )
        self._publish_step_event(session_id, run_id, session_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary("checked session memory", plan.session_memories.memories, query),
                related_memory_ids=[memory.id for memory in plan.session_memories.memories],
            )
        )

        global_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="retrieve_global_memories",
                input_payload=self._with_planner_payload(
                    {"session_id": session_id, "related_paper_ids": list(plan.related_paper_ids)},
                    planner_metadata,
                    "retrieve_global_memories",
                ),
                result_payload={
                    "memory_ids": [memory.id for memory in plan.global_memories.memories],
                    "memory_citations": [
                        asdict(citation)
                        for citation in self._citations_for(plan.global_memories.memories, query, selection_source="retrieval")
                    ],
                    "coverage_score": plan.global_memories.coverage_score,
                    "matched_query_terms": list(plan.global_memories.matched_query_terms),
                    "selection_reasons": list(plan.global_memories.selection_reasons),
                },
            )
        )
        self._publish_step_event(session_id, run_id, global_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary("checked global memory", plan.global_memories.memories, query),
                related_memory_ids=[memory.id for memory in plan.global_memories.memories],
            )
        )

        rerank_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="rerank_context_candidates",
                input_payload=self._with_planner_payload(
                    {
                        "candidate_memory_ids": list(memory_selection.candidate_ids),
                        "top_k": len(memory_selection.selected_ids),
                    },
                    planner_metadata,
                    "rerank_context_candidates",
                ),
                result_payload={
                    "selected_memory_ids": list(memory_selection.selected_ids),
                    "selection_source": memory_selection.selection_source,
                    "fallback_used": memory_selection.fallback_used,
                    "rationale": memory_selection.rationale,
                    "memory_citations": [asdict(citation) for citation in used_memory_citations],
                },
            )
        )
        self._publish_step_event(session_id, run_id, rerank_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary_for_memory_selection(memory_selection),
                related_memory_ids=list(memory_selection.selected_ids),
            )
        )

        decide_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="decide_reread_source",
                input_payload=self._with_planner_payload(
                    {
                        "memory_confidence": self._max_confidence(memory_selection.selected),
                        "related_paper_ids": list(plan.related_paper_ids),
                    },
                    planner_metadata,
                    "decide_reread_source",
                ),
                result_payload={
                    "should_reread_source": should_reread_source,
                    "reason": reread_reason,
                    "memory_ids": [citation.memory_id for citation in used_memory_citations],
                    "memory_reasons": [citation.selection_reason for citation in used_memory_citations],
                },
            )
        )
        self._publish_step_event(session_id, run_id, decide_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary="decided whether to reread",
                related_memory_ids=[citation.memory_id for citation in used_memory_citations],
            )
        )

        if source_selection is not None:
            reread_step = self._trace_repository.save_step(
                TraceStep(
                    run_id=run_id,
                    action="reread_source_passages",
                    input_payload=self._with_planner_payload(
                        {
                            "session_id": session_id,
                            "related_paper_ids": list(plan.related_paper_ids),
                            "candidate_chunk_ids": list(source_selection.candidate_ids),
                            "top_k": len(source_selection.selected_ids),
                        },
                        planner_metadata,
                        "reread_source_passages",
                    ),
                    result_payload={
                        "chunk_ids": [citation.chunk_id for citation in source_reread_chunks],
                        "paper_ids": sorted({citation.paper_id for citation in source_reread_chunks}),
                        "excerpts": [citation.excerpt for citation in source_reread_chunks],
                        "selection_source": source_selection.selection_source,
                        "fallback_used": source_selection.fallback_used,
                        "rationale": source_selection.rationale,
                        "selection_reasons": [citation.selection_reason for citation in source_reread_chunks],
                    },
                )
            )
            self._publish_step_event(session_id, run_id, reread_step)
            self._timeline_repository.save(
                TimelineEvent(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="step_completed",
                    summary=self._timeline_summary_for_source_reread(source_reread_chunks, source_selection),
                    related_paper_ids=sorted({citation.paper_id for citation in source_reread_chunks}),
                )
            )

        answer_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="compose_mock_answer",
                input_payload=self._with_planner_payload(
                    {
                        "should_reread_source": should_reread_source,
                        "memory_selection_source": memory_selection.selection_source,
                        "memory_selection_fallback_used": memory_selection.fallback_used,
                        "session_memory_count": len(plan.session_memories.memories),
                        "global_memory_count": len(plan.global_memories.memories),
                        "matched_query_terms": list(
                            dict.fromkeys(
                                [
                                    *plan.session_memories.matched_query_terms,
                                    *plan.global_memories.matched_query_terms,
                                ]
                            )
                        ),
                        "source_reread_chunk_count": len(source_reread_chunks),
                    },
                    planner_metadata,
                    "compose_mock_answer",
                ),
                result_payload={
                    "answer_preview": final_answer_text or self._compose_mock_answer_preview(plan, memory_selection, should_reread_source, source_selection),
                    "memory_citations": [asdict(citation) for citation in used_memory_citations],
                    "source_reread_chunks": [asdict(citation) for citation in source_reread_chunks],
                },
            )
        )
        self._publish_step_event(session_id, run_id, answer_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="run_finished",
                summary="query run completed",
                related_memory_ids=[citation.memory_id for citation in used_memory_citations],
            )
        )

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

    def _publish_step_event(self, session_id: str, run_id: str, trace_step: TraceStep) -> None:
        if self._runtime_event_broker is None:
            return
        self._runtime_event_broker.publish_step_completed(session_id, run_id, trace_step)

    def _build_memory_citations(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        query: str,
        selection_source: str,
    ) -> tuple[RetrievedMemoryCitation, ...]:
        citations: list[RetrievedMemoryCitation] = []
        seen: set[str] = set()
        for memory in memories:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            citations.append(self._citation_for(memory, query, selection_source))
        return tuple(citations)

    def _unique_memories(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    ) -> tuple[PaperMemory | RelationMemory | OpenQuestionMemory, ...]:
        unique: list[PaperMemory | RelationMemory | OpenQuestionMemory] = []
        seen: set[str] = set()
        for memory in memories:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            unique.append(memory)
        return tuple(unique)

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

    def _build_source_reread_citations(
        self,
        source_reread_chunks: Sequence[Chunk],
        query: str,
        selection_source: str | None,
    ) -> tuple[SourceRereadCitation, ...]:
        if not source_reread_chunks:
            return ()
        return tuple(self._source_reread_citation(chunk, query, selection_source) for chunk in source_reread_chunks)

    def _evaluate_reread_decision(
        self,
        *,
        selected_memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        related_paper_ids: Sequence[str],
    ) -> tuple[bool, str]:
        has_relevant_paper_memory = any(isinstance(memory, PaperMemory) for memory in selected_memories)
        has_evidence_quote = any(self._memory_has_evidence(memory) for memory in selected_memories)
        has_comparison_target = bool(related_paper_ids) or any(isinstance(memory, RelationMemory) for memory in selected_memories)
        memory_confidence = self._max_confidence(selected_memories)
        reread_required = should_reread_source(
            has_relevant_paper_memory=has_relevant_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence,
        )
        return reread_required, build_reread_reason(
            has_relevant_paper_memory=has_relevant_paper_memory,
            has_evidence_quote=has_evidence_quote,
            has_comparison_target=has_comparison_target,
            memory_confidence=memory_confidence,
            reread_required=reread_required,
        )

    def _citation_for(
        self,
        memory: PaperMemory | RelationMemory | OpenQuestionMemory,
        query: str,
        selection_source: str,
    ) -> RetrievedMemoryCitation:
        selection_reason = self._memory_selection_reason(memory, query)
        if selection_source != "retrieval":
            selection_reason = f"{selection_reason}; rerank_strategy={selection_source}"
        if isinstance(memory, PaperMemory):
            summary = self._paper_memory_summary(memory)
            return RetrievedMemoryCitation(
                memory_id=memory.id,
                memory_type="paper_memory",
                summary=summary,
                selection_reason=selection_reason,
            )
        if isinstance(memory, RelationMemory):
            summary = self._relation_memory_summary(memory)
            return RetrievedMemoryCitation(
                memory_id=memory.id,
                memory_type="relation_memory",
                summary=summary,
                selection_reason=selection_reason,
            )
        summary = self._open_question_memory_summary(memory)
        return RetrievedMemoryCitation(
            memory_id=memory.id,
            memory_type="open_question_memory",
            summary=summary,
            selection_reason=selection_reason,
        )

    def _citations_for(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        query: str,
        selection_source: str = "retrieval",
    ) -> tuple[RetrievedMemoryCitation, ...]:
        return tuple(self._citation_for(memory, query, selection_source) for memory in memories)

    def _paper_memory_summary(self, memory: PaperMemory) -> str:
        parts = [memory.problem or memory.method or memory.novelty_claim or "paper memory"]
        if memory.key_results:
            parts.append(memory.key_results[0])
        return " | ".join(parts)

    def _relation_memory_summary(self, memory: RelationMemory) -> str:
        return f"{memory.relation_type.value}: {memory.summary}"

    def _open_question_memory_summary(self, memory: OpenQuestionMemory) -> str:
        return memory.unresolved_question

    def _query_terms(self, query: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", query.lower())

    def _memory_text(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
        if isinstance(memory, PaperMemory):
            source_quotes = " ".join(ref.quote or "" for ref in memory.source_refs)
            return " ".join(
                [
                    memory.problem or "",
                    memory.method or "",
                    " ".join(memory.key_results),
                    " ".join(memory.limitations),
                    memory.novelty_claim or "",
                    source_quotes,
                ]
            ).lower()
        if isinstance(memory, RelationMemory):
            return " ".join([memory.summary, " ".join(memory.evidence), memory.source_paper, memory.target_paper]).lower()
        return " ".join([memory.unresolved_question, " ".join(memory.why_open), " ".join(memory.possible_followup)]).lower()

    def _matched_terms(
        self,
        query: str,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    ) -> list[str]:
        terms = self._query_terms(query)
        matched: list[str] = []
        for term in terms:
            if any(term in self._memory_text(memory) for memory in memories):
                matched.append(term)
        return matched

    def _matched_terms_for_text(
        self,
        query: str,
        items: Sequence[Chunk],
        text_getter,
    ) -> list[str]:
        terms = self._query_terms(query)
        matched: list[str] = []
        for term in terms:
            if any(term in text_getter(item).lower() for item in items):
                matched.append(term)
        return matched

    def _memory_selection_reason(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory, query: str) -> str:
        matched_terms = self._matched_terms(query, [memory])
        if isinstance(memory, PaperMemory):
            memory_type = "paper_memory"
            evidence_score = 2 if any(ref.quote for ref in memory.source_refs) else 1 if memory.source_refs else 0
        elif isinstance(memory, RelationMemory):
            memory_type = "relation_memory"
            evidence_score = 2 if memory.evidence else 0
        else:
            memory_type = "open_question_memory"
            evidence_score = 1 if memory.why_open or memory.possible_followup else 0
        matched_text = ",".join(matched_terms) if matched_terms else "none"
        return f"type={memory_type}; matched_terms={matched_text}; evidence_score={evidence_score}; confidence={memory.confidence.value:.2f}"

    def _chunk_selection_reason(self, chunk: Chunk, query: str, selection_source: str | None) -> str:
        matched_terms = self._matched_terms_for_text(query, [chunk], lambda item: item.text)
        matched_text = ",".join(matched_terms) if matched_terms else "none"
        section = chunk.section or "unknown-section"
        page = str(chunk.page) if chunk.page is not None else "unknown-page"
        suffix = f"; rerank_strategy={selection_source}" if selection_source is not None else ""
        return f"matched_terms={matched_text}; section={section}; page={page}{suffix}"

    def _source_reread_citation(
        self,
        chunk: Chunk,
        query: str,
        selection_source: str | None,
    ) -> SourceRereadCitation:
        excerpt = chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 220:
            excerpt = excerpt[:217].rstrip() + "..."
        return SourceRereadCitation(
            chunk_id=chunk.id,
            paper_id=chunk.paper_id,
            page=chunk.page,
            section=chunk.section,
            excerpt=excerpt,
            selection_reason=self._chunk_selection_reason(chunk, query, selection_source),
        )

    def _timeline_summary(
        self,
        prefix: str,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        query: str,
    ) -> str:
        if not memories:
            return f"{prefix} (no memories)"
        citations = ", ".join(f"{citation.memory_type}:{citation.memory_id}" for citation in self._citations_for(memories, query))
        return f"{prefix}: {citations}"

    def _timeline_summary_for_memory_selection(self, selection: MemoryRerankResult) -> str:
        deduped_selected: list[PaperMemory | RelationMemory | OpenQuestionMemory] = []
        seen: set[str] = set()
        for memory in selection.selected:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            deduped_selected.append(memory)
        if not deduped_selected:
            return "reranked context candidates (no memories)"
        selected_text = ", ".join(self._memory_descriptor(memory) for memory in deduped_selected)
        if selection.fallback_used:
            return f"reranked context candidates via fallback: {selected_text}"
        return f"reranked context candidates: {selected_text}"

    def _timeline_summary_for_source_reread(self, source_reread_chunks: Sequence[SourceRereadCitation], selection: ChunkRerankResult | None) -> str:
        if not source_reread_chunks:
            return "reread source passages (no chunks)"
        chunk_ids = ", ".join(citation.chunk_id for citation in source_reread_chunks)
        if selection is not None and selection.fallback_used:
            return f"reread source passages via fallback: {chunk_ids}"
        return f"reread source passages: {chunk_ids}"

    def _turn_observation(
        self,
        *,
        kind: str,
        summary: str,
        payload: dict[str, object] | None = None,
    ) -> AgentObservation:
        return AgentObservation(kind=kind, summary=summary, payload=payload)

    def _memory_search_observation(
        self,
        *,
        tool_name: QueryToolName,
        scope: str,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        coverage_score: float,
        matched_query_terms: Sequence[str],
        selection_reasons: Sequence[str],
        decision_impact: str,
    ) -> AgentObservation:
        memory_ids = [memory.id for memory in memories]
        summary = f"{scope.title()} memory search returned {len(memory_ids)} memories and should influence the next turn."
        return self._turn_observation(
            kind="memory_search",
            summary=summary,
            payload={
                "tool_name": tool_name.value,
                "scope": scope,
                "memory_ids": memory_ids,
                "coverage_score": coverage_score,
                "matched_query_terms": list(matched_query_terms),
                "selection_reasons": list(selection_reasons),
                "decision_impact": decision_impact,
            },
        )

    def _openviking_search_observation(
        self,
        *,
        scope: str,
        result,
        decision_impact: str,
    ) -> AgentObservation:
        descriptors = tuple(result.memory_descriptors)
        memory_ids = [descriptor.memory_id for descriptor in descriptors]
        hit_ids = [hit.item_id for hit in result.hits]
        summary = (
            f"OpenViking {scope} search mapped {len(memory_ids)} memories from {len(hit_ids)} hits and can steer the next turn."
        )
        return self._turn_observation(
            kind="openviking_memory_search",
            summary=summary,
            payload={
                "tool_name": QueryToolName.SEARCH_OPENVIKING_MEMORY.value,
                "scope": scope,
                "hit_ids": hit_ids,
                "memory_ids": memory_ids,
                "matched_local_memory_ids": list(result.matched_local_memory_ids),
                "matched_local_count": result.matched_local_count,
                "coverage_score": min(1.0, len(memory_ids) / max(1, len(hit_ids))),
                "matched_query_terms": sorted(
                    {term for descriptor in descriptors for term in descriptor.matched_terms}
                ),
                "selection_reasons": [descriptor.selection_reason for descriptor in descriptors],
                "decision_impact": decision_impact,
            },
        )

    def _memory_rerank_observation(
        self,
        *,
        memory_selection: MemoryRerankResult,
        should_reread_source: bool,
        reread_reason: str,
    ) -> AgentObservation:
        summary = (
            f"Reranked memory candidates to {len(memory_selection.selected_ids)} selected memories; "
            f"reread_required={should_reread_source}."
        )
        return self._turn_observation(
            kind="memory_rerank",
            summary=summary,
            payload={
                "tool_name": QueryToolName.RERANK_CANDIDATES.value,
                "candidate_ids": list(memory_selection.candidate_ids),
                "selected_ids": list(memory_selection.selected_ids),
                "selection_source": memory_selection.selection_source,
                "fallback_used": memory_selection.fallback_used,
                "rationale": memory_selection.rationale,
                "should_reread_source": should_reread_source,
                "reread_reason": reread_reason,
                "decision_impact": "The selected memories now gate whether source reread is required.",
            },
        )

    def _reread_decision_observation(
        self,
        *,
        should_reread_source: bool,
        reread_reason: str,
        selected_memory_ids: Sequence[str],
        related_paper_ids: Sequence[str],
    ) -> AgentObservation:
        summary = (
            "Source reread is required." if should_reread_source else "Source reread is not required."
        )
        return self._turn_observation(
            kind="reread_decision",
            summary=summary,
            payload={
                "should_reread_source": should_reread_source,
                "reread_reason": reread_reason,
                "selected_memory_ids": list(selected_memory_ids),
                "related_paper_ids": list(related_paper_ids),
                "decision_impact": (
                    "Use stored chunks next if memory is insufficient."
                    if should_reread_source
                    else "The current memory set is sufficient for answer synthesis."
                ),
            },
        )

    def _source_reread_observation(
        self,
        *,
        source_selection: ChunkRerankResult | None,
        source_reread_chunks: Sequence[SourceRereadCitation],
    ) -> AgentObservation:
        selected_ids = [citation.chunk_id for citation in source_reread_chunks]
        summary = f"Source reread selected {len(selected_ids)} chunks and can support or override memory."
        payload = {
            "tool_name": QueryToolName.READ_SOURCE_PASSAGES.value,
            "chunk_ids": selected_ids,
            "selected_chunk_ids": list(source_selection.selected_ids) if source_selection is not None else [],
            "candidate_ids": list(source_selection.candidate_ids) if source_selection is not None else [],
            "selection_source": source_selection.selection_source if source_selection is not None else None,
            "fallback_used": source_selection.fallback_used if source_selection is not None else False,
            "rationale": source_selection.rationale if source_selection is not None else "",
            "decision_impact": "These passages are the strongest evidence if the model needs source support.",
        }
        if not selected_ids:
            summary = "Source reread produced no chunks."
        return self._turn_observation(
            kind="source_reread",
            summary=summary,
            payload=payload,
        )

    def _memory_descriptor(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> str:
        if isinstance(memory, PaperMemory):
            memory_type = "paper_memory"
        elif isinstance(memory, RelationMemory):
            memory_type = "relation_memory"
        else:
            memory_type = "open_question_memory"
        return f"{memory_type}:{memory.id}"

    def _compose_mock_answer(
        self,
        query: str,
        plan: RetrievalPlan,
        memory_selection: MemoryRerankResult,
        should_reread_source: bool,
        reread_reason: str,
        used_memory_citations: tuple[RetrievedMemoryCitation, ...],
        source_reread_chunks: tuple[SourceRereadCitation, ...],
        source_selection_source: str | None,
    ) -> str:
        memory_notes = [self._trim_answer_text(citation.summary, 120) for citation in used_memory_citations if self._trim_answer_text(citation.summary, 120)]
        source_notes = [self._format_source_note(citation.page, citation.section, citation.excerpt) for citation in source_reread_chunks if self._trim_answer_text(citation.excerpt, 140)]

        if memory_notes:
            lead = f"\u6839\u636e\u5f53\u524d\u8bb0\u5fc6\uff0c\u5173\u4e8e\u300c{self._trim_answer_text(query, 80)}\u300d\u53ef\u4ee5\u5148\u6982\u62ec\u4e3a\uff1a{memory_notes[0]}\u3002"
            if len(memory_notes) > 1:
                lead += f"\u8865\u5145\u8bb0\u5fc6\u8fd8\u5305\u62ec\uff1a{'\uff1b'.join(memory_notes[1:3])}\u3002"
        else:
            lead = f"\u5f53\u524d\u5173\u4e8e\u300c{self._trim_answer_text(query, 80)}\u300d\u7684\u8bb0\u5fc6\u8fd8\u4e0d\u591f\u5b8c\u6574\uff0c\u6682\u65f6\u53ea\u80fd\u7ed9\u51fa\u4fdd\u5b88\u56de\u7b54\u3002"

        if source_notes:
            lead += f"\u539f\u6587\u56de\u8bfb\u5230\u7684\u5173\u952e\u7247\u6bb5\u5305\u62ec\uff1a{'\uff1b'.join(source_notes[:2])}\u3002"

        if should_reread_source:
            lead += f"\u7cfb\u7edf\u4ecd\u5efa\u8bae\u7ee7\u7eed\u56de\u8bfb\u539f\u6587\uff0c\u539f\u56e0\u662f\uff1a{self._trim_answer_text(reread_reason, 120)}\u3002"
        else:
            lead += "\u5f53\u524d\u8bb0\u5fc6\u5df2\u7ecf\u8db3\u591f\u76f4\u63a5\u56de\u7b54\u3002"

        if memory_selection.fallback_used:
            lead += " \u8fd9\u6b21\u8bb0\u5fc6\u9009\u62e9\u5305\u542b\u89c4\u5219\u515c\u5e95\u3002"
        if source_selection_source:
            lead += f" \u539f\u6587\u9009\u62e9\u65b9\u5f0f\uff1a{self._trim_answer_text(source_selection_source, 40)}\u3002"
        return lead.strip()

    def _trim_answer_text(self, text: str | None, limit: int) -> str:
        cleaned = (text or "").strip().replace("\n", " ")
        if not cleaned:
            return ""
        return cleaned[:limit]

    def _format_source_note(self, page: int | None, section: str | None, excerpt: str) -> str:
        location = []
        if page is not None:
            location.append(f"\u7b2c{page}\u9875")
        if section:
            location.append(section)
        location_text = " ".join(location) if location else "\u539f\u6587\u7247\u6bb5"
        excerpt_text = self._trim_answer_text(excerpt, 120)
        return f"{location_text}\uff1a{excerpt_text}"

    def _compose_mock_answer_preview(

        self,
        plan: RetrievalPlan,
        memory_selection: MemoryRerankResult,
        should_reread_source: bool,
        source_selection: ChunkRerankResult | None,
    ) -> str:
        source_strategy = source_selection.selection_source if source_selection is not None else "none"
        return (
            f"session={len(plan.session_memories.memories)} "
            f"global={len(plan.global_memories.memories)} "
            f"memory_rerank={memory_selection.selection_source} "
            f"reread={should_reread_source} "
            f"source_rerank={source_strategy}"
        )

    def _max_confidence(
        self,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    ) -> float:
        if not memories:
            return 0.0
        return max(memory.confidence.value for memory in memories)

    def _memory_has_evidence(self, memory: PaperMemory | RelationMemory | OpenQuestionMemory) -> bool:
        if isinstance(memory, PaperMemory):
            return any(ref.quote for ref in memory.source_refs)
        if isinstance(memory, RelationMemory):
            return any(memory.evidence)
        return any(memory.why_open) or any(memory.possible_followup)
