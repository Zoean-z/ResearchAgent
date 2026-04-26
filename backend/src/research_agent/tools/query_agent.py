"""Compatibility implementations for bounded query turn orchestration."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.runtime.agent_protocol import AgentActionType, AgentStopReason, AgentTurnDecision, AgentTurnRequest
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import QueryToolName
from research_agent.tools.query_planner import QueryToolPlannerClient, QueryToolPlannerState


class PlannerBackedQueryAgentClient:
    """Compatibility agent that still delegates next-tool choice to a planner."""

    def __init__(self, planner: QueryToolPlannerClient, agent_name: str = "planner_backed_agent") -> None:
        self._planner = planner
        self._agent_name = agent_name

    @property
    def agent_name(self) -> str:
        return getattr(self, "_last_agent_name", self._agent_name)

    def decide_next_action(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> QueryTurnDecision | None:
        planner_state = QueryToolPlannerState(
            completed_tools=state.completed_tools,
            session_memories=state.session_memories,
            global_memories=state.global_memories,
            selected_memory_ids=state.selected_memory_ids,
            should_reread_source=state.should_reread_source,
            selected_chunks=state.selected_chunks,
        )
        planner_decision = self._planner.choose_next_tool(
            query=query,
            state=planner_state,
            allowed_tools=allowed_tools,
        )
        if planner_decision is None:
            return None
        decision = QueryTurnDecision(
            action_type="tool_call",
            tool_name=planner_decision.tool_name,
            rationale=planner_decision.rationale,
            agent_name=self._agent_name if planner_decision.planner_name == "heuristic" else planner_decision.planner_name,
            fallback_used=planner_decision.fallback_used,
        )
        self._last_agent_name = decision.agent_name
        self._last_fallback_used = decision.fallback_used
        return decision

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        allowed_tools = tuple(QueryToolName(tool_name) for tool_name in request.allowed_actions)
        state = QueryTurnState(
            completed_tools=tuple(QueryToolName(tool_name) for tool_name in request.completed_actions),
        )
        decision = self.decide_next_action(
            query=request.query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=request.final_answer_allowed,
        )
        if decision is None:
            return None
        self._last_agent_name = decision.agent_name
        self._last_fallback_used = decision.fallback_used
        return decision.to_agent_turn_decision()


class StaticFinalAnswerQueryAgentClient:
    """Deterministic agent used by tests to exercise the final-answer path."""

    def __init__(self, answer_text: str, agent_name: str = "static_final_answer_agent") -> None:
        self._answer_text = answer_text
        self._agent_name = agent_name

    @property
    def agent_name(self) -> str:
        return getattr(self, "_last_agent_name", self._agent_name)

    def decide_next_action(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> QueryTurnDecision | None:
        if final_answer_allowed and QueryToolName.COMPOSE_ANSWER in allowed_tools:
            decision = QueryTurnDecision(
                action_type="final_answer",
                final_answer=self._answer_text,
                rationale="agent_has_enough_context_to_answer_without_compose_tool",
                agent_name=self._agent_name,
                fallback_used=False,
            )
            self._last_agent_name = decision.agent_name
            self._last_fallback_used = decision.fallback_used
            return decision
        if not allowed_tools:
            return None
        decision = QueryTurnDecision(
            action_type="tool_call",
            tool_name=allowed_tools[0],
            rationale="static_agent_falls_back_to_first_allowed_tool_before_final_answer_is_allowed",
            agent_name=self._agent_name,
            fallback_used=False,
        )
        self._last_agent_name = decision.agent_name
        self._last_fallback_used = decision.fallback_used
        return decision

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        allowed_tools = tuple(QueryToolName(tool_name) for tool_name in request.allowed_actions)
        state = QueryTurnState(
            completed_tools=tuple(QueryToolName(tool_name) for tool_name in request.completed_actions),
        )
        if request.final_answer_allowed and "compose_answer" in request.allowed_actions:
            return AgentTurnDecision(
                action_type=AgentActionType.FINAL_ANSWER,
                final_answer=self._answer_text,
                rationale="agent_has_enough_context_to_answer_without_compose_tool",
                stop_reason=AgentStopReason.FINAL_ANSWER_READY,
            )
        if not allowed_tools:
            return None
        fallback = self.decide_next_action(
            query=request.query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=request.final_answer_allowed,
        )
        if fallback is None:
            return None
        self._last_agent_name = fallback.agent_name
        self._last_fallback_used = fallback.fallback_used
        return fallback.to_agent_turn_decision()

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)


__all__ = [
    "PlannerBackedQueryAgentClient",
    "QueryAgentClient",
    "QueryAgentDecision",
    "QueryAgentState",
    "QueryAgentTurnClient",
    "StaticFinalAnswerQueryAgentClient",
]

# Compatibility aliases kept in the tool layer only.
QueryAgentClient = QueryTurnClient
QueryAgentDecision = QueryTurnDecision
QueryAgentState = QueryTurnState
QueryAgentTurnClient = QueryTurnClient
