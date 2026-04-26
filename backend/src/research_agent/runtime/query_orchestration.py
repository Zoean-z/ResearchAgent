"""Framework-neutral query orchestration helpers.

This module owns the host-side query turn selection boundary so the execution
service can stay focused on persistence and tool execution, while different
agents or model adapters only implement the turn decision interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from research_agent.services.errors import InvalidTaskRunStateError
from research_agent.runtime.agent_protocol import AgentObservation
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import QueryToolName


@dataclass(frozen=True, slots=True)
class QueryTurnSelection:
    """Resolved next-step decision together with the runtime constraints used."""

    decision: QueryTurnDecision
    allowed_tools: tuple[QueryToolName, ...]
    final_answer_allowed: bool


class QueryOrchestrationRunner:
    """Host-owned runtime helper for bounded query turn selection."""

    def __init__(self, query_agent_client: QueryTurnClient) -> None:
        self._query_agent_client = query_agent_client

    def choose_next_action_for_query_loop(
        self,
        *,
        query: str,
        state: QueryTurnState,
        run_id: str,
        observations: Sequence[AgentObservation] = (),
    ) -> QueryTurnSelection:
        allowed_tools, final_answer_allowed = self.allowed_actions_for_query_loop(state)
        if not allowed_tools:
            raise InvalidTaskRunStateError(run_id, "available_query_tool", "none")

        decision = self.decide_query_turn(
            query=query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
            observations=observations,
        )
        if decision is None:
            raise InvalidTaskRunStateError(run_id, "agent_decision", "none")

        if decision.action_type == "tool_call":
            if decision.tool_name not in allowed_tools:
                raise InvalidTaskRunStateError(
                    run_id,
                    "allowed_query_tool",
                    decision.tool_name.value if decision.tool_name else "none",
                )
        elif decision.action_type == "final_answer":
            if not final_answer_allowed or QueryToolName.COMPOSE_ANSWER not in allowed_tools:
                raise InvalidTaskRunStateError(run_id, "final_answer_allowed", "false")
            if not decision.final_answer:
                raise InvalidTaskRunStateError(run_id, "final_answer_payload", "empty")
        else:
            raise InvalidTaskRunStateError(run_id, "agent_action_type", decision.action_type)

        return QueryTurnSelection(
            decision=decision,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
        )

    def decide_query_turn(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
        observations: Sequence[AgentObservation] = (),
    ) -> QueryTurnDecision | None:
        request = state.to_agent_turn_request(
            query=query,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
            observations=observations,
        )
        turn_decision = self._query_agent_client.decide_turn(request)
        if turn_decision is None:
            return None
        return QueryTurnDecision.from_agent_turn_decision(
            turn_decision,
            agent_name=self.query_agent_name,
            fallback_used=self.query_agent_fallback_used,
            fallback_reason=self.query_agent_fallback_reason,
        )

    @property
    def query_agent_name(self) -> str:
        return getattr(
            self._query_agent_client,
            "agent_name",
            getattr(self._query_agent_client, "_agent_name", self._query_agent_client.__class__.__name__),
        )

    @property
    def query_agent_fallback_used(self) -> bool:
        return bool(getattr(self._query_agent_client, "fallback_used", False))

    @property
    def query_agent_fallback_reason(self) -> str | None:
        reason = getattr(self._query_agent_client, "fallback_reason", None)
        return str(reason) if reason else None

    def allowed_actions_for_query_loop(self, state: QueryTurnState) -> tuple[tuple[QueryToolName, ...], bool]:
        if state.has_completed(QueryToolName.COMPOSE_ANSWER):
            return (), False

        allowed_tools: list[QueryToolName] = []
        query_tool_pool = (
            QueryToolName.SEARCH_SESSION_MEMORY,
            QueryToolName.SEARCH_GLOBAL_MEMORY,
            QueryToolName.SEARCH_OPENVIKING_MEMORY,
            QueryToolName.RERANK_CANDIDATES,
            QueryToolName.READ_SOURCE_PASSAGES,
            QueryToolName.COMPOSE_ANSWER,
        )

        for tool_name in query_tool_pool:
            if state.has_completed(tool_name):
                continue
            allowed_tools.append(tool_name)

        return tuple(allowed_tools), QueryToolName.COMPOSE_ANSWER in allowed_tools


__all__ = [
    "QueryOrchestrationRunner",
    "QueryTurnSelection",
]
