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
        recent_conversation_context: dict[str, object] | None = None,
    ) -> QueryTurnSelection:
        allowed_tools, final_answer_allowed = self.allowed_actions_for_query_loop(state)
        decision = self.decide_query_turn(
            query=query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
            observations=observations,
            recent_conversation_context=recent_conversation_context,
        )
        if decision is None:
            failure_reason = self.query_agent_fallback_reason or "model_agent_returned_no_decision"
            raise InvalidTaskRunStateError(run_id, "agent_decision", failure_reason)

        if decision.action_type == "tool_call":
            if not allowed_tools:
                raise InvalidTaskRunStateError(
                    run_id,
                    "available_query_tool",
                    decision.tool_name.value if decision.tool_name else "none",
                )
            if decision.tool_name not in allowed_tools:
                raise InvalidTaskRunStateError(
                    run_id,
                    "allowed_query_tool",
                    decision.tool_name.value if decision.tool_name else "none",
                )
        elif decision.action_type == "final_answer":
            if not final_answer_allowed:
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
        recent_conversation_context: dict[str, object] | None = None,
    ) -> QueryTurnDecision | None:
        request = state.to_agent_turn_request(
            query=query,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
            observations=observations,
            recent_conversation_context=recent_conversation_context,
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

    def generate_final_answer_for_query_loop(
        self,
        *,
        query: str,
        state: QueryTurnState,
        observations: Sequence[AgentObservation] = (),
        recent_conversation_context: dict[str, object] | None = None,
    ) -> str | None:
        request = state.to_agent_turn_request(
            query=query,
            allowed_tools=(),
            final_answer_allowed=True,
            observations=observations,
            recent_conversation_context=recent_conversation_context,
        )
        generator = getattr(self._query_agent_client, "generate_final_answer", None)
        if callable(generator):
            final_answer = generator(request)
            if final_answer is None:
                raise RuntimeError("model finalization response contained empty content")
            text = str(final_answer).strip()
            if not text:
                raise RuntimeError("model finalization response contained empty content")
            return text
        turn_decision = self._query_agent_client.decide_turn(request)
        if turn_decision is None:
            return None
        if turn_decision.action_type.value != "final_answer":
            return None
        if not turn_decision.final_answer:
            return None
        return turn_decision.final_answer.strip() or None

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

    @property
    def query_agent_failure_detail(self) -> dict[str, object] | None:
        detail = getattr(self._query_agent_client, "failure_detail", None)
        return detail if isinstance(detail, dict) else None

    def allowed_actions_for_query_loop(self, state: QueryTurnState) -> tuple[tuple[QueryToolName, ...], bool]:
        allowed_tools: list[QueryToolName] = []
        query_tool_pool = (
            QueryToolName.SEARCH_SESSION_MEMORY,
            QueryToolName.SEARCH_GLOBAL_MEMORY,
            QueryToolName.SEARCH_OPENVIKING_MEMORY,
            QueryToolName.LIST_SESSION_PAPERS,
            QueryToolName.GET_PAPER_MEMORY_BUNDLE,
            QueryToolName.SEARCH_SOURCE_CHUNKS,
            QueryToolName.RERANK_CANDIDATES,
            QueryToolName.READ_SOURCE_PASSAGES,
            QueryToolName.COMPOSE_ANSWER,
            QueryToolName.LIST_RECENT_MESSAGES,
            QueryToolName.GET_CONVERSATION_CONTEXT,
        )

        for tool_name in query_tool_pool:
            if state.has_completed(tool_name):
                continue
            allowed_tools.append(tool_name)

        return tuple(allowed_tools), True


__all__ = [
    "QueryOrchestrationRunner",
    "QueryTurnSelection",
]
