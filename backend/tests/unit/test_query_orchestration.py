from __future__ import annotations

import pytest

from research_agent.runtime.query_orchestration import QueryOrchestrationRunner
from research_agent.runtime.agent_protocol import AgentActionType, AgentObservation, AgentStopReason, AgentTurnDecision
from research_agent.runtime.query_turn import QueryTurnState
from research_agent.services.errors import InvalidTaskRunStateError
from research_agent.tools.protocol import QueryToolName


class _StubQueryAgentClient:
    def __init__(self, decision: AgentTurnDecision | None) -> None:
        self._decision = decision
        self.agent_name = "stub_agent"
        self.fallback_used = False
        self.fallback_reason = None
        self.last_request = None

    def decide_turn(self, request):
        self.last_request = request
        return self._decision


def test_query_orchestration_runner_allows_the_next_expected_tool() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.TOOL_CALL,
            tool_name=QueryToolName.SEARCH_GLOBAL_MEMORY,
            rationale="expand recall",
        )
    )
    runner = QueryOrchestrationRunner(client)
    state = QueryTurnState(completed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,))

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=state,
        run_id="run-1",
        observations=(AgentObservation(kind="memory_search", summary="session memory returned 2 candidates"),),
    )

    assert selection.allowed_tools == (
        QueryToolName.SEARCH_GLOBAL_MEMORY,
        QueryToolName.SEARCH_OPENVIKING_MEMORY,
        QueryToolName.RERANK_CANDIDATES,
        QueryToolName.READ_SOURCE_PASSAGES,
        QueryToolName.COMPOSE_ANSWER,
    )
    assert selection.final_answer_allowed is True
    assert selection.decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY
    assert client.last_request is not None
    assert client.last_request.allowed_actions == (
        "search_global_memory",
        "search_openviking_memory",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    )
    assert client.last_request.observations[0].kind == "memory_search"


def test_query_orchestration_runner_exposes_openviking_memory_as_companion_tool() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.TOOL_CALL,
            tool_name=QueryToolName.SEARCH_OPENVIKING_MEMORY,
            rationale="use explicit openviking retrieval",
        )
    )
    runner = QueryOrchestrationRunner(client)
    state = QueryTurnState()

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=state,
        run_id="run-ov",
        observations=(AgentObservation(kind="openviking_search", summary="openviking mapped 1 memory from 2 hits"),),
    )

    assert selection.allowed_tools == (
        QueryToolName.SEARCH_SESSION_MEMORY,
        QueryToolName.SEARCH_GLOBAL_MEMORY,
        QueryToolName.SEARCH_OPENVIKING_MEMORY,
        QueryToolName.RERANK_CANDIDATES,
        QueryToolName.READ_SOURCE_PASSAGES,
        QueryToolName.COMPOSE_ANSWER,
    )
    assert selection.decision.tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY
    assert client.last_request is not None
    assert client.last_request.allowed_actions == (
        "search_session_memory",
        "search_global_memory",
        "search_openviking_memory",
        "rerank_candidates",
        "read_source_passages",
        "compose_answer",
    )


def test_query_orchestration_runner_allows_early_final_answers_when_enabled() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.FINAL_ANSWER,
            final_answer="done",
            rationale="answer too early",
            stop_reason=AgentStopReason.FINAL_ANSWER_READY,
        )
    )
    runner = QueryOrchestrationRunner(client)

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=QueryTurnState(),
        run_id="run-2",
    )

    assert selection.final_answer_allowed is True
    assert selection.decision.action_type == "final_answer"
    assert selection.decision.final_answer == "done"


def test_query_orchestration_runner_preserves_fallback_reason() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.TOOL_CALL,
            tool_name=QueryToolName.COMPOSE_ANSWER,
            rationale="fallback chose compose",
        )
    )
    client.agent_name = "planner_backed_agent"
    client.fallback_used = True
    client.fallback_reason = "ModelHTTPError: unsupported tool_choice"
    runner = QueryOrchestrationRunner(client)

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=QueryTurnState(),
        run_id="run-fallback",
    )

    assert selection.decision.agent_name == "planner_backed_agent"
    assert selection.decision.fallback_used is True
    assert selection.decision.fallback_reason == "ModelHTTPError: unsupported tool_choice"


def test_query_orchestration_runner_broadens_the_later_tool_pool_after_rerank() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.TOOL_CALL,
            tool_name=QueryToolName.READ_SOURCE_PASSAGES,
            rationale="read source after rerank",
        )
    )
    runner = QueryOrchestrationRunner(client)
    state = QueryTurnState(
        completed_tools=(
            QueryToolName.SEARCH_SESSION_MEMORY,
            QueryToolName.SEARCH_GLOBAL_MEMORY,
            QueryToolName.RERANK_CANDIDATES,
        ),
        should_reread_source=False,
    )

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=state,
        run_id="run-late",
        observations=(AgentObservation(kind="memory_rerank", summary="rerank complete"),),
    )

    assert selection.allowed_tools == (
        QueryToolName.SEARCH_OPENVIKING_MEMORY,
        QueryToolName.READ_SOURCE_PASSAGES,
        QueryToolName.COMPOSE_ANSWER,
    )
    assert selection.final_answer_allowed is True
    assert selection.decision.tool_name is QueryToolName.READ_SOURCE_PASSAGES


def test_query_orchestration_runner_does_not_reoffer_completed_tools_in_the_unified_pool() -> None:
    client = _StubQueryAgentClient(
        AgentTurnDecision(
            action_type=AgentActionType.TOOL_CALL,
            tool_name=QueryToolName.READ_SOURCE_PASSAGES,
            rationale="avoid repeating completed tools",
        )
    )
    runner = QueryOrchestrationRunner(client)
    state = QueryTurnState(
        completed_tools=(
            QueryToolName.SEARCH_SESSION_MEMORY,
            QueryToolName.SEARCH_GLOBAL_MEMORY,
            QueryToolName.RERANK_CANDIDATES,
        ),
        should_reread_source=True,
    )

    selection = runner.choose_next_action_for_query_loop(
        query="what does it say?",
        state=state,
        run_id="run-no-repeat",
    )

    assert QueryToolName.SEARCH_SESSION_MEMORY not in selection.allowed_tools
    assert QueryToolName.SEARCH_GLOBAL_MEMORY not in selection.allowed_tools
    assert QueryToolName.RERANK_CANDIDATES not in selection.allowed_tools
    assert selection.allowed_tools == (
        QueryToolName.SEARCH_OPENVIKING_MEMORY,
        QueryToolName.READ_SOURCE_PASSAGES,
        QueryToolName.COMPOSE_ANSWER,
    )
