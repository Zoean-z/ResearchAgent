"""Unit tests for the framework-agnostic agent turn protocol."""

from __future__ import annotations

from research_agent.adapters.llm import ModelBackedQueryAgentClient, StaticStructuredQueryAgentTransport
from research_agent.runtime.agent_protocol import AgentActionType, AgentObservation, AgentStopReason, AgentTurnRequest
from research_agent.tools import (
    QueryAgentDecision,
    QueryAgentState,
    QueryToolName,
    StaticFinalAnswerQueryAgentClient,
)


def test_query_agent_state_builds_agent_turn_request() -> None:
    state = QueryAgentState(
        completed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
        selected_memory_ids=("memory-1",),
        should_reread_source=False,
    )

    request = state.to_agent_turn_request(
        query="Did it improve accuracy?",
        allowed_tools=(QueryToolName.SEARCH_GLOBAL_MEMORY, QueryToolName.COMPOSE_ANSWER),
        final_answer_allowed=True,
        observations=(AgentObservation(kind="memory_snapshot", summary="session memory ready"),),
    )

    assert request.query == "Did it improve accuracy?"
    assert request.allowed_actions == ("search_global_memory", "compose_answer")
    assert request.completed_actions == ("search_session_memory",)
    assert request.final_answer_allowed is True
    assert request.state_summary.startswith("completed=search_session_memory")
    assert "evidence" in request.tool_descriptions["compose_answer"].lower()
    assert request.observations[0].kind == "memory_snapshot"


def test_query_agent_decision_round_trips_through_frozen_agent_turn_decision() -> None:
    decision = QueryAgentDecision(
        action_type="final_answer",
        final_answer="Agent final answer",
        rationale="memory_is_sufficient",
        agent_name="model_adapter",
    )

    agent_turn_decision = decision.to_agent_turn_decision()

    assert agent_turn_decision.action_type is AgentActionType.FINAL_ANSWER
    assert agent_turn_decision.stop_reason is AgentStopReason.FINAL_ANSWER_READY

    round_trip = QueryAgentDecision.from_agent_turn_decision(
        agent_turn_decision,
        agent_name="model_adapter",
        fallback_used=False,
    )

    assert round_trip.action_type == "final_answer"
    assert round_trip.final_answer == "Agent final answer"
    assert round_trip.agent_name == "model_adapter"


def test_query_agent_decision_round_trips_through_agent_turn_decision() -> None:
    agent = StaticFinalAnswerQueryAgentClient("Agent final answer")
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("compose_answer",),
        final_answer_allowed=True,
        completed_actions=("search_session_memory", "search_global_memory", "rerank_candidates"),
        state_summary="completed=search_session_memory,search_global_memory,rerank_candidates",
    )

    decision = agent.decide_turn(request)

    assert decision is not None
    assert decision.action_type is AgentActionType.FINAL_ANSWER
    assert decision.final_answer == "Agent final answer"
    assert decision.stop_reason is AgentStopReason.FINAL_ANSWER_READY


def test_model_backed_query_agent_turn_accepts_model_choice() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="tool_call",
            tool_name="search_global_memory",
            rationale="session_memory_checked_expand_recall",
        ),
        fallback=None,
    )
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("search_global_memory",),
        final_answer_allowed=False,
        completed_actions=("search_session_memory",),
        state_summary="completed=search_session_memory; session_memories=1; global_memories=0",
    )

    decision = agent.decide_turn(request)

    assert decision is not None
    assert decision.action_type is AgentActionType.TOOL_CALL
    assert decision.tool_name == "search_global_memory"


def test_model_backed_query_agent_turn_falls_back_on_invalid_choice() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="tool_call",
            tool_name="compose_answer",
            rationale="invalid_choice",
        ),
        fallback=None,
    )
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("search_session_memory",),
        final_answer_allowed=False,
        state_summary="completed=none",
    )

    decision = agent.decide_turn(request)

    assert decision is None
