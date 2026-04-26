"""Unit tests for the PydanticAI query-turn adapter boundary."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from research_agent.adapters.pydantic_ai import PydanticAIQueryTurnClient
from research_agent.runtime.agent_protocol import AgentActionType, AgentStopReason, AgentTurnDecision, AgentTurnRequest
from research_agent.tools import HeuristicQueryToolPlannerClient, PlannerBackedQueryAgentClient, QueryToolName


def test_pydantic_ai_query_turn_client_returns_structured_final_answer() -> None:
    agent = Agent(
        TestModel(
            custom_output_args={
                "action_type": "final_answer",
                "final_answer": "PydanticAI answer",
                "rationale": "enough_context",
                "stop_reason": "final_answer_ready",
            }
        ),
        deps_type=AgentTurnRequest,
        output_type=AgentTurnDecision,
    )
    client = PydanticAIQueryTurnClient(
        agent=agent,
        fallback=PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient()),
    )
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("compose_answer",),
        final_answer_allowed=True,
        completed_actions=("search_session_memory", "search_global_memory", "rerank_candidates"),
        state_summary="completed=search_session_memory,search_global_memory,rerank_candidates",
    )

    decision = client.decide_turn(request)

    assert decision is not None
    assert decision.action_type is AgentActionType.FINAL_ANSWER
    assert decision.final_answer == "PydanticAI answer"
    assert decision.stop_reason is AgentStopReason.FINAL_ANSWER_READY
    assert client.fallback_used is False
    assert client.agent_name.startswith("pydantic_ai:")


def test_pydantic_ai_query_turn_client_falls_back_when_agent_raises() -> None:
    class BrokenAgent:
        def run_sync(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    client = PydanticAIQueryTurnClient(
        agent=BrokenAgent(),
        fallback=PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient()),
    )
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("search_session_memory",),
        final_answer_allowed=False,
        state_summary="completed=none",
    )

    decision = client.decide_turn(request)

    assert decision is not None
    assert decision.action_type is AgentActionType.TOOL_CALL
    assert decision.tool_name == QueryToolName.SEARCH_SESSION_MEMORY
    assert client.fallback_used is True
