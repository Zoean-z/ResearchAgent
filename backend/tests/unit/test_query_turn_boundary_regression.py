from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from research_agent.adapters.pydantic_ai import PydanticAIQueryTurnClient
from research_agent.runtime.agent_protocol import AgentTurnDecision, AgentTurnRequest
from research_agent.runtime.query_orchestration import QueryOrchestrationRunner
from research_agent.runtime.query_turn import QueryTurnState
from research_agent.tools import HeuristicQueryToolPlannerClient, PlannerBackedQueryAgentClient, QueryToolName


def test_query_turn_runtime_boundary_accepts_heuristic_and_pydantic_ai_paths() -> None:
    request = AgentTurnRequest(
        query="Did it improve accuracy?",
        allowed_actions=("search_global_memory",),
        final_answer_allowed=False,
        completed_actions=("search_session_memory",),
        state_summary="completed=search_session_memory; session_memories=1; global_memories=0",
    )
    state = QueryTurnState(completed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,))

    heuristic_runner = QueryOrchestrationRunner(
        PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient())
    )
    heuristic_selection = heuristic_runner.choose_next_action_for_query_loop(
        query=request.query,
        state=state,
        run_id="run-heuristic",
    )

    model_agent = Agent(
        TestModel(
            custom_output_args={
                "action_type": "tool_call",
                "tool_name": "search_global_memory",
                "rationale": "model_prefers_global_memory_next",
            }
        ),
        deps_type=AgentTurnRequest,
        output_type=AgentTurnDecision,
    )
    pydantic_ai_runner = QueryOrchestrationRunner(
        PydanticAIQueryTurnClient(
            agent=model_agent,
            fallback=PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient()),
        )
    )
    pydantic_ai_selection = pydantic_ai_runner.choose_next_action_for_query_loop(
        query=request.query,
        state=state,
        run_id="run-pydantic-ai",
    )

    assert heuristic_selection.allowed_tools == pydantic_ai_selection.allowed_tools == (
        QueryToolName.SEARCH_GLOBAL_MEMORY,
        QueryToolName.SEARCH_OPENVIKING_MEMORY,
        QueryToolName.RERANK_CANDIDATES,
        QueryToolName.READ_SOURCE_PASSAGES,
        QueryToolName.COMPOSE_ANSWER,
    )
    assert heuristic_selection.final_answer_allowed is True
    assert pydantic_ai_selection.final_answer_allowed is True
    assert heuristic_selection.decision.action_type == "tool_call"
    assert pydantic_ai_selection.decision.action_type == "tool_call"
    assert heuristic_selection.decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY
    assert pydantic_ai_selection.decision.tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY
