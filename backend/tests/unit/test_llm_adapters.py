"""Unit tests for provider-specific LLM adapter boundaries."""

from __future__ import annotations

import json

import pytest

from research_agent.adapters.llm import (
    DeepSeekStructuredPlannerTransport,
    DeepSeekStructuredQueryAgentTransport,
    ModelBackedQueryAgentClient,
    StaticStructuredPlannerTransport,
    StaticStructuredQueryAgentTransport,
    StructuredPlannerPrompt,
    StructuredQueryAgentPrompt,
)
from research_agent.tools import HeuristicQueryToolPlannerClient, PlannerBackedQueryAgentClient, QueryAgentState
from research_agent.tools.protocol import QueryToolName


def test_deepseek_transport_builds_chat_completion_request_and_parses_choice() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["timeout_seconds"] = timeout_seconds
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_global_memory",
                                    "rationale": "session memory already checked, expand recall first",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredPlannerTransport(
        api_key="test-key",
        model="deepseekv4flash",
        http_post=fake_http_post,
        timeout_seconds=12.5,
    )

    choice = transport.choose_next_tool(
        StructuredPlannerPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("search_global_memory",),
            completed_tools=("search_session_memory",),
            state_summary="completed=search_session_memory; session_memories=1; global_memories=0",
            tool_descriptions={"search_global_memory": "Search globally stored memories."},
        )
    )

    assert transport.normalized_model == "deepseek-v4-flash"
    assert choice.tool_name == "search_global_memory"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert captured["timeout_seconds"] == 12.5
    request_body = captured["body"]
    assert request_body["model"] == "deepseek-v4-flash"
    assert request_body["response_format"] == {"type": "json_object"}
    assert request_body["messages"][0]["role"] == "system"
    assert "json" in request_body["messages"][0]["content"].lower()
    assert "allowed_tools" in request_body["messages"][1]["content"]


def test_deepseek_transport_requires_api_key() -> None:
    transport = DeepSeekStructuredPlannerTransport(api_key=None)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        transport.choose_next_tool(
            StructuredPlannerPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("search_session_memory",),
                state_summary="completed=none",
            )
        )


def test_deepseek_transport_rejects_empty_content() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        return json.dumps({"choices": [{"message": {"content": ""}}]}).encode("utf-8")

    transport = DeepSeekStructuredPlannerTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    with pytest.raises(RuntimeError, match="empty content"):
        transport.choose_next_tool(
            StructuredPlannerPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("search_session_memory",),
                state_summary="completed=none",
            )
        )


def test_deepseek_query_agent_transport_builds_chat_completion_request_and_parses_tool_call() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["timeout_seconds"] = timeout_seconds
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action_type": "tool_call",
                                    "tool_name": "search_global_memory",
                                    "rationale": "expand recall after session memory",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        model="deepseekv4flash",
        http_post=fake_http_post,
        timeout_seconds=9.5,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("search_global_memory",),
            final_answer_allowed=False,
            completed_tools=("search_session_memory",),
            state_summary="completed=search_session_memory; session_memories=1; global_memories=0",
            tool_descriptions={"search_global_memory": "Search globally stored memories."},
        )
    )

    assert transport.normalized_model == "deepseek-v4-flash"
    assert choice.action_type == "tool_call"
    assert choice.tool_name == "search_global_memory"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert captured["timeout_seconds"] == 9.5
    request_body = captured["body"]
    assert request_body["model"] == "deepseek-v4-flash"
    assert request_body["response_format"] == {"type": "json_object"}
    assert request_body["messages"][0]["role"] == "system"
    assert "final answer" in request_body["messages"][0]["content"].lower()
    assert "prefer final_answer whenever the query is already answerable without retrieval" in request_body["messages"][0]["content"].lower()
    assert "final_answer_allowed" in request_body["messages"][1]["content"]
    assert "if the query can already be answered well without more evidence, choose final_answer" in request_body["messages"][1]["content"].lower()


def test_deepseek_query_agent_transport_parses_final_answer_choice() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action_type": "final_answer",
                                    "final_answer": "The model improved accuracy by 12%.",
                                    "rationale": "sufficient memory and reread evidence",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("compose_answer",),
            final_answer_allowed=True,
            state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
        )
    )

    assert choice.action_type == "final_answer"
    assert choice.final_answer == "The model improved accuracy by 12%."


def test_model_backed_query_agent_client_accepts_valid_final_answer_choice() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="final_answer",
            final_answer="Agent final answer",
            rationale="model_has_enough_context_to_answer",
        ),
        fallback=PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient()),
    )

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(
            completed_tools=(
                QueryToolName.SEARCH_SESSION_MEMORY,
                QueryToolName.SEARCH_GLOBAL_MEMORY,
                QueryToolName.RERANK_CANDIDATES,
            ),
            selected_memory_ids=("memory-1",),
            should_reread_source=False,
        ),
        allowed_tools=(QueryToolName.COMPOSE_ANSWER,),
        final_answer_allowed=True,
    )

    assert decision is not None
    assert decision.action_type == "final_answer"
    assert decision.final_answer == "Agent final answer"
    assert decision.agent_name == "model_adapter"
    assert decision.fallback_used is False


def test_model_backed_query_agent_client_falls_back_when_choice_is_invalid() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="tool_call",
            tool_name="compose_answer",
            rationale="invalid_tool_choice",
        ),
        fallback=PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient()),
    )

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(),
        allowed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
        final_answer_allowed=False,
    )

    assert decision is not None
    assert decision.action_type == "tool_call"
    assert decision.tool_name is QueryToolName.SEARCH_SESSION_MEMORY
    assert decision.agent_name == "model_adapter"
    assert decision.fallback_used is True
    assert "fallback_after_model_adapter_error" in decision.rationale
