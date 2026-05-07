"""Unit tests for provider-specific LLM adapter boundaries."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
import json
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from research_agent.runtime.agent_protocol import AgentObservation, AgentTurnRequest

from research_agent.adapters.llm import (
    DeepSeekStructuredPlannerTransport,
    DeepSeekStructuredQueryAgentTransport,
    DeepSeekHttpResponse,
    DeepSeekQueryAgentResponseError,
    DeepSeekStructuredIngestExtractionTransport,
    ModelBackedQueryAgentClient,
    QueryAgentFailureDetail,
    StructuredIngestExtractionPrompt,
    StaticStructuredPlannerTransport,
    StaticStructuredQueryAgentTransport,
    StructuredPlannerPrompt,
    StructuredQueryAgentPrompt,
)
from research_agent.utils import reset_request_api_key_override, set_request_api_key_override
from research_agent.tools import QueryAgentState
from research_agent.tools.protocol import QueryToolName


def _deepseek_response(payload: dict[str, object], status_code: int = 200) -> DeepSeekHttpResponse:
    return DeepSeekHttpResponse(status_code=status_code, body=json.dumps(payload).encode("utf-8"))


class _ObservationEnum(str, Enum):
    READY = "ready"


class _ObservationModel(BaseModel):
    value: int


@dataclass
class _ObservationData:
    created_at: datetime
    state: _ObservationEnum


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
    assert request_body["max_tokens"] == 256
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


def test_deepseek_planner_transport_prefers_request_scoped_api_key_override() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        captured["headers"] = headers
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_session_memory",
                                    "rationale": "override key worked",
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredPlannerTransport(api_key="env-key", http_post=fake_http_post)
    token = set_request_api_key_override("header-key")
    try:
        choice = transport.choose_next_tool(
            StructuredPlannerPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("search_session_memory",),
                state_summary="completed=none",
            )
        )
    finally:
        reset_request_api_key_override(token)

    assert choice.tool_name == "search_session_memory"
    assert captured["headers"] == {
        "Authorization": "Bearer header-key",
        "Content-Type": "application/json",
    }


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


def test_deepseek_ingest_transport_disables_thinking_in_request() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        captured["body"] = json.loads(body.decode("utf-8"))
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "understanding": {
                                        "topic": None,
                                        "problem": None,
                                        "method": None,
                                        "novelty_claims": [],
                                        "key_results": [],
                                        "experiment_design": None,
                                        "limitations": [],
                                        "open_questions": [],
                                        "evidence_chunk_ids": [],
                                        "confidence": 0.5,
                                    },
                                    "needs_more_context": True,
                                    "context_hints": [],
                                    "rationale": "needs more evidence",
                                }
                            )
                        },
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredIngestExtractionTransport(api_key="test-key", http_post=fake_http_post)
    prompt = StructuredIngestExtractionPrompt(
        session_id="session-1",
        paper_title="Paper Title",
        paper_abstract=None,
        related_paper_titles=(),
        window_kind="broad",
        extraction_stage="full_text",
        context_summary="summary",
        candidate_passages=(),
    )

    choice = transport.extract(prompt)

    assert choice.needs_more_context is True
    assert captured["body"]["thinking"] == {"type": "disabled"}


def test_deepseek_ingest_transport_prefers_request_scoped_api_key_override() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        captured["headers"] = headers
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "understanding": {
                                        "topic": None,
                                        "problem": None,
                                        "method": None,
                                        "novelty_claims": [],
                                        "key_results": [],
                                        "experiment_design": None,
                                        "limitations": [],
                                        "open_questions": [],
                                        "evidence_chunk_ids": [],
                                        "confidence": 0.5,
                                    },
                                    "needs_more_context": True,
                                    "context_hints": [],
                                    "rationale": "needs more evidence",
                                }
                            )
                        },
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredIngestExtractionTransport(api_key="env-key", http_post=fake_http_post)
    prompt = StructuredIngestExtractionPrompt(
        session_id="session-1",
        paper_title="Paper Title",
        paper_abstract=None,
        related_paper_titles=(),
        window_kind="broad",
        extraction_stage="full_text",
        context_summary="summary",
        candidate_passages=(),
    )
    token = set_request_api_key_override("header-key")
    try:
        transport.extract(prompt)
    finally:
        reset_request_api_key_override(token)

    assert captured["headers"] == {
        "Authorization": "Bearer header-key",
        "Content-Type": "application/json",
    }


def test_deepseek_ingest_transport_normalizes_value_and_single_object_shapes() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "understanding": {
                                        "topic": {"value": "Topic from value", "evidence_chunk_ids": ["chunk-1"]},
                                        "problem": {"text": "Problem from text", "evidence_chunk_ids": ["chunk-2"]},
                                        "method": "Method as plain string",
                                        "novelty_claims": {"value": "Novel claim from single object"},
                                        "key_results": [{"value": "Result from array object"}],
                                        "experiment_design": {"text": "Experiment design", "evidence_chunk_ids": ["chunk-3"]},
                                        "limitations": None,
                                        "open_questions": {"text": "Open question", "evidence_chunk_ids": ["chunk-4"]},
                                        "evidence_chunk_ids": ["chunk-1", "chunk-2"],
                                        "confidence": 0.8,
                                    },
                                    "paper": {
                                        "problem": {"value": "Paper problem"},
                                        "method": {"text": "Paper method"},
                                        "key_results": {"value": "Paper result"},
                                        "limitations": None,
                                        "novelty_claim": {"value": "Paper novelty"},
                                        "evidence_candidate_ids": ["chunk-1"],
                                        "confidence": 0.7,
                                    },
                                    "relation": {
                                        "relation_type": {"value": "improves_on"},
                                        "summary": {"text": "Related summary"},
                                        "evidence_candidate_ids": [],
                                        "confidence": 0.6,
                                    },
                                    "open_question": {
                                        "unresolved_question": {"value": "Why is it open?"},
                                        "why_open": {"value": "Because evidence is weak."},
                                        "possible_followup": "Follow up question",
                                        "evidence_candidate_ids": ["chunk-4"],
                                        "confidence": 0.5,
                                    },
                                    "paper_summary": {
                                        "what_it_is_about": {"value": "About the paper"},
                                        "problem_solved": {"text": "Solved the issue"},
                                        "new_ideas": {"value": "Idea from summary"},
                                        "limitations": {"value": "Summary limitation"},
                                        "suggestions_or_questions": {"text": "Summary follow-up"},
                                        "evidence_candidate_ids": ["chunk-1"],
                                        "confidence": 0.9,
                                    },
                                    "needs_more_context": False,
                                    "context_hints": {"value": "Need more benchmark evidence"},
                                    "rationale": "normalized payload",
                                }
                            )
                        },
                    }
                ]
            }
        ).encode("utf-8")

    transport = DeepSeekStructuredIngestExtractionTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.extract(
        StructuredIngestExtractionPrompt(
            session_id="session-1",
            paper_title="Paper Title",
            paper_abstract=None,
            related_paper_titles=(),
            window_kind="broad",
            extraction_stage="full_text",
            context_summary="summary",
            candidate_passages=(),
        )
    )

    assert choice.understanding is not None
    assert choice.understanding.topic is not None
    assert choice.understanding.topic.text == "Topic from value"
    assert choice.understanding.topic.evidence_status == "strong"
    assert choice.understanding.method is not None
    assert choice.understanding.method.text == "Method as plain string"
    assert choice.understanding.method.evidence_status == "weak"
    assert len(choice.understanding.novelty_claims) == 1
    assert choice.understanding.novelty_claims[0].text == "Novel claim from single object"
    assert choice.understanding.novelty_claims[0].evidence_status == "weak"
    assert len(choice.understanding.key_results) == 1
    assert choice.understanding.key_results[0].text == "Result from array object"
    assert choice.understanding.key_results[0].evidence_status == "weak"
    assert choice.understanding.limitations == ()
    assert len(choice.understanding.open_questions) == 1
    assert choice.understanding.open_questions[0].text == "Open question"
    assert choice.paper is not None
    assert choice.paper.problem == "Paper problem"
    assert choice.paper.key_results == ("Paper result",)
    assert choice.open_question is not None
    assert choice.open_question.unresolved_question == "Why is it open?"
    assert choice.open_question.why_open == ("Because evidence is weak.",)
    assert choice.open_question.possible_followup == ("Follow up question",)
    assert choice.paper_summary is not None
    assert choice.paper_summary.what_it_is_about == "About the paper"
    assert choice.paper_summary.new_ideas == ("Idea from summary",)
    assert choice.context_hints == ("Need more benchmark evidence",)


def test_deepseek_query_agent_transport_builds_chat_completion_request_and_parses_tool_call() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json.loads(body.decode("utf-8"))
        captured["timeout_seconds"] = timeout_seconds
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "action_type": "tool_call",
                                    "tool_name": "search_global_memory",
                                    "rationale": "expand recall after session memory",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

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
    assert request_body["max_tokens"] == 640
    assert request_body["messages"][0]["role"] == "system"
    assert "final answer" in request_body["messages"][0]["content"].lower()
    assert "prefer final_answer whenever the query is already answerable without retrieval" in request_body["messages"][0]["content"].lower()
    assert "tool_parameters" in request_body["messages"][0]["content"]
    assert "example tool_call" in request_body["messages"][0]["content"].lower()
    assert "arguments" not in request_body["messages"][0]["content"].lower()
    assert "final_answer_allowed" in request_body["messages"][1]["content"]
    assert "output valid json and follow one of the example json shapes exactly" in request_body["messages"][1]["content"].lower()


def test_deepseek_query_agent_transport_prefers_request_scoped_api_key_override() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        captured["headers"] = headers
        return _deepseek_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action_type": "tool_call",
                                    "tool_name": "search_global_memory",
                                    "rationale": "override key worked",
                                }
                            )
                        }
                    }
                ]
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(api_key="env-key", http_post=fake_http_post)
    token = set_request_api_key_override("header-key")
    try:
        choice = transport.choose_next_action(
            StructuredQueryAgentPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("search_global_memory",),
                final_answer_allowed=False,
                completed_tools=(),
                state_summary="completed=none",
            )
        )
    finally:
        reset_request_api_key_override(token)

    assert choice.tool_name == "search_global_memory"
    assert captured["headers"] == {
        "Authorization": "Bearer header-key",
        "Content-Type": "application/json",
    }


def test_deepseek_query_agent_transport_parses_final_answer_choice() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "final_answer": "The model improved accuracy by 12%.",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

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
    assert choice.rationale == ""


def test_deepseek_query_agent_transport_parses_tool_parameters_only_choice() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_global_memory",
                                    "tool_parameters": {"top_k": 3},
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("search_global_memory",),
            final_answer_allowed=False,
            state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
        )
    )

    assert choice.action_type == "tool_call"
    assert choice.tool_name == "search_global_memory"
    assert choice.tool_parameters == {"top_k": 3}
    assert choice.arguments is None
    assert choice.rationale == ""


def test_deepseek_query_agent_transport_parses_legacy_arguments_only_choice() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_global_memory",
                                    "arguments": {"top_k": 3},
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("search_global_memory",),
            final_answer_allowed=False,
            state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
        )
    )

    assert choice.action_type == "tool_call"
    assert choice.tool_name == "search_global_memory"
    assert choice.tool_parameters == {"top_k": 3}
    assert choice.arguments == {"top_k": 3}
    assert choice.rationale == ""


def test_deepseek_query_agent_transport_accepts_matching_tool_parameters_and_arguments() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_global_memory",
                                    "tool_parameters": {"top_k": 3},
                                    "arguments": {"top_k": 3},
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("search_global_memory",),
            final_answer_allowed=False,
            state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
        )
    )

    assert choice.action_type == "tool_call"
    assert choice.tool_name == "search_global_memory"
    assert choice.tool_parameters == {"top_k": 3}
    assert choice.arguments == {"top_k": 3}
    assert choice.rationale == ""


def test_deepseek_query_agent_transport_rejects_conflicting_tool_parameters_and_arguments() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "tool_name": "search_global_memory",
                                    "tool_parameters": {"top_k": 3},
                                    "arguments": {"top_k": 5},
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    with pytest.raises(DeepSeekQueryAgentResponseError) as error:
        transport.choose_next_action(
            StructuredQueryAgentPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("search_global_memory",),
                final_answer_allowed=False,
                state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
            )
        )

    assert "conflicting tool_parameters and arguments" in str(error.value)
    assert error.value.failure_detail.failure_stage_detail == "normalize_choice"


def test_deepseek_query_agent_transport_repairs_invalid_json_once() -> None:
    call_count = {"count": 0}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "not-json"},
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            )
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "final_answer": "The model improved accuracy by 12%.",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        )

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

    assert call_count["count"] == 2
    assert choice.action_type == "final_answer"
    assert choice.final_answer == "The model improved accuracy by 12%."


def test_deepseek_query_agent_transport_fails_after_repair_retry() -> None:
    call_count = {"count": 0}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        call_count["count"] += 1
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not-json"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    with pytest.raises(DeepSeekQueryAgentResponseError, match="could not be repaired"):
        transport.choose_next_action(
            StructuredQueryAgentPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("compose_answer",),
                final_answer_allowed=True,
                state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
            )
        )

    assert call_count["count"] == 2


def test_deepseek_query_agent_transport_surfaces_structured_failure_detail() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not-json"},
                    }
                ]
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    with pytest.raises(DeepSeekQueryAgentResponseError) as error:
        transport.choose_next_action(
            StructuredQueryAgentPrompt(
                query="Did it improve accuracy?",
                allowed_tools=("compose_answer",),
                final_answer_allowed=True,
                state_summary="completed=rerank_candidates; session_memories=1; global_memories=1",
            )
        )

    assert error.value.failure_detail.failure_stage_detail == "parse_message_content"
    assert error.value.failure_detail.repair_attempted is True
    assert error.value.failure_detail.content_preview == "not-json"
    assert error.value.failure_detail.raw_response_preview is not None


def test_deepseek_query_agent_transport_retries_empty_content_and_succeeds() -> None:
    call_count = {"count": 0}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        call_count["count"] += 1
        if call_count["count"] == 1:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "",
                                "reasoning_content": "",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                }
            )
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "final_answer": "The model improved accuracy by 12%.",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    choice = transport.choose_next_action(
        StructuredQueryAgentPrompt(
            query="Did it improve accuracy?",
            allowed_tools=("compose_answer",),
            final_answer_allowed=True,
            completed_tools=("list_session_papers",),
            state_summary="completed=list_session_papers; session_memories=0; global_memories=0",
            tool_descriptions={"compose_answer": "Return the final answer."},
            observations=(
                {
                    "kind": "session_papers",
                    "summary": "Current session has 2 imported papers/documents.",
                    "payload": {
                        "tool_name": "list_session_papers",
                        "papers": [{"paper_id": "paper-1", "title": "Paper 1"}],
                    },
                },
            ),
        )
    )

    assert call_count["count"] == 2
    assert choice.action_type == "final_answer"
    assert choice.final_answer == "The model improved accuracy by 12%."


def test_deepseek_query_agent_transport_uses_reasoning_content_when_content_is_empty() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "",
                            "reasoning_content": json.dumps(
                                {
                                    "final_answer": "The model improved accuracy by 12%.",
                                }
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

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


def test_deepseek_query_agent_transport_serializes_json_unsafe_observations() -> None:
    captured: dict[str, object] = {}

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        captured["body"] = json.loads(body.decode("utf-8"))
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "final_answer": "OK",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    when = datetime(2026, 4, 27, 15, 30, 0, tzinfo=timezone.utc)
    payload = {
        "datetime": when,
        "enum": _ObservationEnum.READY,
        "uuid": uuid4(),
        "path": Path("C:/tmp/example.pdf"),
        "model": _ObservationModel(value=7),
        "dataclass": _ObservationData(created_at=when, state=_ObservationEnum.READY),
        "nested": {
            "tuple": (when, _ObservationEnum.READY),
            "set": {_ObservationEnum.READY},
        },
    }

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
            observations=(AgentObservation(kind="debug", summary="debug payload", payload=payload),),
        )
    )

    assert choice.action_type == "final_answer"
    assert captured["body"]["messages"][1]["content"]
    user_prompt = json.loads(captured["body"]["messages"][1]["content"])
    observations = user_prompt["observations"]
    assert observations[0]["payload"]["datetime"] == when.isoformat()
    assert observations[0]["payload"]["enum"] == "ready"
    assert observations[0]["payload"]["path"] == str(Path("C:/tmp/example.pdf"))
    assert observations[0]["payload"]["model"] == {"value": 7}
    assert observations[0]["payload"]["dataclass"] == {"created_at": when.isoformat(), "state": "ready"}
    assert observations[0]["payload"]["nested"]["tuple"] == [when.isoformat(), "ready"]
    assert observations[0]["payload"]["nested"]["set"] == ["ready"]


def test_deepseek_query_agent_transport_generates_plain_text_final_answer() -> None:
    captured: list[dict[str, object]] = []

    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        request_body = json.loads(body.decode("utf-8"))
        captured.append(request_body)
        if len(captured) == 1:
            return _deepseek_response(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": "我们被问到 CRE_v2.pdf 的新想法是什么，因此先梳理这些记忆和证据，然后再回答",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 64, "total_tokens": 67},
                }
            )
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "CRE_v2 的新想法是位置敏感评估。",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    answer = transport.generate_final_answer(
        AgentTurnRequest(
            query="CRE_v2.pdf 的新想法是什么",
            allowed_actions=(),
            final_answer_allowed=True,
            state_summary="completed=list_session_papers,get_paper_memory_bundle",
            observations=(AgentObservation(kind="paper_memory_bundle", summary="bundle loaded"),),
        )
    )

    assert answer == "CRE_v2 的新想法是位置敏感评估。"
    assert len(captured) == 2
    assert captured[0]["max_tokens"] == 1536
    assert "response_format" not in captured[0]
    assert "return only the final user-facing answer" in captured[0]["messages"][0]["content"].lower()
    assert "do not output analysis" in captured[0]["messages"][0]["content"].lower()
    assert "state_summary" not in captured[0]["messages"][1]["content"]
    assert "completed_actions" not in captured[0]["messages"][1]["content"]
    assert "tool_descriptions" not in captured[0]["messages"][1]["content"]
    assert "evidence_view" in captured[0]["messages"][1]["content"]
    assert "concise complete final answer" in captured[1]["messages"][2]["content"].lower()


def test_deepseek_query_agent_transport_logs_finalization_metrics(caplog: pytest.LogCaptureFixture) -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "这是最终答案。",
                            "reasoning_content": "internal reasoning",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )

    with caplog.at_level(logging.DEBUG):
        answer = transport.generate_final_answer(
            AgentTurnRequest(
                query="CRE_v2 的新想法是什么？",
                allowed_actions=(),
                final_answer_allowed=True,
                state_summary="completed=list_session_papers,get_paper_memory_bundle",
                observations=(AgentObservation(kind="paper_memory_bundle", summary="bundle loaded"),),
            )
        )

    assert answer == "这是最终答案。"
    assert "DeepSeek query-finalization summary" in caplog.text
    assert "prompt_chars=" in caplog.text
    assert "output_length=" in caplog.text
    assert "finish_reason=stop" in caplog.text
    assert "reasoning_content_length=" in caplog.text


def test_model_backed_query_agent_client_accepts_valid_final_answer_choice() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="final_answer",
            final_answer="Agent final answer",
            rationale="model_has_enough_context_to_answer",
        ),
        fallback=None,
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


def test_model_backed_query_agent_client_accepts_final_answer_only_transport_choice() -> None:
    def fake_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> DeepSeekHttpResponse:
        return _deepseek_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "final_answer": "Agent final answer",
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

    transport = DeepSeekStructuredQueryAgentTransport(
        api_key="test-key",
        http_post=fake_http_post,
    )
    agent = ModelBackedQueryAgentClient(
        transport=transport,
        fallback=None,
    )

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(
            completed_tools=(),
            selected_memory_ids=(),
            should_reread_source=False,
        ),
        allowed_tools=(QueryToolName.COMPOSE_ANSWER,),
        final_answer_allowed=True,
    )

    assert decision is not None
    assert decision.action_type == "final_answer"
    assert decision.final_answer == "Agent final answer"
    assert decision.fallback_used is False
    assert agent.fallback_reason is None


def test_model_backed_query_agent_client_returns_none_and_records_reason_when_choice_is_invalid() -> None:
    agent = ModelBackedQueryAgentClient(
        transport=StaticStructuredQueryAgentTransport(
            action_type="tool_call",
            tool_name="compose_answer",
            rationale="invalid_tool_choice",
        ),
        fallback=None,
    )

    decision = agent.decide_next_action(
        query="Did it improve accuracy?",
        state=QueryAgentState(),
        allowed_tools=(QueryToolName.SEARCH_SESSION_MEMORY,),
        final_answer_allowed=False,
    )

    assert decision is None
    assert agent.fallback_used is False
    assert agent.fallback_reason is not None
    assert "Chosen tool 'compose_answer' is outside the allowed tool set." in agent.fallback_reason


def test_model_backed_query_agent_client_records_failure_detail_from_transport() -> None:
    class FailingTransport:
        def choose_next_action(self, prompt: StructuredQueryAgentPrompt):
            raise DeepSeekQueryAgentResponseError(
                "DeepSeek query-agent response contained empty content.",
                failure_detail=QueryAgentFailureDetail(
                    failure_stage_detail="empty_content",
                    status_code=200,
                    repair_attempted=True,
                    raw_response_preview="{\"choices\":[]}",
                    content_preview=None,
                ),
            )

        def generate_final_answer(self, request: AgentTurnRequest) -> str | None:
            return "unused"

    agent = ModelBackedQueryAgentClient(
        transport=FailingTransport(),
        fallback=None,
    )

    decision = agent.decide_turn(
        AgentTurnRequest(
            query="What happened?",
            allowed_actions=("compose_answer",),
            final_answer_allowed=True,
            state_summary="completed=none; session_memories=0; global_memories=0",
        )
    )

    assert decision is None
    assert agent.fallback_reason is not None
    assert agent.failure_detail == {
        "failure_stage_detail": "empty_content",
        "status_code": 200,
        "repair_attempted": True,
        "raw_response_preview": "{\"choices\":[]}",
        "content_preview": None,
    }
