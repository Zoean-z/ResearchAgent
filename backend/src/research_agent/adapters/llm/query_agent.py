"""Model-backed adapter boundary for bounded query agent decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal, Protocol
from urllib import request as urllib_request

import json

from pydantic import BaseModel, Field

from research_agent.runtime.agent_protocol import AgentTurnDecision, AgentTurnRequest
from research_agent.runtime.query_turn import QueryTurnClient, QueryTurnDecision, QueryTurnState
from research_agent.tools.protocol import QueryToolName, get_query_tool_definition


class StructuredQueryAgentPrompt(BaseModel):
    """Structured prompt payload sent to a query-agent model transport."""

    query: str = Field(description="Original follow-up query")
    allowed_tools: tuple[str, ...] = Field(description="Only these tools may be selected")
    final_answer_allowed: bool = Field(description="Whether the model may answer directly")
    completed_tools: tuple[str, ...] = Field(default_factory=tuple, description="Already executed tools in this run")
    state_summary: str = Field(description="Compact host-generated summary of current retrieval state")
    tool_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Short descriptions of each allowed tool",
    )


class StructuredQueryAgentChoice(BaseModel):
    """Structured agent choice returned by a model transport."""

    action_type: Literal["tool_call", "final_answer"] = Field(description="Next bounded action")
    tool_name: str | None = Field(default=None, description="Selected next tool name when action_type is tool_call")
    final_answer: str | None = Field(default=None, description="Final answer when action_type is final_answer")
    rationale: str = Field(description="Why this action should happen next")


class StructuredQueryAgentTransport(Protocol):
    """Transport that can obtain a structured next-action choice from a model."""

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        """Return the structured next action choice."""


HttpPost = Callable[[str, dict[str, str], bytes, float], bytes]


class UnavailableStructuredQueryAgentTransport:
    """Default transport placeholder until a provider-specific client is configured."""

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:  # pragma: no cover - defensive default
        raise RuntimeError("No structured query agent transport is configured.")


@dataclass(frozen=True, slots=True)
class StaticStructuredQueryAgentTransport:
    """Deterministic transport used by tests to simulate a model response."""

    action_type: Literal["tool_call", "final_answer"]
    tool_name: str | None = None
    final_answer: str | None = None
    rationale: str = "model_selected_next_action"

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        return StructuredQueryAgentChoice(
            action_type=self.action_type,
            tool_name=self.tool_name,
            final_answer=self.final_answer,
            rationale=self.rationale,
        )


def _default_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class DeepSeekStructuredQueryAgentTransport:
    """DeepSeek chat-completions transport for bounded query-agent decisions."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 30.0,
        http_post: HttpPost | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = self._normalize_model_name(model)
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_post = http_post or _default_http_post

    @property
    def normalized_model(self) -> str:
        return self._model

    def choose_next_action(self, prompt: StructuredQueryAgentPrompt) -> StructuredQueryAgentChoice:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        payload = {
            "model": self._model,
            "messages": self._messages_for(prompt),
            "response_format": {"type": "json_object"},
            "max_tokens": 384,
            "temperature": 0.0,
            "stream": False,
        }
        raw_response = self._http_post(
            f"{self._base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json.dumps(payload).encode("utf-8"),
            self._timeout_seconds,
        )
        return self._parse_response(raw_response)

    def _messages_for(self, prompt: StructuredQueryAgentPrompt) -> list[dict[str, str]]:
        system_prompt = (
            "You are a query agent for a memory-routed paper system. Respond in json only. "
            "You may either request one next tool call or return a final answer. "
            "Prefer final_answer whenever the query is already answerable without retrieval. "
            "Use tool_call only when another bounded tool will materially improve the answer. "
            "For greetings, acknowledgements, capability questions, or other low-context conversational turns, prefer final_answer immediately. "
            "Default final_answer language is Chinese; use English only if the user explicitly asks for English. "
            "Do not call retrieval tools just to be safe when there is no clear need. "
            "Never invent tools outside allowed_tools. "
            'Return JSON with keys "action_type", "tool_name" (for tool_call), '
            '"final_answer" (for final_answer), and "rationale".'
        )
        user_prompt = json.dumps(
            {
                "query": prompt.query,
                "allowed_tools": prompt.allowed_tools,
                "final_answer_allowed": prompt.final_answer_allowed,
                "completed_tools": prompt.completed_tools,
                "state_summary": prompt.state_summary,
                "tool_descriptions": prompt.tool_descriptions,
                "instruction": (
                    "Choose either tool_call or final_answer. If tool_call, tool_name must be one of allowed_tools. "
                    "If final_answer, provide final_answer text and do not invent extra tools. "
                    "Default final_answer language is Chinese unless the user explicitly asks for another language. "
                    "If the query can already be answered well without more evidence, choose final_answer. "
                    "If the turn is ordinary conversation and not a research retrieval request, choose final_answer. "
                    "Output valid json."
                ),
            },
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, raw_response: bytes) -> StructuredQueryAgentChoice:
        payload = json.loads(raw_response.decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek query-agent response contained no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("DeepSeek query-agent response contained empty content.")
        return StructuredQueryAgentChoice.model_validate(json.loads(content))

    def _normalize_model_name(self, model: str) -> str:
        normalized = model.strip()
        aliases = {
            "deepseekv4flash": "deepseek-v4-flash",
            "deepseek_v4_flash": "deepseek-v4-flash",
            "deepseek-v4flash": "deepseek-v4-flash",
            "deepseekv4pro": "deepseek-v4-pro",
            "deepseek_v4_pro": "deepseek-v4-pro",
            "deepseek-v4pro": "deepseek-v4-pro",
        }
        return aliases.get(normalized.lower(), normalized)


class ModelBackedQueryAgentClient:
    """Query agent that delegates next-action selection to a model adapter with fallback."""

    def __init__(
        self,
        *,
        transport: StructuredQueryAgentTransport,
        fallback: QueryTurnClient,
        agent_name: str = "model_adapter",
    ) -> None:
        self._transport = transport
        self._fallback = fallback
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
        if not allowed_tools:
            return None

        prompt = self._build_prompt(
            query=query,
            state=state,
            allowed_tools=allowed_tools,
            final_answer_allowed=final_answer_allowed,
        )
        try:
            choice = self._transport.choose_next_action(prompt)
            decision = self._decision_from_choice(choice, allowed_tools, final_answer_allowed)
            if decision is None:
                raise ValueError("Model query-agent choice could not be converted to a decision.")
            self._last_agent_name = decision.agent_name
            self._last_fallback_used = decision.fallback_used
            return decision
        except Exception as exc:
            fallback_decision = self._fallback.decide_next_action(
                query=query,
                state=state,
                allowed_tools=allowed_tools,
                final_answer_allowed=final_answer_allowed,
            )
            if fallback_decision is None:
                return None
            rationale = f"{fallback_decision.rationale}; fallback_after_{self._agent_name}_error={type(exc).__name__}"
            decision = QueryTurnDecision(
                action_type=fallback_decision.action_type,
                tool_name=fallback_decision.tool_name,
                final_answer=fallback_decision.final_answer,
                rationale=rationale,
                agent_name=self._agent_name,
                fallback_used=True,
            )
            self._last_agent_name = decision.agent_name
            self._last_fallback_used = decision.fallback_used
            return decision

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        if not request.allowed_actions:
            return None

        prompt = StructuredQueryAgentPrompt(
            query=request.query,
            allowed_tools=request.allowed_actions,
            final_answer_allowed=request.final_answer_allowed,
            completed_tools=request.completed_actions,
            state_summary=request.state_summary,
            tool_descriptions=request.tool_descriptions,
        )
        try:
            choice = self._transport.choose_next_action(prompt)
            decision = self._decision_from_choice(choice, tuple(QueryToolName(tool) for tool in request.allowed_actions), request.final_answer_allowed)
            if decision is None:
                raise ValueError("Model query-agent choice could not be converted to a decision.")
            self._last_agent_name = decision.agent_name
            self._last_fallback_used = decision.fallback_used
            return decision.to_agent_turn_decision()
        except Exception as exc:
            fallback_decision = self._fallback.decide_next_action(
                query=request.query,
                state=QueryTurnState(
                    completed_tools=tuple(QueryToolName(tool) for tool in request.completed_actions),
                ),
                allowed_tools=tuple(QueryToolName(tool) for tool in request.allowed_actions),
                final_answer_allowed=request.final_answer_allowed,
            )
            if fallback_decision is None:
                return None
            rationale = f"{fallback_decision.rationale}; fallback_after_{self._agent_name}_error={type(exc).__name__}"
            decision = QueryTurnDecision(
                action_type=fallback_decision.action_type,
                tool_name=fallback_decision.tool_name,
                final_answer=fallback_decision.final_answer,
                rationale=rationale,
                agent_name=self._agent_name,
                fallback_used=True,
            )
            self._last_agent_name = decision.agent_name
            self._last_fallback_used = decision.fallback_used
            return decision.to_agent_turn_decision()

    @property
    def fallback_used(self) -> bool:
        return getattr(self, "_last_fallback_used", False)

    def _decision_from_choice(
        self,
        choice: StructuredQueryAgentChoice,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> QueryTurnDecision:
        if choice.action_type == "tool_call":
            if not choice.tool_name:
                raise ValueError("Model query-agent tool_call choice is missing tool_name.")
            chosen_tool = QueryToolName(choice.tool_name)
            if chosen_tool not in allowed_tools:
                raise ValueError(f"Chosen tool '{chosen_tool.value}' is outside the allowed tool set.")
            return QueryTurnDecision(
                action_type="tool_call",
                tool_name=chosen_tool,
                rationale=choice.rationale,
                agent_name=self._agent_name,
                fallback_used=False,
            )
        if not final_answer_allowed:
            raise ValueError("Model query-agent final_answer choice is not allowed at this step.")
        if not choice.final_answer:
            raise ValueError("Model query-agent final_answer choice is missing final_answer text.")
        return QueryTurnDecision(
            action_type="final_answer",
            final_answer=choice.final_answer,
            rationale=choice.rationale,
            agent_name=self._agent_name,
            fallback_used=False,
        )

    def _build_prompt(
        self,
        *,
        query: str,
        state: QueryTurnState,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
    ) -> StructuredQueryAgentPrompt:
        tool_descriptions = {
            tool.value: (definition.description if (definition := get_query_tool_definition(tool)) is not None else tool.value)
            for tool in allowed_tools
        }
        return StructuredQueryAgentPrompt(
            query=query,
            allowed_tools=tuple(tool.value for tool in allowed_tools),
            final_answer_allowed=final_answer_allowed,
            completed_tools=tuple(tool.value for tool in state.completed_tools),
            state_summary=self._state_summary(state),
            tool_descriptions=tool_descriptions,
        )

    def _state_summary(self, state: QueryTurnState) -> str:
        return (
            f"completed={','.join(tool.value for tool in state.completed_tools) or 'none'}; "
            f"session_memories={len(state.session_memories)}; "
            f"global_memories={len(state.global_memories)}; "
            f"selected_memory_ids={len(state.selected_memory_ids)}; "
            f"should_reread_source={state.should_reread_source}; "
            f"selected_chunks={len(state.selected_chunks)}"
        )


__all__ = [
    "DeepSeekStructuredQueryAgentTransport",
    "ModelBackedQueryAgentClient",
    "StaticStructuredQueryAgentTransport",
    "StructuredQueryAgentChoice",
    "StructuredQueryAgentPrompt",
    "StructuredQueryAgentTransport",
    "UnavailableStructuredQueryAgentTransport",
]
