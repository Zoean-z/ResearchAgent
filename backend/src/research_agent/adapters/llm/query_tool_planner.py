"""Model-backed adapter boundary for bounded query tool planning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any, Callable
from typing import Protocol
from urllib import request as urllib_request

from pydantic import BaseModel, Field

from research_agent.tools.protocol import QueryToolName, get_query_tool_definition
from research_agent.tools.query_planner import QueryToolPlannerClient, QueryToolPlannerDecision, QueryToolPlannerState


class StructuredPlannerPrompt(BaseModel):
    """Structured prompt payload sent to a planner model transport."""

    query: str = Field(description="Original follow-up query")
    allowed_tools: tuple[str, ...] = Field(description="Only these tools may be selected")
    completed_tools: tuple[str, ...] = Field(default_factory=tuple, description="Already executed tools in this run")
    state_summary: str = Field(description="Compact host-generated summary of current retrieval state")
    tool_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Short descriptions of each allowed tool",
    )


class StructuredPlannerChoice(BaseModel):
    """Structured planner choice returned by a model transport."""

    tool_name: str = Field(description="Selected next tool name")
    rationale: str = Field(description="Why this tool should be called next")


class StructuredPlannerTransport(Protocol):
    """Transport that can obtain a structured next-tool choice from a model."""

    def choose_next_tool(self, prompt: StructuredPlannerPrompt) -> StructuredPlannerChoice:
        """Return the structured planner choice."""


HttpPost = Callable[[str, dict[str, str], bytes, float], bytes]


class UnavailableStructuredPlannerTransport:
    """Default transport placeholder until a provider-specific client is configured."""

    def choose_next_tool(self, prompt: StructuredPlannerPrompt) -> StructuredPlannerChoice:  # pragma: no cover - defensive default
        raise RuntimeError("No structured planner transport is configured.")


@dataclass(frozen=True, slots=True)
class StaticStructuredPlannerTransport:
    """Deterministic transport used by tests to simulate a model response."""

    tool_name: str
    rationale: str = "model_selected_next_tool"

    def choose_next_tool(self, prompt: StructuredPlannerPrompt) -> StructuredPlannerChoice:
        return StructuredPlannerChoice(tool_name=self.tool_name, rationale=self.rationale)


def _default_http_post(url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> bytes:
    request = urllib_request.Request(url=url, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class DeepSeekStructuredPlannerTransport:
    """DeepSeek chat-completions transport for bounded next-tool planning.

    Uses DeepSeek JSON output mode so the model returns a small structured object:
    `{"tool_name": "...", "rationale": "..."}`.
    """

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

    def choose_next_tool(self, prompt: StructuredPlannerPrompt) -> StructuredPlannerChoice:
        if not self._api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        payload = {
            "model": self._model,
            "messages": self._messages_for(prompt),
            "response_format": {"type": "json_object"},
            "max_tokens": 256,
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

    def _messages_for(self, prompt: StructuredPlannerPrompt) -> list[dict[str, str]]:
        system_prompt = (
            "You are a query tool planner. Respond in json only. "
            "Select exactly one next tool from the allowed_tools list. "
            "Never invent a tool outside allowed_tools. "
            'Return JSON with keys "tool_name" and "rationale".'
        )
        user_prompt = json.dumps(
            {
                "query": prompt.query,
                "allowed_tools": prompt.allowed_tools,
                "completed_tools": prompt.completed_tools,
                "state_summary": prompt.state_summary,
                "tool_descriptions": prompt.tool_descriptions,
                "instruction": (
                    "Choose the single best next tool for this bounded host-controlled query run. "
                    "Output valid json."
                ),
            },
            ensure_ascii=True,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, raw_response: bytes) -> StructuredPlannerChoice:
        payload = json.loads(raw_response.decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek planner response contained no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("DeepSeek planner response contained empty content.")
        return StructuredPlannerChoice.model_validate(json.loads(content))

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


class ModelBackedQueryToolPlannerClient:
    """Query planner that delegates next-tool selection to a model adapter with fallback."""

    def __init__(
        self,
        *,
        transport: StructuredPlannerTransport,
        fallback: QueryToolPlannerClient,
        planner_name: str = "model_adapter",
    ) -> None:
        self._transport = transport
        self._fallback = fallback
        self._planner_name = planner_name

    def choose_next_tool(
        self,
        *,
        query: str,
        state: QueryToolPlannerState,
        allowed_tools: Sequence[QueryToolName],
    ) -> QueryToolPlannerDecision | None:
        if not allowed_tools:
            return None

        prompt = self._build_prompt(query=query, state=state, allowed_tools=allowed_tools)
        try:
            choice = self._transport.choose_next_tool(prompt)
            chosen_tool = QueryToolName(choice.tool_name)
            if chosen_tool not in allowed_tools:
                raise ValueError(f"Chosen tool '{chosen_tool.value}' is outside the allowed tool set.")
            return QueryToolPlannerDecision(
                tool_name=chosen_tool,
                rationale=choice.rationale,
                planner_name=self._planner_name,
                fallback_used=False,
            )
        except Exception as exc:
            fallback_decision = self._fallback.choose_next_tool(
                query=query,
                state=state,
                allowed_tools=allowed_tools,
            )
            if fallback_decision is None:
                return None
            fallback_rationale = (
                f"{fallback_decision.rationale}; "
                f"fallback_after_{self._planner_name}_error={type(exc).__name__}"
            )
            return QueryToolPlannerDecision(
                tool_name=fallback_decision.tool_name,
                rationale=fallback_rationale,
                planner_name=self._planner_name,
                fallback_used=True,
            )

    def _build_prompt(
        self,
        *,
        query: str,
        state: QueryToolPlannerState,
        allowed_tools: Sequence[QueryToolName],
    ) -> StructuredPlannerPrompt:
        tool_descriptions = {
            tool.value: (definition.description if (definition := get_query_tool_definition(tool)) is not None else tool.value)
            for tool in allowed_tools
        }
        return StructuredPlannerPrompt(
            query=query,
            allowed_tools=tuple(tool.value for tool in allowed_tools),
            completed_tools=tuple(tool.value for tool in state.completed_tools),
            state_summary=self._state_summary(state),
            tool_descriptions=tool_descriptions,
        )

    def _state_summary(self, state: QueryToolPlannerState) -> str:
        return (
            f"completed={','.join(tool.value for tool in state.completed_tools) or 'none'}; "
            f"session_memories={len(state.session_memories)}; "
            f"global_memories={len(state.global_memories)}; "
            f"selected_memory_ids={len(state.selected_memory_ids)}; "
            f"should_reread_source={state.should_reread_source}; "
            f"selected_chunks={len(state.selected_chunks)}"
        )


__all__ = [
    "ModelBackedQueryToolPlannerClient",
    "StaticStructuredPlannerTransport",
    "StructuredPlannerChoice",
    "StructuredPlannerPrompt",
    "StructuredPlannerTransport",
    "UnavailableStructuredPlannerTransport",
]
