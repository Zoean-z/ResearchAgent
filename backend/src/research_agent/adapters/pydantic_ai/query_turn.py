"""PydanticAI-backed query turn orchestration boundary.

This adapter keeps the framework at the orchestration edge only:
it converts a frozen agent-turn request into a structured turn decision,
while the host runtime still owns lifecycle, validation, and persistence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider

from research_agent.runtime.agent_protocol import AgentTurnDecision, AgentTurnRequest
from research_agent.runtime.query_turn import QueryTurnClient
from research_agent.tools.protocol import QueryToolName

logger = logging.getLogger(__name__)


def _normalize_model_name(model: str) -> str:
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


@dataclass
class PydanticAIQueryTurnClient:
    """Query-turn client powered by a PydanticAI Agent."""

    model: str = "deepseek-v4-flash"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    fallback: QueryTurnClient | None = None
    framework_name: str = "pydantic_ai"
    agent: Any | None = None

    def __post_init__(self) -> None:
        if self.agent is not None:
            self._agent = self.agent
            self._agent_name = f"{self.framework_name}:{_normalize_model_name(self.model)}"
            self._last_agent_name = self._agent_name
            self._last_fallback_used = False
            self._last_fallback_reason = None
            return

        if not self.api_key:
            self._agent = None
            self._agent_name = f"{self.framework_name}:{_normalize_model_name(self.model)}"
            self._last_agent_name = self._agent_name
            self._last_fallback_used = True
            self._last_fallback_reason = "pydantic_ai_agent_not_configured"
            return

        normalized_model = _normalize_model_name(self.model)
        self._agent_name = f"{self.framework_name}:{normalized_model}"
        self._last_agent_name = self._agent_name
        self._last_fallback_used = False
        self._last_fallback_reason = None
        provider = self._provider_for(normalized_model)
        self._agent = Agent(
            OpenAIChatModel(normalized_model, provider=provider),
            deps_type=AgentTurnRequest,
            output_type=AgentTurnDecision,
            name=self._agent_name,
        )

        @self._agent.instructions
        def _turn_instructions(ctx: RunContext[AgentTurnRequest]) -> str:
            request = ctx.deps
            allowed_actions = ", ".join(request.allowed_actions) or "none"
            completed_actions = ", ".join(request.completed_actions) or "none"
            tool_lines = "\n".join(
                f"- {name}: {description}"
                for name, description in request.tool_descriptions.items()
            )
            observation_lines = "\n".join(
                f"- {observation.kind}: {observation.summary}"
                + (
                    f" | impact={observation.payload.get('decision_impact')}"
                    if observation.payload and observation.payload.get("decision_impact")
                    else ""
                )
                for observation in request.observations
            )
            return "\n".join(
                [
                    "You are the query-turn orchestrator for a memory-first paper agent.",
                    "Return a single structured AgentTurnDecision.",
                    "Choose only from the allowed actions.",
                    "Default to final_answer when the user query can already be answered well without another tool call.",
                    "Use tool_call only when another bounded tool will materially improve the answer.",
                    "Use final_answer for greetings, acknowledgements, capability questions, and other low-context turns that do not need retrieval.",
                    "Default final_answer language is Chinese; use English only if the user explicitly asks for English.",
                    "Do not call memory or source tools just to be safe when the question is already answerable.",
                    "Use final_answer only when the host says final_answer_allowed is true.",
                    "Do not invent tools outside the allowed set.",
                    "Do not emit stop for this runtime slice.",
                    "Treat the observations as the latest execution evidence and let them influence the next turn.",
                    "If the observations are empty and the query is ordinary conversation rather than a research question, answer directly.",
                    f"Allowed actions: {allowed_actions}.",
                    f"Completed actions: {completed_actions}.",
                    f"Final answer allowed: {request.final_answer_allowed}.",
                    f"State summary: {request.state_summary}.",
                    f"Tool descriptions:\n{tool_lines or '- none'}",
                    f"Observations:\n{observation_lines or '- none'}",
                ]
            )

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        if not request.allowed_actions:
            return None

        try:
            if self._agent is None:
                raise RuntimeError("PydanticAI query-turn agent is not configured.")
            result = self._agent.run_sync(request.query, deps=request)
            decision = result.output
            self._validate_decision(decision, request.allowed_actions, request.final_answer_allowed)
            self._last_agent_name = self._agent_name
            self._last_fallback_used = False
            self._last_fallback_reason = None
            return decision
        except Exception as error:
            self._last_fallback_reason = self._format_fallback_reason(error)
            logger.warning(
                "PydanticAI query-turn agent fell back to the local planner: %s",
                self._last_fallback_reason,
            )
            if self.fallback is None:
                return None
            fallback_decision = self.fallback.decide_turn(request)
            if fallback_decision is None:
                return None
            self._last_agent_name = getattr(self.fallback, "agent_name", self._agent_name)
            self._last_fallback_used = True
            return fallback_decision

    @property
    def fallback_used(self) -> bool:
        return self._last_fallback_used

    @property
    def agent_name(self) -> str:
        return self._last_agent_name

    @property
    def fallback_reason(self) -> str | None:
        return self._last_fallback_reason

    def _provider_for(self, model: str):
        if self.base_url.rstrip("/") == "https://api.deepseek.com":
            return DeepSeekProvider(api_key=self.api_key)
        return OpenAIProvider(base_url=self.base_url, api_key=self.api_key)

    def _validate_decision(
        self,
        decision: AgentTurnDecision,
        allowed_actions: Sequence[str],
        final_answer_allowed: bool,
    ) -> None:
        if decision.action_type.value == "tool_call":
            if decision.tool_name not in allowed_actions:
                raise ValueError(f"Chosen tool '{decision.tool_name}' is outside the allowed turn set.")
            return
        if decision.action_type.value != "final_answer":
            raise ValueError(f"Unsupported agent action type: {decision.action_type.value}")
        if not final_answer_allowed:
            raise ValueError("final_answer is not allowed in this turn.")
        if not decision.final_answer:
            raise ValueError("final_answer is missing.")

    def _format_fallback_reason(self, error: Exception) -> str:
        message = str(error).replace("\n", " ").strip()
        if len(message) > 300:
            message = message[:297].rstrip() + "..."
        return f"{error.__class__.__name__}: {message}"


__all__ = ["PydanticAIQueryTurnClient"]
