"""Framework-agnostic agent turn protocol for constrained orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class AgentActionType(str, Enum):
    """High-level action categories an agent may emit."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    STOP = "stop"


class AgentStopReason(str, Enum):
    """Host-visible reasons an agent turn may stop."""

    FINAL_ANSWER_READY = "final_answer_ready"
    STEP_LIMIT_REACHED = "step_limit_reached"
    NO_MORE_ACTIONS = "no_more_actions"
    INVALID_OUTPUT = "invalid_output"
    HOST_REJECTED = "host_rejected"


class AgentObservation(BaseModel):
    """Generic observation visible to an agent between turns."""

    kind: str = Field(description="Stable observation kind")
    summary: str = Field(description="Host-rendered summary of the observation")
    payload: dict[str, Any] | None = Field(default=None, description="Structured observation payload")


class AgentTurnRequest(BaseModel):
    """Framework-agnostic request for a single agent turn."""

    query: str = Field(description="Original user query")
    allowed_actions: tuple[str, ...] = Field(description="Actions the host currently allows")
    completed_actions: tuple[str, ...] = Field(default_factory=tuple, description="Actions already completed in this run")
    final_answer_allowed: bool = Field(default=False, description="Whether the agent may answer directly")
    state_summary: str = Field(description="Compact host-generated state summary")
    tool_descriptions: dict[str, str] = Field(default_factory=dict, description="Human-readable descriptions for allowed actions")
    observations: tuple[AgentObservation, ...] = Field(default_factory=tuple, description="Host observations visible to the agent")
    recent_conversation_context: dict[str, Any] | None = Field(
        default=None,
        description="Compact host-injected recent conversation context for follow-up disambiguation",
    )


class AgentTurnDecision(BaseModel):
    """Framework-agnostic next-turn decision emitted by an agent."""

    action_type: AgentActionType = Field(description="The next action kind")
    tool_name: str | None = Field(default=None, description="Tool selected when action_type is tool_call")
    tool_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Business parameters for a tool call. Runtime context such as session_id must not be supplied by the model.",
    )
    final_answer: str | None = Field(default=None, description="Final answer when action_type is final_answer")
    rationale: str = Field(description="Why this action should happen next")
    stop_reason: AgentStopReason | None = Field(default=None, description="Why the turn stopped, if applicable")


class AgentTurnClient(Protocol):
    """Protocol for a component that chooses the next bounded agent turn."""

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        """Return the next turn decision, or None when no action is available."""


__all__ = [
    "AgentActionType",
    "AgentObservation",
    "AgentStopReason",
    "AgentTurnClient",
    "AgentTurnDecision",
    "AgentTurnRequest",
]
