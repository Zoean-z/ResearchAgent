"""Runtime-facing query turn protocol.

This module owns the query-specific turn state and decision types so
orchestrators and model adapters can target runtime types instead of the tool
layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from research_agent.runtime.agent_protocol import AgentActionType, AgentObservation, AgentStopReason, AgentTurnDecision, AgentTurnRequest
from research_agent.tools.protocol import ChunkDescriptor, MemoryDescriptor, QueryToolName, get_query_tool_definition
from research_agent.tools.query_planner import QueryToolPlannerState


@dataclass(frozen=True, slots=True)
class QueryTurnState:
    """Serializable snapshot exposed to a bounded query turn orchestrator."""

    completed_tools: tuple[QueryToolName, ...] = ()
    session_memories: tuple[MemoryDescriptor, ...] = ()
    global_memories: tuple[MemoryDescriptor, ...] = ()
    selected_memory_ids: tuple[str, ...] = ()
    should_reread_source: bool | None = None
    selected_chunks: tuple[ChunkDescriptor, ...] = ()

    def has_completed(self, tool_name: QueryToolName) -> bool:
        return tool_name in self.completed_tools

    @classmethod
    def from_planner_state(cls, state: QueryToolPlannerState) -> "QueryTurnState":
        return cls(
            completed_tools=state.completed_tools,
            session_memories=state.session_memories,
            global_memories=state.global_memories,
            selected_memory_ids=state.selected_memory_ids,
            should_reread_source=state.should_reread_source,
            selected_chunks=state.selected_chunks,
        )

    def to_agent_turn_request(
        self,
        *,
        query: str,
        allowed_tools: Sequence[QueryToolName],
        final_answer_allowed: bool,
        observations: Sequence[AgentObservation] = (),
    ) -> AgentTurnRequest:
        return AgentTurnRequest(
            query=query,
            allowed_actions=tuple(tool.value for tool in allowed_tools),
            completed_actions=tuple(tool.value for tool in self.completed_tools),
            final_answer_allowed=final_answer_allowed,
            state_summary=self.state_summary(),
            tool_descriptions={
                tool.value: (
                    definition.description if (definition := get_query_tool_definition(tool)) is not None else tool.value
                )
                for tool in allowed_tools
            },
            observations=tuple(observations),
        )

    def state_summary(self) -> str:
        return (
            f"completed={','.join(tool.value for tool in self.completed_tools) or 'none'}; "
            f"session_memories={len(self.session_memories)}; "
            f"global_memories={len(self.global_memories)}; "
            f"selected_memory_ids={len(self.selected_memory_ids)}; "
            f"should_reread_source={self.should_reread_source}; "
            f"selected_chunks={len(self.selected_chunks)}"
        )


@dataclass(frozen=True, slots=True)
class QueryTurnDecision:
    """Single bounded next-step decision produced by a query orchestrator."""

    action_type: str
    rationale: str
    tool_name: QueryToolName | None = None
    final_answer: str | None = None
    agent_name: str = "heuristic_agent"
    fallback_used: bool = False
    fallback_reason: str | None = None

    def to_agent_turn_decision(self) -> AgentTurnDecision:
        return AgentTurnDecision(
            action_type=AgentActionType(self.action_type),
            tool_name=self.tool_name.value if self.tool_name is not None else None,
            final_answer=self.final_answer,
            rationale=self.rationale,
            stop_reason=AgentStopReason.FINAL_ANSWER_READY if self.action_type == "final_answer" else None,
        )

    @classmethod
    def from_agent_turn_decision(
        cls,
        decision: AgentTurnDecision,
        *,
        agent_name: str,
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> "QueryTurnDecision":
        return cls(
            action_type=decision.action_type.value,
            tool_name=QueryToolName(decision.tool_name) if decision.tool_name is not None else None,
            final_answer=decision.final_answer,
            rationale=decision.rationale,
            agent_name=agent_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )


class QueryTurnClient(Protocol):
    """Protocol for a component that chooses the next bounded query turn."""

    def decide_turn(self, request: AgentTurnRequest) -> AgentTurnDecision | None:
        """Return the next turn decision, or None when no action is available."""


__all__ = [
    "QueryTurnClient",
    "QueryTurnDecision",
    "QueryTurnState",
]
