"""Host-controlled planner contract for query-only tool calling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from research_agent.tools.protocol import ChunkDescriptor, MemoryDescriptor, QueryToolName


HOST_CONTROLLED_QUERY_TOOLS: tuple[QueryToolName, ...] = (
    QueryToolName.SEARCH_SESSION_MEMORY,
    QueryToolName.SEARCH_GLOBAL_MEMORY,
    QueryToolName.SEARCH_SOURCE_CHUNKS,
    QueryToolName.RERANK_CANDIDATES,
    QueryToolName.READ_SOURCE_PASSAGES,
    QueryToolName.COMPOSE_ANSWER,
    QueryToolName.SEARCH_OPENVIKING_MEMORY,
)
"""Small query-only subset exposed to the Phase 2 host-controlled planner."""


@dataclass(frozen=True, slots=True)
class QueryToolPlannerState:
    """Serializable snapshot used by the host planner."""

    completed_tools: tuple[QueryToolName, ...] = ()
    session_memories: tuple[MemoryDescriptor, ...] = ()
    global_memories: tuple[MemoryDescriptor, ...] = ()
    selected_memory_ids: tuple[str, ...] = ()
    should_reread_source: bool | None = None
    selected_chunks: tuple[ChunkDescriptor, ...] = ()

    def has_completed(self, tool_name: QueryToolName) -> bool:
        return tool_name in self.completed_tools


@dataclass(frozen=True, slots=True)
class QueryToolPlannerDecision:
    """Single next-tool decision produced by the planner."""

    tool_name: QueryToolName
    rationale: str
    planner_name: str = "heuristic"
    fallback_used: bool = False


class QueryToolPlannerClient(Protocol):
    """Planner that chooses the next tool within a host-provided allowed set."""

    def choose_next_tool(
        self,
        *,
        query: str,
        state: QueryToolPlannerState,
        allowed_tools: Sequence[QueryToolName],
    ) -> QueryToolPlannerDecision | None:
        """Return the next tool to call, or None when no further tool is needed."""


class HeuristicQueryToolPlannerClient:
    """Deterministic planner used before a real model-backed planner exists."""

    def choose_next_tool(
        self,
        *,
        query: str,
        state: QueryToolPlannerState,
        allowed_tools: Sequence[QueryToolName],
    ) -> QueryToolPlannerDecision | None:
        if not allowed_tools:
            return None

        allowed = tuple(tool for tool in allowed_tools if tool in HOST_CONTROLLED_QUERY_TOOLS)
        if not allowed:
            return None

        if (
            state.should_reread_source is False
            and QueryToolName.COMPOSE_ANSWER in allowed
            and QueryToolName.READ_SOURCE_PASSAGES in allowed
        ):
            return QueryToolPlannerDecision(
                tool_name=QueryToolName.COMPOSE_ANSWER,
                rationale=self._rationale_for(tool_name=QueryToolName.COMPOSE_ANSWER, state=state, query=query),
                planner_name="heuristic",
                fallback_used=False,
            )
        if (
            state.should_reread_source is True
            and QueryToolName.READ_SOURCE_PASSAGES in allowed
        ):
            return QueryToolPlannerDecision(
                tool_name=QueryToolName.READ_SOURCE_PASSAGES,
                rationale=self._rationale_for(tool_name=QueryToolName.READ_SOURCE_PASSAGES, state=state, query=query),
                planner_name="heuristic",
                fallback_used=False,
            )

        for tool_name in HOST_CONTROLLED_QUERY_TOOLS:
            if tool_name in allowed:
                return QueryToolPlannerDecision(
                    tool_name=tool_name,
                    rationale=self._rationale_for(tool_name=tool_name, state=state, query=query),
                    planner_name="heuristic",
                    fallback_used=False,
                )
        return None

    def _rationale_for(
        self,
        *,
        tool_name: QueryToolName,
        state: QueryToolPlannerState,
        query: str,
    ) -> str:
        if tool_name is QueryToolName.SEARCH_SESSION_MEMORY:
            return "memory_first_rule_requires_session_memory_before_any_other_context"
        if tool_name is QueryToolName.SEARCH_GLOBAL_MEMORY:
            return "session_memory_checked_expand_recall_with_global_memory_before_sufficiency_decision"
        if tool_name is QueryToolName.SEARCH_OPENVIKING_MEMORY:
            return "openviking_memory_is_an_explicit_retrieval_surface_before_final_answer_selection"
        if tool_name is QueryToolName.SEARCH_SOURCE_CHUNKS:
            return "source_chunk_search_can_surface_original_passages_before_reread_selection"
        if tool_name is QueryToolName.RERANK_CANDIDATES:
            if state.session_memories or state.global_memories:
                return "bounded_memory_candidates_available_rerank_before_reread_gating"
            return "no_memory_candidates_available_record_rerank_fallback_before_reread_gating"
        if tool_name is QueryToolName.READ_SOURCE_PASSAGES:
            return "memory_evidence_insufficient_reread_bound_source_passages_before_answer"
        if tool_name is QueryToolName.COMPOSE_ANSWER:
            if state.selected_chunks:
                return "memory_and_reread_chunks_are_ready_compose_final_answer"
            if state.selected_memory_ids:
                return "selected_memory_is_sufficient_compose_without_source_reread"
            return "no_supporting_memory_selected_compose_with_explicit_fallback_context"
        return f"heuristic_planner_selected_{tool_name.value}"


__all__ = [
    "HOST_CONTROLLED_QUERY_TOOLS",
    "HeuristicQueryToolPlannerClient",
    "QueryToolPlannerClient",
    "QueryToolPlannerDecision",
    "QueryToolPlannerState",
]
