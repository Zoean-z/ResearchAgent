"""Pure functions for building AgentObservation objects from query step results."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory
from research_agent.runtime.agent_protocol import AgentObservation
from research_agent.services.context_rerank_service import ChunkRerankResult, MemoryRerankResult
from research_agent.services.query_citation_builder import SourceRereadCitation, memory_descriptor
from research_agent.tools.protocol import QueryToolName
from research_agent.utils import to_json_safe


def turn_observation(
    *,
    kind: str,
    summary: str,
    payload: dict[str, object] | None = None,
) -> AgentObservation:
    return AgentObservation(kind=kind, summary=summary, payload=to_json_safe(payload) if payload is not None else None)


def memory_search_observation(
    *,
    tool_name: QueryToolName,
    scope: str,
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
    coverage_score: float,
    matched_query_terms: Sequence[str],
    selection_reasons: Sequence[str],
    decision_impact: str,
) -> AgentObservation:
    memory_ids = [memory.id for memory in memories]
    summary = f"{scope.title()} memory search returned {len(memory_ids)} memories and should influence the next turn."
    return turn_observation(
        kind="memory_search",
        summary=summary,
        payload={
            "tool_name": tool_name.value,
            "scope": scope,
            "memory_ids": memory_ids,
            "coverage_score": coverage_score,
            "matched_query_terms": list(matched_query_terms),
            "selection_reasons": list(selection_reasons),
            "decision_impact": decision_impact,
        },
    )


def openviking_search_observation(
    *,
    scope: str,
    result,
    decision_impact: str,
) -> AgentObservation:
    descriptors = tuple(result.memory_descriptors)
    memory_ids = [descriptor.memory_id for descriptor in descriptors]
    hit_ids = [hit.item_id for hit in result.hits]
    summary = (
        f"OpenViking {scope} search mapped {len(memory_ids)} memories from {len(hit_ids)} hits and can steer the next turn."
    )
    return turn_observation(
        kind="openviking_memory_search",
        summary=summary,
        payload={
            "tool_name": QueryToolName.SEARCH_OPENVIKING_MEMORY.value,
            "scope": scope,
            "hit_ids": hit_ids,
            "memory_ids": memory_ids,
            "matched_local_memory_ids": list(result.matched_local_memory_ids),
            "matched_local_count": result.matched_local_count,
            "coverage_score": min(1.0, len(memory_ids) / max(1, len(hit_ids))),
            "matched_query_terms": sorted(
                {term for descriptor in descriptors for term in descriptor.matched_terms}
            ),
            "selection_reasons": [descriptor.selection_reason for descriptor in descriptors],
            "decision_impact": decision_impact,
        },
    )


def memory_rerank_observation(
    *,
    memory_selection: MemoryRerankResult,
    should_reread_source: bool,
    reread_reason: str,
) -> AgentObservation:
    summary = (
        f"Reranked memory candidates to {len(memory_selection.selected_ids)} selected memories; "
        f"reread_required={should_reread_source}."
    )
    return turn_observation(
        kind="memory_rerank",
        summary=summary,
        payload={
            "tool_name": QueryToolName.RERANK_CANDIDATES.value,
            "candidate_ids": list(memory_selection.candidate_ids),
            "selected_ids": list(memory_selection.selected_ids),
            "selection_source": memory_selection.selection_source,
            "fallback_used": memory_selection.fallback_used,
            "rationale": memory_selection.rationale,
            "should_reread_source": should_reread_source,
            "reread_reason": reread_reason,
            "decision_impact": "The selected memories now gate whether source reread is required.",
        },
    )


def reread_decision_observation(
    *,
    should_reread_source: bool,
    reread_reason: str,
    selected_memory_ids: Sequence[str],
    related_paper_ids: Sequence[str],
) -> AgentObservation:
    summary = (
        "Source reread is required." if should_reread_source else "Source reread is not required."
    )
    return turn_observation(
        kind="reread_decision",
        summary=summary,
        payload={
            "should_reread_source": should_reread_source,
            "reread_reason": reread_reason,
            "selected_memory_ids": list(selected_memory_ids),
            "related_paper_ids": list(related_paper_ids),
            "decision_impact": (
                "Use stored chunks next if memory is insufficient."
                if should_reread_source
                else "The current memory set is sufficient for answer synthesis."
            ),
        },
    )


def source_reread_observation(
    *,
    source_selection: ChunkRerankResult | None,
    source_reread_chunks: Sequence[SourceRereadCitation],
) -> AgentObservation:
    selected_ids = [citation.chunk_id for citation in source_reread_chunks]
    summary = f"Source reread selected {len(selected_ids)} chunks and can support or override memory."
    payload = {
        "tool_name": QueryToolName.READ_SOURCE_PASSAGES.value,
        "chunk_ids": selected_ids,
        "selected_chunk_ids": list(source_selection.selected_ids) if source_selection is not None else [],
        "candidate_ids": list(source_selection.candidate_ids) if source_selection is not None else [],
        "selection_source": source_selection.selection_source if source_selection is not None else None,
        "fallback_used": source_selection.fallback_used if source_selection is not None else False,
        "rationale": source_selection.rationale if source_selection is not None else "",
        "decision_impact": "These passages are the strongest evidence if the model needs source support.",
    }
    if not selected_ids:
        summary = "Source reread produced no chunks."
    return turn_observation(
        kind="source_reread",
        summary=summary,
        payload=payload,
    )
