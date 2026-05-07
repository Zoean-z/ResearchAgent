"""Trace and timeline persistence for query execution steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory, TimelineEvent, TraceStep
from research_agent.domain.ports import TimelineRepositoryPort, TraceRepositoryPort
from research_agent.runtime.streaming import RuntimeEventBroker
from research_agent.services.context_rerank_service import ChunkRerankResult, MemoryRerankResult
from research_agent.services.query_citation_builder import (
    RetrievedMemoryCitation,
    SourceRereadCitation,
    citations_for,
    memory_descriptor,
)
from research_agent.services.retrieval_service import RetrievalPlan
from research_agent.utils import to_json_safe


class QueryTraceWriter:
    """Writes trace steps and timeline events for query execution."""

    def __init__(
        self,
        trace_repository: TraceRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        runtime_event_broker: RuntimeEventBroker | None = None,
    ) -> None:
        self._trace_repository = trace_repository
        self._timeline_repository = timeline_repository
        self._runtime_event_broker = runtime_event_broker

    def save_tool_trace_step(
        self,
        *,
        session_id: str,
        run_id: str,
        action: str,
        input_payload: dict[str, object],
        result_payload: dict[str, object],
    ) -> None:
        trace_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action=action,
                input_payload=to_json_safe(input_payload),
                result_payload=to_json_safe(result_payload),
            )
        )
        self._publish_step_event(session_id, run_id, trace_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=f"{action} completed",
            )
        )

    def save_failure_trace_step(
        self,
        *,
        session_id: str,
        run_id: str,
        action: str,
        input_payload: dict[str, object],
        error_payload: dict[str, object],
    ) -> None:
        trace_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action=action,
                input_payload=to_json_safe(input_payload),
                result_payload=to_json_safe(error_payload),
                status="failed",
            )
        )
        self._publish_step_event(session_id, run_id, trace_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_failed",
                summary=f"{action} failed",
            )
        )

    def publish_step_event(self, session_id: str, run_id: str, trace_step: TraceStep) -> None:
        self._publish_step_event(session_id, run_id, trace_step)

    def write_trace_and_timeline(
        self,
        *,
        session_id: str,
        run_id: str,
        query: str,
        plan: RetrievalPlan,
        memory_selection: MemoryRerankResult,
        should_reread_source: bool,
        reread_reason: str,
        used_memory_citations: tuple[RetrievedMemoryCitation, ...],
        source_selection: ChunkRerankResult | None,
        source_reread_chunks: tuple[SourceRereadCitation, ...],
        planner_metadata: dict[str, dict[str, object]] | None = None,
        final_answer_text: str | None = None,
    ) -> None:
        session_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="retrieve_session_memories",
                input_payload=self._with_planner_payload(
                    {"session_id": session_id, "top_k": len(plan.session_memories.memories)},
                    planner_metadata,
                    "retrieve_session_memories",
                ),
                result_payload={
                    "memory_ids": [memory.id for memory in plan.session_memories.memories],
                    "memory_citations": [
                        asdict(citation)
                        for citation in citations_for(plan.session_memories.memories, query, selection_source="retrieval")
                    ],
                    "coverage_score": plan.session_memories.coverage_score,
                    "matched_query_terms": list(plan.session_memories.matched_query_terms),
                    "selection_reasons": list(plan.session_memories.selection_reasons),
                },
            )
        )
        self._publish_step_event(session_id, run_id, session_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary("checked session memory", plan.session_memories.memories, query),
                related_memory_ids=[memory.id for memory in plan.session_memories.memories],
            )
        )

        global_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="retrieve_global_memories",
                input_payload=self._with_planner_payload(
                    {"session_id": session_id, "related_paper_ids": list(plan.related_paper_ids)},
                    planner_metadata,
                    "retrieve_global_memories",
                ),
                result_payload={
                    "memory_ids": [memory.id for memory in plan.global_memories.memories],
                    "memory_citations": [
                        asdict(citation)
                        for citation in citations_for(plan.global_memories.memories, query, selection_source="retrieval")
                    ],
                    "coverage_score": plan.global_memories.coverage_score,
                    "matched_query_terms": list(plan.global_memories.matched_query_terms),
                    "selection_reasons": list(plan.global_memories.selection_reasons),
                },
            )
        )
        self._publish_step_event(session_id, run_id, global_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary("checked global memory", plan.global_memories.memories, query),
                related_memory_ids=[memory.id for memory in plan.global_memories.memories],
            )
        )

        rerank_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="rerank_context_candidates",
                input_payload=self._with_planner_payload(
                    {
                        "candidate_memory_ids": list(memory_selection.candidate_ids),
                        "top_k": len(memory_selection.selected_ids),
                    },
                    planner_metadata,
                    "rerank_context_candidates",
                ),
                result_payload={
                    "selected_memory_ids": list(memory_selection.selected_ids),
                    "selection_source": memory_selection.selection_source,
                    "fallback_used": memory_selection.fallback_used,
                    "rationale": memory_selection.rationale,
                    "memory_citations": [asdict(citation) for citation in used_memory_citations],
                },
            )
        )
        self._publish_step_event(session_id, run_id, rerank_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary=self._timeline_summary_for_memory_selection(memory_selection),
                related_memory_ids=list(memory_selection.selected_ids),
            )
        )

        decide_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="decide_reread_source",
                input_payload=self._with_planner_payload(
                    {
                        "memory_confidence": _max_confidence(memory_selection.selected),
                        "related_paper_ids": list(plan.related_paper_ids),
                    },
                    planner_metadata,
                    "decide_reread_source",
                ),
                result_payload={
                    "should_reread_source": should_reread_source,
                    "reason": reread_reason,
                    "memory_ids": [citation.memory_id for citation in used_memory_citations],
                    "memory_reasons": [citation.selection_reason for citation in used_memory_citations],
                },
            )
        )
        self._publish_step_event(session_id, run_id, decide_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="step_completed",
                summary="decided whether to reread",
                related_memory_ids=[citation.memory_id for citation in used_memory_citations],
            )
        )

        if source_selection is not None:
            reread_step = self._trace_repository.save_step(
                TraceStep(
                    run_id=run_id,
                    action="reread_source_passages",
                    input_payload=self._with_planner_payload(
                        {
                            "session_id": session_id,
                            "related_paper_ids": list(plan.related_paper_ids),
                            "candidate_chunk_ids": list(source_selection.candidate_ids),
                            "top_k": len(source_selection.selected_ids),
                        },
                        planner_metadata,
                        "reread_source_passages",
                    ),
                    result_payload={
                        "chunk_ids": [citation.chunk_id for citation in source_reread_chunks],
                        "paper_ids": sorted({citation.paper_id for citation in source_reread_chunks}),
                        "excerpts": [citation.excerpt for citation in source_reread_chunks],
                        "selection_source": source_selection.selection_source,
                        "fallback_used": source_selection.fallback_used,
                        "rationale": source_selection.rationale,
                        "selection_reasons": [citation.selection_reason for citation in source_reread_chunks],
                    },
                )
            )
            self._publish_step_event(session_id, run_id, reread_step)
            self._timeline_repository.save(
                TimelineEvent(
                    session_id=session_id,
                    run_id=run_id,
                    event_type="step_completed",
                    summary=self._timeline_summary_for_source_reread(source_reread_chunks, source_selection),
                    related_paper_ids=sorted({citation.paper_id for citation in source_reread_chunks}),
                )
            )

        answer_step = self._trace_repository.save_step(
            TraceStep(
                run_id=run_id,
                action="final_answer",
                input_payload=self._with_planner_payload(
                    {
                        "should_reread_source": should_reread_source,
                        "memory_selection_source": memory_selection.selection_source,
                        "memory_selection_fallback_used": memory_selection.fallback_used,
                        "session_memory_count": len(plan.session_memories.memories),
                        "global_memory_count": len(plan.global_memories.memories),
                        "matched_query_terms": list(
                            dict.fromkeys(
                                [
                                    *plan.session_memories.matched_query_terms,
                                    *plan.global_memories.matched_query_terms,
                                ]
                            )
                        ),
                        "source_reread_chunk_count": len(source_reread_chunks),
                    },
                    planner_metadata,
                    "final_answer",
                ),
                result_payload={
                    "answer_preview": final_answer_text or "",
                    "memory_citations": [asdict(citation) for citation in used_memory_citations],
                    "source_reread_chunks": [asdict(citation) for citation in source_reread_chunks],
                },
            )
        )
        self._publish_step_event(session_id, run_id, answer_step)
        self._timeline_repository.save(
            TimelineEvent(
                session_id=session_id,
                run_id=run_id,
                event_type="run_finished",
                summary="query run completed",
                related_memory_ids=[citation.memory_id for citation in used_memory_citations],
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_step_event(self, session_id: str, run_id: str, trace_step: TraceStep) -> None:
        if self._runtime_event_broker is None:
            return
        self._runtime_event_broker.publish_step_completed(session_id, run_id, trace_step)

    def _with_planner_payload(
        self,
        payload: dict[str, object],
        planner_metadata: dict[str, dict[str, object]] | None,
        action: str,
    ) -> dict[str, object]:
        if planner_metadata is None or action not in planner_metadata:
            return payload
        return {
            **payload,
            "planner_decision": planner_metadata[action],
        }

    def _timeline_summary(
        self,
        prefix: str,
        memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
        query: str,
    ) -> str:
        if not memories:
            return f"{prefix} (no memories)"
        cit = citations_for(memories, query)
        citations_text = ", ".join(f"{c.memory_type}:{c.memory_id}" for c in cit)
        return f"{prefix}: {citations_text}"

    def _timeline_summary_for_memory_selection(self, selection: MemoryRerankResult) -> str:
        deduped_selected: list[PaperMemory | RelationMemory | OpenQuestionMemory] = []
        seen: set[str] = set()
        for memory in selection.selected:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            deduped_selected.append(memory)
        if not deduped_selected:
            return "reranked context candidates (no memories)"
        selected_text = ", ".join(memory_descriptor(memory) for memory in deduped_selected)
        if selection.fallback_used:
            return f"reranked context candidates via fallback: {selected_text}"
        return f"reranked context candidates: {selected_text}"

    def _timeline_summary_for_source_reread(self, source_reread_chunks: Sequence[SourceRereadCitation], selection: ChunkRerankResult | None) -> str:
        if not source_reread_chunks:
            return "reread source passages (no chunks)"
        chunk_ids = ", ".join(citation.chunk_id for citation in source_reread_chunks)
        if selection is not None and selection.fallback_used:
            return f"reread source passages via fallback: {chunk_ids}"
        return f"reread source passages: {chunk_ids}"


def _max_confidence(
    memories: Sequence[PaperMemory | RelationMemory | OpenQuestionMemory],
) -> float:
    if not memories:
        return 0.0
    return max(memory.confidence.value for memory in memories)
