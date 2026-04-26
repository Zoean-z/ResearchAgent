"""Pure memory upsert policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory, SourceRef, utc_now

MemoryRecordT = TypeVar("MemoryRecordT", PaperMemory, RelationMemory, OpenQuestionMemory)


@dataclass(frozen=True)
class MemoryUpsertDecision(Generic[MemoryRecordT]):
    """Result of applying an upsert policy to a memory record."""

    operation: str
    record: MemoryRecordT
    requires_conflict_followup: bool = False


def merge_paper_memory(existing: PaperMemory | None, incoming: PaperMemory) -> MemoryUpsertDecision[PaperMemory]:
    """Merge paper memories using the project upsert rules."""

    if existing is None:
        return MemoryUpsertDecision(operation="created", record=incoming)

    merged_refs = _dedupe_source_refs([*existing.source_refs, *incoming.source_refs])
    merged = existing.model_copy(
        update={
            "problem": _prefer_existing_text(existing.problem, incoming.problem, existing.confidence.value, incoming.confidence.value),
            "method": _prefer_existing_text(existing.method, incoming.method, existing.confidence.value, incoming.confidence.value),
            "novelty_claim": _prefer_existing_text(
                existing.novelty_claim,
                incoming.novelty_claim,
                existing.confidence.value,
                incoming.confidence.value,
            ),
            "key_results": _merge_unique(existing.key_results, incoming.key_results),
            "limitations": _merge_unique(existing.limitations, incoming.limitations),
            "source_refs": merged_refs,
            "confidence": incoming.confidence
            if incoming.confidence.value >= existing.confidence.value
            else existing.confidence,
            "updated_at": utc_now(),
        }
    )
    conflict = _has_conflicting_text(existing.problem, incoming.problem) or _has_conflicting_text(
        existing.method,
        incoming.method,
    )
    return MemoryUpsertDecision(operation="updated", record=merged, requires_conflict_followup=conflict)


def merge_relation_memory(
    existing: RelationMemory | None,
    incoming: RelationMemory,
) -> MemoryUpsertDecision[RelationMemory]:
    """Merge relation memories by appending evidence and taking the stronger confidence."""

    if existing is None:
        return MemoryUpsertDecision(operation="created", record=incoming)

    merged = existing.model_copy(
        update={
            "summary": incoming.summary if incoming.confidence.value >= existing.confidence.value else existing.summary,
            "evidence": _merge_unique(existing.evidence, incoming.evidence),
            "confidence": incoming.confidence
            if incoming.confidence.value >= existing.confidence.value
            else existing.confidence,
            "updated_at": utc_now(),
        }
    )
    return MemoryUpsertDecision(operation="updated", record=merged)


def merge_open_question_memory(
    existing: OpenQuestionMemory | None,
    incoming: OpenQuestionMemory,
) -> MemoryUpsertDecision[OpenQuestionMemory]:
    """Merge open questions by incrementally supplementing why-open and follow-ups."""

    if existing is None:
        return MemoryUpsertDecision(operation="created", record=incoming)

    merged = existing.model_copy(
        update={
            "related_papers": _merge_unique(existing.related_papers, incoming.related_papers),
            "why_open": _merge_unique(existing.why_open, incoming.why_open),
            "possible_followup": _merge_unique(existing.possible_followup, incoming.possible_followup),
            "confidence": incoming.confidence
            if incoming.confidence.value >= existing.confidence.value
            else existing.confidence,
            "updated_at": utc_now(),
        }
    )
    return MemoryUpsertDecision(operation="updated", record=merged)


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*existing, *incoming]:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged


def _dedupe_source_refs(source_refs: list[SourceRef]) -> list[SourceRef]:
    seen: set[tuple[str, str, int | None, str | None, str | None, str | None]] = set()
    deduped: list[SourceRef] = []
    for source_ref in source_refs:
        key = (
            source_ref.paper_id,
            source_ref.artifact_id,
            source_ref.page,
            source_ref.section,
            source_ref.chunk_id,
            source_ref.quote,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(source_ref)
    return deduped


def _prefer_existing_text(
    existing_value: str | None,
    incoming_value: str | None,
    existing_confidence: float,
    incoming_confidence: float,
) -> str | None:
    if not incoming_value:
        return existing_value
    if not existing_value:
        return incoming_value
    return incoming_value if incoming_confidence > existing_confidence else existing_value


def _has_conflicting_text(existing_value: str | None, incoming_value: str | None) -> bool:
    if not existing_value or not incoming_value:
        return False
    return existing_value.strip().lower() != incoming_value.strip().lower()
