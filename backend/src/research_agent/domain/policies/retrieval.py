"""Follow-up retrieval order and reread gating policies."""

from __future__ import annotations

from dataclasses import dataclass


SESSION_MEMORY_TOP_K = 5
GLOBAL_MEMORY_TOP_K = 5
CONTEXT_CANDIDATE_TOP_K = 10
CONTEXT_RERANK_TOP_K = 3
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class RetrievalStep:
    """A single retrieval stage in the follow-up path."""

    source: str
    top_k: int


def get_followup_retrieval_plan() -> tuple[RetrievalStep, ...]:
    """Return the required memory-first retrieval order."""

    return (
        RetrievalStep(source="session_memory", top_k=SESSION_MEMORY_TOP_K),
        RetrievalStep(source="global_memory", top_k=GLOBAL_MEMORY_TOP_K),
        RetrievalStep(source="source_reread", top_k=GLOBAL_MEMORY_TOP_K),
    )


def should_reread_source(
    *,
    has_relevant_paper_memory: bool,
    has_evidence_quote: bool,
    has_comparison_target: bool,
    memory_confidence: float,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Decide whether follow-up handling must reread source passages."""

    if not has_relevant_paper_memory:
        return True
    if not has_evidence_quote or not has_comparison_target:
        return True
    if memory_confidence < confidence_threshold:
        return True
    return False


def build_reread_reason(
    *,
    has_relevant_paper_memory: bool,
    has_evidence_quote: bool,
    has_comparison_target: bool,
    memory_confidence: float,
    reread_required: bool,
) -> str:
    """Build a concise reason string for the reread decision."""

    if not reread_required:
        return "memory_is_sufficient_for_mock_answer"
    reasons: list[str] = []
    if not has_relevant_paper_memory:
        reasons.append("missing_relevant_paper_memory")
    if not has_evidence_quote:
        reasons.append("missing_evidence_quote")
    if not has_comparison_target:
        reasons.append("missing_comparison_target")
    if memory_confidence < DEFAULT_CONFIDENCE_THRESHOLD:
        reasons.append("low_memory_confidence")
    return ",".join(reasons) if reasons else "mock_reread_required"
