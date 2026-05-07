"""Runtime-facing protocol for ingest extraction decisions.

The ingest boundary keeps the candidate window broad enough to preserve recall,
then lets a model-backed adapter or a heuristic fallback decide how to distill
paper, relation, and open-question memories.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from research_agent.domain.models import Paper


@dataclass(frozen=True, slots=True)
class IngestExtractionCandidate:
    """A single evidence candidate presented to the ingest extractor."""

    candidate_id: str
    chunk_id: str | None
    page: int | None
    section: str | None
    cleaned_text: str
    excerpt: str
    relevance_reason: str
    source_chunk_ids: tuple[str, ...] = ()
    content_role: Literal["title", "abstract", "main", "appendix", "table", "reference", "unknown"] = "unknown"
    quality_flags: tuple[str, ...] = ()
    removed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IngestExtractionWindow:
    """A bounded input window for ingest extraction."""

    kind: Literal["broad", "expanded"]
    context_summary: str
    candidate_passages: tuple[IngestExtractionCandidate, ...]


@dataclass(frozen=True, slots=True)
class IngestExtractionRequest:
    """Structured ingest extraction request."""

    session_id: str
    paper: Paper
    related_papers: tuple[Paper, ...]
    window: IngestExtractionWindow
    extraction_stage: Literal["full_text", "batch", "merge"] = "full_text"
    batch_index: int | None = None
    batch_count: int | None = None
    batch_label: str | None = None
    batch_summaries: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class IngestEvidenceFieldDraft:
    """A single evidence-bound field returned by ingest extraction."""

    text: str | None
    evidence_chunk_ids: tuple[str, ...] = ()
    confidence: float = 0.5
    evidence_status: Literal["strong", "weak"] = "strong"


@dataclass(frozen=True, slots=True)
class IngestUnderstandingDraft:
    """Model-facing paper understanding extracted from cleaned evidence."""

    topic: IngestEvidenceFieldDraft | None
    problem: IngestEvidenceFieldDraft | None
    method: IngestEvidenceFieldDraft | None
    novelty_claims: tuple[IngestEvidenceFieldDraft, ...]
    key_results: tuple[IngestEvidenceFieldDraft, ...]
    experiment_design: IngestEvidenceFieldDraft | None
    limitations: tuple[IngestEvidenceFieldDraft, ...]
    open_questions: tuple[IngestEvidenceFieldDraft, ...]
    evidence_chunk_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestPaperMemoryDraft:
    """Structured draft for a paper memory."""

    problem: str | None
    method: str | None
    key_results: tuple[str, ...]
    limitations: tuple[str, ...]
    novelty_claim: str | None
    evidence_candidate_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestRelationMemoryDraft:
    """Structured draft for a relation memory."""

    relation_type: str
    summary: str
    evidence_candidate_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestOpenQuestionMemoryDraft:
    """Structured draft for an open-question memory."""

    unresolved_question: str
    why_open: tuple[str, ...]
    possible_followup: tuple[str, ...]
    evidence_candidate_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestPaperSummaryDraft:
    """Structured draft for a paper summary and closing suggestions."""

    what_it_is_about: str
    problem_solved: str
    new_ideas: tuple[str, ...]
    limitations: tuple[str, ...]
    suggestions_or_questions: tuple[str, ...]
    evidence_candidate_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class IngestExtractionDecision:
    """Structured ingest extraction result."""

    understanding: IngestUnderstandingDraft | None
    paper: IngestPaperMemoryDraft
    relation: IngestRelationMemoryDraft | None
    open_question: IngestOpenQuestionMemoryDraft
    paper_summary: IngestPaperSummaryDraft
    needs_more_context: bool
    context_hints: tuple[str, ...]
    rationale: str


class IngestExtractionClient(Protocol):
    """Protocol for a model-backed ingest extractor."""

    def extract(self, request: IngestExtractionRequest) -> IngestExtractionDecision:
        """Return a structured ingest extraction decision."""


__all__ = [
    "IngestExtractionCandidate",
    "IngestExtractionClient",
    "IngestExtractionDecision",
    "IngestExtractionRequest",
    "IngestExtractionWindow",
    "IngestEvidenceFieldDraft",
    "IngestOpenQuestionMemoryDraft",
    "IngestPaperMemoryDraft",
    "IngestPaperSummaryDraft",
    "IngestUnderstandingDraft",
    "IngestRelationMemoryDraft",
]
