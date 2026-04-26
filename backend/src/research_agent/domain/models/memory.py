"""Structured research memory models."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from research_agent.domain.enums import RelationType
from research_agent.domain.models.base import DomainModel, utc_now
from research_agent.domain.models.paper import SourceRef
from research_agent.domain.value_objects import ConfidenceScore


class PaperMemory(DomainModel):
    """Paper-level distilled memory used before rereading source passages."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    problem: str | None = None
    method: str | None = None
    key_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    novelty_claim: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=lambda: ConfidenceScore(value=0.5))
    updated_at: datetime = Field(default_factory=utc_now)


class RelationMemory(DomainModel):
    """Cross-paper relationship memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_paper: str
    target_paper: str
    relation_type: RelationType
    summary: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=lambda: ConfidenceScore(value=0.5))
    updated_at: datetime = Field(default_factory=utc_now)


class OpenQuestionMemory(DomainModel):
    """Persisted unresolved questions discovered during ingest or follow-up."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    unresolved_question: str = Field(min_length=1)
    related_papers: list[str] = Field(default_factory=list)
    why_open: list[str] = Field(default_factory=list)
    possible_followup: list[str] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=lambda: ConfidenceScore(value=0.5))
    updated_at: datetime = Field(default_factory=utc_now)
