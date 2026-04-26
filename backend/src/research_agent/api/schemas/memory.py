"""Memory snapshot API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory


class PaperMemoryResponse(BaseModel):
    """Serialized paper memory for session snapshots."""

    model_config = ConfigDict(extra="forbid")

    id: str
    paper_id: str
    problem: str | None
    method: str | None
    key_results: list[str]
    limitations: list[str]
    novelty_claim: str | None
    confidence: float
    updated_at: datetime

    @classmethod
    def from_domain(cls, memory: PaperMemory) -> "PaperMemoryResponse":
        return cls(
            id=memory.id,
            paper_id=memory.paper_id,
            problem=memory.problem,
            method=memory.method,
            key_results=memory.key_results,
            limitations=memory.limitations,
            novelty_claim=memory.novelty_claim,
            confidence=memory.confidence.value,
            updated_at=memory.updated_at,
        )


class RelationMemoryResponse(BaseModel):
    """Serialized relation memory for session snapshots."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_paper: str
    target_paper: str
    relation_type: str
    summary: str
    evidence: list[str]
    confidence: float
    updated_at: datetime

    @classmethod
    def from_domain(cls, memory: RelationMemory) -> "RelationMemoryResponse":
        return cls(
            id=memory.id,
            source_paper=memory.source_paper,
            target_paper=memory.target_paper,
            relation_type=memory.relation_type.value,
            summary=memory.summary,
            evidence=memory.evidence,
            confidence=memory.confidence.value,
            updated_at=memory.updated_at,
        )


class OpenQuestionMemoryResponse(BaseModel):
    """Serialized open-question memory for session snapshots."""

    model_config = ConfigDict(extra="forbid")

    id: str
    unresolved_question: str
    related_papers: list[str]
    why_open: list[str]
    possible_followup: list[str]
    confidence: float
    updated_at: datetime

    @classmethod
    def from_domain(cls, memory: OpenQuestionMemory) -> "OpenQuestionMemoryResponse":
        return cls(
            id=memory.id,
            unresolved_question=memory.unresolved_question,
            related_papers=memory.related_papers,
            why_open=memory.why_open,
            possible_followup=memory.possible_followup,
            confidence=memory.confidence.value,
            updated_at=memory.updated_at,
        )


class MemorySnapshotResponse(BaseModel):
    """Grouped memory snapshot for a session."""

    model_config = ConfigDict(extra="forbid")

    paper_memories: list[PaperMemoryResponse]
    relation_memories: list[RelationMemoryResponse]
    open_question_memories: list[OpenQuestionMemoryResponse]
