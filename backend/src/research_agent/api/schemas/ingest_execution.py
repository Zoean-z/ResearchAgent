"""Ingest execution API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from research_agent.runtime.ingest_extraction import IngestPaperSummaryDraft
from research_agent.api.schemas.task_runs import TaskRunResponse


class PaperSummaryResponse(BaseModel):
    """Structured ingest summary for a paper."""

    model_config = ConfigDict(extra="forbid")

    what_it_is_about: str
    problem_solved: str
    new_ideas: list[str]
    limitations: list[str]
    suggestions_or_questions: list[str]
    evidence_candidate_ids: list[str]
    confidence: float

    @classmethod
    def from_domain(cls, summary: IngestPaperSummaryDraft) -> "PaperSummaryResponse":
        return cls(
            what_it_is_about=summary.what_it_is_about,
            problem_solved=summary.problem_solved,
            new_ideas=list(summary.new_ideas),
            limitations=list(summary.limitations),
            suggestions_or_questions=list(summary.suggestions_or_questions),
            evidence_candidate_ids=list(summary.evidence_candidate_ids),
            confidence=summary.confidence,
        )


class IngestExecutionResponse(BaseModel):
    """Serialized ingest execution result."""

    model_config = ConfigDict(extra="forbid")

    task_run: TaskRunResponse
    source_type: str
    paper_id: str
    artifact_id: str
    session_document_id: str
    chunk_count: int
    operation: str
    summary: str
    paper_summary: PaperSummaryResponse
