"""Data models and exceptions for query execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RetrievedMemoryCitation:
    """Human-readable pointer to a retrieved memory record."""

    memory_id: str
    memory_type: str
    summary: str
    selection_reason: str


@dataclass(frozen=True, slots=True)
class SourceRereadCitation:
    """Human-readable pointer to a reread source chunk."""

    chunk_id: str
    paper_id: str
    page: int | None
    section: str | None
    excerpt: str
    selection_reason: str


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    """Single host-controlled tool-planning decision captured during query execution."""

    turn_index: int
    action_type: str
    tool_name: str | None
    tool_parameters: dict[str, object]
    final_answer: str | None
    allowed_tools: tuple[str, ...]
    rationale: str
    agent_name: str
    fallback_used: bool
    validation_error: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QueryFailureDetail:
    """Structured query failure details for API and streaming surfaces."""

    error_code: str
    failed_stage: str
    error_message: str
    run_id: str
    tool_name: str | None = None
    fallback_reason: str | None = None
    validation_error: str | None = None
    failure_stage_detail: str | None = None
    status_code: int | None = None
    repair_attempted: bool | None = None
    raw_response_preview: str | None = None
    content_preview: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class QueryExecutionError(RuntimeError):
    """Query failure that preserves machine-readable context."""

    def __init__(self, detail: QueryFailureDetail) -> None:
        super().__init__(detail.error_message)
        self.detail = detail

    def to_dict(self) -> dict[str, object]:
        return self.detail.to_dict()
