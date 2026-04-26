"""Runtime, trace, and timeline models."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import Field

from research_agent.domain.enums import TaskRunStatus
from research_agent.domain.models.base import DomainModel, utc_now


class TaskRun(DomainModel):
    """A bounded host-controlled execution for a single user message."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    message_id: str
    status: TaskRunStatus = Field(default=TaskRunStatus.PENDING)
    step_count: int = Field(default=0, ge=0)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    finish_reason: str | None = None


class TraceStep(DomainModel):
    """Raw execution record for a single runtime action."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    action: str = Field(min_length=1)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    result_payload: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="completed")
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class TraceNarrative(DomainModel):
    """Generated narrative layered on top of raw trace records."""

    trace_step_id: str
    reason_text: str = Field(min_length=1)
    impact_text: str = Field(min_length=1)


class TimelineEvent(DomainModel):
    """Session-facing timeline event used by the right-hand panel."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    run_id: str | None = None
    event_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    related_memory_ids: list[str] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
