"""Task-run API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from research_agent.domain.models import TaskRun


class TaskRunResponse(BaseModel):
    """Serialized task-run record returned by the API."""

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    message_id: str
    status: str
    step_count: int
    started_at: datetime
    finished_at: datetime | None
    finish_reason: str | None

    @classmethod
    def from_domain(cls, task_run: TaskRun) -> "TaskRunResponse":
        return cls(
            id=task_run.id,
            session_id=task_run.session_id,
            message_id=task_run.message_id,
            status=task_run.status.value,
            step_count=task_run.step_count,
            started_at=task_run.started_at,
            finished_at=task_run.finished_at,
            finish_reason=task_run.finish_reason,
        )


class TaskRunListResponse(BaseModel):
    """List response for session-scoped task runs."""

    model_config = ConfigDict(extra="forbid")

    items: list[TaskRunResponse]
