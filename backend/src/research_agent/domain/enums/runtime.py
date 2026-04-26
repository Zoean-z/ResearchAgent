"""Runtime-related enums for task control and execution state."""

from __future__ import annotations

from enum import StrEnum


class TaskRunStatus(StrEnum):
    """Lifecycle states for a single task run."""

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    STEP_LIMIT_REACHED = "step_limit_reached"
