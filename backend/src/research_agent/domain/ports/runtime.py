"""Trace, timeline, and task-run repository ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_agent.domain.models import TaskRun, TimelineEvent, TraceNarrative, TraceStep


@runtime_checkable
class TraceRepositoryPort(Protocol):
    """Storage boundary for task runs and raw/generated trace data."""

    def save_run(self, task_run: TaskRun) -> TaskRun:
        """Create or replace a task run."""

    def get_run(self, run_id: str) -> TaskRun | None:
        """Fetch a task run by id."""

    def list_runs_by_session(self, session_id: str) -> Sequence[TaskRun]:
        """List task runs scoped to a session."""

    def save_step(self, trace_step: TraceStep) -> TraceStep:
        """Create or replace a trace step."""

    def list_steps(self, run_id: str) -> Sequence[TraceStep]:
        """List raw trace steps for a run."""

    def save_narrative(self, trace_narrative: TraceNarrative) -> TraceNarrative:
        """Create or replace a trace narrative."""

    def list_narratives(self, run_id: str) -> Sequence[TraceNarrative]:
        """List trace narratives for a run."""

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its associated trace data."""

    def delete_runs_for_session(self, session_id: str) -> int:
        """Delete all runs and trace data scoped to a session."""


@runtime_checkable
class TimelineRepositoryPort(Protocol):
    """Storage boundary for session-facing timeline events."""

    def save(self, event: TimelineEvent) -> TimelineEvent:
        """Create or replace a timeline event."""

    def list_by_session(self, session_id: str) -> Sequence[TimelineEvent]:
        """List timeline events scoped to a session."""

    def delete_by_session(self, session_id: str) -> int:
        """Delete timeline events scoped to a session."""
