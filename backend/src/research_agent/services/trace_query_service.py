"""Thin read service for task-run trace data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from research_agent.domain.models import TraceNarrative, TraceStep
from research_agent.domain.ports import SessionRepositoryPort, TraceRepositoryPort
from research_agent.services.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class TraceQueryResult:
    """Grouped trace data for a single task run."""

    steps: tuple[TraceStep, ...]
    narratives: tuple[TraceNarrative, ...]


class TraceQueryService:
    """Read-side trace queries scoped to a session and task run."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        trace_repository: TraceRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._trace_repository = trace_repository

    def get_trace(self, session_id: str, run_id: str) -> TraceQueryResult:
        """Return raw steps and narratives for a run after verifying session ownership."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)

        task_run = self._trace_repository.get_run(run_id)
        if task_run is None or task_run.session_id != session_id:
            raise EntityNotFoundError("TaskRun", run_id)

        return TraceQueryResult(
            steps=tuple(self._trace_repository.list_steps(run_id)),
            narratives=tuple(self._trace_repository.list_narratives(run_id)),
        )
