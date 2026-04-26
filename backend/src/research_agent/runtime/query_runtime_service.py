"""Compatibility wrapper for the task runtime service."""

from __future__ import annotations

from research_agent.domain.ports import TraceRepositoryPort
from research_agent.services.ingest_execution_service import IngestExecutionResult
from research_agent.runtime.task_runtime_service import TaskRuntimeService


class _NoopIngestExecutionService:
    """Placeholder ingest executor kept for QueryRuntimeService compatibility."""

    def execute_ingest_run(self, session_id: str, run_id: str) -> IngestExecutionResult:  # pragma: no cover - safety net
        raise RuntimeError("QueryRuntimeService does not support ingest execution.")


class QueryRuntimeService(TaskRuntimeService):
    """Backward-compatible runtime wrapper for query execution."""

    def __init__(
        self,
        task_run_service,
        query_execution_service,
        trace_repository: TraceRepositoryPort,
        max_steps: int = 8,
    ) -> None:
        super().__init__(
            task_run_service=task_run_service,
            query_execution_service=query_execution_service,
            ingest_execution_service=_NoopIngestExecutionService(),
            trace_repository=trace_repository,
            max_steps=max_steps,
        )


__all__ = ["QueryRuntimeService"]
