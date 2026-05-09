"""Thin service for model-triggered arXiv imports through the existing ingest run flow."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.services.ingest_execution_service import IngestExecutionResult
from research_agent.services.message_intake_service import MessageIntakeService, SubmittedMessage
from research_agent.runtime.task_runtime_service import TaskRuntimeService
from research_agent.tools.arxiv_reference import normalize_arxiv_id_or_url


@dataclass(frozen=True, slots=True)
class ArxivImportToolResult:
    """Accepted arXiv ingest submission plus its completed ingest execution."""

    submitted: SubmittedMessage
    execution: IngestExecutionResult


class ArxivImportToolService:
    """Reuse the current arXiv ingest acceptance and runtime chain for tool calls."""

    def __init__(
        self,
        message_intake_service: MessageIntakeService,
        task_runtime_service: TaskRuntimeService,
    ) -> None:
        self._message_intake_service = message_intake_service
        self._task_runtime_service = task_runtime_service

    def import_arxiv_paper(self, *, session_id: str, arxiv_id_or_url: str) -> ArxivImportToolResult:
        """Accept and execute an arXiv ingest run for the current session."""

        canonical_abs_url = normalize_arxiv_id_or_url(arxiv_id_or_url)
        submitted = self._message_intake_service.submit_arxiv_ingest(
            session_id=session_id,
            arxiv_url=canonical_abs_url,
        )
        execution = self._task_runtime_service.execute_ingest_run(
            session_id=session_id,
            run_id=submitted.task_run.id,
        )
        return ArxivImportToolResult(submitted=submitted, execution=execution)


__all__ = [
    "ArxivImportToolResult",
    "ArxivImportToolService",
]
