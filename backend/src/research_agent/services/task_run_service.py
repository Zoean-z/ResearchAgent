"""Thin application service for request acceptance and task-run lookup."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.domain.enums import MessageType
from research_agent.domain.enums import TaskRunStatus
from research_agent.domain.models import Message, TaskRun, utc_now
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort, TraceRepositoryPort
from research_agent.services.errors import EntityNotFoundError, InvalidTaskRunStateError


@dataclass(frozen=True, slots=True)
class AcceptedTaskRun:
    """Accepted request result returned by the thin task-run service."""

    message: Message
    task_run: TaskRun


class TaskRunService:
    """Accept ingest/query requests and expose minimal task-run queries."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        message_repository: MessageRepositoryPort,
        trace_repository: TraceRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._message_repository = message_repository
        self._trace_repository = trace_repository

    def accept_arxiv_ingest(self, session_id: str, arxiv_url: str) -> AcceptedTaskRun:
        """Create the message and task-run records for an arXiv ingest request."""

        return self._accept(session_id=session_id, content=arxiv_url, message_type=MessageType.INGEST_ARXIV)

    def accept_pdf_ingest(self, session_id: str, file_path: str) -> AcceptedTaskRun:
        """Create the message and task-run records for a local PDF ingest request."""

        return self._accept(session_id=session_id, content=file_path, message_type=MessageType.INGEST_PDF)

    def accept_followup_query(self, session_id: str, query: str) -> AcceptedTaskRun:
        """Create the message and task-run records for a follow-up query request."""

        return self._accept(session_id=session_id, content=query, message_type=MessageType.FOLLOWUP_QUERY)

    def get_run(self, session_id: str, run_id: str) -> TaskRun:
        """Fetch a task run scoped to the requested session."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)

        task_run = self._trace_repository.get_run(run_id)
        if task_run is None or task_run.session_id != session_id:
            raise EntityNotFoundError("TaskRun", run_id)
        return task_run

    def get_message_for_run(self, session_id: str, run_id: str) -> Message:
        """Fetch the user message associated with a task run."""

        task_run = self.get_run(session_id, run_id)
        message = self._message_repository.get_by_id(task_run.message_id)
        if message is None:
            raise EntityNotFoundError("Message", task_run.message_id)
        return message

    def list_runs(self, session_id: str) -> list[TaskRun]:
        """List task runs scoped to a session."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)
        return list(self._trace_repository.list_runs_by_session(session_id))

    def mark_running(self, session_id: str, run_id: str) -> TaskRun:
        """Transition a task run from pending to running."""

        task_run = self.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.PENDING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.PENDING.value, task_run.status.value)

        updated = task_run.model_copy(update={"status": TaskRunStatus.RUNNING})
        return self._trace_repository.save_run(updated)

    def update_step_count(self, session_id: str, run_id: str, step_count: int) -> TaskRun:
        """Persist the current host-controlled step count."""

        task_run = self.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        updated = task_run.model_copy(update={"step_count": step_count})
        return self._trace_repository.save_run(updated)

    def mark_step_limit_reached(self, session_id: str, run_id: str, step_count: int) -> TaskRun:
        """Transition a task run to step_limit_reached."""

        task_run = self.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        updated = task_run.model_copy(
            update={
                "status": TaskRunStatus.STEP_LIMIT_REACHED,
                "step_count": step_count,
                "finished_at": utc_now(),
                "finish_reason": TaskRunStatus.STEP_LIMIT_REACHED.value,
            }
        )
        return self._trace_repository.save_run(updated)

    def finish_run(self, session_id: str, run_id: str, finish_reason: str) -> TaskRun:
        """Transition a task run to finished."""

        task_run = self.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        updated = task_run.model_copy(
            update={
                "status": TaskRunStatus.FINISHED,
                "finish_reason": finish_reason,
                "finished_at": utc_now(),
            }
        )
        return self._trace_repository.save_run(updated)

    def fail_run(self, session_id: str, run_id: str, finish_reason: str) -> TaskRun:
        """Transition a task run to failed."""

        task_run = self.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.RUNNING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.RUNNING.value, task_run.status.value)

        updated = task_run.model_copy(
            update={
                "status": TaskRunStatus.FAILED,
                "finish_reason": finish_reason,
                "finished_at": utc_now(),
            }
        )
        return self._trace_repository.save_run(updated)

    def _accept(self, *, session_id: str, content: str, message_type: MessageType) -> AcceptedTaskRun:
        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)

        message = self._message_repository.save(
            Message(
                session_id=session_id,
                role="user",
                type=message_type,
                content=content,
                status="accepted",
            )
        )
        task_run = self._trace_repository.save_run(
            TaskRun(
                session_id=session_id,
                message_id=message.id,
            )
        )
        return AcceptedTaskRun(message=message, task_run=task_run)
