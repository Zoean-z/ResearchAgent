"""Live task-run startup and stream subscription helpers."""

from __future__ import annotations

from threading import Thread

from research_agent.domain.enums import MessageType, TaskRunStatus
from research_agent.runtime.streaming import RuntimeEventBroker, RuntimeEventSubscription
from research_agent.runtime.task_runtime_service import TaskRuntimeService
from research_agent.services.errors import InvalidTaskRunStateError
from research_agent.services.task_run_service import TaskRunService


class TaskRunStreamingService:
    """Start query or ingest runs in the background and expose event subscriptions."""

    def __init__(
        self,
        task_run_service: TaskRunService,
        task_runtime_service: TaskRuntimeService,
        runtime_event_broker: RuntimeEventBroker,
    ) -> None:
        self._task_run_service = task_run_service
        self._task_runtime_service = task_runtime_service
        self._runtime_event_broker = runtime_event_broker

    def start_query_run(self, session_id: str, run_id: str):
        """Transition a pending query run to running and execute it in a background thread."""

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        if message.type is not MessageType.FOLLOWUP_QUERY:
            raise InvalidTaskRunStateError(run_id, MessageType.FOLLOWUP_QUERY.value, message.type.value)
        return self._start_task_run(session_id=session_id, run_id=run_id, thread_name=f"research-agent-query-run-{run_id}")

    def start_ingest_run(self, session_id: str, run_id: str):
        """Transition a pending ingest run to running and execute it in a background thread."""

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        if message.type not in {MessageType.INGEST_ARXIV, MessageType.INGEST_PDF}:
            raise InvalidTaskRunStateError(run_id, "ingest_message", message.type.value)
        return self._start_task_run(session_id=session_id, run_id=run_id, thread_name=f"research-agent-ingest-run-{run_id}")

    def subscribe(self, session_id: str, run_id: str) -> RuntimeEventSubscription:
        """Subscribe to live or replayed events for a task run."""

        self._task_run_service.get_run(session_id, run_id)
        return self._runtime_event_broker.subscribe(run_id, replay=True)

    def _start_task_run(self, *, session_id: str, run_id: str, thread_name: str):
        task_run = self._task_run_service.get_run(session_id, run_id)
        if task_run.status is not TaskRunStatus.PENDING:
            raise InvalidTaskRunStateError(run_id, TaskRunStatus.PENDING.value, task_run.status.value)

        message = self._task_run_service.get_message_for_run(session_id, run_id)
        running_run = self._task_run_service.mark_running(session_id, run_id)
        self._runtime_event_broker.publish_run_started(running_run, message)

        thread = Thread(
            target=self._run_task_thread,
            args=(session_id, run_id),
            daemon=True,
            name=thread_name,
        )
        thread.start()
        return running_run

    def _run_task_thread(self, session_id: str, run_id: str) -> None:
        try:
            self._task_runtime_service.execute_running_task_run(session_id=session_id, run_id=run_id)
        except Exception as error:
            current_run = self._task_run_service.get_run(session_id, run_id)
            if current_run.status is not TaskRunStatus.RUNNING:
                return
            error_detail = error.to_dict() if hasattr(error, "to_dict") else None
            self._task_runtime_service.fail_running_task_run(session_id, run_id, str(error), error_detail)
