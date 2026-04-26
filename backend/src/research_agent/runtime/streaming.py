"""Runtime event streaming primitives for live query observation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from queue import Empty, Queue
from threading import Lock

from research_agent.domain.models import Message, TaskRun, TraceStep, utc_now


@dataclass(frozen=True, slots=True)
class RuntimeStreamEvent:
    """Single stream event published during a task run."""

    event_type: str
    session_id: str
    run_id: str
    timestamp: datetime = field(default_factory=utc_now)
    payload: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class RuntimeEventSubscription:
    """Thread-safe event subscription for one task run."""

    def __init__(self, broker: "RuntimeEventBroker", run_id: str, queue: Queue[RuntimeStreamEvent]) -> None:
        self._broker = broker
        self._run_id = run_id
        self._queue = queue
        self._closed = False

    def iter_events(self, *, timeout_seconds: float = 15.0) -> Iterator[RuntimeStreamEvent | None]:
        while not self._closed:
            try:
                event = self._queue.get(timeout=timeout_seconds)
            except Empty:
                yield None
                continue
            yield event
            if event.event_type in {"run_finished", "run_failed", "run_step_limit_reached"}:
                self.close()
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._broker.unsubscribe(self._run_id, self._queue)


class RuntimeEventBroker:
    """In-process pubsub broker with per-run replayable history."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: dict[str, list[Queue[RuntimeStreamEvent]]] = defaultdict(list)
        self._history: dict[str, list[RuntimeStreamEvent]] = defaultdict(list)

    def publish(self, event: RuntimeStreamEvent) -> None:
        with self._lock:
            self._history[event.run_id].append(event)
            subscribers = list(self._subscribers.get(event.run_id, []))
        for subscriber in subscribers:
            subscriber.put(event)

    def subscribe(self, run_id: str, *, replay: bool = True) -> RuntimeEventSubscription:
        queue: Queue[RuntimeStreamEvent] = Queue()
        with self._lock:
            self._subscribers[run_id].append(queue)
            history = list(self._history.get(run_id, []))
        if replay:
            for event in history:
                queue.put(event)
        return RuntimeEventSubscription(self, run_id, queue)

    def unsubscribe(self, run_id: str, queue: Queue[RuntimeStreamEvent]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is None:
                return
            self._subscribers[run_id] = [existing for existing in subscribers if existing is not queue]
            if not self._subscribers[run_id]:
                self._subscribers.pop(run_id, None)

    def publish_run_started(self, task_run: TaskRun, message: Message) -> None:
        self.publish(
            RuntimeStreamEvent(
                event_type="run_started",
                session_id=task_run.session_id,
                run_id=task_run.id,
                payload={
                    "task_run": {
                        "id": task_run.id,
                        "status": task_run.status.value,
                        "step_count": task_run.step_count,
                        "started_at": task_run.started_at.isoformat(),
                    },
                    "message": {
                        "id": message.id,
                        "type": message.type.value,
                        "role": message.role,
                        "content": message.content,
                    },
                },
            )
        )

    def publish_step_completed(self, session_id: str, run_id: str, trace_step: TraceStep) -> None:
        self.publish(
            RuntimeStreamEvent(
                event_type="step_completed",
                session_id=session_id,
                run_id=run_id,
                payload={
                    "trace_step": {
                        "id": trace_step.id,
                        "run_id": trace_step.run_id,
                        "action": trace_step.action,
                        "input_payload": trace_step.input_payload,
                        "result_payload": trace_step.result_payload,
                        "status": trace_step.status,
                        "started_at": trace_step.started_at.isoformat(),
                        "finished_at": trace_step.finished_at.isoformat() if trace_step.finished_at is not None else None,
                    }
                },
            )
        )

    def publish_assistant_message(self, run_id: str, message: Message) -> None:
        self.publish(
            RuntimeStreamEvent(
                event_type="assistant_message_committed",
                session_id=message.session_id,
                run_id=run_id,
                payload={
                    "message": {
                        "id": message.id,
                        "session_id": message.session_id,
                        "role": message.role,
                        "type": message.type.value,
                        "content": message.content,
                        "created_at": message.created_at.isoformat(),
                        "status": message.status,
                    }
                },
            )
        )

    def publish_run_finished(self, task_run: TaskRun) -> None:
        self.publish(
            RuntimeStreamEvent(
                event_type="run_finished",
                session_id=task_run.session_id,
                run_id=task_run.id,
                payload={
                    "task_run": {
                        "id": task_run.id,
                        "status": task_run.status.value,
                        "step_count": task_run.step_count,
                        "finished_at": task_run.finished_at.isoformat() if task_run.finished_at is not None else None,
                        "finish_reason": task_run.finish_reason,
                    }
                },
            )
        )

    def publish_run_failed(self, task_run: TaskRun, reason: str) -> None:
        self.publish(
            RuntimeStreamEvent(
                event_type="run_failed",
                session_id=task_run.session_id,
                run_id=task_run.id,
                payload={
                    "task_run": {
                        "id": task_run.id,
                        "status": task_run.status.value,
                        "step_count": task_run.step_count,
                        "finished_at": task_run.finished_at.isoformat() if task_run.finished_at is not None else None,
                        "finish_reason": task_run.finish_reason,
                    },
                    "reason": reason,
                },
            )
        )
