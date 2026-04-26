"""In-memory storage for task runs, trace records, and timeline events."""

from __future__ import annotations

from research_agent.domain.models import TaskRun, TimelineEvent, TraceNarrative, TraceStep
from research_agent.domain.ports import TimelineRepositoryPort, TraceRepositoryPort


class InMemoryTraceRepository(TraceRepositoryPort):
    """Simple in-memory storage for task runs and trace records."""

    def __init__(self) -> None:
        self._runs: dict[str, TaskRun] = {}
        self._steps: dict[str, list[TraceStep]] = {}
        self._narratives_by_step_id: dict[str, TraceNarrative] = {}

    def save_run(self, task_run: TaskRun) -> TaskRun:
        self._runs[task_run.id] = task_run
        return task_run

    def get_run(self, run_id: str) -> TaskRun | None:
        return self._runs.get(run_id)

    def list_runs_by_session(self, session_id: str) -> list[TaskRun]:
        runs = [run for run in self._runs.values() if run.session_id == session_id]
        runs.sort(key=lambda item: item.started_at, reverse=True)
        return runs

    def save_step(self, trace_step: TraceStep) -> TraceStep:
        steps = [existing for existing in self._steps.get(trace_step.run_id, []) if existing.id != trace_step.id]
        steps.append(trace_step)
        steps.sort(key=lambda item: item.started_at)
        self._steps[trace_step.run_id] = steps
        return trace_step

    def list_steps(self, run_id: str) -> list[TraceStep]:
        return list(self._steps.get(run_id, []))

    def save_narrative(self, trace_narrative: TraceNarrative) -> TraceNarrative:
        self._narratives_by_step_id[trace_narrative.trace_step_id] = trace_narrative
        return trace_narrative

    def list_narratives(self, run_id: str) -> list[TraceNarrative]:
        step_ids = {step.id for step in self._steps.get(run_id, [])}
        return [
            narrative
            for step_id, narrative in self._narratives_by_step_id.items()
            if step_id in step_ids
        ]

    def delete_run(self, run_id: str) -> None:
        steps = self._steps.pop(run_id, [])
        for step in steps:
            self._narratives_by_step_id.pop(step.id, None)
        self._runs.pop(run_id, None)

    def delete_runs_for_session(self, session_id: str) -> int:
        run_ids = [run_id for run_id, run in self._runs.items() if run.session_id == session_id]
        for run_id in run_ids:
            self.delete_run(run_id)
        return len(run_ids)


class InMemoryTimelineRepository(TimelineRepositoryPort):
    """Simple in-memory storage for timeline events."""

    def __init__(self) -> None:
        self._events_by_session: dict[str, list[TimelineEvent]] = {}

    def save(self, event: TimelineEvent) -> TimelineEvent:
        events = [existing for existing in self._events_by_session.get(event.session_id, []) if existing.id != event.id]
        events.append(event)
        events.sort(key=lambda item: item.created_at)
        self._events_by_session[event.session_id] = events
        return event

    def list_by_session(self, session_id: str) -> list[TimelineEvent]:
        return list(self._events_by_session.get(session_id, []))

    def delete_by_session(self, session_id: str) -> int:
        events = self._events_by_session.pop(session_id, [])
        return len(events)
