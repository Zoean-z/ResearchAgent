"""SQLite repositories for task runs, trace steps, narratives, and timeline events."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.adapters.storage.sqlite_common import SQLiteDatabase
from research_agent.domain.models import TaskRun, TimelineEvent, TraceNarrative, TraceStep
from research_agent.domain.ports import TimelineRepositoryPort, TraceRepositoryPort


class SQLiteTraceRepository(TraceRepositoryPort):
    """SQLite-backed task run and trace repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save_run(self, task_run: TaskRun) -> TaskRun:
        self._database.execute(
            """
            INSERT INTO task_runs (id, session_id, message_id, status, step_count, started_at, finished_at, finish_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                message_id = excluded.message_id,
                status = excluded.status,
                step_count = excluded.step_count,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                finish_reason = excluded.finish_reason
            """,
            (
                task_run.id,
                task_run.session_id,
                task_run.message_id,
                task_run.status.value,
                task_run.step_count,
                task_run.started_at.isoformat(),
                task_run.finished_at.isoformat() if task_run.finished_at is not None else None,
                task_run.finish_reason,
            ),
        )
        return task_run

    def get_run(self, run_id: str) -> TaskRun | None:
        row = self._database.query_one("SELECT * FROM task_runs WHERE id = ?", (run_id,))
        return self._row_to_run(row) if row is not None else None

    def list_runs_by_session(self, session_id: str) -> Sequence[TaskRun]:
        rows = self._database.query_all(
            "SELECT * FROM task_runs WHERE session_id = ? ORDER BY started_at DESC",
            (session_id,),
        )
        return [self._row_to_run(row) for row in rows]

    def save_step(self, trace_step: TraceStep) -> TraceStep:
        self._database.execute(
            """
            INSERT INTO trace_steps (id, run_id, action, input_payload_json, result_payload_json, status, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                run_id = excluded.run_id,
                action = excluded.action,
                input_payload_json = excluded.input_payload_json,
                result_payload_json = excluded.result_payload_json,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at
            """,
            (
                trace_step.id,
                trace_step.run_id,
                trace_step.action,
                SQLiteDatabase.encode_json(trace_step.input_payload),
                SQLiteDatabase.encode_json(trace_step.result_payload),
                trace_step.status,
                trace_step.started_at.isoformat(),
                trace_step.finished_at.isoformat() if trace_step.finished_at is not None else None,
            ),
        )
        return trace_step

    def list_steps(self, run_id: str) -> Sequence[TraceStep]:
        rows = self._database.query_all("SELECT * FROM trace_steps WHERE run_id = ? ORDER BY started_at ASC", (run_id,))
        return [self._row_to_step(row) for row in rows]

    def save_narrative(self, trace_narrative: TraceNarrative) -> TraceNarrative:
        self._database.execute(
            """
            INSERT INTO trace_narratives (trace_step_id, reason_text, impact_text)
            VALUES (?, ?, ?)
            ON CONFLICT(trace_step_id) DO UPDATE SET
                reason_text = excluded.reason_text,
                impact_text = excluded.impact_text
            """,
            (trace_narrative.trace_step_id, trace_narrative.reason_text, trace_narrative.impact_text),
        )
        return trace_narrative

    def list_narratives(self, run_id: str) -> Sequence[TraceNarrative]:
        rows = self._database.query_all(
            """
            SELECT tn.trace_step_id, tn.reason_text, tn.impact_text
            FROM trace_narratives tn
            JOIN trace_steps ts ON ts.id = tn.trace_step_id
            WHERE ts.run_id = ?
            ORDER BY ts.started_at ASC
            """,
            (run_id,),
        )
        return [TraceNarrative.model_validate(dict(row)) for row in rows]

    def delete_run(self, run_id: str) -> None:
        self._database.execute(
            "DELETE FROM trace_narratives WHERE trace_step_id IN (SELECT id FROM trace_steps WHERE run_id = ?)",
            (run_id,),
        )
        self._database.execute("DELETE FROM trace_steps WHERE run_id = ?", (run_id,))
        self._database.execute("DELETE FROM task_runs WHERE id = ?", (run_id,))

    def delete_runs_for_session(self, session_id: str) -> int:
        rows = self._database.query_all("SELECT id FROM task_runs WHERE session_id = ?", (session_id,))
        run_ids = [row["id"] for row in rows]
        for run_id in run_ids:
            self.delete_run(run_id)
        return len(run_ids)

    def _row_to_run(self, row) -> TaskRun:
        return TaskRun.model_validate(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "message_id": row["message_id"],
                "status": row["status"],
                "step_count": row["step_count"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "finish_reason": row["finish_reason"],
            }
        )

    def _row_to_step(self, row) -> TraceStep:
        return TraceStep.model_validate(
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "action": row["action"],
                "input_payload": SQLiteDatabase.decode_json(row["input_payload_json"]),
                "result_payload": SQLiteDatabase.decode_json(row["result_payload_json"]),
                "status": row["status"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
            }
        )


class SQLiteTimelineRepository(TimelineRepositoryPort):
    """SQLite-backed timeline repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(self, event: TimelineEvent) -> TimelineEvent:
        self._database.execute(
            """
            INSERT INTO timeline_events (
                id, session_id, run_id, event_type, summary,
                related_memory_ids_json, related_paper_ids_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                run_id = excluded.run_id,
                event_type = excluded.event_type,
                summary = excluded.summary,
                related_memory_ids_json = excluded.related_memory_ids_json,
                related_paper_ids_json = excluded.related_paper_ids_json,
                created_at = excluded.created_at
            """,
            (
                event.id,
                event.session_id,
                event.run_id,
                event.event_type,
                event.summary,
                SQLiteDatabase.encode_json(event.related_memory_ids),
                SQLiteDatabase.encode_json(event.related_paper_ids),
                event.created_at.isoformat(),
            ),
        )
        return event

    def list_by_session(self, session_id: str) -> Sequence[TimelineEvent]:
        rows = self._database.query_all(
            "SELECT * FROM timeline_events WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        return [self._row_to_event(row) for row in rows]

    def delete_by_session(self, session_id: str) -> int:
        rows = self._database.query_all("SELECT id FROM timeline_events WHERE session_id = ?", (session_id,))
        deleted = len(rows)
        self._database.execute("DELETE FROM timeline_events WHERE session_id = ?", (session_id,))
        return deleted

    def _row_to_event(self, row) -> TimelineEvent:
        return TimelineEvent.model_validate(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "run_id": row["run_id"],
                "event_type": row["event_type"],
                "summary": row["summary"],
                "related_memory_ids": SQLiteDatabase.decode_json(row["related_memory_ids_json"]),
                "related_paper_ids": SQLiteDatabase.decode_json(row["related_paper_ids_json"]),
                "created_at": row["created_at"],
            }
        )
