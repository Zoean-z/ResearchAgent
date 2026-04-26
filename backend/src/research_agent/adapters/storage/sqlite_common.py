"""Shared SQLite helper utilities for storage adapters."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


_DEFAULT_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_documents (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    canonical_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    arxiv_id TEXT,
    pdf_fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    uri_or_path TEXT NOT NULL,
    checksum TEXT NOT NULL,
    page_count INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    page INTEGER,
    section TEXT
);

CREATE TABLE IF NOT EXISTS paper_memories (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    problem TEXT,
    method TEXT,
    key_results_json TEXT NOT NULL,
    limitations_json TEXT NOT NULL,
    novelty_claim TEXT,
    source_refs_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relation_memories (
    id TEXT PRIMARY KEY,
    source_paper TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    target_paper TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS open_question_memories (
    id TEXT PRIMARY KEY,
    unresolved_question TEXT NOT NULL,
    related_papers_json TEXT NOT NULL,
    why_open_json TEXT NOT NULL,
    possible_followup_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    step_count INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    finish_reason TEXT
);

CREATE TABLE IF NOT EXISTS trace_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    input_payload_json TEXT NOT NULL,
    result_payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS trace_narratives (
    trace_step_id TEXT PRIMARY KEY REFERENCES trace_steps(id) ON DELETE CASCADE,
    reason_text TEXT NOT NULL,
    impact_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES task_runs(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    related_memory_ids_json TEXT NOT NULL,
    related_paper_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_documents_session_id ON session_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_memories_paper_id ON paper_memories(paper_id);
CREATE INDEX IF NOT EXISTS idx_relation_memories_source_paper ON relation_memories(source_paper);
CREATE INDEX IF NOT EXISTS idx_relation_memories_target_paper ON relation_memories(target_paper);
CREATE INDEX IF NOT EXISTS idx_task_runs_session_id ON task_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_trace_steps_run_id ON trace_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_timeline_events_session_id ON timeline_events(session_id);
CREATE INDEX IF NOT EXISTS idx_timeline_events_run_id ON timeline_events(run_id);
"""

_SCHEMA_MIGRATION_PATH = Path(__file__).resolve().parents[4] / "migrations" / "0001_initial_schema.sql"


def _load_schema_sql() -> str:
    """Load the shared schema from the migration stub when available."""

    if _SCHEMA_MIGRATION_PATH.exists():
        return _SCHEMA_MIGRATION_PATH.read_text(encoding="utf-8")
    return _DEFAULT_SCHEMA_SQL


SCHEMA_SQL = _load_schema_sql()


class SQLiteDatabase:
    """Small thread-safe wrapper around a shared SQLite connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path not in {":memory:"} and not self.path.startswith("file:"):
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON;")
            self._connection.executescript(SCHEMA_SQL)
            self._ensure_messages_role_column()

    def _ensure_messages_role_column(self) -> None:
        rows = self._connection.execute("PRAGMA table_info(messages)").fetchall()
        column_names = {row[1] for row in rows}
        if "role" not in column_names:
            self._connection.execute("ALTER TABLE messages ADD COLUMN role TEXT")
            self._connection.execute(
                """
                UPDATE messages
                SET role = CASE
                    WHEN status = 'completed' THEN 'assistant'
                    WHEN status = 'accepted' THEN 'user'
                    ELSE 'user'
                END
                WHERE role IS NULL OR role = ''
                """
            )
            self._connection.commit()
            return
        self._connection.execute(
            """
            UPDATE messages
            SET role = CASE
                WHEN status = 'completed' THEN 'assistant'
                WHEN status = 'accepted' THEN 'user'
                ELSE COALESCE(role, 'user')
            END
            WHERE role IS NULL OR role = ''
            """
        )
        self._connection.commit()

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._connection.execute(sql, parameters)
            self._connection.commit()
            return cursor

    def query_one(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._lock:
            cursor = self._connection.execute(sql, parameters)
            return cursor.fetchone()

    def query_all(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            cursor = self._connection.execute(sql, parameters)
            return cursor.fetchall()

    def executemany(self, sql: str, parameters: list[tuple[Any, ...]]) -> None:
        if not parameters:
            return
        with self._lock:
            self._connection.executemany(sql, parameters)
            self._connection.commit()

    @staticmethod
    def encode_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def decode_json(value: str) -> Any:
        return json.loads(value)
