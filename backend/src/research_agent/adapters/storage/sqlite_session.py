"""SQLite repositories for sessions, messages, and session documents."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.adapters.storage.sqlite_common import SQLiteDatabase
from research_agent.domain.models import Message, Session, SessionDocument, utc_now
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort


class SQLiteSessionRepository(SessionRepositoryPort):
    """SQLite-backed session and session-document repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(self, session: Session) -> Session:
        self._database.execute(
            """
            INSERT INTO sessions (id, title, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                status = excluded.status,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.title,
                session.status,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )
        return session

    def get_by_id(self, session_id: str) -> Session | None:
        row = self._database.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        return self._row_to_session(row) if row is not None else None

    def list_all(self) -> Sequence[Session]:
        rows = self._database.query_all("SELECT * FROM sessions ORDER BY created_at ASC")
        return [self._row_to_session(row) for row in rows]

    def delete(self, session_id: str) -> Session | None:
        session = self.get_by_id(session_id)
        if session is None:
            return None
        deleted = session.model_copy(update={"status": "deleted", "updated_at": utc_now()})
        self._database.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (deleted.status, deleted.updated_at.isoformat(), deleted.id),
        )
        return deleted

    def save_document(self, session_document: SessionDocument) -> SessionDocument:
        self._database.execute(
            """
            INSERT INTO session_documents (id, session_id, paper_id, source_type, artifact_id, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                paper_id = excluded.paper_id,
                source_type = excluded.source_type,
                artifact_id = excluded.artifact_id,
                added_at = excluded.added_at
            """,
            (
                session_document.id,
                session_document.session_id,
                session_document.paper_id,
                session_document.source_type.value,
                session_document.artifact_id,
                session_document.added_at.isoformat(),
            ),
        )
        return session_document

    def list_documents(self, session_id: str) -> Sequence[SessionDocument]:
        rows = self._database.query_all(
            "SELECT * FROM session_documents WHERE session_id = ? ORDER BY added_at ASC",
            (session_id,),
        )
        return [self._row_to_document(row) for row in rows]

    def delete_documents(self, session_id: str) -> int:
        rows = self._database.query_all("SELECT id FROM session_documents WHERE session_id = ?", (session_id,))
        deleted = len(rows)
        self._database.execute("DELETE FROM session_documents WHERE session_id = ?", (session_id,))
        return deleted

    def _row_to_session(self, row) -> Session:
        return Session.model_validate(
            {
                "id": row["id"],
                "title": row["title"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    def _row_to_document(self, row) -> SessionDocument:
        return SessionDocument.model_validate(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "paper_id": row["paper_id"],
                "source_type": row["source_type"],
                "artifact_id": row["artifact_id"],
                "added_at": row["added_at"],
            }
        )


class SQLiteMessageRepository(MessageRepositoryPort):
    """SQLite-backed session message repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(self, message: Message) -> Message:
        self._database.execute(
            """
            INSERT INTO messages (id, session_id, role, type, content, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                session_id = excluded.session_id,
                role = excluded.role,
                type = excluded.type,
                content = excluded.content,
                status = excluded.status,
                created_at = excluded.created_at
            """,
            (
                message.id,
                message.session_id,
                message.role,
                message.type.value,
                message.content,
                message.status,
                message.created_at.isoformat(),
            ),
        )
        return message

    def get_by_id(self, message_id: str) -> Message | None:
        row = self._database.query_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return self._row_to_message(row) if row is not None else None

    def list_by_session(self, session_id: str) -> Sequence[Message]:
        rows = self._database.query_all("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
        return [self._row_to_message(row) for row in rows]

    def delete_by_session(self, session_id: str) -> int:
        rows = self._database.query_all("SELECT id FROM messages WHERE session_id = ?", (session_id,))
        deleted = len(rows)
        self._database.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        return deleted

    def _row_to_message(self, row) -> Message:
        return Message.model_validate(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "role": row["role"],
                "type": row["type"],
                "content": row["content"],
                "created_at": row["created_at"],
                "status": row["status"],
            }
        )
