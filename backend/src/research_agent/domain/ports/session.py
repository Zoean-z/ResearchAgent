"""Session and message repository ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_agent.domain.models import Message, Session, SessionDocument


@runtime_checkable
class SessionRepositoryPort(Protocol):
    """Storage boundary for sessions and session-document bindings."""

    def save(self, session: Session) -> Session:
        """Create or replace a session record."""

    def get_by_id(self, session_id: str) -> Session | None:
        """Fetch a single session by id."""

    def list_all(self) -> Sequence[Session]:
        """Return all known sessions."""

    def delete(self, session_id: str) -> Session | None:
        """Delete or tombstone a session record."""

    def save_document(self, session_document: SessionDocument) -> SessionDocument:
        """Create or replace a session-document binding."""

    def list_documents(self, session_id: str) -> Sequence[SessionDocument]:
        """List documents linked to a session."""

    def list_all_documents(self) -> Sequence[SessionDocument]:
        """List all session-document bindings across sessions."""

    def get_document(self, session_id: str, paper_id: str) -> SessionDocument | None:
        """Fetch a session-document binding by session and paper."""

    def delete_documents(self, session_id: str) -> int:
        """Delete all session-document bindings for a session."""


@runtime_checkable
class MessageRepositoryPort(Protocol):
    """Storage boundary for session messages."""

    def save(self, message: Message) -> Message:
        """Create or replace a message record."""

    def get_by_id(self, message_id: str) -> Message | None:
        """Fetch a single message by id."""

    def list_by_session(self, session_id: str) -> Sequence[Message]:
        """List messages scoped to a session."""

    def delete_by_session(self, session_id: str) -> int:
        """Delete all messages scoped to a session."""
