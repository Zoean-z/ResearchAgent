"""Thin application service for session lifecycle operations."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.domain.models import Session
from research_agent.domain.ports import SessionRepositoryPort
from research_agent.services.errors import EntityNotFoundError


class SessionService:
    """Session-focused orchestration over repository ports."""

    def __init__(self, session_repository: SessionRepositoryPort) -> None:
        self._session_repository = session_repository

    def create_session(self, title: str) -> Session:
        """Create and persist a new session."""

        session = Session(title=title)
        return self._session_repository.save(session)

    def list_sessions(self) -> Sequence[Session]:
        """Return all sessions."""

        return self._session_repository.list_all()

    def get_session(self, session_id: str) -> Session:
        """Fetch a session or raise a stable service-layer error."""

        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)
        if session.status == "deleted":
            raise EntityNotFoundError("Session", session_id)
        return session
