"""Thin read service for session messages."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.domain.models import Message
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort
from research_agent.services.errors import EntityNotFoundError


class MessageQueryService:
    """Read-side message queries scoped to a session."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        message_repository: MessageRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._message_repository = message_repository

    def list_messages(self, session_id: str) -> Sequence[Message]:
        """Return messages for a session after verifying the session exists."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)
        return self._message_repository.list_by_session(session_id)
