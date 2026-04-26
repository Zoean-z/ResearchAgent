"""Thin read service for session timeline events."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.domain.models import TimelineEvent
from research_agent.domain.ports import SessionRepositoryPort, TimelineRepositoryPort
from research_agent.services.errors import EntityNotFoundError


class TimelineQueryService:
    """Read-side timeline queries scoped to a session."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._timeline_repository = timeline_repository

    def list_timeline(self, session_id: str) -> Sequence[TimelineEvent]:
        """Return timeline events for a session after verifying the session exists."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)
        return self._timeline_repository.list_by_session(session_id)

    def list_events_for_run(self, session_id: str, run_id: str) -> Sequence[TimelineEvent]:
        """Return timeline events for a particular run within a session."""

        if self._session_repository.get_by_id(session_id) is None:
            raise EntityNotFoundError("Session", session_id)
        return [event for event in self._timeline_repository.list_by_session(session_id) if event.run_id == run_id]
