"""Session API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from research_agent.domain.models import Session


class CreateSessionRequest(BaseModel):
    """Request payload for creating a session."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)


class SessionResponse(BaseModel):
    """Serialized session record returned by the API."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    status: str

    @classmethod
    def from_domain(cls, session: Session) -> "SessionResponse":
        return cls(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            status=session.status,
        )


class SessionListResponse(BaseModel):
    """List response for sessions."""

    model_config = ConfigDict(extra="forbid")

    items: list[SessionResponse]
