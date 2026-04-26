"""Session-scoped domain entities."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import Field

from research_agent.domain.enums import MessageType, SourceType
from research_agent.domain.models.base import DomainModel, utc_now


class Session(DomainModel):
    """A user workspace that groups documents, memories, and follow-up queries."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: str = Field(default="active")


class SessionDocument(DomainModel):
    """Binding between a session and a paper source artifact."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    paper_id: str
    source_type: SourceType
    artifact_id: str
    added_at: datetime = Field(default_factory=utc_now)


class Message(DomainModel):
    """Session message record for ingest or follow-up requests."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: str = Field(default="user", min_length=1)
    type: MessageType
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    status: str = Field(default="accepted")
