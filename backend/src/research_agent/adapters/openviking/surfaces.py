"""OpenViking message, memory, and session adapter surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from research_agent.domain.models.base import DomainModel, utc_now


class OpenVikingMessageRecord(DomainModel):
    """A message payload mirrored into OpenViking."""

    session_id: str
    message_id: str
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenVikingMemoryRecord(DomainModel):
    """A structured memory payload mirrored into OpenViking."""

    memory_id: str
    memory_kind: str = Field(min_length=1)
    session_id: str | None = None
    paper_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class OpenVikingSearchHit(DomainModel):
    """A bounded retrieval result from OpenViking."""

    item_kind: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    session_id: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenVikingSessionSnapshot(DomainModel):
    """A compact session view useful for sync and deletion planning."""

    session_id: str
    title: str | None = None
    message_count: int = Field(default=0, ge=0)
    memory_count: int = Field(default=0, ge=0)
    committed_at: datetime | None = None
    deleted: bool = False


@runtime_checkable
class OpenVikingMessageSurface(Protocol):
    """Message-history operations exposed to the rest of the repo."""

    def mirror_message(self, message: OpenVikingMessageRecord) -> None:
        """Mirror a single message into OpenViking."""

    def list_messages(self, session_id: str) -> tuple[OpenVikingMessageRecord, ...]:
        """Return mirrored messages for a session."""

    def delete_message(self, session_id: str, message_id: str) -> None:
        """Delete a single mirrored message."""


@runtime_checkable
class OpenVikingMemorySurface(Protocol):
    """Memory operations exposed to the rest of the repo."""

    def mirror_memory(self, memory: OpenVikingMemoryRecord) -> None:
        """Mirror a structured memory payload into OpenViking."""

    def search_session_memory(self, session_id: str, query: str, top_k: int = 5) -> tuple[OpenVikingSearchHit, ...]:
        """Search session-scoped OpenViking memory."""

    def search_global_memory(
        self,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> tuple[OpenVikingSearchHit, ...]:
        """Search globally available OpenViking memory."""

    def delete_memory(self, memory_id: str) -> None:
        """Delete a single mirrored memory."""


@runtime_checkable
class OpenVikingSessionSurface(Protocol):
    """Session lifecycle operations exposed to the rest of the repo."""

    def ensure_session(self, session_id: str, title: str | None = None) -> OpenVikingSessionSnapshot:
        """Ensure a mirrored OpenViking session exists."""

    def commit_session(self, session_id: str) -> OpenVikingSessionSnapshot:
        """Commit the mirrored OpenViking session."""

    def delete_session(self, session_id: str) -> None:
        """Delete a mirrored OpenViking session."""


@dataclass(slots=True)
class OpenVikingAdapterSurfaceBundle:
    """Grouped OpenViking surfaces for dependency wiring."""

    messages: OpenVikingMessageSurface = field(default_factory=lambda: NoopOpenVikingMessageSurface())
    memories: OpenVikingMemorySurface = field(default_factory=lambda: NoopOpenVikingMemorySurface())
    sessions: OpenVikingSessionSurface = field(default_factory=lambda: NoopOpenVikingSessionSurface())


class NoopOpenVikingMessageSurface:
    """No-op message surface used until OpenViking is wired in."""

    def mirror_message(self, message: OpenVikingMessageRecord) -> None:
        return None

    def list_messages(self, session_id: str) -> tuple[OpenVikingMessageRecord, ...]:
        return ()

    def delete_message(self, session_id: str, message_id: str) -> None:
        return None


class NoopOpenVikingMemorySurface:
    """No-op memory surface used until OpenViking is wired in."""

    def mirror_memory(self, memory: OpenVikingMemoryRecord) -> None:
        return None

    def search_session_memory(self, session_id: str, query: str, top_k: int = 5) -> tuple[OpenVikingSearchHit, ...]:
        return ()

    def search_global_memory(
        self,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> tuple[OpenVikingSearchHit, ...]:
        return ()

    def delete_memory(self, memory_id: str) -> None:
        return None


class NoopOpenVikingSessionSurface:
    """No-op session surface used until OpenViking is wired in."""

    def ensure_session(self, session_id: str, title: str | None = None) -> OpenVikingSessionSnapshot:
        return OpenVikingSessionSnapshot(session_id=session_id, title=title)

    def commit_session(self, session_id: str) -> OpenVikingSessionSnapshot:
        return OpenVikingSessionSnapshot(session_id=session_id)

    def delete_session(self, session_id: str) -> None:
        return None


__all__ = [
    "NoopOpenVikingMemorySurface",
    "NoopOpenVikingMessageSurface",
    "NoopOpenVikingSessionSurface",
    "OpenVikingAdapterSurfaceBundle",
    "OpenVikingMemoryRecord",
    "OpenVikingMemorySurface",
    "OpenVikingMessageRecord",
    "OpenVikingMessageSurface",
    "OpenVikingSearchHit",
    "OpenVikingSessionSnapshot",
    "OpenVikingSessionSurface",
]
