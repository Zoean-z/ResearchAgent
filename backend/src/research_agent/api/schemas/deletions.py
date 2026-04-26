"""Deletion API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from research_agent.api.schemas.sessions import SessionResponse


class DeleteSessionResponse(BaseModel):
    """Response payload for deleting a session."""

    model_config = ConfigDict(extra="forbid")

    session: SessionResponse
    deleted_documents: int
    deleted_messages: int
    deleted_runs: int
    deleted_timeline_events: int
    deleted_memories: int
    mirrored_to_openviking: bool


class DeleteMemoryResponse(BaseModel):
    """Response payload for deleting a single memory item."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    memory_kind: str
    memory_id: str
    deleted: bool
    mirrored_to_openviking: bool
