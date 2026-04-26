"""Follow-up query API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FollowupQueryRequest(BaseModel):
    """Request payload for a session follow-up query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class QueryAcceptedResponse(BaseModel):
    """Accepted query response for future runtime execution."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = True
    session_id: str
    message_id: str
    run_id: str
    status: str = "accepted"
