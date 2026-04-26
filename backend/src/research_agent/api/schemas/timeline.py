"""Timeline API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from research_agent.domain.models import TimelineEvent


class TimelineEventResponse(BaseModel):
    """Serialized session timeline event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    run_id: str | None
    event_type: str
    summary: str
    related_memory_ids: list[str]
    related_paper_ids: list[str]
    created_at: datetime

    @classmethod
    def from_domain(cls, event: TimelineEvent) -> "TimelineEventResponse":
        return cls(
            id=event.id,
            session_id=event.session_id,
            run_id=event.run_id,
            event_type=event.event_type,
            summary=event.summary,
            related_memory_ids=event.related_memory_ids,
            related_paper_ids=event.related_paper_ids,
            created_at=event.created_at,
        )


class TimelineResponse(BaseModel):
    """List response for timeline events."""

    model_config = ConfigDict(extra="forbid")

    items: list[TimelineEventResponse]
