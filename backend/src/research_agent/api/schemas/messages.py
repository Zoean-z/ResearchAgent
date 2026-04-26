"""Message API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.domain.models import Message


class MessageIntakeRequest(BaseModel):
    """Unified message intake payload."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, min_length=1)
    arxiv_url: str | None = Field(default=None, min_length=1)
    file_path: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_oneof(self) -> "MessageIntakeRequest":
        provided = [value for value in (self.text, self.arxiv_url, self.file_path) if value is not None and value.strip()]
        if len(provided) != 1:
            raise ValueError("Provide exactly one of text, arxiv_url, or file_path.")
        return self


class MessageSubmissionResponse(BaseModel):
    """Unified accepted-message response."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = True
    session_id: str
    message_id: str
    run_id: str
    message_type: str
    status: str = "accepted"


class MessageResponse(BaseModel):
    """Serialized message record returned by the API."""

    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    role: str
    type: str
    content: str
    created_at: datetime
    status: str

    @classmethod
    def from_domain(cls, message: Message) -> "MessageResponse":
        return cls(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            type=message.type.value,
            content=message.content,
            created_at=message.created_at,
            status=message.status,
        )


class MessageListResponse(BaseModel):
    """List response for session messages."""

    model_config = ConfigDict(extra="forbid")

    items: list[MessageResponse]
