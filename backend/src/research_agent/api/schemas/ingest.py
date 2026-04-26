"""Ingest API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IngestArxivRequest(BaseModel):
    """Request payload for arXiv ingestion."""

    model_config = ConfigDict(extra="forbid")

    arxiv_url: str = Field(min_length=1)


class IngestPdfRequest(BaseModel):
    """Request payload for local PDF ingestion."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1)


class IngestPdfResponse(BaseModel):
    """Placeholder response for local PDF loading."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    checksum: str
    page_count: int | None = None


class IngestAcceptedResponse(BaseModel):
    """Accepted ingest request response for future async runtime execution."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool = True
    session_id: str
    message_id: str
    run_id: str
    status: str = "accepted"
