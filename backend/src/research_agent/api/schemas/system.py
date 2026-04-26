"""System status API schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Minimal health payload."""

    model_config = ConfigDict(extra="forbid")

    status: str


class RuntimeStatusResponse(BaseModel):
    """Safe runtime configuration snapshot for the frontend settings drawer."""

    model_config = ConfigDict(extra="forbid")

    app_name: str
    storage_backend: str
    sqlite_path: str | None
    query_agent_backend: str
    query_agent_provider: str
    query_agent_model: str
    ingest_extraction_backend: str
    ingest_extraction_provider: str
    ingest_extraction_model: str
    openviking_backend: str
    openviking_data_path: str | None
    openviking_url: str | None
