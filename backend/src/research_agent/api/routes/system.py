"""System routes."""

from __future__ import annotations

import os

from fastapi import APIRouter

from research_agent.api.deps import get_app_name
from research_agent.api.schemas import HealthResponse, RuntimeStatusResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a minimal health payload."""

    return HealthResponse(status="ok")


@router.get("/api/system/runtime", response_model=RuntimeStatusResponse)
def runtime_status() -> RuntimeStatusResponse:
    """Return a safe runtime status snapshot for the frontend."""

    query_agent_backend = os.getenv("RESEARCH_AGENT_QUERY_AGENT_BACKEND", "pydantic_ai")

    return RuntimeStatusResponse(
        app_name=get_app_name(),
        storage_backend=os.getenv("RESEARCH_AGENT_STORAGE_BACKEND", "sqlite"),
        sqlite_path=os.getenv("RESEARCH_AGENT_SQLITE_PATH"),
        query_agent_backend=query_agent_backend,
        query_agent_provider=os.getenv("RESEARCH_AGENT_QUERY_AGENT_PROVIDER", "deepseek"),
        query_agent_model=os.getenv("RESEARCH_AGENT_QUERY_AGENT_MODEL", "deepseek-v4-flash"),
        ingest_extraction_backend=os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND", "heuristic"),
        ingest_extraction_provider=os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_PROVIDER", "deepseek"),
        ingest_extraction_model=os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_MODEL", "deepseek-v4-flash"),
        openviking_backend=os.getenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "noop"),
        openviking_data_path=os.getenv("RESEARCH_AGENT_OPENVIKING_DATA_PATH"),
        openviking_url=os.getenv("RESEARCH_AGENT_OPENVIKING_URL"),
    )
