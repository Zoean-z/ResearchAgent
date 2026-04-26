"""OpenViking memory gateway implementations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol, runtime_checkable

from research_agent.adapters.openviking.surfaces import (
    NoopOpenVikingMemorySurface,
    NoopOpenVikingSessionSurface,
    OpenVikingMemoryRecord,
    OpenVikingMemorySurface,
    OpenVikingSessionSurface,
)
from research_agent.domain.models import Paper
from research_agent.services.ingest_analysis_service import MemoryAnalysisResult


@dataclass(frozen=True, slots=True)
class OpenVikingMemoryGatewayConfig:
    """Configuration for the optional OpenViking memory gateway."""

    enabled: bool = False
    url: str = "http://127.0.0.1:1933"
    api_key: str | None = None
    agent_id: str = "research-agent"


@runtime_checkable
class OpenVikingMemoryGateway(Protocol):
    """Mirror structured ingest output into OpenViking."""

    def mirror_ingest_result(
        self,
        *,
        session_id: str,
        paper: Paper,
        analysis: MemoryAnalysisResult,
    ) -> None:
        """Mirror a structured ingest bundle into OpenViking."""


class NoopOpenVikingMemoryGateway:
    """Fallback gateway used when OpenViking is not configured."""

    def mirror_ingest_result(
        self,
        *,
        session_id: str,
        paper: Paper,
        analysis: MemoryAnalysisResult,
    ) -> None:
        return None


class SurfaceBackedOpenVikingMemoryGateway:
    """Mirror structured memories through the repo-owned OpenViking surfaces."""

    def __init__(
        self,
        memory_surface: OpenVikingMemorySurface | None = None,
        session_surface: OpenVikingSessionSurface | None = None,
    ) -> None:
        self._memory_surface = memory_surface or NoopOpenVikingMemorySurface()
        self._session_surface = session_surface or NoopOpenVikingSessionSurface()

    def mirror_ingest_result(
        self,
        *,
        session_id: str,
        paper: Paper,
        analysis: MemoryAnalysisResult,
    ) -> None:
        self._session_surface.ensure_session(session_id, title=paper.title)
        self._memory_surface.mirror_memory(
            OpenVikingMemoryRecord(
                memory_id=analysis.paper_memory.id,
                memory_kind="paper_memory",
                session_id=session_id,
                paper_id=paper.id,
                payload=analysis.paper_memory.model_dump(mode="json"),
            )
        )
        if analysis.relation_memory is not None:
            self._memory_surface.mirror_memory(
                OpenVikingMemoryRecord(
                    memory_id=analysis.relation_memory.id,
                    memory_kind="relation_memory",
                    session_id=session_id,
                    paper_id=paper.id,
                    payload=analysis.relation_memory.model_dump(mode="json"),
                )
            )
        self._memory_surface.mirror_memory(
            OpenVikingMemoryRecord(
                memory_id=analysis.open_question_memory.id,
                memory_kind="open_question_memory",
                session_id=session_id,
                paper_id=paper.id,
                payload=analysis.open_question_memory.model_dump(mode="json"),
            )
        )
        self._session_surface.commit_session(session_id)


class HttpOpenVikingMemoryGateway:
    """Compatibility gateway that mirrors the whole ingest bundle as one session message."""

    def __init__(self, config: OpenVikingMemoryGatewayConfig) -> None:
        self._config = config
        self._client = self._create_client(config)

    def mirror_ingest_result(
        self,
        *,
        session_id: str,
        paper: Paper,
        analysis: MemoryAnalysisResult,
    ) -> None:
        session = self._get_session(session_id)
        payload = {
            "paper": paper.model_dump(mode="json"),
            "paper_memory": analysis.paper_memory.model_dump(mode="json"),
            "relation_memory": analysis.relation_memory.model_dump(mode="json") if analysis.relation_memory else None,
            "open_question_memory": analysis.open_question_memory.model_dump(mode="json"),
            "context_summary": analysis.context_summary,
        }
        session.add_message(
            "assistant",
            [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        )
        session.commit()

    def _create_client(self, config: OpenVikingMemoryGatewayConfig):
        if not config.enabled:
            return None
        try:
            import openviking as ov
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("openviking package is not installed") from exc
        client = ov.SyncHTTPClient(url=config.url, api_key=config.api_key)
        client.initialize()
        return client

    def _get_session(self, session_id: str):
        client = self._client
        if client is None:
            raise RuntimeError("OpenViking mirror is disabled")
        if hasattr(client, "get_session"):
            return client.get_session(session_id=session_id, auto_create=True)
        if hasattr(client, "session"):
            return client.session(session_id=session_id)
        raise RuntimeError("OpenViking client does not expose a session API")


__all__ = [
    "HttpOpenVikingMemoryGateway",
    "NoopOpenVikingMemoryGateway",
    "OpenVikingMemoryGateway",
    "OpenVikingMemoryGatewayConfig",
    "SurfaceBackedOpenVikingMemoryGateway",
]
