"""Dependency wiring for the API layer."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from fastapi import Request

from research_agent.adapters.storage import (
    InMemoryArtifactRepository,
    InMemoryChunkRepository,
    InMemoryMemoryRepository,
    InMemoryMessageRepository,
    InMemoryPaperRepository,
    InMemorySessionRepository,
    InMemoryTimelineRepository,
    InMemoryTraceRepository,
    SQLiteArtifactRepository,
    SQLiteDatabase,
    SQLiteChunkRepository,
    SQLiteMemoryRepository,
    SQLiteMessageRepository,
    SQLitePaperRepository,
    SQLiteSessionRepository,
    SQLiteTimelineRepository,
    SQLiteTraceRepository,
)
from research_agent.adapters.llm import (
    DeepSeekStructuredIngestExtractionTransport,
    DeepSeekStructuredQueryAgentTransport,
    ModelBackedIngestExtractionClient,
    ModelBackedQueryAgentClient,
    UnavailableStructuredQueryAgentTransport,
)
from research_agent.adapters.pydantic_ai import PydanticAIQueryTurnClient
from research_agent.adapters.openviking import HttpOpenVikingMemoryGateway, NoopOpenVikingMemoryGateway, OpenVikingMemoryGatewayConfig
from research_agent.adapters.openviking import (
    OpenVikingAdapterSurfaceBundle,
    OpenVikingRetrievalAdapter,
    OpenVikingSurfaceConfig,
    SurfaceBackedOpenVikingMemoryGateway,
    build_embedded_openviking_surface_bundle,
    build_inmemory_openviking_surface_bundle,
    build_sdk_openviking_surface_bundle,
)
from research_agent.domain.ports import (
    ArtifactRepositoryPort,
    ChunkRepositoryPort,
    MemoryRepositoryPort,
    MessageRepositoryPort,
    PaperRepositoryPort,
    SessionRepositoryPort,
    TimelineRepositoryPort,
    TraceRepositoryPort,
)
from research_agent.services.message_intake_service import MessageIntakeService
from research_agent.services import (
    ContextRerankService,
    DeletionService,
    MemorySnapshotService,
    MessageQueryService,
    QueryExecutionService,
    RetrievalService,
    SessionService,
    TaskRunService,
    TimelineQueryService,
    TraceQueryService,
)
from research_agent.services.ingest_execution_service import IngestExecutionService
from research_agent.services.ingest_analysis_service import IngestAnalysisService
from research_agent.services.ingest_materialization_service import IngestMaterializationService
from research_agent.services.memory_extraction_service import MemoryExtractionService
from research_agent.services.task_run_streaming_service import TaskRunStreamingService
from research_agent.runtime import RuntimeEventBroker, TaskRuntimeService
from research_agent.tools import HeuristicQueryToolPlannerClient, InternalToolRegistry, PlannerBackedQueryAgentClient, QueryToolExecutor


@dataclass(slots=True)
class RepositoryBundle:
    """Container for the repositories used by mock API routes."""

    sessions: SessionRepositoryPort
    messages: MessageRepositoryPort
    papers: PaperRepositoryPort
    artifacts: ArtifactRepositoryPort
    memories: MemoryRepositoryPort
    trace: TraceRepositoryPort
    timeline: TimelineRepositoryPort
    chunks: ChunkRepositoryPort


@dataclass(slots=True)
class ServiceBundle:
    """Container for thin application services used by the API layer."""

    sessions: SessionService
    message_intake: MessageIntakeService
    messages: MessageQueryService
    timeline: TimelineQueryService
    memory_snapshot: MemorySnapshotService
    task_runs: TaskRunService
    deletions: DeletionService
    tools: InternalToolRegistry
    retrieval: RetrievalService
    context_rerank: ContextRerankService
    query_execution: QueryExecutionService
    task_stream: TaskRunStreamingService
    task_runtime: TaskRuntimeService
    query_runtime: TaskRuntimeService
    trace: TraceQueryService


def get_app_name() -> str:
    """Return the API application name."""

    return "OpenViking Memory-Routed Paper Agent"


def create_repository_bundle(
    storage_backend: str | None = None,
    sqlite_path: str | Path | None = None,
) -> RepositoryBundle:
    """Create repositories used by the API boundary."""

    backend_name = (storage_backend or os.getenv("RESEARCH_AGENT_STORAGE_BACKEND", "sqlite")).lower()
    if backend_name == "sqlite":
        database_path = _resolve_sqlite_path(sqlite_path)
        database = SQLiteDatabase(database_path)
        return RepositoryBundle(
            sessions=SQLiteSessionRepository(database),
            messages=SQLiteMessageRepository(database),
            papers=SQLitePaperRepository(database),
            artifacts=SQLiteArtifactRepository(database),
            memories=SQLiteMemoryRepository(database),
            trace=SQLiteTraceRepository(database),
            timeline=SQLiteTimelineRepository(database),
            chunks=SQLiteChunkRepository(database),
        )

    return RepositoryBundle(
        sessions=InMemorySessionRepository(),
        messages=InMemoryMessageRepository(),
        papers=InMemoryPaperRepository(),
        artifacts=InMemoryArtifactRepository(),
        memories=InMemoryMemoryRepository(),
        trace=InMemoryTraceRepository(),
        timeline=InMemoryTimelineRepository(),
        chunks=InMemoryChunkRepository(),
    )


def _resolve_sqlite_path(sqlite_path: str | Path | None) -> str:
    if sqlite_path is not None:
        return str(sqlite_path)
    env_path = os.getenv("RESEARCH_AGENT_SQLITE_PATH")
    if env_path:
        return env_path
    return str(Path(__file__).resolve().parents[4] / "data" / "sqlite" / "research_agent.sqlite3")


def create_service_bundle(repositories: RepositoryBundle) -> ServiceBundle:
    """Create thin application services on top of repository ports."""

    runtime_event_broker = RuntimeEventBroker()
    openviking_bundle = _create_openviking_surface_bundle()
    openviking_retrieval_adapter = OpenVikingRetrievalAdapter(
        session_repository=repositories.sessions,
        memory_repository=repositories.memories,
        memory_surface=openviking_bundle.memories,
    )
    task_run_service = TaskRunService(
        session_repository=repositories.sessions,
        message_repository=repositories.messages,
        trace_repository=repositories.trace,
    )
    deletion_service = DeletionService(
        session_repository=repositories.sessions,
        message_repository=repositories.messages,
        memory_repository=repositories.memories,
        trace_repository=repositories.trace,
        timeline_repository=repositories.timeline,
        openviking_bundle=openviking_bundle,
    )
    retrieval_service = RetrievalService(
        session_repository=repositories.sessions,
        memory_repository=repositories.memories,
        chunk_repository=repositories.chunks,
        openviking_retrieval_adapter=openviking_retrieval_adapter,
    )
    context_rerank_service = ContextRerankService()
    memory_extraction_service = MemoryExtractionService(
        session_repository=repositories.sessions,
        paper_repository=repositories.papers,
        chunk_repository=repositories.chunks,
        memory_repository=repositories.memories,
        analysis_service=IngestAnalysisService(
            session_repository=repositories.sessions,
            paper_repository=repositories.papers,
            chunk_repository=repositories.chunks,
            memory_repository=repositories.memories,
            extraction_client=_create_ingest_extraction_client(),
        ),
        openviking_gateway=_create_openviking_memory_gateway(openviking_bundle),
    )
    tool_registry = InternalToolRegistry(
        paper_repository=repositories.papers,
        retrieval_service=retrieval_service,
        context_rerank_service=context_rerank_service,
        memory_extraction_service=memory_extraction_service,
        openviking_retrieval_adapter=openviking_retrieval_adapter,
    )
    query_tool_executor = QueryToolExecutor(tool_registry)
    query_tool_planner = _create_query_tool_planner()
    query_agent_client = _create_query_agent_client()
    query_execution_service = QueryExecutionService(
        message_repository=repositories.messages,
        retrieval_service=retrieval_service,
        context_rerank_service=context_rerank_service,
        session_repository=repositories.sessions,
        trace_repository=repositories.trace,
        timeline_repository=repositories.timeline,
        tool_registry=tool_registry,
        query_tool_executor=query_tool_executor,
        query_tool_planner=query_tool_planner,
        query_agent_client=query_agent_client,
        openviking_bundle=openviking_bundle,
        runtime_event_broker=runtime_event_broker,
    )
    ingest_execution_service = IngestExecutionService(
        message_repository=repositories.messages,
        materialization_service=IngestMaterializationService(
            session_repository=repositories.sessions,
            paper_repository=repositories.papers,
            artifact_repository=repositories.artifacts,
            chunk_repository=repositories.chunks,
            tool_registry=tool_registry,
        ),
        memory_extraction_service=memory_extraction_service,
        trace_repository=repositories.trace,
        timeline_repository=repositories.timeline,
        tool_registry=tool_registry,
        openviking_bundle=openviking_bundle,
        runtime_event_broker=runtime_event_broker,
    )
    task_runtime_service = TaskRuntimeService(
        task_run_service=task_run_service,
        query_execution_service=query_execution_service,
        ingest_execution_service=ingest_execution_service,
        trace_repository=repositories.trace,
        runtime_event_broker=runtime_event_broker,
    )

    return ServiceBundle(
        sessions=SessionService(session_repository=repositories.sessions),
        message_intake=MessageIntakeService(
            task_run_service=task_run_service,
            openviking_bundle=openviking_bundle,
        ),
        messages=MessageQueryService(
            session_repository=repositories.sessions,
            message_repository=repositories.messages,
        ),
        timeline=TimelineQueryService(
            session_repository=repositories.sessions,
            timeline_repository=repositories.timeline,
        ),
        memory_snapshot=MemorySnapshotService(
            session_repository=repositories.sessions,
            memory_repository=repositories.memories,
        ),
        task_runs=task_run_service,
        deletions=deletion_service,
        tools=tool_registry,
        retrieval=retrieval_service,
        context_rerank=context_rerank_service,
        query_execution=query_execution_service,
        task_stream=TaskRunStreamingService(
            task_run_service=task_run_service,
            task_runtime_service=task_runtime_service,
            runtime_event_broker=runtime_event_broker,
        ),
        task_runtime=task_runtime_service,
        query_runtime=task_runtime_service,
        trace=TraceQueryService(
            session_repository=repositories.sessions,
            trace_repository=repositories.trace,
        ),
    )


def _create_query_tool_planner():
    return HeuristicQueryToolPlannerClient()


def _create_ingest_extraction_client():
    backend_name = os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND", "heuristic").lower()
    if backend_name != "model_adapter":
        return None
    provider = os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_PROVIDER", "deepseek").lower()
    model = os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_MODEL", "deepseek-v4-flash")
    timeout_seconds = float(os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_TIMEOUT_SECONDS", "30"))
    if provider == "deepseek":
        transport = DeepSeekStructuredIngestExtractionTransport(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model=model,
            base_url=os.getenv("RESEARCH_AGENT_INGEST_EXTRACTION_BASE_URL", "https://api.deepseek.com"),
            timeout_seconds=timeout_seconds,
        )
        return ModelBackedIngestExtractionClient(transport=transport)
    return None


def _create_query_agent_client():
    agent_backend = os.getenv("RESEARCH_AGENT_QUERY_AGENT_BACKEND", "turn_adapter").lower()
    if agent_backend == "pydantic_ai":
        provider = os.getenv("RESEARCH_AGENT_QUERY_AGENT_PROVIDER", "deepseek").lower()
        model = os.getenv("RESEARCH_AGENT_QUERY_AGENT_MODEL", "deepseek-v4-flash")
        base_url = os.getenv("RESEARCH_AGENT_QUERY_AGENT_BASE_URL", "https://api.deepseek.com")
        fallback = PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient())
        if provider == "deepseek":
            return PydanticAIQueryTurnClient(
                model=model,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url,
                fallback=fallback,
                framework_name="pydantic_ai",
            )
        return fallback

    backend_name = os.getenv("RESEARCH_AGENT_QUERY_PLANNER_BACKEND", "heuristic").lower()
    heuristic = PlannerBackedQueryAgentClient(HeuristicQueryToolPlannerClient())
    if backend_name == "model_adapter":
        provider = os.getenv("RESEARCH_AGENT_QUERY_PLANNER_PROVIDER", "unconfigured")
        model = os.getenv("RESEARCH_AGENT_QUERY_PLANNER_MODEL", "unconfigured")
        timeout_seconds = float(os.getenv("RESEARCH_AGENT_QUERY_PLANNER_TIMEOUT_SECONDS", "30"))
        if provider == "deepseek":
            transport = DeepSeekStructuredQueryAgentTransport(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                model=model,
                base_url=os.getenv("RESEARCH_AGENT_QUERY_PLANNER_BASE_URL", "https://api.deepseek.com"),
                timeout_seconds=timeout_seconds,
            )
            agent_name = f"deepseek:{transport.normalized_model}"
        else:
            transport = UnavailableStructuredQueryAgentTransport()
            agent_name = f"{provider}:{model}"
        return ModelBackedQueryAgentClient(
            transport=transport,
            fallback=heuristic,
            agent_name=agent_name,
        )
    return heuristic


def _create_openviking_surface_bundle() -> OpenVikingAdapterSurfaceBundle:
    backend_name = os.getenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "noop").lower()
    if backend_name == "inmemory":
        return build_inmemory_openviking_surface_bundle()
    if backend_name == "embedded":
        try:
            openviking_config_file = _resolve_openviking_config_file()
            os.environ.setdefault("OPENVIKING_CONFIG_FILE", openviking_config_file)
            return build_embedded_openviking_surface_bundle(
                OpenVikingSurfaceConfig(
                    enabled=True,
                    path=os.getenv(
                        "RESEARCH_AGENT_OPENVIKING_DATA_PATH",
                        str(Path(__file__).resolve().parents[4] / "data" / "openviking"),
                    ),
                )
            )
        except Exception:
            return OpenVikingAdapterSurfaceBundle()
    if backend_name not in {"mirror", "sdk"}:
        return OpenVikingAdapterSurfaceBundle()
    try:
        return build_sdk_openviking_surface_bundle(
            OpenVikingSurfaceConfig(
                enabled=True,
                url=os.getenv("RESEARCH_AGENT_OPENVIKING_URL", "http://127.0.0.1:1933"),
                api_key=os.getenv("RESEARCH_AGENT_OPENVIKING_API_KEY"),
            )
        )
    except Exception:
        return OpenVikingAdapterSurfaceBundle()


def _create_openviking_memory_gateway(openviking_bundle: OpenVikingAdapterSurfaceBundle):
    backend_name = os.getenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "noop").lower()
    if backend_name in {"mirror", "sdk", "inmemory"}:
        return SurfaceBackedOpenVikingMemoryGateway(
            memory_surface=openviking_bundle.memories,
            session_surface=openviking_bundle.sessions,
        )
    if backend_name != "legacy_http":
        return NoopOpenVikingMemoryGateway()
    try:
        return HttpOpenVikingMemoryGateway(
            OpenVikingMemoryGatewayConfig(
                enabled=True,
                url=os.getenv("RESEARCH_AGENT_OPENVIKING_URL", "http://127.0.0.1:1933"),
                api_key=os.getenv("RESEARCH_AGENT_OPENVIKING_API_KEY"),
                agent_id=os.getenv("RESEARCH_AGENT_OPENVIKING_AGENT_ID", "research-agent"),
            )
        )
    except Exception:
        return NoopOpenVikingMemoryGateway()


def _resolve_openviking_config_file() -> str:
    """Resolve the repo-local OpenViking config path for embedded mode."""

    configured = os.getenv("RESEARCH_AGENT_OPENVIKING_CONFIG_FILE")
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[4] / "data" / "openviking" / "ov.conf")


def get_repository_bundle(request: Request) -> RepositoryBundle:
    """Resolve the app-scoped repository bundle."""

    return request.app.state.repositories


def get_service_bundle(request: Request) -> ServiceBundle:
    """Resolve the app-scoped service bundle."""

    return request.app.state.services
