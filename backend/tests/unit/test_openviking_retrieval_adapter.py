"""Tests for the explicit OpenViking retrieval and mapping boundary."""

from __future__ import annotations

from research_agent.adapters.openviking import (
    OpenVikingMemoryRecord,
    OpenVikingRetrievalAdapter,
    build_inmemory_openviking_surface_bundle,
)
from research_agent.adapters.storage import (
    InMemoryChunkRepository,
    InMemoryMemoryRepository,
    InMemoryPaperRepository,
    InMemorySessionRepository,
)
from research_agent.domain.enums import SourceType
from research_agent.domain.models import OpenQuestionMemory, PaperMemory, SessionDocument
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.services import ContextRerankService, MemoryExtractionService, RetrievalService, SessionService
from research_agent.tools import InternalToolRegistry


def test_openviking_retrieval_adapter_maps_session_hits_to_local_memory_ids() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("OV Adapter")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_paper_memory(
        PaperMemory(
            id="paper-memory-1",
            paper_id="paper-1",
            key_results=["accuracy improved over baseline"],
            confidence=ConfidenceScore(value=0.8),
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="paper-memory-1",
            memory_kind="paper_memory",
            session_id=session.id,
            paper_id="paper-1",
            payload={"key_results": ["accuracy improved over baseline"]},
        )
    )
    adapter = OpenVikingRetrievalAdapter(
        session_repository=session_repository,
        memory_repository=memory_repository,
        memory_surface=bundle.memories,
    )

    result = adapter.search_session_memory(session_id=session.id, query="accuracy baseline", top_k=5)

    assert result.scope == "session"
    assert result.matched_local_memory_ids == ("paper-memory-1",)
    assert result.matched_local_count == 1
    assert result.hits[0].item_id == "paper-memory-1"
    assert result.memory_descriptors[0].memory_id == "paper-memory-1"


def test_internal_tool_registry_exposes_search_openviking_memory() -> None:
    session_repository = InMemorySessionRepository()
    memory_repository = InMemoryMemoryRepository()
    chunk_repository = InMemoryChunkRepository()
    paper_repository = InMemoryPaperRepository()
    bundle = build_inmemory_openviking_surface_bundle()
    session = SessionService(session_repository=session_repository).create_session("OV Tool")
    session_repository.save(session)
    session_repository.save_document(
        SessionDocument(
            session_id=session.id,
            paper_id="paper-1",
            source_type=SourceType.PDF,
            artifact_id="artifact-1",
        )
    )
    memory_repository.upsert_open_question_memory(
        OpenQuestionMemory(
            id="open-question-1",
            unresolved_question="Why does the accuracy delta remain open?",
            related_papers=["paper-1"],
            confidence=ConfidenceScore(value=0.6),
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="open-question-1",
            memory_kind="open_question_memory",
            session_id=session.id,
            paper_id="paper-1",
            payload={"unresolved_question": "accuracy delta remains open"},
        )
    )
    adapter = OpenVikingRetrievalAdapter(
        session_repository=session_repository,
        memory_repository=memory_repository,
        memory_surface=bundle.memories,
    )
    registry = InternalToolRegistry(
        paper_repository=paper_repository,
        retrieval_service=RetrievalService(
            session_repository=session_repository,
            memory_repository=memory_repository,
            chunk_repository=chunk_repository,
            openviking_retrieval_adapter=adapter,
        ),
        context_rerank_service=ContextRerankService(),
        memory_extraction_service=MemoryExtractionService(
            session_repository=session_repository,
            paper_repository=paper_repository,
            chunk_repository=chunk_repository,
            memory_repository=memory_repository,
        ),
        openviking_retrieval_adapter=adapter,
    )

    result = registry.search_openviking_memory(
        scope="session",
        session_id=session.id,
        query="accuracy delta",
        top_k=5,
    )

    assert any(entry.name == "search_openviking_memory" for entry in registry.list_tools())
    assert result.scope == "session"
    assert result.matched_local_memory_ids == ("open-question-1",)
    assert result.hits[0].item_id == "open-question-1"
    assert result.memory_descriptors[0].memory_id == "open-question-1"
