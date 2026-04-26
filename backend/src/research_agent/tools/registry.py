"""Internal tool registry for runtime-invoked paper research actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from research_agent.adapters.openviking.retrieval_adapter import OpenVikingMemorySearchResult, OpenVikingRetrievalAdapter
from research_agent.domain.models import Paper
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.ports import PaperRepositoryPort

if TYPE_CHECKING:
    from research_agent.services.context_rerank_service import (
        ChunkRerankResult,
        ContextRerankService,
        MemoryRerankResult,
    )
    from research_agent.services.memory_extraction_service import MemoryExtractionResult, MemoryExtractionService
    from research_agent.services.retrieval_service import MemoryRetrievalResult, RetrievalService, SourceRereadResult


@dataclass(frozen=True, slots=True)
class RegistryToolEntry:
    """Internal registry entry for a tool implementation."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class PaperRegistrationResult:
    """Structured result for paper registration."""

    paper: Paper
    operation: str


class InternalToolRegistry:
    """Registry for the first batch of internal, runtime-callable tools."""

    def __init__(
        self,
        *,
        paper_repository: PaperRepositoryPort,
        retrieval_service: RetrievalService,
        context_rerank_service: ContextRerankService,
        memory_extraction_service: MemoryExtractionService,
        openviking_retrieval_adapter: OpenVikingRetrievalAdapter | None = None,
    ) -> None:
        self._paper_repository = paper_repository
        self._retrieval_service = retrieval_service
        self._context_rerank_service = context_rerank_service
        self._memory_extraction_service = memory_extraction_service
        self._openviking_retrieval_adapter = openviking_retrieval_adapter
        self._definitions = {
            "register_paper": RegistryToolEntry("register_paper", "Register or match a canonical paper record."),
            "extract_memories": RegistryToolEntry("extract_memories", "Extract paper, relation, and open-question memories."),
            "search_openviking_memory": RegistryToolEntry("search_openviking_memory", "Search OpenViking-backed memory and return bounded hits plus local-mapping metadata."),
            "search_session_memory": RegistryToolEntry("search_session_memory", "Search memories scoped to the current session."),
            "search_global_memory": RegistryToolEntry("search_global_memory", "Search globally stored memories."),
            "search_source_chunks": RegistryToolEntry("search_source_chunks", "Retrieve source chunks for a paper-backed reread."),
            "rerank_candidates": RegistryToolEntry("rerank_candidates", "Rerank a bounded candidate pool with fallback."),
            "read_source_passages": RegistryToolEntry("read_source_passages", "Select source passages from stored chunks."),
            "compose_answer": RegistryToolEntry("compose_answer", "Compose a mock answer from selected memories and reread chunks."),
        }

    def list_tools(self) -> tuple[RegistryToolEntry, ...]:
        """Return the registered tool definitions."""

        return tuple(self._definitions.values())

    def get_tool(self, name: str) -> RegistryToolEntry:
        """Return a single tool definition."""

        if name not in self._definitions:
            raise KeyError(name)
        return self._definitions[name]

    def invoke(self, tool_name: str, **kwargs: Any) -> Any:
        """Invoke a tool by registry name."""

        if tool_name == "register_paper":
            return self.register_paper(**kwargs)
        if tool_name == "extract_memories":
            return self.extract_memories(**kwargs)
        if tool_name == "search_openviking_memory":
            return self.search_openviking_memory(**kwargs)
        if tool_name == "search_session_memory":
            return self.search_session_memory(**kwargs)
        if tool_name == "search_global_memory":
            return self.search_global_memory(**kwargs)
        if tool_name == "search_source_chunks":
            return self.search_source_chunks(**kwargs)
        if tool_name == "rerank_candidates":
            return self.rerank_candidates(**kwargs)
        if tool_name == "read_source_passages":
            return self.read_source_passages(**kwargs)
        if tool_name == "compose_answer":
            return self.compose_answer(**kwargs)
        raise KeyError(tool_name)

    def register_paper(
        self,
        *,
        title: str,
        authors: list[str] | None = None,
        arxiv_id: str | None = None,
        pdf_fingerprint: str | None = None,
        checksum: str | None = None,
    ) -> PaperRegistrationResult:
        """Create or match a canonical paper record."""

        canonical_key = build_canonical_key(arxiv_id=arxiv_id, pdf_checksum=pdf_fingerprint or checksum)
        existing = self._paper_repository.get_by_canonical_key(canonical_key)
        if existing is not None:
            updated = existing.model_copy(
                update={
                    "title": existing.title or title,
                    "authors": existing.authors or (authors or existing.authors),
                    "arxiv_id": existing.arxiv_id or arxiv_id,
                    "pdf_fingerprint": existing.pdf_fingerprint or pdf_fingerprint or checksum,
                }
            )
            return PaperRegistrationResult(paper=self._paper_repository.save(updated), operation="matched")

        paper = self._paper_repository.save(
            Paper(
                id=str(uuid4()),
                canonical_key=canonical_key,
                title=title,
                authors=authors or [],
                arxiv_id=arxiv_id,
                pdf_fingerprint=pdf_fingerprint or checksum,
            )
        )
        return PaperRegistrationResult(paper=paper, operation="created")

    def extract_memories(self, session_id: str, paper_id: str) -> MemoryExtractionResult:
        """Extract and store all first-pass memory types for a paper."""

        return self._memory_extraction_service.extract_and_store_memories(session_id=session_id, paper_id=paper_id)

    def search_session_memory(self, session_id: str, query: str, top_k: int) -> MemoryRetrievalResult:
        """Search session-scoped memories."""

        return self._retrieval_service.retrieve_session_memories(session_id=session_id, query=query, top_k=top_k)

    def search_openviking_memory(
        self,
        *,
        scope: str,
        query: str,
        session_id: str | None = None,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> OpenVikingMemorySearchResult:
        """Search OpenViking explicitly through its retrieval adapter."""

        adapter = self._openviking_retrieval_adapter
        if adapter is None:
            return OpenVikingMemorySearchResult(
                scope=scope,
                hits=(),
                memory_descriptors=(),
                matched_local_memory_ids=(),
                matched_local_count=0,
            )
        if scope == "session":
            if session_id is None:
                raise ValueError("session_id is required for session-scoped OpenViking search")
            return adapter.search_session_memory(session_id=session_id, query=query, top_k=top_k)
        if scope == "global":
            return adapter.search_global_memory(
                query=query,
                related_paper_ids=related_paper_ids,
                top_k=top_k,
            )
        raise ValueError(f"Unsupported OpenViking search scope: {scope}")

    def search_global_memory(
        self,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None,
        top_k: int,
    ) -> MemoryRetrievalResult:
        """Search global memories with an optional paper filter."""

        return self._retrieval_service.retrieve_global_memories(
            query=query,
            related_paper_ids=related_paper_ids,
            top_k=top_k,
        )

    def search_source_chunks(
        self,
        session_id: str,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None,
        top_k: int,
    ) -> SourceRereadResult:
        """Retrieve candidate chunks for a source reread."""

        return self._retrieval_service.retrieve_source_passages(
            session_id=session_id,
            query=query,
            related_paper_ids=related_paper_ids,
            top_k=top_k,
        )

    def rerank_candidates(
        self,
        *,
        candidate_kind: str,
        query: str,
        candidates,
        top_k: int,
    ) -> MemoryRerankResult | ChunkRerankResult:
        """Rerank a bounded candidate pool."""

        if candidate_kind == "memory":
            return self._context_rerank_service.rerank_memories(query=query, candidates=candidates, top_k=top_k)
        if candidate_kind == "chunk":
            return self._context_rerank_service.rerank_chunks(query=query, candidates=candidates, top_k=top_k)
        raise ValueError(f"Unsupported candidate kind: {candidate_kind}")

    def read_source_passages(
        self,
        *,
        session_id: str,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None,
        top_k: int,
    ) -> ChunkRerankResult:
        """Select the final source passages from stored chunks."""

        candidates = self.search_source_chunks(
            session_id=session_id,
            query=query,
            related_paper_ids=related_paper_ids,
            top_k=max(top_k, 10),
        )
        return self._context_rerank_service.rerank_chunks(query=query, candidates=candidates.chunks, top_k=top_k)

    def compose_answer(
        self,
        *,
        query: str,
        session_memory_count: int,
        global_memory_count: int,
        memory_selection_source: str,
        memory_selection_fallback_used: bool,
        should_reread_source: bool,
        reread_reason: str,
        used_memory_citations,
        source_reread_chunks,
        source_selection_source: str | None,
    ) -> str:
        """Compose the final answer in Chinese from retrieved context."""

        memory_notes = [self._trim_answer_text(citation.summary, 120) for citation in used_memory_citations if self._trim_answer_text(citation.summary, 120)]
        source_notes = [
            self._format_source_note(citation.page, citation.section, citation.excerpt)
            for citation in source_reread_chunks
            if self._trim_answer_text(citation.excerpt, 140)
        ]

        if memory_notes:
            lead = f"根据当前记忆，关于「{self._trim_answer_text(query, 80)}」可以先概括为：{memory_notes[0]}。"
            if len(memory_notes) > 1:
                lead += f"补充记忆还包括：{'；'.join(memory_notes[1:3])}。"
        else:
            lead = f"当前关于「{self._trim_answer_text(query, 80)}」的记忆还不够完整，暂时只能给出保守回答。"

        if source_notes:
            lead += f"原文回读到的关键片段包括：{'；'.join(source_notes[:2])}。"

        if should_reread_source:
            lead += f"系统仍建议继续回读原文，原因是：{self._trim_answer_text(reread_reason, 120)}。"
        else:
            lead += "当前记忆已经足以直接回答。"

        if memory_selection_fallback_used:
            lead += " 这次记忆选择包含规则兜底。"
        if source_selection_source:
            lead += f" 原文选择方式：{self._trim_answer_text(source_selection_source, 40)}。"
        return lead.strip()

    def _trim_answer_text(self, text: str | None, limit: int) -> str:
        cleaned = (text or "").strip().replace("\n", " ")
        if not cleaned:
            return ""
        return cleaned[:limit]

    def _format_source_note(self, page: int | None, section: str | None, excerpt: str) -> str:
        location = []
        if page is not None:
            location.append(f"第{page}页")
        if section:
            location.append(section)
        location_text = " ".join(location) if location else "原文片段"
        excerpt_text = self._trim_answer_text(excerpt, 120)
        return f"{location_text}：{excerpt_text}"


__all__ = [
    "InternalToolRegistry",
    "PaperRegistrationResult",
    "RegistryToolEntry",
]
