"""Internal tool registry for runtime-invoked paper research actions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pathlib import Path

from research_agent.adapters.openviking.retrieval_adapter import OpenVikingMemorySearchResult, OpenVikingRetrievalAdapter

from research_agent.domain.enums import MessageType
from research_agent.domain.models import Message, Paper, TaskRun, TraceStep
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.ports import ArtifactRepositoryPort, ChunkRepositoryPort, MemoryRepositoryPort, MessageRepositoryPort, PaperRepositoryPort, SessionRepositoryPort, TraceRepositoryPort
from research_agent.tools.protocol import (
    ConversationEvidenceRefDescriptor,
    ChunkDescriptor,
    GetConversationContextOutput,
    GetPaperMemoryBundleOutput,
    ListRecentMessagesOutput,
    ListSessionPapersOutput,
    PaperInfoDescriptor,
    PaperMemoryBundleDescriptor,
    RecentConversationContextDescriptor,
    RecentConversationMessageDescriptor,
    SessionPaperDescriptor,
)

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
        session_repository: SessionRepositoryPort | None = None,
        message_repository: MessageRepositoryPort | None = None,
        trace_repository: TraceRepositoryPort | None = None,
        memory_repository: MemoryRepositoryPort | None = None,
        chunk_repository: ChunkRepositoryPort | None = None,
        artifact_repository: ArtifactRepositoryPort | None = None,
    ) -> None:
        self._paper_repository = paper_repository
        self._retrieval_service = retrieval_service
        self._context_rerank_service = context_rerank_service
        self._memory_extraction_service = memory_extraction_service
        self._openviking_retrieval_adapter = openviking_retrieval_adapter
        self._session_repository = session_repository
        self._message_repository = message_repository
        self._trace_repository = trace_repository
        self._memory_repository = memory_repository
        self._chunk_repository = chunk_repository
        self._artifact_repository = artifact_repository
        self._definitions = {
            "register_paper": RegistryToolEntry("register_paper", "Register or match a canonical paper record."),
            "extract_memories": RegistryToolEntry("extract_memories", "Extract paper, relation, and open-question memories."),
            "search_openviking_memory": RegistryToolEntry("search_openviking_memory", "Search OpenViking-backed memory and return bounded hits plus local-mapping metadata."),
            "search_session_memory": RegistryToolEntry("search_session_memory", "Search memories scoped to the current session."),
            "search_global_memory": RegistryToolEntry("search_global_memory", "Search globally stored memories."),
            "search_source_chunks": RegistryToolEntry("search_source_chunks", "Retrieve source chunks for a paper-backed reread."),
            "list_recent_messages": RegistryToolEntry("list_recent_messages", "List the most recent follow-up conversation messages for the current session."),
            "get_conversation_context": RegistryToolEntry("get_conversation_context", "Return the compact recent conversation context for the current session."),
            "rerank_candidates": RegistryToolEntry("rerank_candidates", "Rerank a bounded candidate pool with fallback."),
            "read_source_passages": RegistryToolEntry("read_source_passages", "Select source passages from stored chunks."),
            "compose_answer": RegistryToolEntry("compose_answer", "Package evidence for the model's final answer."),
            "list_session_papers": RegistryToolEntry("list_session_papers", "List papers/documents imported into the current session."),
            "get_paper_memory_bundle": RegistryToolEntry("get_paper_memory_bundle", "Return memory and evidence for one paper."),
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
        if tool_name == "list_recent_messages":
            return self.list_recent_messages(**kwargs)
        if tool_name == "get_conversation_context":
            return self.get_conversation_context(**kwargs)
        if tool_name == "rerank_candidates":
            return self.rerank_candidates(**kwargs)
        if tool_name == "read_source_passages":
            return self.read_source_passages(**kwargs)
        if tool_name == "compose_answer":
            return self.compose_answer(**kwargs)
        if tool_name == "list_session_papers":
            return self.list_session_papers(**kwargs)
        if tool_name == "get_paper_memory_bundle":
            return self.get_paper_memory_bundle(**kwargs)
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

    def list_session_papers(self, *, session_id: str, limit: int = 20) -> ListSessionPapersOutput:
        """List papers/documents imported into a session."""

        self._require_query_repositories()
        documents = list(self._session_repository.list_documents(session_id))[:limit]  # type: ignore[union-attr]
        papers_by_id = {paper.id: paper for paper in self._paper_repository.list_by_ids([document.paper_id for document in documents])}
        paper_memories = self._memory_repository.list_paper_memories_for_papers([document.paper_id for document in documents])  # type: ignore[union-attr]
        relation_memories = self._memory_repository.list_relation_memories_for_papers([document.paper_id for document in documents])  # type: ignore[union-attr]
        open_question_memories = self._memory_repository.list_open_question_memories_for_papers([document.paper_id for document in documents])  # type: ignore[union-attr]
        memory_count_by_paper = {document.paper_id: 0 for document in documents}
        for memory in paper_memories:
            memory_count_by_paper[memory.paper_id] = memory_count_by_paper.get(memory.paper_id, 0) + 1
        for memory in relation_memories:
            memory_count_by_paper[memory.source_paper] = memory_count_by_paper.get(memory.source_paper, 0) + 1
            memory_count_by_paper[memory.target_paper] = memory_count_by_paper.get(memory.target_paper, 0) + 1
        for memory in open_question_memories:
            for paper_id in memory.related_papers:
                memory_count_by_paper[paper_id] = memory_count_by_paper.get(paper_id, 0) + 1

        return ListSessionPapersOutput(
            papers=tuple(
                SessionPaperDescriptor(
                    paper_id=document.paper_id,
                    title=papers_by_id[document.paper_id].title if document.paper_id in papers_by_id else document.paper_id,
                    file_name=self._artifact_file_name(document.artifact_id),
                    created_at=document.added_at.isoformat(),
                    memory_count=memory_count_by_paper.get(document.paper_id, 0),
                    summary_status="available" if memory_count_by_paper.get(document.paper_id, 0) else "missing",
                )
                for document in documents
            ),
            total_count=len(documents),
        )

    def get_paper_memory_bundle(self, *, paper_id: str, source_chunk_limit: int = 5) -> GetPaperMemoryBundleOutput:
        """Return all query-visible memory and evidence for one paper."""

        self._require_query_repositories()
        paper = self._paper_repository.get_by_id(paper_id)
        if paper is None:
            raise ValueError(f"paper not found: {paper_id}")
        documents = [document for session in self._session_repository.list_all() for document in self._session_repository.list_documents(session.id)]  # type: ignore[union-attr]
        document = next((item for item in documents if item.paper_id == paper_id), None)
        paper_memories = tuple(self._memory_repository.list_paper_memories_for_papers([paper_id]))  # type: ignore[union-attr]
        open_questions = tuple(self._memory_repository.list_open_question_memories_for_papers([paper_id]))  # type: ignore[union-attr]
        relations = tuple(self._memory_repository.list_relation_memories_for_papers([paper_id]))  # type: ignore[union-attr]
        chunks = tuple(self._chunk_repository.list_by_paper_ids([paper_id]))[:source_chunk_limit]  # type: ignore[union-attr]
        empty_fields = []
        if not paper_memories:
            empty_fields.append("paper_memory")
        if not open_questions:
            empty_fields.append("open_questions")
        if not relations:
            empty_fields.append("relations")
        if not chunks:
            empty_fields.append("evidence_source_chunks")

        return GetPaperMemoryBundleOutput(
            bundle=PaperMemoryBundleDescriptor(
                paper=PaperInfoDescriptor(
                    paper_id=paper.id,
                    title=paper.title,
                    authors=tuple(paper.authors),
                    abstract=paper.abstract,
                    year=paper.year,
                    arxiv_id=paper.arxiv_id,
                    file_name=self._artifact_file_name(document.artifact_id) if document is not None else None,
                    created_at=document.added_at.isoformat() if document is not None else None,
                ),
                paper_memory=paper_memories[0].model_dump(mode="python") if paper_memories else None,
                open_questions=tuple(memory.model_dump(mode="python") for memory in open_questions),
                relations=tuple(memory.model_dump(mode="python") for memory in relations),
                evidence_source_chunks=tuple(
                    ChunkDescriptor(
                        chunk_id=chunk.id,
                        paper_id=chunk.paper_id,
                        excerpt=self._trim_answer_text(chunk.text, 220),
                        page=chunk.page,
                        section=chunk.section,
                        matched_terms=(),
                        selection_reason="paper_memory_bundle_evidence",
                    )
                    for chunk in chunks
                ),
                empty_fields=tuple(empty_fields),
            )
        )

    def list_recent_messages(
        self,
        *,
        session_id: str,
        limit: int = 8,
        exclude_message_id: str | None = None,
    ) -> ListRecentMessagesOutput:
        """List the most recent follow-up conversation messages for a session."""

        context = self._build_recent_conversation_context(
            session_id=session_id,
            limit=limit,
            exclude_message_id=exclude_message_id,
        )
        messages = self._flatten_recent_conversation_messages(context)
        return ListRecentMessagesOutput(
            messages=tuple(messages),
            total_count=context.recent_message_count,
            window_count=len(messages),
        )

    def get_conversation_context(
        self,
        *,
        session_id: str,
        limit: int = 8,
        exclude_message_id: str | None = None,
    ) -> GetConversationContextOutput:
        """Return the compact recent conversation context for a session."""

        return GetConversationContextOutput(
            context=self._build_recent_conversation_context(
                session_id=session_id,
                limit=limit,
                exclude_message_id=exclude_message_id,
            ),
        )

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
        """Package selected evidence for the model without drafting a final answer."""

        memory_notes = [
            self._trim_answer_text(citation.summary, 120)
            for citation in used_memory_citations
            if self._trim_answer_text(citation.summary, 120)
        ]
        source_notes = [
            self._format_source_note(citation.page, citation.section, citation.excerpt)
            for citation in source_reread_chunks
            if self._trim_answer_text(citation.excerpt, 140)
        ]

        package_parts = [
            f"query={self._trim_answer_text(query, 80)}",
            f"session_memory_count={session_memory_count}",
            f"global_memory_count={global_memory_count}",
            f"memory_selection_source={memory_selection_source}",
            f"memory_selection_fallback_used={memory_selection_fallback_used}",
            f"should_reread_source={should_reread_source}",
            f"reread_reason={self._trim_answer_text(reread_reason, 120)}",
            f"memory_notes={memory_notes[:3]}",
            f"source_notes={source_notes[:2]}",
        ]
        if source_selection_source:
            package_parts.append(f"source_selection_source={self._trim_answer_text(source_selection_source, 40)}")
        return "Evidence package for model answer: " + "; ".join(package_parts)

    def _build_recent_conversation_context(
        self,
        *,
        session_id: str,
        limit: int,
        exclude_message_id: str | None = None,
    ) -> RecentConversationContextDescriptor:
        if self._session_repository is None:
            return RecentConversationContextDescriptor()
        query_messages = []
        if self._message_repository is not None:
            query_messages = [
                message
                for message in self._message_repository.list_by_session(session_id)
                if message.type == MessageType.FOLLOWUP_QUERY and message.id != exclude_message_id
            ]
        if limit > 0:
            query_messages = query_messages[-limit:]

        runs_by_message_id = {}
        if self._trace_repository is not None:
            runs_by_message_id = {
                run.message_id: run
                for run in self._trace_repository.list_runs_by_session(session_id)
            }
        conversation_turns = self._build_conversation_turns(
            session_id=session_id,
            messages=query_messages,
            runs_by_message_id=runs_by_message_id,
        )
        recent_user_messages = tuple(
            turn["user_message"]
            for turn in conversation_turns
            if turn["user_message"] is not None
        )
        recent_assistant_answers = tuple(
            turn["assistant_message"]
            for turn in conversation_turns
            if turn["assistant_message"] is not None
        )
        active_turn = next(
            (turn for turn in reversed(conversation_turns) if turn.get("paper_id") or turn.get("last_answer_summary") or turn.get("active_topic")),
            None,
        )
        active_paper_id = active_turn.get("paper_id") if active_turn is not None else self._latest_session_paper_id(session_id)
        active_paper_file_name = active_turn.get("paper_file_name") if active_turn is not None else self._latest_session_paper_file_name(session_id)
        active_topic = active_turn.get("active_topic") if active_turn is not None else None
        last_answer_summary = active_turn.get("last_answer_summary") if active_turn is not None else None
        last_evidence_refs = active_turn.get("evidence_refs") if active_turn is not None else ()
        if active_paper_id is None:
            active_paper_id = self._latest_session_paper_id(session_id)
        if active_paper_file_name is None and active_paper_id is not None:
            active_paper_file_name = self._latest_session_paper_file_name(session_id)

        return RecentConversationContextDescriptor(
            recent_user_messages=recent_user_messages,
            recent_assistant_answers=recent_assistant_answers,
            active_paper_id=active_paper_id,
            active_paper_file_name=active_paper_file_name,
            active_topic=active_topic,
            last_answer_summary=last_answer_summary,
            last_evidence_refs=tuple(last_evidence_refs),
            recent_message_count=len(query_messages),
            recent_turn_count=len(conversation_turns),
        )

    def _build_conversation_turns(
        self,
        *,
        session_id: str,
        messages: Sequence[Message],
        runs_by_message_id: dict[str, TaskRun],
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                run = runs_by_message_id.get(message.id)
                turns.append(
                    {
                        "user_message": self._conversation_message_descriptor(
                            message,
                            run_id=run.id if run is not None else None,
                        ),
                        "assistant_message": None,
                        "paper_id": None,
                        "paper_file_name": None,
                        "active_topic": self._trim_answer_text(message.content, 160),
                        "last_answer_summary": None,
                        "evidence_refs": (),
                    }
                )
                continue

            if message.role != "assistant":
                continue

            if not turns or turns[-1]["assistant_message"] is not None:
                turns.append(
                    {
                        "user_message": None,
                        "assistant_message": self._conversation_message_descriptor(message),
                        "paper_id": None,
                        "paper_file_name": None,
                        "active_topic": self._trim_answer_text(message.content, 160),
                        "last_answer_summary": self._trim_answer_text(message.content, 240),
                        "evidence_refs": (),
                    }
                )
                continue

            turn = turns[-1]
            user_message = turn["user_message"]
            run = runs_by_message_id.get(user_message.message_id) if user_message is not None else None
            evidence_refs = self._evidence_refs_for_run(session_id=session_id, run=run)
            paper_id = self._paper_id_from_evidence_refs(evidence_refs)
            if paper_id is None:
                paper_id = self._latest_session_paper_id(session_id)
            paper_file_name = self._paper_file_name_for_paper_id(session_id, paper_id)
            turn["assistant_message"] = self._conversation_message_descriptor(
                message,
                paper_id=paper_id,
                run_id=run.id if run is not None else None,
                source_refs=evidence_refs,
            )
            if user_message is not None:
                turn["user_message"] = user_message.model_copy(update={"paper_id": paper_id})
            turn["paper_id"] = paper_id
            turn["paper_file_name"] = paper_file_name
            turn["active_topic"] = self._trim_answer_text(message.content, 160) or turn["active_topic"]
            turn["last_answer_summary"] = self._trim_answer_text(message.content, 240)
            turn["evidence_refs"] = evidence_refs
        return turns

    def _flatten_recent_conversation_messages(self, context: RecentConversationContextDescriptor) -> list[RecentConversationMessageDescriptor]:
        messages = [message for message in (*context.recent_user_messages, *context.recent_assistant_answers)]
        return sorted(messages, key=lambda message: message.created_at)

    def _conversation_message_descriptor(
        self,
        message: Message,
        *,
        paper_id: str | None = None,
        run_id: str | None = None,
        source_refs: Sequence[ConversationEvidenceRefDescriptor] = (),
    ) -> RecentConversationMessageDescriptor:
        return RecentConversationMessageDescriptor(
            message_id=message.id,
            role=message.role if message.role in {"user", "assistant"} else "user",
            content=self._trim_answer_text(message.content, 280),
            created_at=message.created_at.isoformat(),
            paper_id=paper_id,
            run_id=run_id,
            source_refs=tuple(source_refs),
        )

    def _evidence_refs_for_run(
        self,
        *,
        session_id: str,
        run: TaskRun | None,
    ) -> tuple[ConversationEvidenceRefDescriptor, ...]:
        if run is None or self._trace_repository is None:
            return ()
        steps = list(self._trace_repository.list_steps(run.id))
        answer_step = next((step for step in reversed(steps) if step.action == "final_answer"), None)
        if answer_step is None:
            return ()
        result_payload = answer_step.result_payload if isinstance(answer_step.result_payload, dict) else {}
        refs: list[ConversationEvidenceRefDescriptor] = []

        memory_citations = result_payload.get("memory_citations")
        if isinstance(memory_citations, list):
            for citation in memory_citations[:3]:
                if not isinstance(citation, dict):
                    continue
                memory_id = str(citation.get("memory_id") or "").strip()
                if not memory_id:
                    continue
                refs.append(
                    ConversationEvidenceRefDescriptor(
                        ref_type="memory",
                        ref_id=memory_id,
                        paper_id=self._paper_id_for_memory_id(memory_id),
                        summary=self._trim_answer_text(str(citation.get("summary") or ""), 180) or "memory citation",
                        quote=None,
                        page=None,
                        section=None,
                        memory_type=self._trim_answer_text(str(citation.get("memory_type") or ""), 40) or None,
                    )
                )

        source_reread_chunks = result_payload.get("source_reread_chunks")
        if isinstance(source_reread_chunks, list):
            for chunk in source_reread_chunks[:3]:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                excerpt = self._trim_answer_text(str(chunk.get("excerpt") or ""), 180)
                paper_id = self._trim_answer_text(str(chunk.get("paper_id") or ""), 80) or self._latest_session_paper_id(session_id)
                refs.append(
                    ConversationEvidenceRefDescriptor(
                        ref_type="chunk",
                        ref_id=chunk_id,
                        paper_id=paper_id,
                        summary=excerpt or self._trim_answer_text(str(chunk.get("selection_reason") or ""), 180) or "source chunk",
                        quote=excerpt or None,
                        page=chunk.get("page"),
                        section=self._trim_answer_text(str(chunk.get("section") or ""), 80) or None,
                        memory_type=None,
                    )
                )
        return tuple(refs)

    def _paper_id_for_memory_id(self, memory_id: str) -> str | None:
        if self._memory_repository is None:
            return None
        for memory in self._memory_repository.list_all_paper_memories():
            if memory.id == memory_id:
                return memory.paper_id
        for memory in self._memory_repository.list_all_relation_memories():
            if memory.id == memory_id:
                return memory.source_paper
        for memory in self._memory_repository.list_all_open_question_memories():
            if memory.id == memory_id and memory.related_papers:
                return memory.related_papers[0]
        return None

    def _latest_session_paper_id(self, session_id: str) -> str | None:
        if self._session_repository is None:
            return None
        documents = list(self._session_repository.list_documents(session_id))
        if not documents:
            return None
        return documents[-1].paper_id

    def _latest_session_paper_file_name(self, session_id: str) -> str | None:
        paper_id = self._latest_session_paper_id(session_id)
        if paper_id is None:
            return None
        return self._paper_file_name_for_paper_id(session_id, paper_id)

    def _paper_file_name_for_paper_id(self, session_id: str, paper_id: str | None) -> str | None:
        if paper_id is None or self._session_repository is None:
            return None
        for document in reversed(list(self._session_repository.list_documents(session_id))):
            if document.paper_id == paper_id:
                return self._artifact_file_name(document.artifact_id)
        return None

    def _paper_id_from_evidence_refs(self, refs: Sequence[ConversationEvidenceRefDescriptor]) -> str | None:
        for ref in refs:
            if ref.paper_id:
                return ref.paper_id
        return None

    def _trim_answer_text(self, text: str | None, limit: int) -> str:
        cleaned = (text or "").strip().replace("\n", " ")
        if not cleaned:
            return ""
        return cleaned[:limit]

    def _artifact_file_name(self, artifact_id: str | None) -> str | None:
        if artifact_id is None or self._artifact_repository is None:
            return None
        artifact = self._artifact_repository.get_by_id(artifact_id)
        if artifact is None:
            return None
        return Path(artifact.uri_or_path).name

    def _require_query_repositories(self) -> None:
        if (
            self._session_repository is None
            or self._memory_repository is None
            or self._chunk_repository is None
        ):
            raise RuntimeError("Query paper tools require session, memory, and chunk repositories.")

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
