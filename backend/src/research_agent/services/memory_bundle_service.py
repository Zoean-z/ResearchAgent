"""Read service for paper-centric memory bundle views."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from research_agent.domain.models import OpenQuestionMemory, Paper, PaperMemory, RelationMemory, SessionDocument, utc_now
from research_agent.domain.ports import ArtifactRepositoryPort, ChunkRepositoryPort, MemoryRepositoryPort, PaperRepositoryPort, SessionRepositoryPort
from research_agent.services.errors import EntityNotFoundError


@dataclass(frozen=True, slots=True)
class MemoryBundleSourceChunk:
    """Compact source chunk summary for a paper group."""

    chunk_id: str
    paper_id: str
    page: int | None
    section: str | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class MemoryBundleItem:
    """Normalized memory item rendered in the paper-centric view."""

    id: str
    memory_type: str
    content: str
    updated_at: datetime
    created_at: datetime | None = None
    paper_id: str | None = None
    source_paper: str | None = None
    target_paper: str | None = None
    relation_direction: str | None = None
    relation_type: str | None = None
    related_papers: tuple[str, ...] = field(default_factory=tuple)
    source_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_count: int = 0


@dataclass(frozen=True, slots=True)
class MemoryBundlePaperInfo:
    """Paper header shown in the memory modal."""

    paper_id: str
    title: str
    file_name: str | None
    created_at: datetime | None
    updated_at: datetime
    memory_count: int


@dataclass(frozen=True, slots=True)
class MemoryBundleGroup:
    """Paper-centric group of memories and source evidence."""

    paper: MemoryBundlePaperInfo
    paper_memories: tuple[MemoryBundleItem, ...]
    open_question_memories: tuple[MemoryBundleItem, ...]
    relation_memories: tuple[MemoryBundleItem, ...]
    source_chunks: tuple[MemoryBundleSourceChunk, ...]
    source_chunk_count: int
    empty_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryBundleCatalog:
    """Paper-centric memory catalog for a session."""

    papers: tuple[MemoryBundleGroup, ...]
    unscoped_memories: tuple[MemoryBundleItem, ...]


@dataclass(slots=True)
class MemoryBundleGroupBuilder:
    """Mutable builder used to assemble paper groups."""

    paper_id: str
    paper: Paper | None
    document: SessionDocument | None
    file_name: str | None
    source_chunks: tuple[MemoryBundleSourceChunk, ...]
    paper_memories: list[MemoryBundleItem] = field(default_factory=list)
    open_question_memories: list[MemoryBundleItem] = field(default_factory=list)
    relation_memories: list[MemoryBundleItem] = field(default_factory=list)

    def build(self) -> MemoryBundleGroup:
        """Finalize a paper-centric group."""

        paper_title = self.paper.title if self.paper is not None else self.paper_id
        created_at = self.document.added_at if self.document is not None else None
        memory_count = len(self.paper_memories) + len(self.open_question_memories) + len(self.relation_memories)
        updated_at = self._updated_at(created_at)
        empty_fields = []
        if not self.paper_memories:
            empty_fields.append("paper_memories")
        if not self.open_question_memories:
            empty_fields.append("open_question_memories")
        if not self.relation_memories:
            empty_fields.append("relation_memories")
        if not self.source_chunks:
            empty_fields.append("source_chunks")
        return MemoryBundleGroup(
            paper=MemoryBundlePaperInfo(
                paper_id=self.paper_id,
                title=paper_title,
                file_name=self.file_name,
                created_at=created_at,
                updated_at=updated_at,
                memory_count=memory_count,
            ),
            paper_memories=tuple(sorted(self.paper_memories, key=lambda item: item.updated_at, reverse=True)),
            open_question_memories=tuple(sorted(self.open_question_memories, key=lambda item: item.updated_at, reverse=True)),
            relation_memories=tuple(sorted(self.relation_memories, key=lambda item: item.updated_at, reverse=True)),
            source_chunks=self.source_chunks,
            source_chunk_count=len(self.source_chunks),
            empty_fields=tuple(empty_fields),
        )

    def _updated_at(self, created_at: datetime | None) -> datetime:
        candidates = [created_at] if created_at is not None else []
        candidates.extend(item.updated_at for item in self.paper_memories)
        candidates.extend(item.updated_at for item in self.open_question_memories)
        candidates.extend(item.updated_at for item in self.relation_memories)
        if candidates:
            return max(candidates)
        if created_at is not None:
            return created_at
        return utc_now()


class MemoryBundleService:
    """Compose paper-centric memory bundles from existing repositories."""

    def __init__(
        self,
        session_repository: SessionRepositoryPort,
        paper_repository: PaperRepositoryPort,
        artifact_repository: ArtifactRepositoryPort,
        chunk_repository: ChunkRepositoryPort,
        memory_repository: MemoryRepositoryPort,
    ) -> None:
        self._session_repository = session_repository
        self._paper_repository = paper_repository
        self._artifact_repository = artifact_repository
        self._chunk_repository = chunk_repository
        self._memory_repository = memory_repository

    def get_bundle(self, session_id: str, source_chunk_limit: int = 3) -> MemoryBundleCatalog:
        """Return paper-centric memory groups for a session."""

        session = self._session_repository.get_by_id(session_id)
        if session is None:
            raise EntityNotFoundError("Session", session_id)

        documents = list(self._session_repository.list_documents(session.id))
        latest_documents = self._latest_document_by_paper_id(documents)
        paper_ids = list(latest_documents)
        papers_by_id = {paper.id: paper for paper in self._paper_repository.list_by_ids(paper_ids)}

        paper_memories = list(self._memory_repository.list_paper_memories_for_papers(paper_ids))
        relation_memories = list(self._memory_repository.list_relation_memories_for_papers(paper_ids))
        open_question_memories = list(self._memory_repository.list_open_question_memories_for_papers(paper_ids))

        groups: dict[str, MemoryBundleGroupBuilder] = {
            paper_id: MemoryBundleGroupBuilder(
                paper_id=paper_id,
                paper=papers_by_id.get(paper_id),
                document=latest_documents.get(paper_id),
                file_name=self._paper_file_name(latest_documents[paper_id].artifact_id),
                source_chunks=self._source_chunks_for_paper(paper_id, source_chunk_limit),
            )
            for paper_id in paper_ids
        }
        unscoped: list[MemoryBundleItem] = []

        for memory in paper_memories:
            item = self._build_paper_memory_item(memory)
            builder = groups.get(memory.paper_id)
            if builder is None:
                unscoped.append(item)
                continue
            builder.paper_memories.append(item)

        for memory in relation_memories:
            item = self._build_relation_memory_item(memory, relation_direction=None)
            matched = False
            if memory.source_paper in groups:
                groups[memory.source_paper].relation_memories.append(
                    self._build_relation_memory_item(memory, relation_direction="source")
                )
                matched = True
            if memory.target_paper in groups:
                groups[memory.target_paper].relation_memories.append(
                    self._build_relation_memory_item(memory, relation_direction="target")
                )
                matched = True
            if not matched:
                unscoped.append(item)

        for memory in open_question_memories:
            item = self._build_open_question_memory_item(memory)
            matched = False
            for paper_id in memory.related_papers:
                if paper_id in groups:
                    groups[paper_id].open_question_memories.append(item)
                    matched = True
            if not matched:
                unscoped.append(item)

        built_groups = [builder.build() for builder in groups.values()]
        built_groups.sort(key=lambda group: (group.paper.updated_at, group.paper.title.lower()), reverse=True)

        return MemoryBundleCatalog(
            papers=tuple(built_groups),
            unscoped_memories=tuple(unscoped),
        )

    def get_global_bundle(self, source_chunk_limit: int = 3) -> MemoryBundleCatalog:
        """Return paper-centric memory groups for all persisted memories."""

        paper_memories = list(self._memory_repository.list_all_paper_memories())
        relation_memories = list(self._memory_repository.list_all_relation_memories())
        open_question_memories = list(self._memory_repository.list_all_open_question_memories())
        latest_documents = self._latest_document_by_paper_id(list(self._session_repository.list_all_documents()))

        paper_ids = self._paper_ids_for_global_bundle(
            paper_memories=paper_memories,
            relation_memories=relation_memories,
            open_question_memories=open_question_memories,
        )
        papers_by_id = {paper.id: paper for paper in self._paper_repository.list_by_ids(paper_ids)}

        groups: dict[str, MemoryBundleGroupBuilder] = {
            paper_id: MemoryBundleGroupBuilder(
                paper_id=paper_id,
                paper=papers_by_id.get(paper_id),
                document=latest_documents.get(paper_id),
                file_name=(
                    self._paper_file_name(latest_documents[paper_id].artifact_id)
                    if paper_id in latest_documents
                    else self._global_paper_file_name(papers_by_id.get(paper_id))
                ),
                source_chunks=self._source_chunks_for_paper(paper_id, source_chunk_limit),
            )
            for paper_id in paper_ids
        }
        unscoped: list[MemoryBundleItem] = []

        for memory in paper_memories:
            item = self._build_paper_memory_item(memory)
            builder = groups.get(memory.paper_id)
            if builder is None:
                unscoped.append(item)
                continue
            builder.paper_memories.append(item)

        for memory in relation_memories:
            item = self._build_relation_memory_item(memory, relation_direction=None)
            matched = False
            if memory.source_paper in groups:
                groups[memory.source_paper].relation_memories.append(
                    self._build_relation_memory_item(memory, relation_direction="source")
                )
                matched = True
            if memory.target_paper in groups:
                groups[memory.target_paper].relation_memories.append(
                    self._build_relation_memory_item(memory, relation_direction="target")
                )
                matched = True
            if not matched:
                unscoped.append(item)

        for memory in open_question_memories:
            item = self._build_open_question_memory_item(memory)
            matched = False
            for paper_id in memory.related_papers:
                if paper_id in groups:
                    groups[paper_id].open_question_memories.append(item)
                    matched = True
            if not matched:
                unscoped.append(item)

        built_groups = [builder.build() for builder in groups.values()]
        built_groups.sort(key=lambda group: (group.paper.updated_at, group.paper.title.lower()), reverse=True)

        return MemoryBundleCatalog(
            papers=tuple(built_groups),
            unscoped_memories=tuple(unscoped),
        )

    def _source_chunks_for_paper(self, paper_id: str, source_chunk_limit: int) -> tuple[MemoryBundleSourceChunk, ...]:
        chunks = list(self._chunk_repository.list_by_paper_ids([paper_id]))
        chunks.sort(key=lambda chunk: (chunk.page is None, chunk.page or 0, chunk.id))
        return tuple(
            MemoryBundleSourceChunk(
                chunk_id=chunk.id,
                paper_id=chunk.paper_id,
                page=chunk.page,
                section=chunk.section,
                excerpt=self._trim_text(chunk.text, 200),
            )
            for chunk in chunks[:source_chunk_limit]
        )

    def _build_paper_memory_item(self, memory: PaperMemory) -> MemoryBundleItem:
        content_parts = [
            memory.problem,
            memory.method,
            ", ".join(memory.key_results),
            ", ".join(memory.limitations),
            memory.novelty_claim,
        ]
        source_chunk_ids = tuple(ref.chunk_id for ref in memory.source_refs if ref.chunk_id is not None)
        return MemoryBundleItem(
            id=memory.id,
            memory_type="paper_memory",
            content=self._join_content_parts(content_parts),
            created_at=None,
            updated_at=memory.updated_at,
            paper_id=memory.paper_id,
            source_chunk_ids=source_chunk_ids,
            evidence_count=len(memory.source_refs),
        )

    def _build_relation_memory_item(
        self,
        memory: RelationMemory,
        relation_direction: str | None,
    ) -> MemoryBundleItem:
        evidence = tuple(memory.evidence)
        return MemoryBundleItem(
            id=memory.id,
            memory_type="relation_memory",
            content=self._join_content_parts((memory.summary, " ".join(evidence) if evidence else None)),
            created_at=None,
            updated_at=memory.updated_at,
            source_paper=memory.source_paper,
            target_paper=memory.target_paper,
            relation_direction=relation_direction,
            relation_type=memory.relation_type.value,
            evidence_count=len(evidence),
        )

    def _build_open_question_memory_item(self, memory: OpenQuestionMemory) -> MemoryBundleItem:
        parts = [
            memory.unresolved_question,
            " ".join(memory.why_open),
            " ".join(memory.possible_followup),
        ]
        return MemoryBundleItem(
            id=memory.id,
            memory_type="open_question_memory",
            content=self._join_content_parts(parts),
            created_at=None,
            updated_at=memory.updated_at,
            related_papers=tuple(memory.related_papers),
            evidence_count=max(len(memory.why_open), len(memory.possible_followup)),
        )

    def _latest_document_by_paper_id(self, documents: list[SessionDocument]) -> dict[str, SessionDocument]:
        latest: dict[str, SessionDocument] = {}
        for document in documents:
            current = latest.get(document.paper_id)
            if current is None or document.added_at >= current.added_at:
                latest[document.paper_id] = document
        return latest

    def _paper_file_name(self, artifact_id: str | None) -> str | None:
        if artifact_id is None:
            return None
        artifact = self._artifact_repository.get_by_id(artifact_id)
        if artifact is None:
            return None
        return Path(artifact.uri_or_path).name

    def _global_paper_file_name(self, paper: Paper | None) -> str | None:
        if paper is None or paper.pdf_fingerprint is None:
            return None
        artifact = self._artifact_repository.get_by_checksum(paper.pdf_fingerprint)
        if artifact is None:
            return None
        return Path(artifact.uri_or_path).name

    def _paper_ids_for_global_bundle(
        self,
        *,
        paper_memories: list[PaperMemory],
        relation_memories: list[RelationMemory],
        open_question_memories: list[OpenQuestionMemory],
    ) -> list[str]:
        ordered_ids: list[str] = []
        seen: set[str] = set()

        def add(paper_id: str | None) -> None:
            if not paper_id or paper_id in seen:
                return
            seen.add(paper_id)
            ordered_ids.append(paper_id)

        for memory in paper_memories:
            add(memory.paper_id)
        for memory in relation_memories:
            add(memory.source_paper)
            add(memory.target_paper)
        for memory in open_question_memories:
            for paper_id in memory.related_papers:
                add(paper_id)
        return ordered_ids

    def _trim_text(self, text: str, limit: int) -> str:
        cleaned = " ".join(text.split())
        return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}..."

    def _join_content_parts(self, parts: tuple[str | None, ...] | list[str | None]) -> str:
        cleaned = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
        return " | ".join(cleaned) if cleaned else ""
