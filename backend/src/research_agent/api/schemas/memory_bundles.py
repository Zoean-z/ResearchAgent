"""Paper-centric memory bundle API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from research_agent.services.memory_bundle_service import (
    MemoryBundleCatalog,
    MemoryBundleGroup,
    MemoryBundleItem,
    MemoryBundlePaperInfo,
    MemoryBundleSourceChunk,
)


class MemoryBundleSourceChunkResponse(BaseModel):
    """Compact source chunk summary for a paper group."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    paper_id: str
    page: int | None
    section: str | None
    excerpt: str

    @classmethod
    def from_domain(cls, chunk: MemoryBundleSourceChunk) -> "MemoryBundleSourceChunkResponse":
        return cls(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            page=chunk.page,
            section=chunk.section,
            excerpt=chunk.excerpt,
        )


class MemoryBundleItemResponse(BaseModel):
    """Normalized memory card shown in the paper-centric modal."""

    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    content: str
    created_at: datetime | None = None
    updated_at: datetime
    paper_id: str | None = None
    source_paper: str | None = None
    target_paper: str | None = None
    relation_direction: Literal["source", "target"] | None = None
    relation_type: str | None = None
    related_papers: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0

    @classmethod
    def from_domain(cls, item: MemoryBundleItem) -> "MemoryBundleItemResponse":
        return cls(
            id=item.id,
            memory_type=item.memory_type,
            content=item.content,
            created_at=item.created_at,
            updated_at=item.updated_at,
            paper_id=item.paper_id,
            source_paper=item.source_paper,
            target_paper=item.target_paper,
            relation_direction=item.relation_direction,  # type: ignore[arg-type]
            relation_type=item.relation_type,
            related_papers=list(item.related_papers),
            source_chunk_ids=list(item.source_chunk_ids),
            evidence_count=item.evidence_count,
        )


class MemoryBundlePaperInfoResponse(BaseModel):
    """Paper header visible in the modal."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    file_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime
    memory_count: int

    @classmethod
    def from_domain(cls, paper: MemoryBundlePaperInfo) -> "MemoryBundlePaperInfoResponse":
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            file_name=paper.file_name,
            created_at=paper.created_at,
            updated_at=paper.updated_at,
            memory_count=paper.memory_count,
        )


class MemoryBundleGroupResponse(BaseModel):
    """A paper-centric group of memories."""

    model_config = ConfigDict(extra="forbid")

    paper: MemoryBundlePaperInfoResponse
    paper_memories: list[MemoryBundleItemResponse]
    open_question_memories: list[MemoryBundleItemResponse]
    relation_memories: list[MemoryBundleItemResponse]
    source_chunks: list[MemoryBundleSourceChunkResponse]
    source_chunk_count: int
    empty_fields: list[str]

    @classmethod
    def from_domain(cls, group: MemoryBundleGroup) -> "MemoryBundleGroupResponse":
        return cls(
            paper=MemoryBundlePaperInfoResponse.from_domain(group.paper),
            paper_memories=[MemoryBundleItemResponse.from_domain(item) for item in group.paper_memories],
            open_question_memories=[MemoryBundleItemResponse.from_domain(item) for item in group.open_question_memories],
            relation_memories=[MemoryBundleItemResponse.from_domain(item) for item in group.relation_memories],
            source_chunks=[MemoryBundleSourceChunkResponse.from_domain(chunk) for chunk in group.source_chunks],
            source_chunk_count=group.source_chunk_count,
            empty_fields=list(group.empty_fields),
        )


class MemoryBundlesResponse(BaseModel):
    """Paper-centric memory catalog for a session."""

    model_config = ConfigDict(extra="forbid")

    papers: list[MemoryBundleGroupResponse]
    unscoped_memories: list[MemoryBundleItemResponse]

    @classmethod
    def from_domain(cls, bundle: MemoryBundleCatalog) -> "MemoryBundlesResponse":
        return cls(
            papers=[MemoryBundleGroupResponse.from_domain(group) for group in bundle.papers],
            unscoped_memories=[MemoryBundleItemResponse.from_domain(item) for item in bundle.unscoped_memories],
        )
