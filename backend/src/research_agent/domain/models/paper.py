"""Paper, artifact, and source reference models."""

from __future__ import annotations

from pydantic import Field

from research_agent.domain.enums import ArtifactKind
from research_agent.domain.models.base import DomainModel
from research_agent.domain.value_objects import CanonicalKey, SourceLocator


class Paper(DomainModel):
    """Canonical paper record across sessions."""

    id: str
    canonical_key: CanonicalKey
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = Field(default=None, ge=0)
    arxiv_id: str | None = None
    pdf_fingerprint: str | None = None


class Artifact(DomainModel):
    """Persistent artifact record for a PDF source."""

    id: str
    kind: ArtifactKind
    uri_or_path: str = Field(min_length=1)
    checksum: str = Field(min_length=1)
    page_count: int | None = Field(default=None, ge=1)


class SourceRef(DomainModel):
    """Citation-style reference back into an original source."""

    paper_id: str
    artifact_id: str
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_id: str | None = None
    quote: str | None = None

    @property
    def locator(self) -> SourceLocator:
        return SourceLocator(page=self.page, section=self.section, chunk_id=self.chunk_id)


class Chunk(DomainModel):
    """Extracted paper chunk placeholder for future retrieval adapters."""

    id: str
    paper_id: str
    artifact_id: str
    text: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    section: str | None = None
