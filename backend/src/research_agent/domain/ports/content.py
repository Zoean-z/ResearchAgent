"""Paper, artifact, and chunk repository ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from research_agent.domain.models import Artifact, Chunk, Paper
from research_agent.domain.value_objects import CanonicalKey


@runtime_checkable
class PaperRepositoryPort(Protocol):
    """Storage boundary for canonical paper records."""

    def save(self, paper: Paper) -> Paper:
        """Create or replace a paper record."""

    def get_by_id(self, paper_id: str) -> Paper | None:
        """Fetch a paper by id."""

    def get_by_canonical_key(self, canonical_key: CanonicalKey) -> Paper | None:
        """Fetch a paper by canonical identity."""

    def list_by_ids(self, paper_ids: Sequence[str]) -> Sequence[Paper]:
        """Fetch papers by id."""


@runtime_checkable
class ArtifactRepositoryPort(Protocol):
    """Storage boundary for PDF artifacts."""

    def save(self, artifact: Artifact) -> Artifact:
        """Create or replace an artifact record."""

    def get_by_id(self, artifact_id: str) -> Artifact | None:
        """Fetch an artifact by id."""


@runtime_checkable
class ChunkRepositoryPort(Protocol):
    """Storage boundary for extracted paper chunks."""

    def save_many(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Persist chunk records."""

    def list_by_paper_ids(self, paper_ids: Sequence[str]) -> Sequence[Chunk]:
        """List chunks linked to one or more paper ids."""
