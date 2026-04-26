"""In-memory content repositories for papers, artifacts, and chunks."""

from __future__ import annotations

from research_agent.domain.models import Artifact, Chunk, Paper
from research_agent.domain.ports import ArtifactRepositoryPort, ChunkRepositoryPort, PaperRepositoryPort
from research_agent.domain.value_objects import CanonicalKey


class InMemoryPaperRepository(PaperRepositoryPort):
    """Simple in-memory storage for canonical papers."""

    def __init__(self) -> None:
        self._papers: dict[str, Paper] = {}
        self._paper_ids_by_key: dict[str, str] = {}

    def save(self, paper: Paper) -> Paper:
        self._papers[paper.id] = paper
        self._paper_ids_by_key[paper.canonical_key.value] = paper.id
        return paper

    def get_by_id(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)

    def get_by_canonical_key(self, canonical_key: CanonicalKey) -> Paper | None:
        paper_id = self._paper_ids_by_key.get(canonical_key.value)
        return self._papers.get(paper_id) if paper_id else None

    def list_by_ids(self, paper_ids: list[str]) -> list[Paper]:
        return [paper for paper_id in paper_ids if (paper := self._papers.get(paper_id)) is not None]


class InMemoryArtifactRepository(ArtifactRepositoryPort):
    """Simple in-memory storage for source artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def save(self, artifact: Artifact) -> Artifact:
        self._artifacts[artifact.id] = artifact
        return artifact

    def get_by_id(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)


class InMemoryChunkRepository(ChunkRepositoryPort):
    """Simple in-memory storage for extracted chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def save_many(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        return chunks

    def list_by_paper_ids(self, paper_ids: list[str]) -> list[Chunk]:
        paper_id_set = set(paper_ids)
        return [chunk for chunk in self._chunks.values() if chunk.paper_id in paper_id_set]


__all__ = [
    "InMemoryArtifactRepository",
    "InMemoryChunkRepository",
    "InMemoryPaperRepository",
]
