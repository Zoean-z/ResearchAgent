"""In-memory content repositories for papers, artifacts, and chunks."""

from __future__ import annotations

from hashlib import sha256

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
        self._artifact_ids_by_checksum: dict[str, str] = {}

    def save(self, artifact: Artifact) -> Artifact:
        existing = self.get_by_checksum(artifact.checksum)
        if existing is not None and existing.id != artifact.id:
            return existing
        self._artifacts[artifact.id] = artifact
        self._artifact_ids_by_checksum[artifact.checksum] = artifact.id
        return artifact

    def get_by_id(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def get_by_checksum(self, checksum: str) -> Artifact | None:
        artifact_id = self._artifact_ids_by_checksum.get(checksum)
        return self._artifacts.get(artifact_id) if artifact_id else None


class InMemoryChunkRepository(ChunkRepositoryPort):
    """Simple in-memory storage for extracted chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._chunk_ids_by_business_key: dict[tuple[str, int | None, str], str] = {}

    def save_many(self, chunks: list[Chunk]) -> list[Chunk]:
        persisted: list[Chunk] = []
        for chunk in chunks:
            business_key = (chunk.artifact_id, chunk.page, sha256(chunk.text.encode("utf-8")).hexdigest())
            existing_id = self._chunk_ids_by_business_key.get(business_key)
            if existing_id is not None:
                persisted.append(self._chunks[existing_id])
                continue
            previous = self._chunks.get(chunk.id)
            if previous is not None:
                previous_key = (previous.artifact_id, previous.page, sha256(previous.text.encode("utf-8")).hexdigest())
                self._chunk_ids_by_business_key.pop(previous_key, None)
            self._chunks[chunk.id] = chunk
            self._chunk_ids_by_business_key[business_key] = chunk.id
            persisted.append(chunk)
        return persisted

    def list_by_paper_ids(self, paper_ids: list[str]) -> list[Chunk]:
        paper_id_set = set(paper_ids)
        return [chunk for chunk in self._chunks.values() if chunk.paper_id in paper_id_set]

    def list_by_artifact_id(self, artifact_id: str) -> list[Chunk]:
        return [chunk for chunk in self._chunks.values() if chunk.artifact_id == artifact_id]


__all__ = [
    "InMemoryArtifactRepository",
    "InMemoryChunkRepository",
    "InMemoryPaperRepository",
]
