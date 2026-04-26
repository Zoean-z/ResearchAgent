"""SQLite repositories for papers and artifacts."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.adapters.storage.sqlite_common import SQLiteDatabase
from research_agent.domain.models import Artifact, Chunk, Paper
from research_agent.domain.ports import ArtifactRepositoryPort, ChunkRepositoryPort, PaperRepositoryPort
from research_agent.domain.value_objects import CanonicalKey


class SQLitePaperRepository(PaperRepositoryPort):
    """SQLite-backed canonical paper repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(self, paper: Paper) -> Paper:
        self._database.execute(
            """
            INSERT INTO papers (id, canonical_key, title, authors_json, abstract, year, arxiv_id, pdf_fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                canonical_key = excluded.canonical_key,
                title = excluded.title,
                authors_json = excluded.authors_json,
                abstract = excluded.abstract,
                year = excluded.year,
                arxiv_id = excluded.arxiv_id,
                pdf_fingerprint = excluded.pdf_fingerprint
            """,
            (
                paper.id,
                paper.canonical_key.value,
                paper.title,
                SQLiteDatabase.encode_json(paper.authors),
                paper.abstract,
                paper.year,
                paper.arxiv_id,
                paper.pdf_fingerprint,
            ),
        )
        return paper

    def get_by_id(self, paper_id: str) -> Paper | None:
        row = self._database.query_one("SELECT * FROM papers WHERE id = ?", (paper_id,))
        return self._row_to_paper(row) if row is not None else None

    def get_by_canonical_key(self, canonical_key: CanonicalKey) -> Paper | None:
        row = self._database.query_one("SELECT * FROM papers WHERE canonical_key = ?", (canonical_key.value,))
        return self._row_to_paper(row) if row is not None else None

    def list_by_ids(self, paper_ids: Sequence[str]) -> Sequence[Paper]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"SELECT * FROM papers WHERE id IN ({placeholders})",
            tuple(paper_ids),
        )
        indexed = {row["id"]: self._row_to_paper(row) for row in rows}
        return [indexed[paper_id] for paper_id in paper_ids if paper_id in indexed]

    def _row_to_paper(self, row) -> Paper:
        return Paper.model_validate(
            {
                "id": row["id"],
                "canonical_key": {"value": row["canonical_key"]},
                "title": row["title"],
                "authors": SQLiteDatabase.decode_json(row["authors_json"]),
                "abstract": row["abstract"],
                "year": row["year"],
                "arxiv_id": row["arxiv_id"],
                "pdf_fingerprint": row["pdf_fingerprint"],
            }
        )


class SQLiteArtifactRepository(ArtifactRepositoryPort):
    """SQLite-backed artifact repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save(self, artifact: Artifact) -> Artifact:
        self._database.execute(
            """
            INSERT INTO artifacts (id, kind, uri_or_path, checksum, page_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                uri_or_path = excluded.uri_or_path,
                checksum = excluded.checksum,
                page_count = excluded.page_count
            """,
            (
                artifact.id,
                artifact.kind.value,
                artifact.uri_or_path,
                artifact.checksum,
                artifact.page_count,
            ),
        )
        return artifact

    def get_by_id(self, artifact_id: str) -> Artifact | None:
        row = self._database.query_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))
        return self._row_to_artifact(row) if row is not None else None

    def _row_to_artifact(self, row) -> Artifact:
        return Artifact.model_validate(
            {
                "id": row["id"],
                "kind": row["kind"],
                "uri_or_path": row["uri_or_path"],
                "checksum": row["checksum"],
                "page_count": row["page_count"],
            }
        )


class SQLiteChunkRepository(ChunkRepositoryPort):
    """SQLite-backed chunk repository."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def save_many(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        for chunk in chunks:
            self._database.execute(
                """
                INSERT INTO chunks (id, paper_id, artifact_id, text, page, section)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    paper_id = excluded.paper_id,
                    artifact_id = excluded.artifact_id,
                    text = excluded.text,
                    page = excluded.page,
                    section = excluded.section
                """,
                (chunk.id, chunk.paper_id, chunk.artifact_id, chunk.text, chunk.page, chunk.section),
            )
        return chunks

    def list_by_paper_ids(self, paper_ids: Sequence[str]) -> Sequence[Chunk]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"SELECT * FROM chunks WHERE paper_id IN ({placeholders}) ORDER BY id ASC",
            tuple(paper_ids),
        )
        return [self._row_to_chunk(row) for row in rows]

    def _row_to_chunk(self, row) -> Chunk:
        return Chunk.model_validate(
            {
                "id": row["id"],
                "paper_id": row["paper_id"],
                "artifact_id": row["artifact_id"],
                "text": row["text"],
                "page": row["page"],
                "section": row["section"],
            }
        )
