"""SQLite repository for structured research memories."""

from __future__ import annotations

from collections.abc import Sequence

from research_agent.adapters.storage.sqlite_common import SQLiteDatabase
from research_agent.domain.enums import RelationType
from research_agent.domain.models import OpenQuestionMemory, PaperMemory, RelationMemory, SourceRef
from research_agent.domain.ports import MemoryRepositoryPort
from research_agent.domain.value_objects import ConfidenceScore


class SQLiteMemoryRepository(MemoryRepositoryPort):
    """SQLite-backed storage for all three memory types."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def upsert_paper_memory(self, memory: PaperMemory) -> PaperMemory:
        self._database.execute(
            """
            INSERT INTO paper_memories (
                id, paper_id, problem, method, key_results_json, limitations_json,
                novelty_claim, source_refs_json, confidence, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                paper_id = excluded.paper_id,
                problem = excluded.problem,
                method = excluded.method,
                key_results_json = excluded.key_results_json,
                limitations_json = excluded.limitations_json,
                novelty_claim = excluded.novelty_claim,
                source_refs_json = excluded.source_refs_json,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                memory.id,
                memory.paper_id,
                memory.problem,
                memory.method,
                SQLiteDatabase.encode_json(memory.key_results),
                SQLiteDatabase.encode_json(memory.limitations),
                memory.novelty_claim,
                SQLiteDatabase.encode_json([ref.model_dump(mode="json") for ref in memory.source_refs]),
                memory.confidence.value,
                memory.updated_at.isoformat(),
            ),
        )
        return memory

    def upsert_relation_memory(self, memory: RelationMemory) -> RelationMemory:
        self._database.execute(
            """
            INSERT INTO relation_memories (
                id, source_paper, target_paper, relation_type, summary, evidence_json, confidence, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_paper = excluded.source_paper,
                target_paper = excluded.target_paper,
                relation_type = excluded.relation_type,
                summary = excluded.summary,
                evidence_json = excluded.evidence_json,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                memory.id,
                memory.source_paper,
                memory.target_paper,
                memory.relation_type.value,
                memory.summary,
                SQLiteDatabase.encode_json(memory.evidence),
                memory.confidence.value,
                memory.updated_at.isoformat(),
            ),
        )
        return memory

    def upsert_open_question_memory(self, memory: OpenQuestionMemory) -> OpenQuestionMemory:
        self._database.execute(
            """
            INSERT INTO open_question_memories (
                id, unresolved_question, related_papers_json, why_open_json,
                possible_followup_json, confidence, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                unresolved_question = excluded.unresolved_question,
                related_papers_json = excluded.related_papers_json,
                why_open_json = excluded.why_open_json,
                possible_followup_json = excluded.possible_followup_json,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (
                memory.id,
                memory.unresolved_question,
                SQLiteDatabase.encode_json(memory.related_papers),
                SQLiteDatabase.encode_json(memory.why_open),
                SQLiteDatabase.encode_json(memory.possible_followup),
                memory.confidence.value,
                memory.updated_at.isoformat(),
            ),
        )
        return memory

    def list_paper_memories_for_papers(self, paper_ids: Sequence[str]) -> Sequence[PaperMemory]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"SELECT * FROM paper_memories WHERE paper_id IN ({placeholders}) ORDER BY updated_at ASC",
            tuple(paper_ids),
        )
        return [self._row_to_paper_memory(row) for row in rows]

    def list_relation_memories_for_papers(self, paper_ids: Sequence[str]) -> Sequence[RelationMemory]:
        if not paper_ids:
            return []
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"""
            SELECT * FROM relation_memories
            WHERE source_paper IN ({placeholders}) OR target_paper IN ({placeholders})
            ORDER BY updated_at ASC
            """,
            tuple(paper_ids) + tuple(paper_ids),
        )
        return [self._row_to_relation_memory(row) for row in rows]

    def list_open_question_memories_for_papers(self, paper_ids: Sequence[str]) -> Sequence[OpenQuestionMemory]:
        if not paper_ids:
            return []
        paper_id_set = set(paper_ids)
        rows = self._database.query_all("SELECT * FROM open_question_memories ORDER BY updated_at ASC")
        return [
            memory
            for memory in (self._row_to_open_question_memory(row) for row in rows)
            if paper_id_set.intersection(memory.related_papers)
        ]

    def list_all_paper_memories(self) -> Sequence[PaperMemory]:
        rows = self._database.query_all("SELECT * FROM paper_memories ORDER BY updated_at ASC")
        return [self._row_to_paper_memory(row) for row in rows]

    def list_all_relation_memories(self) -> Sequence[RelationMemory]:
        rows = self._database.query_all("SELECT * FROM relation_memories ORDER BY updated_at ASC")
        return [self._row_to_relation_memory(row) for row in rows]

    def list_all_open_question_memories(self) -> Sequence[OpenQuestionMemory]:
        rows = self._database.query_all("SELECT * FROM open_question_memories ORDER BY updated_at ASC")
        return [self._row_to_open_question_memory(row) for row in rows]

    def delete_paper_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        if not paper_ids:
            return 0
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"SELECT id FROM paper_memories WHERE paper_id IN ({placeholders})",
            tuple(paper_ids),
        )
        deleted = len(rows)
        self._database.execute(f"DELETE FROM paper_memories WHERE paper_id IN ({placeholders})", tuple(paper_ids))
        return deleted

    def delete_relation_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        if not paper_ids:
            return 0
        placeholders = ",".join("?" for _ in paper_ids)
        rows = self._database.query_all(
            f"""
            SELECT id FROM relation_memories
            WHERE source_paper IN ({placeholders}) OR target_paper IN ({placeholders})
            """,
            tuple(paper_ids) + tuple(paper_ids),
        )
        deleted = len(rows)
        self._database.execute(
            f"""
            DELETE FROM relation_memories
            WHERE source_paper IN ({placeholders}) OR target_paper IN ({placeholders})
            """,
            tuple(paper_ids) + tuple(paper_ids),
        )
        return deleted

    def delete_open_question_memories_for_papers(self, paper_ids: Sequence[str]) -> int:
        if not paper_ids:
            return 0
        paper_id_set = set(paper_ids)
        rows = self._database.query_all("SELECT id, related_papers_json FROM open_question_memories")
        memory_ids = []
        for row in rows:
            related_papers = set(SQLiteDatabase.decode_json(row["related_papers_json"]))
            if paper_id_set.intersection(related_papers):
                memory_ids.append(row["id"])
        if not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        self._database.execute(
            f"DELETE FROM open_question_memories WHERE id IN ({placeholders})",
            tuple(memory_ids),
        )
        return len(memory_ids)

    def delete_paper_memory(self, memory_id: str) -> bool:
        cursor = self._database.execute("DELETE FROM paper_memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def delete_relation_memory(self, memory_id: str) -> bool:
        cursor = self._database.execute("DELETE FROM relation_memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def delete_open_question_memory(self, memory_id: str) -> bool:
        cursor = self._database.execute("DELETE FROM open_question_memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def _row_to_paper_memory(self, row) -> PaperMemory:
        return PaperMemory.model_validate(
            {
                "id": row["id"],
                "paper_id": row["paper_id"],
                "problem": row["problem"],
                "method": row["method"],
                "key_results": SQLiteDatabase.decode_json(row["key_results_json"]),
                "limitations": SQLiteDatabase.decode_json(row["limitations_json"]),
                "novelty_claim": row["novelty_claim"],
                "source_refs": SQLiteDatabase.decode_json(row["source_refs_json"]),
                "confidence": {"value": row["confidence"]},
                "updated_at": row["updated_at"],
            }
        )

    def _row_to_relation_memory(self, row) -> RelationMemory:
        return RelationMemory.model_validate(
            {
                "id": row["id"],
                "source_paper": row["source_paper"],
                "target_paper": row["target_paper"],
                "relation_type": row["relation_type"],
                "summary": row["summary"],
                "evidence": SQLiteDatabase.decode_json(row["evidence_json"]),
                "confidence": {"value": row["confidence"]},
                "updated_at": row["updated_at"],
            }
        )

    def _row_to_open_question_memory(self, row) -> OpenQuestionMemory:
        return OpenQuestionMemory.model_validate(
            {
                "id": row["id"],
                "unresolved_question": row["unresolved_question"],
                "related_papers": SQLiteDatabase.decode_json(row["related_papers_json"]),
                "why_open": SQLiteDatabase.decode_json(row["why_open_json"]),
                "possible_followup": SQLiteDatabase.decode_json(row["possible_followup_json"]),
                "confidence": {"value": row["confidence"]},
                "updated_at": row["updated_at"],
            }
        )
