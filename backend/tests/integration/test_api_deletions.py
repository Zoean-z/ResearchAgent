"""Integration tests for delete endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from research_agent.api.app import create_app
from research_agent.domain.enums import ArtifactKind, RelationType, SourceType
from research_agent.domain.models import Artifact, Paper, PaperMemory, RelationMemory, SessionDocument
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore


def _build_client(tmp_path) -> TestClient:
    return TestClient(create_app(storage_backend="sqlite", sqlite_path=tmp_path / "delete-api.sqlite3"))


def test_delete_session_endpoint_tombstones_and_clears_dialogue_state(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Delete session"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.12345"),
                title="Delete me",
            )
        )
        repositories.papers.save(
            Paper(
                id="paper-2",
                canonical_key=build_canonical_key(pdf_checksum="paper-2-checksum"),
                title="Delete target",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/delete.pdf",
                checksum="delete-checksum",
            )
        )
        repositories.sessions.save_document(
            SessionDocument(
                session_id=session_id,
                paper_id="paper-1",
                source_type=SourceType.PDF,
                artifact_id="artifact-1",
            )
        )
        repositories.memories.upsert_paper_memory(
            PaperMemory(
                id="paper-memory-1",
                paper_id="paper-1",
                key_results=["Improved accuracy"],
                confidence=ConfidenceScore(value=0.8),
            )
        )
        repositories.memories.upsert_relation_memory(
            RelationMemory(
                id="relation-memory-1",
                source_paper="paper-1",
                target_paper="paper-2",
                relation_type=RelationType.COMPARES_WITH,
                summary="Compares on the same benchmark.",
                confidence=ConfidenceScore(value=0.7),
            )
        )

        accept_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        run_id = accept_response.json()["run_id"]
        delete_response = client.delete(f"/api/sessions/{session_id}")
        session_after_delete = client.get(f"/api/sessions/{session_id}")
        sessions_after_delete = client.get("/api/sessions")
        messages_response = client.get(f"/api/sessions/{session_id}/messages")
        timeline_response = client.get(f"/api/sessions/{session_id}/timeline")
        snapshot_response = client.get(f"/api/sessions/{session_id}/memory-snapshot")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}")

    assert delete_response.status_code == 200
    payload = delete_response.json()
    assert payload["deleted_documents"] == 1
    assert payload["deleted_messages"] == 1
    assert payload["deleted_runs"] == 1
    assert payload["deleted_timeline_events"] == 0
    assert payload["deleted_memories"] == 0
    assert payload["session"]["status"] == "deleted"
    assert payload["mirrored_to_openviking"] is False

    assert session_after_delete.status_code == 404
    assert all(item["id"] != session_id for item in sessions_after_delete.json()["items"])
    assert messages_response.json() == {"items": []}
    assert timeline_response.json() == {"items": []}
    assert snapshot_response.json() == {
        "paper_memories": [],
        "relation_memories": [],
        "open_question_memories": [],
    }
    assert run_response.status_code == 404
    assert {memory.id for memory in repositories.memories.list_paper_memories_for_papers(["paper-1"])} == {"paper-memory-1"}
    assert {memory.id for memory in repositories.memories.list_relation_memories_for_papers(["paper-1"])} == {"relation-memory-1"}


def test_delete_memory_endpoint_removes_only_the_selected_memory(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Delete memory"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.12345"),
                title="Memory paper",
            )
        )
        repositories.papers.save(
            Paper(
                id="paper-2",
                canonical_key=build_canonical_key(pdf_checksum="paper-2-checksum"),
                title="Memory target",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/memory.pdf",
                checksum="memory-checksum",
            )
        )
        repositories.sessions.save_document(
            SessionDocument(
                session_id=session_id,
                paper_id="paper-1",
                source_type=SourceType.PDF,
                artifact_id="artifact-1",
            )
        )
        repositories.memories.upsert_paper_memory(
            PaperMemory(
                id="paper-memory-1",
                paper_id="paper-1",
                key_results=["Improved accuracy"],
                confidence=ConfidenceScore(value=0.8),
            )
        )
        repositories.memories.upsert_relation_memory(
            RelationMemory(
                id="relation-memory-1",
                source_paper="paper-1",
                target_paper="paper-2",
                relation_type=RelationType.COMPARES_WITH,
                summary="Compares on the same benchmark.",
                confidence=ConfidenceScore(value=0.7),
            )
        )

        delete_response = client.delete(f"/api/sessions/{session_id}/memories/paper_memory/paper-memory-1")
        snapshot_response = client.get(f"/api/sessions/{session_id}/memory-snapshot")

    assert delete_response.status_code == 200
    payload = delete_response.json()
    assert payload["session_id"] == session_id
    assert payload["memory_kind"] == "paper_memory"
    assert payload["memory_id"] == "paper-memory-1"
    assert payload["deleted"] is True
    assert payload["mirrored_to_openviking"] is False
    snapshot = snapshot_response.json()
    assert snapshot["paper_memories"] == []
    assert len(snapshot["relation_memories"]) == 1
