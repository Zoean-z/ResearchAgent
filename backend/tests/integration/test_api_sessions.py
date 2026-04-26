"""Integration tests for the mock session API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from research_agent.domain.enums import ArtifactKind, RelationType, SourceType
from research_agent.domain.models import Artifact, Chunk, Paper, PaperMemory, RelationMemory, SessionDocument, SourceRef
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore
from research_agent.api.app import create_app
from research_agent.services.ingest_materialization_service import IngestMaterializationService


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf_bytes(text: str) -> bytes:
    content_stream = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET\n".encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n" + content_stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(payload)
        if not payload.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(b"endobj\n")
    xref_start = len(output)
    output.extend(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(b"trailer << /Root 1 0 R /Size 6 >>\nstartxref\n")
    output.extend(f"{xref_start}\n".encode("ascii"))
    output.extend(b"%%EOF\n")
    return bytes(output)


def _build_client(tmp_path) -> TestClient:
    return TestClient(create_app(storage_backend="sqlite", sqlite_path=tmp_path / "api.sqlite3"))


@pytest.fixture(autouse=True)
def _stub_arxiv_download(monkeypatch) -> None:
    monkeypatch.setattr(
        IngestMaterializationService,
        "_download_arxiv_pdf",
        lambda self, pdf_url, source_value: _build_minimal_pdf_bytes("ArXiv integration text that should be extracted."),
    )


def test_health_endpoint(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runtime_status_endpoint_exposes_safe_backend_configuration(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_AGENT_STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_AGENT_BACKEND", "pydantic_ai")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_AGENT_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("RESEARCH_AGENT_INGEST_EXTRACTION_BACKEND", "model_adapter")
    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_BACKEND", "embedded")
    monkeypatch.setenv("RESEARCH_AGENT_OPENVIKING_DATA_PATH", str(tmp_path / "openviking-data"))

    with _build_client(tmp_path) as client:
        response = client.get("/api/system/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["storage_backend"] == "sqlite"
    assert payload["query_agent_backend"] == "pydantic_ai"
    assert payload["query_agent_model"] == "deepseek-v4-flash"
    assert payload["ingest_extraction_backend"] == "model_adapter"
    assert payload["openviking_backend"] == "embedded"
    assert payload["openviking_data_path"] == str(tmp_path / "openviking-data")


def test_create_and_list_sessions(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        create_response = client.post("/api/sessions", json={"title": "First session"})
        list_response = client.get("/api/sessions")

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["title"] == "First session"
    assert created["status"] == "active"

    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["id"] == created["id"]


def test_empty_session_views_return_stable_shapes(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Empty views"})
        session_id = session_response.json()["id"]

        messages_response = client.get(f"/api/sessions/{session_id}/messages")
        timeline_response = client.get(f"/api/sessions/{session_id}/timeline")
        snapshot_response = client.get(f"/api/sessions/{session_id}/memory-snapshot")

    assert messages_response.status_code == 200
    assert messages_response.json() == {"items": []}

    assert timeline_response.status_code == 200
    assert timeline_response.json() == {"items": []}

    assert snapshot_response.status_code == 200
    assert snapshot_response.json() == {
        "paper_memories": [],
        "relation_memories": [],
        "open_question_memories": [],
    }


def test_acceptance_endpoints_create_messages_and_task_runs(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Acceptance"})
        session_id = session_response.json()["id"]

        ingest_response = client.post(
            f"/api/sessions/{session_id}/ingest/arxiv",
            json={"arxiv_url": "https://arxiv.org/abs/2401.12345"},
        )
        query_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "What changed after reading this paper?"},
        )
        messages_response = client.get(f"/api/sessions/{session_id}/messages")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{ingest_response.json()['run_id']}")
        runs_response = client.get(f"/api/sessions/{session_id}/runs")

    assert ingest_response.status_code == 202
    assert ingest_response.json()["accepted"] is True

    assert query_response.status_code == 202
    assert query_response.json()["accepted"] is True

    messages = messages_response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["type"] == "ingest_arxiv"
    assert messages[1]["type"] == "followup_query"

    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["session_id"] == session_id
    assert run_payload["status"] == "pending"
    assert run_payload["step_count"] == 0
    assert runs_response.status_code == 200
    assert len(runs_response.json()["items"]) == 2


def test_unified_message_endpoint_routes_by_payload_type(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Unified"})
        session_id = session_response.json()["id"]

        query_response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"text": "What changed after reading this paper?"},
        )
        arxiv_response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"arxiv_url": "https://arxiv.org/abs/2401.12345"},
        )
        pdf_response = client.post(
            f"/api/sessions/{session_id}/messages",
            json={"file_path": "C:/papers/example.pdf"},
        )
        messages_response = client.get(f"/api/sessions/{session_id}/messages")

    assert query_response.status_code == 202
    assert query_response.json()["message_type"] == "followup_query"
    assert arxiv_response.status_code == 202
    assert arxiv_response.json()["message_type"] == "ingest_arxiv"
    assert pdf_response.status_code == 202
    assert pdf_response.json()["message_type"] == "ingest_pdf"

    messages = messages_response.json()["items"]
    assert [item["type"] for item in messages] == [
        "followup_query",
        "ingest_arxiv",
        "ingest_pdf",
    ]


def test_pdf_acceptance_and_missing_run_behavior(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "PDF"})
        session_id = session_response.json()["id"]

        pdf_response = client.post(
            f"/api/sessions/{session_id}/ingest/pdf",
            json={"file_path": "C:/papers/example.pdf"},
        )
        missing_run_response = client.get(f"/api/sessions/{session_id}/runs/missing-run")

    assert pdf_response.status_code == 202
    assert pdf_response.json()["accepted"] is True
    assert missing_run_response.status_code == 404


def test_pdf_upload_endpoint_creates_ingest_run_from_browser_file(tmp_path) -> None:
    pdf_bytes = _build_minimal_pdf_bytes("Uploaded PDF text for browser ingest.")

    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Browser upload"})
        session_id = session_response.json()["id"]

        upload_response = client.post(
            f"/api/sessions/{session_id}/uploads/pdf",
            files={"file": ("browser-upload.pdf", pdf_bytes, "application/pdf")},
        )
        run_id = upload_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/ingest/{run_id}/execute")
        messages_response = client.get(f"/api/sessions/{session_id}/messages")

    assert upload_response.status_code == 202
    assert upload_response.json()["message_type"] == "ingest_pdf"
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["chunk_count"] == 1
    assert payload["source_type"] == "pdf"

    messages = messages_response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["type"] == "ingest_pdf"
    assert messages[0]["content"].endswith(".pdf")


def test_ingest_execution_endpoint_runs_mock_ingest_chain(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Ingest Execute"})
        session_id = session_response.json()["id"]

        accept_response = client.post(
            f"/api/sessions/{session_id}/ingest/arxiv",
            json={"arxiv_url": "https://arxiv.org/abs/2401.12345"},
        )
        run_id = accept_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/ingest/{run_id}/execute")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}")
        trace_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/trace")
        events_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/events")
        messages_response = client.get(f"/api/sessions/{session_id}/messages")

    assert accept_response.status_code == 202
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["task_run"]["status"] == "finished"
    assert payload["task_run"]["step_count"] == 7
    assert payload["source_type"] == "arxiv"
    assert payload["operation"] == "created"
    assert payload["summary"].startswith("已解析 arXiv PDF")
    assert payload["paper_summary"]["what_it_is_about"]
    assert payload["paper_summary"]["problem_solved"]
    assert isinstance(payload["paper_summary"]["new_ideas"], list)
    assert payload["paper_id"]
    assert payload["artifact_id"]
    assert payload["session_document_id"]

    documents = client.app.state.repositories.sessions.list_documents(session_id)
    assert len(documents) == 1
    assert documents[0].paper_id == payload["paper_id"]
    assert documents[0].artifact_id == payload["artifact_id"]
    assert client.app.state.repositories.papers.get_by_id(payload["paper_id"]) is not None
    assert client.app.state.repositories.artifacts.get_by_id(payload["artifact_id"]) is not None

    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["status"] == "finished"
    assert run_payload["step_count"] == 7

    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert [step["action"] for step in trace_payload["steps"]] == [
        "inspect_ingest_request",
        "extract_arxiv_pdf_text",
        "persist_arxiv_chunks",
        "compose_ingest_summary",
        "extract_paper_memory",
        "derive_relation_memory",
        "capture_open_questions",
    ]
    assert len(trace_payload["narratives"]) == 7
    assert trace_payload["narratives"][0]["reason_text"].startswith(
        "The runtime validates the accepted ingest request"
    )

    assert events_response.status_code == 200
    assert [item["summary"] for item in events_response.json()["items"]] == [
        "已检查导入请求",
        "已抽取 arXiv PDF 文本",
        "已保存 arXiv 分块",
        "已生成arXiv PDF摘要",
        "已抽取论文记忆",
        "已生成关系记忆",
        "已记录开放问题",
        "导入运行已完成",
    ]

    assert messages_response.status_code == 200
    messages = messages_response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == payload["summary"]


def test_query_execution_endpoint_runs_mock_retrieval_chain(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Execute"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.12345"),
                title="SQLite-backed paper",
            )
        )
        repositories.papers.save(
            Paper(
                id="paper-2",
                canonical_key=build_canonical_key(pdf_checksum="paper-2-checksum"),
                title="Comparison target",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/example.pdf",
                checksum="artifact-checksum",
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
                key_results=["Higher accuracy"],
                source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
                confidence=ConfidenceScore(value=0.9),
            )
        )
        repositories.memories.upsert_relation_memory(
            RelationMemory(
                id="relation-memory-1",
                source_paper="paper-1",
                target_paper="paper-2",
                relation_type=RelationType.COMPARES_WITH,
                summary="Compares on the same benchmark.",
                evidence=["same benchmark"],
                confidence=ConfidenceScore(value=0.8),
            )
        )

        accept_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        run_id = accept_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/queries/{run_id}/execute")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}")
        trace_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/trace")
        events_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/events")
        timeline_response = client.get(f"/api/sessions/{session_id}/timeline")
        messages_response = client.get(f"/api/sessions/{session_id}/messages")

    assert accept_response.status_code == 202
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["task_run"]["status"] == "finished"
    assert payload["task_run"]["step_count"] == 6
    assert payload["should_reread_source"] is False
    assert payload["session_memory_count"] >= 1
    assert payload["global_memory_count"] >= 1
    assert "Mock answer for:" not in payload["answer"]
    assert "记忆" in payload["answer"]
    assert "当前" in payload["answer"] or "根据当前记忆" in payload["answer"]
    assert payload["used_memory_citations"]
    assert payload["used_memory_citations"][0]["selection_reason"].startswith("type=")
    assert "rerank_strategy=model" in payload["used_memory_citations"][0]["selection_reason"]
    assert payload["matched_query_terms"]
    assert payload["memory_selection_source"] == "model"
    assert payload["memory_selection_fallback_used"] is False
    assert payload["source_selection_source"] == "rule_fallback"
    assert payload["source_selection_fallback_used"] is True

    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["status"] == "finished"
    assert run_payload["step_count"] == 6

    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert [step["action"] for step in trace_payload["steps"]] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "reread_source_passages",
        "compose_mock_answer",
    ]
    assert trace_payload["steps"][0]["input_payload"]["planner_decision"]["selected_tool"] == "search_session_memory"
    assert trace_payload["steps"][1]["input_payload"]["planner_decision"]["selected_tool"] == "search_global_memory"
    assert trace_payload["steps"][2]["input_payload"]["planner_decision"]["selected_tool"] == "rerank_candidates"
    assert trace_payload["steps"][3]["input_payload"]["planner_decision"]["selected_tool"] == "read_source_passages"
    assert trace_payload["steps"][4]["input_payload"]["planner_decision"]["selected_tool"] == "read_source_passages"
    assert trace_payload["steps"][5]["input_payload"]["planner_decision"]["selected_tool"] == "compose_answer"
    assert len(trace_payload["narratives"]) == 6
    assert trace_payload["narratives"][0]["reason_text"].startswith(
        "Session memory is checked first"
    )
    assert "paper_memory:paper-memory-1" in trace_payload["narratives"][0]["impact_text"]
    assert "reranked context candidates" in trace_payload["narratives"][2]["reason_text"].lower()
    assert "model" in trace_payload["narratives"][2]["impact_text"].lower()
    assert "source passages" in trace_payload["narratives"][4]["reason_text"].lower()

    assert events_response.status_code == 200
    event_summaries = [item["summary"] for item in events_response.json()["items"]]
    assert event_summaries[0].startswith("checked session memory: paper_memory:paper-memory-1")
    assert event_summaries[1].startswith("checked global memory: paper_memory:paper-memory-1")
    assert event_summaries[2].startswith("reranked context candidates:")
    assert event_summaries[3] == "decided whether to reread"
    assert event_summaries[4].startswith("reread source passages")
    assert event_summaries[5] == "query run completed"

    assert timeline_response.status_code == 200
    timeline_summaries = [item["summary"] for item in timeline_response.json()["items"]]
    assert timeline_summaries[0].startswith("checked session memory: paper_memory:paper-memory-1")
    assert timeline_summaries[1].startswith("checked global memory: paper_memory:paper-memory-1")
    assert timeline_summaries[2].startswith("reranked context candidates:")
    assert timeline_summaries[3] == "decided whether to reread"
    assert timeline_summaries[4].startswith("reread source passages")
    assert timeline_summaries[5] == "query run completed"

    assert messages_response.status_code == 200
    messages = messages_response.json()["items"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == payload["answer"]


def test_query_execution_endpoint_rereads_source_chunks_when_memory_is_insufficient(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Reread"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.99999"),
                title="Reread paper",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/reread.pdf",
                checksum="artifact-reread-checksum",
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
        repositories.chunks.save_many(
            [
                Chunk(
                    id="chunk-1",
                    paper_id="paper-1",
                    artifact_id="artifact-1",
                    text="The method improves accuracy over the baseline.",
                    page=1,
                    section="Abstract",
                )
            ]
        )

        accept_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        run_id = accept_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/queries/{run_id}/execute")
        trace_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/trace")
        events_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/events")

    assert accept_response.status_code == 202
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["should_reread_source"] is True
    assert payload["source_reread_chunk_count"] == 1
    assert payload["source_reread_chunks"][0]["chunk_id"] == "chunk-1"
    assert "selection_reason" in payload["source_reread_chunks"][0]
    assert "rerank_strategy=model" in payload["source_reread_chunks"][0]["selection_reason"]
    assert "\u539f\u6587\u56de\u8bfb\u5230\u7684\u5173\u952e\u7247\u6bb5" in payload["answer"]
    assert payload["used_memory_citations"] == []
    assert payload["memory_selection_source"] == "rule_fallback"
    assert payload["memory_selection_fallback_used"] is True
    assert payload["source_selection_source"] == "model"
    assert payload["source_selection_fallback_used"] is False

    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert [step["action"] for step in trace_payload["steps"]] == [
        "retrieve_session_memories",
        "retrieve_global_memories",
        "rerank_context_candidates",
        "decide_reread_source",
        "reread_source_passages",
        "compose_mock_answer",
    ]
    assert trace_payload["steps"][3]["input_payload"]["planner_decision"]["selected_tool"] == "read_source_passages"
    assert trace_payload["steps"][4]["input_payload"]["planner_decision"]["selected_tool"] == "read_source_passages"
    assert trace_payload["steps"][5]["input_payload"]["planner_decision"]["selected_tool"] == "compose_answer"
    assert len(trace_payload["narratives"]) == 6
    assert trace_payload["narratives"][4]["reason_text"].startswith(
        "The runtime reranks source passages"
    )

    assert events_response.status_code == 200
    assert [item["summary"] for item in events_response.json()["items"]] == [
        "checked session memory (no memories)",
        "checked global memory (no memories)",
        "reranked context candidates (no memories)",
        "decided whether to reread",
        "reread source passages: chunk-1",
        "query run completed",
    ]


def test_query_start_and_stream_endpoint_publish_live_events(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Live query"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.12345"),
                title="Live query paper",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/live-query.pdf",
                checksum="live-query-checksum",
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
                key_results=["Higher accuracy"],
                source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
                confidence=ConfidenceScore(value=0.9),
            )
        )

        accept_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        run_id = accept_response.json()["run_id"]

        start_response = client.post(f"/api/sessions/{session_id}/queries/{run_id}/start")
        stream_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/stream")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}")

    assert start_response.status_code == 202
    assert start_response.json()["status"] == "running"
    assert stream_response.status_code == 200
    stream_text = stream_response.text
    assert "event: run_started" in stream_text
    assert "event: step_completed" in stream_text
    assert "event: assistant_message_committed" in stream_text
    assert "event: run_finished" in stream_text
    assert "\"selected_tool\": \"search_session_memory\"" in stream_text
    assert "\"action\": \"compose_mock_answer\"" in stream_text
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "finished"


def test_ingest_start_and_stream_endpoint_publish_live_events(tmp_path) -> None:
    with _build_client(tmp_path) as client:
        session_response = client.post("/api/sessions", json={"title": "Live ingest"})
        session_id = session_response.json()["id"]

        accept_response = client.post(
            f"/api/sessions/{session_id}/ingest/arxiv",
            json={"arxiv_url": "https://arxiv.org/abs/2401.12345"},
        )
        run_id = accept_response.json()["run_id"]

        start_response = client.post(f"/api/sessions/{session_id}/ingest/{run_id}/start")
        stream_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/stream")
        run_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}")

    assert start_response.status_code == 202
    assert start_response.json()["status"] == "running"
    assert stream_response.status_code == 200
    stream_text = stream_response.text
    assert "event: run_started" in stream_text
    assert "event: step_completed" in stream_text
    assert "event: assistant_message_committed" in stream_text
    assert "event: run_finished" in stream_text
    assert "\"action\": \"compose_ingest_summary\"" in stream_text
    assert "\"action\": \"capture_open_questions\"" in stream_text
    assert run_response.status_code == 200
    assert run_response.json()["status"] == "finished"


def test_sqlite_config_runs_query_and_ingest_paths(tmp_path) -> None:
    sqlite_path = tmp_path / "api.sqlite3"
    with TestClient(create_app(storage_backend="sqlite", sqlite_path=sqlite_path)) as client:
        session_response = client.post("/api/sessions", json={"title": "SQLite session"})
        session_id = session_response.json()["id"]

        query_accept = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        query_run_id = query_accept.json()["run_id"]
        query_execute = client.post(f"/api/sessions/{session_id}/queries/{query_run_id}/execute")

        ingest_accept = client.post(
            f"/api/sessions/{session_id}/ingest/arxiv",
            json={"arxiv_url": "https://arxiv.org/abs/2401.12345"},
        )
        ingest_run_id = ingest_accept.json()["run_id"]
        ingest_execute = client.post(f"/api/sessions/{session_id}/ingest/{ingest_run_id}/execute")

        repositories = client.app.state.repositories
        session = repositories.sessions.get_by_id(session_id)
        query_run = repositories.trace.get_run(query_run_id)
        ingest_run = repositories.trace.get_run(ingest_run_id)
        documents = repositories.sessions.list_documents(session_id)

    assert session is not None
    assert query_execute.status_code == 200
    assert query_run is not None and query_run.status == "finished"
    assert ingest_execute.status_code == 200
    assert ingest_run is not None and ingest_run.status == "finished"
    assert len(documents) == 1
    assert repositories.papers.get_by_id(documents[0].paper_id) is not None
    assert repositories.artifacts.get_by_id(documents[0].artifact_id) is not None


def test_pdf_ingest_execution_extracts_chunks_in_sqlite(tmp_path) -> None:
    sqlite_path = tmp_path / "pdf-api.sqlite3"
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(_build_minimal_pdf_bytes("Local PDF text for the API path."))

    with TestClient(create_app(storage_backend="sqlite", sqlite_path=sqlite_path)) as client:
        session_response = client.post("/api/sessions", json={"title": "PDF session"})
        session_id = session_response.json()["id"]

        accept_response = client.post(
            f"/api/sessions/{session_id}/ingest/pdf",
            json={"file_path": str(pdf_path)},
        )
        run_id = accept_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/ingest/{run_id}/execute")

        repositories = client.app.state.repositories
        documents = repositories.sessions.list_documents(session_id)
        chunks = repositories.chunks.list_by_paper_ids([documents[0].paper_id])

    assert accept_response.status_code == 202
    assert execute_response.status_code == 200
    payload = execute_response.json()
    assert payload["chunk_count"] == 1
    assert payload["summary"].startswith("已解析 本地 PDF")
    assert len(documents) == 1
    assert len(chunks) == 1
    assert chunks[0].text == "Local PDF text for the API path."
    assert len(repositories.memories.list_paper_memories_for_papers([documents[0].paper_id])) == 1
    assert len(repositories.memories.list_open_question_memories_for_papers([documents[0].paper_id])) == 1
    assert len(repositories.trace.list_steps(run_id)) == 7


def test_create_app_defaults_to_sqlite_when_env_points_to_tmp_path(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_AGENT_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("RESEARCH_AGENT_SQLITE_PATH", str(tmp_path / "default.sqlite3"))

    with TestClient(create_app()) as client:
        response = client.post("/api/sessions", json={"title": "Default sqlite"})

    assert response.status_code == 201
    assert response.json()["title"] == "Default sqlite"


def test_model_adapter_planner_backend_falls_back_cleanly_when_transport_is_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_PLANNER_BACKEND", "model_adapter")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_PLANNER_PROVIDER", "deepseek")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_PLANNER_MODEL", "deepseekv4flash")
    monkeypatch.setenv("RESEARCH_AGENT_QUERY_AGENT_BACKEND", "turn_adapter")
    sqlite_path = tmp_path / "model-planner.sqlite3"
    with TestClient(create_app(storage_backend="sqlite", sqlite_path=sqlite_path)) as client:
        session_response = client.post("/api/sessions", json={"title": "Model planner fallback"})
        session_id = session_response.json()["id"]
        repositories = client.app.state.repositories
        repositories.papers.save(
            Paper(
                id="paper-1",
                canonical_key=build_canonical_key(arxiv_id="2401.12345"),
                title="Fallback paper",
            )
        )
        repositories.artifacts.save(
            Artifact(
                id="artifact-1",
                kind=ArtifactKind.LOCAL_PDF,
                uri_or_path="C:/papers/example.pdf",
                checksum="artifact-checksum",
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
                key_results=["Higher accuracy"],
                source_refs=[SourceRef(paper_id="paper-1", artifact_id="artifact-1", quote="higher accuracy")],
                confidence=ConfidenceScore(value=0.9),
            )
        )
        accept_response = client.post(
            f"/api/sessions/{session_id}/queries",
            json={"query": "Did it improve accuracy?"},
        )
        run_id = accept_response.json()["run_id"]
        execute_response = client.post(f"/api/sessions/{session_id}/queries/{run_id}/execute")
        trace_response = client.get(f"/api/sessions/{session_id}/runs/{run_id}/trace")

    assert execute_response.status_code == 200
    assert trace_response.status_code == 200
    steps = trace_response.json()["steps"]
    assert steps[0]["input_payload"]["planner_decision"]["agent_name"] == "deepseek:deepseek-v4-flash"
    assert steps[0]["input_payload"]["planner_decision"]["fallback_used"] is True


def test_create_app_loads_settings_from_env_file(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / "repo.env"
    sqlite_path = tmp_path / "from-env.sqlite3"
    env_path.write_text(
        "\n".join(
            [
                "RESEARCH_AGENT_STORAGE_BACKEND=sqlite",
                f"RESEARCH_AGENT_SQLITE_PATH={sqlite_path}",
                "RESEARCH_AGENT_QUERY_PLANNER_BACKEND=model_adapter",
                "RESEARCH_AGENT_QUERY_PLANNER_PROVIDER=deepseek",
                "RESEARCH_AGENT_QUERY_PLANNER_MODEL=deepseekv4flash",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RESEARCH_AGENT_ENV_FILE", str(env_path))
    monkeypatch.delenv("RESEARCH_AGENT_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_SQLITE_PATH", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_QUERY_PLANNER_BACKEND", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_QUERY_PLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_QUERY_PLANNER_MODEL", raising=False)

    with TestClient(create_app()) as client:
        response = client.post("/api/sessions", json={"title": "Loaded from env file"})

    assert response.status_code == 201
    assert sqlite_path.exists()
