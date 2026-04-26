"""Tests for OpenViking message, memory, and session surfaces."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from research_agent.adapters.openviking import (
    NoopOpenVikingMemorySurface,
    NoopOpenVikingMessageSurface,
    NoopOpenVikingSessionSurface,
    OpenVikingAdapterSurfaceBundle,
    OpenVikingMemoryRecord,
    OpenVikingMemorySurface,
    OpenVikingMessageRecord,
    OpenVikingMessageSurface,
    OpenVikingSearchHit,
    OpenVikingSessionSurface,
    OpenVikingSurfaceConfig,
    build_embedded_openviking_surface_bundle,
)


def test_openviking_surface_bundle_defaults_to_noop_implementations() -> None:
    bundle = OpenVikingAdapterSurfaceBundle()

    assert isinstance(bundle.messages, OpenVikingMessageSurface)
    assert isinstance(bundle.memories, OpenVikingMemorySurface)
    assert isinstance(bundle.sessions, OpenVikingSessionSurface)
    assert isinstance(bundle.messages, NoopOpenVikingMessageSurface)
    assert isinstance(bundle.memories, NoopOpenVikingMemorySurface)
    assert isinstance(bundle.sessions, NoopOpenVikingSessionSurface)


def test_noop_message_surface_is_safe_to_call() -> None:
    surface = NoopOpenVikingMessageSurface()
    record = OpenVikingMessageRecord(
        session_id="session-1",
        message_id="message-1",
        role="user",
        content="hello",
    )

    assert surface.mirror_message(record) is None
    assert surface.list_messages("session-1") == ()
    assert surface.delete_message("session-1", "message-1") is None


def test_noop_memory_surface_is_safe_to_call() -> None:
    surface = NoopOpenVikingMemorySurface()
    memory = OpenVikingMemoryRecord(
        memory_id="memory-1",
        memory_kind="paper_memory",
        session_id="session-1",
        paper_id="paper-1",
        payload={"summary": "paper"},
    )

    assert surface.mirror_memory(memory) is None
    assert surface.search_session_memory("session-1", "query") == ()
    assert surface.search_global_memory("query", related_paper_ids=["paper-1"]) == ()
    assert surface.delete_memory("memory-1") is None


def test_noop_session_surface_is_safe_to_call() -> None:
    surface = NoopOpenVikingSessionSurface()

    snapshot = surface.ensure_session("session-1", title="Session Title")

    assert snapshot.session_id == "session-1"
    assert snapshot.title == "Session Title"
    assert snapshot.message_count == 0
    assert snapshot.memory_count == 0
    assert surface.commit_session("session-1").session_id == "session-1"
    assert surface.delete_session("session-1") is None


def test_openviking_search_hit_model_is_structured() -> None:
    hit = OpenVikingSearchHit(
        item_kind="memory",
        item_id="memory-1",
        session_id="session-1",
        score=0.75,
        summary="relevant result",
    )

    assert hit.item_kind == "memory"
    assert hit.score == 0.75


def test_embedded_openviking_surface_bundle_uses_local_client(monkeypatch, tmp_path) -> None:
    created_clients = []

    class FakeOpenVikingClient:
        def __init__(self, path: str) -> None:
            self.path = path
            self.initialized = False
            self.sessions: dict[str, FakeOpenVikingSession] = {}
            created_clients.append(self)

        def initialize(self) -> None:
            self.initialized = True

        def get_session(self, session_id: str, auto_create: bool = True):  # noqa: FBT001, FBT002
            if auto_create or session_id in self.sessions:
                self.sessions.setdefault(session_id, FakeOpenVikingSession(session_id))
                return self.sessions[session_id]
            return None

        def find(self, query: str, target_uri: str):  # noqa: ARG002
            hits = []
            for session in self.sessions.values():
                for memory in session.memory_payloads:
                    text = " ".join(str(value) for value in memory["payload"].values()).lower()
                    if query.lower().split()[0] in text:
                        hits.append(
                            {
                                "metadata": {
                                    "memory_id": memory["memory_id"],
                                    "memory_kind": memory["memory_kind"],
                                    "session_id": memory["session_id"],
                                    "paper_id": memory["paper_id"],
                                },
                                "score": 0.9,
                                "summary": text,
                            }
                        )
            return SimpleNamespace(memories=hits)

    class FakeOpenVikingSession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            self.messages: list[tuple[str, list[dict[str, str]]]] = []
            self.memory_payloads: list[dict[str, object]] = []
            self.commits = 0

        def add_message(self, role: str, parts) -> None:  # noqa: ANN001
            self.messages.append((role, parts))
            if role == "assistant" and parts:
                text = parts[0]["text"]
                if "\"memory_kind\"" in text and "\"memory_id\"" in text:
                    self.memory_payloads.append(
                        {
                            "memory_id": f"memory-{len(self.memory_payloads) + 1}",
                            "memory_kind": "paper_memory",
                            "session_id": self.session_id,
                            "paper_id": "paper-1",
                            "payload": {"summary": text},
                        }
                    )

        def commit(self) -> dict[str, str]:
            self.commits += 1
            return {"status": "accepted", "task_id": f"task-{self.commits}"}

    monkeypatch.setitem(sys.modules, "openviking", SimpleNamespace(OpenViking=FakeOpenVikingClient))

    bundle = build_embedded_openviking_surface_bundle(
        OpenVikingSurfaceConfig(enabled=True, path=str(tmp_path / "ov-data"))
    )

    bundle.sessions.ensure_session("session-1", title="Embedded")
    bundle.messages.mirror_message(
        OpenVikingMessageRecord(
            session_id="session-1",
            message_id="message-1",
            role="user",
            content="embedded hello",
        )
    )
    bundle.memories.mirror_memory(
        OpenVikingMemoryRecord(
            memory_id="memory-1",
            memory_kind="paper_memory",
            session_id="session-1",
            paper_id="paper-1",
            payload={"summary": "embedded hello world"},
        )
    )

    hits = bundle.memories.search_session_memory("session-1", "hello", top_k=5)

    assert created_clients[0].path == str(tmp_path / "ov-data")
    assert created_clients[0].initialized is True
    assert bundle.messages.list_messages("session-1")[0].content == "embedded hello"
    assert hits and hits[0].item_id == "memory-1"
