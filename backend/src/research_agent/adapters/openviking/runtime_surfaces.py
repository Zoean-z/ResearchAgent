"""Concrete OpenViking surface implementations used by services and tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import json
import re

from research_agent.adapters.openviking.surfaces import (
    OpenVikingAdapterSurfaceBundle,
    OpenVikingMemoryRecord,
    OpenVikingSearchHit,
    OpenVikingSessionSnapshot,
)


@dataclass(frozen=True, slots=True)
class OpenVikingSurfaceConfig:
    """Configuration for the repo-owned OpenViking surface adapter."""

    enabled: bool = False
    path: str = "./data/openviking"
    url: str = "http://127.0.0.1:1933"
    api_key: str | None = None
    user_memory_target_uri: str = "viking://user/memories/"


@dataclass(slots=True)
class _OpenVikingRuntimeState:
    """Shared mirrored state used by both test and SDK-backed surfaces."""

    messages_by_session: dict[str, list] = field(default_factory=lambda: defaultdict(list))
    memories_by_id: dict[str, OpenVikingMemoryRecord] = field(default_factory=dict)
    session_titles: dict[str, str | None] = field(default_factory=dict)
    committed_sessions: set[str] = field(default_factory=set)


class InMemoryOpenVikingMessageSurface:
    """Simple in-process OpenViking message mirror used by tests."""

    def __init__(self, state: _OpenVikingRuntimeState | None = None) -> None:
        self._state = state or _OpenVikingRuntimeState()

    def mirror_message(self, message) -> None:
        messages = [item for item in self._state.messages_by_session[message.session_id] if item.message_id != message.message_id]
        messages.append(message)
        messages.sort(key=lambda item: item.created_at)
        self._state.messages_by_session[message.session_id] = messages

    def list_messages(self, session_id: str) -> tuple:
        return tuple(self._state.messages_by_session.get(session_id, ()))

    def delete_message(self, session_id: str, message_id: str) -> None:
        self._state.messages_by_session[session_id] = [
            item for item in self._state.messages_by_session.get(session_id, ()) if item.message_id != message_id
        ]


class InMemoryOpenVikingMemorySurface:
    """Simple in-process memory mirror that mimics bounded OpenViking retrieval."""

    def __init__(self, state: _OpenVikingRuntimeState | None = None) -> None:
        self._state = state or _OpenVikingRuntimeState()

    def mirror_memory(self, memory: OpenVikingMemoryRecord) -> None:
        self._state.memories_by_id[memory.memory_id] = memory

    def search_session_memory(self, session_id: str, query: str, top_k: int = 5) -> tuple[OpenVikingSearchHit, ...]:
        memories = [memory for memory in self._state.memories_by_id.values() if memory.session_id == session_id]
        return self._search(memories, query=query, top_k=top_k)

    def search_global_memory(
        self,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> tuple[OpenVikingSearchHit, ...]:
        paper_filter = set(related_paper_ids or [])
        memories = list(self._state.memories_by_id.values())
        if paper_filter:
            memories = [
                memory
                for memory in memories
                if memory.paper_id in paper_filter
                or bool(paper_filter.intersection(memory.payload.get("related_papers", ())))
            ]
        return self._search(memories, query=query, top_k=top_k)

    def delete_memory(self, memory_id: str) -> None:
        self._state.memories_by_id.pop(memory_id, None)

    def _search(self, memories: list[OpenVikingMemoryRecord], *, query: str, top_k: int) -> tuple[OpenVikingSearchHit, ...]:
        scored = sorted(
            memories,
            key=lambda memory: (_match_score(self._memory_text(memory), query), memory.updated_at, memory.memory_id),
            reverse=True,
        )
        hits: list[OpenVikingSearchHit] = []
        for memory in scored[:top_k]:
            score = _normalized_score(self._memory_text(memory), query)
            hits.append(
                OpenVikingSearchHit(
                    item_kind=memory.memory_kind,
                    item_id=memory.memory_id,
                    session_id=memory.session_id,
                    score=score,
                    summary=_memory_summary(memory),
                    metadata={"paper_id": memory.paper_id, **memory.payload},
                )
            )
        return tuple(hits)

    def _memory_text(self, memory: OpenVikingMemoryRecord) -> str:
        return _memory_text(memory)


class InMemoryOpenVikingSessionSurface:
    """Simple in-process session mirror used by tests."""

    def __init__(self, state: _OpenVikingRuntimeState | None = None) -> None:
        self._state = state or _OpenVikingRuntimeState()

    def ensure_session(self, session_id: str, title: str | None = None) -> OpenVikingSessionSnapshot:
        if title is not None:
            self._state.session_titles[session_id] = title
        else:
            self._state.session_titles.setdefault(session_id, None)
        return self._snapshot(session_id)

    def commit_session(self, session_id: str) -> OpenVikingSessionSnapshot:
        self._state.committed_sessions.add(session_id)
        return self._snapshot(session_id)

    def delete_session(self, session_id: str) -> None:
        self._state.messages_by_session.pop(session_id, None)
        self._state.session_titles.pop(session_id, None)
        self._state.committed_sessions.discard(session_id)
        for memory_id in [
            memory_id
            for memory_id, memory in self._state.memories_by_id.items()
            if memory.session_id == session_id
        ]:
            self._state.memories_by_id.pop(memory_id, None)

    def _snapshot(self, session_id: str) -> OpenVikingSessionSnapshot:
        return OpenVikingSessionSnapshot(
            session_id=session_id,
            title=self._state.session_titles.get(session_id),
            message_count=len(self._state.messages_by_session.get(session_id, ())),
            memory_count=sum(1 for memory in self._state.memories_by_id.values() if memory.session_id == session_id),
            deleted=False,
        )


class SDKBackedOpenVikingMessageSurface(InMemoryOpenVikingMessageSurface):
    """Message mirror that writes to OpenViking sessions and keeps a local fallback cache."""

    def __init__(self, client: Any, state: _OpenVikingRuntimeState | None = None) -> None:
        super().__init__(state)
        self._client = client

    def mirror_message(self, message) -> None:
        super().mirror_message(message)
        session = _get_session(self._client, message.session_id)
        if session is None:
            return
        parts = [{"type": "text", "text": message.content}]
        session.add_message(message.role, parts)
        if hasattr(session, "commit"):
            session.commit()


class SDKBackedOpenVikingMemorySurface(InMemoryOpenVikingMemorySurface):
    """Memory surface that tries OpenViking search first and falls back to mirrored cache."""

    def __init__(self, client: Any, config: OpenVikingSurfaceConfig, state: _OpenVikingRuntimeState | None = None) -> None:
        super().__init__(state)
        self._client = client
        self._config = config

    def mirror_memory(self, memory: OpenVikingMemoryRecord) -> None:
        super().mirror_memory(memory)
        session = _get_session(self._client, memory.session_id) if memory.session_id else None
        if session is None:
            return
        payload = {
            "memory_id": memory.memory_id,
            "memory_kind": memory.memory_kind,
            "paper_id": memory.paper_id,
            "session_id": memory.session_id,
            "payload": memory.payload,
        }
        session.add_message(
            "assistant",
            [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        )
        if hasattr(session, "commit"):
            session.commit()

    def search_session_memory(self, session_id: str, query: str, top_k: int = 5) -> tuple[OpenVikingSearchHit, ...]:
        remote_hits = self._remote_find(query=query, top_k=top_k, session_id=session_id)
        if remote_hits:
            return remote_hits
        return super().search_session_memory(session_id=session_id, query=query, top_k=top_k)

    def search_global_memory(
        self,
        query: str,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> tuple[OpenVikingSearchHit, ...]:
        remote_hits = self._remote_find(query=query, top_k=top_k, related_paper_ids=related_paper_ids)
        if remote_hits:
            return remote_hits
        return super().search_global_memory(query=query, related_paper_ids=related_paper_ids, top_k=top_k)

    def delete_memory(self, memory_id: str) -> None:
        super().delete_memory(memory_id)
        _best_effort_delete(self._client, f"{self._config.user_memory_target_uri}{memory_id}")

    def _remote_find(
        self,
        *,
        query: str,
        top_k: int,
        session_id: str | None = None,
        related_paper_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[OpenVikingSearchHit, ...]:
        client = self._client
        if client is None or not hasattr(client, "find"):
            return ()
        try:
            raw_results = client.find(query, target_uri=self._config.user_memory_target_uri)
        except Exception:
            return ()
        hits = _coerce_find_results(raw_results)
        paper_filter = set(related_paper_ids or [])
        filtered: list[OpenVikingSearchHit] = []
        for hit in hits:
            if session_id is not None and hit.session_id not in (None, session_id):
                continue
            if paper_filter:
                hit_paper_id = hit.metadata.get("paper_id")
                hit_related_papers = set(hit.metadata.get("related_papers", ()))
                if hit_paper_id not in paper_filter and not hit_related_papers.intersection(paper_filter):
                    continue
            filtered.append(hit)
        return tuple(filtered[:top_k])


class SDKBackedOpenVikingSessionSurface(InMemoryOpenVikingSessionSurface):
    """Session surface that keeps OpenViking session creation/commit aligned with host flow."""

    def __init__(self, client: Any, state: _OpenVikingRuntimeState | None = None) -> None:
        super().__init__(state)
        self._client = client

    def ensure_session(self, session_id: str, title: str | None = None) -> OpenVikingSessionSnapshot:
        snapshot = super().ensure_session(session_id, title=title)
        _get_session(self._client, session_id)
        return snapshot

    def commit_session(self, session_id: str) -> OpenVikingSessionSnapshot:
        snapshot = super().commit_session(session_id)
        session = _get_session(self._client, session_id)
        if session is not None and hasattr(session, "commit"):
            try:
                session.commit()
            except Exception:
                pass
        return snapshot

    def delete_session(self, session_id: str) -> None:
        super().delete_session(session_id)
        _best_effort_delete(self._client, f"viking://session/{session_id}/")


def build_inmemory_openviking_surface_bundle() -> OpenVikingAdapterSurfaceBundle:
    """Create a shared in-memory OpenViking bundle for tests and local fallbacks."""

    state = _OpenVikingRuntimeState()
    return OpenVikingAdapterSurfaceBundle(
        messages=InMemoryOpenVikingMessageSurface(state),
        memories=InMemoryOpenVikingMemorySurface(state),
        sessions=InMemoryOpenVikingSessionSurface(state),
    )


def build_sdk_openviking_surface_bundle(config: OpenVikingSurfaceConfig) -> OpenVikingAdapterSurfaceBundle:
    """Create a shared SDK-backed OpenViking bundle with local fallback state."""

    client = _create_client(config)
    state = _OpenVikingRuntimeState()
    return OpenVikingAdapterSurfaceBundle(
        messages=SDKBackedOpenVikingMessageSurface(client, state),
        memories=SDKBackedOpenVikingMemorySurface(client, config, state),
        sessions=SDKBackedOpenVikingSessionSurface(client, state),
    )


def build_embedded_openviking_surface_bundle(config: OpenVikingSurfaceConfig) -> OpenVikingAdapterSurfaceBundle:
    """Create a shared embedded OpenViking bundle with local fallback state."""

    client = _create_embedded_client(config)
    state = _OpenVikingRuntimeState()
    return OpenVikingAdapterSurfaceBundle(
        messages=SDKBackedOpenVikingMessageSurface(client, state),
        memories=SDKBackedOpenVikingMemorySurface(client, config, state),
        sessions=SDKBackedOpenVikingSessionSurface(client, state),
    )


def _create_client(config: OpenVikingSurfaceConfig):
    if not config.enabled:
        return None
    try:
        import openviking as ov
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openviking package is not installed") from exc
    client = ov.SyncHTTPClient(url=config.url, api_key=config.api_key)
    client.initialize()
    return client


def _create_embedded_client(config: OpenVikingSurfaceConfig):
    if not config.enabled:
        return None
    try:
        import openviking as ov
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openviking package is not installed") from exc
    client = ov.OpenViking(path=config.path)
    client.initialize()
    return client


def _get_session(client: Any, session_id: str):
    if client is None:
        return None
    if hasattr(client, "get_session"):
        return client.get_session(session_id=session_id, auto_create=True)
    if hasattr(client, "session"):
        return client.session(session_id=session_id)
    return None


def _best_effort_delete(client: Any, uri: str) -> None:
    if client is None:
        return
    for method_name in ("rm", "remove", "delete"):
        method = getattr(client, method_name, None)
        if method is None:
            continue
        try:
            method(uri)
        except Exception:
            pass
        return


def _coerce_find_results(raw_results: Any) -> tuple[OpenVikingSearchHit, ...]:
    collections: list[Any] = []
    for attr in ("memories", "resources", "matches", "items"):
        value = getattr(raw_results, attr, None)
        if value:
            collections.extend(value)
    if not collections and isinstance(raw_results, list):
        collections.extend(raw_results)

    hits: list[OpenVikingSearchHit] = []
    for item in collections:
        metadata = _item_value(item, "metadata", {}) or {}
        uri = _item_value(item, "uri", "") or ""
        item_id = metadata.get("memory_id") or _memory_id_from_uri(uri)
        if not item_id:
            continue
        hits.append(
            OpenVikingSearchHit(
                item_kind=str(metadata.get("memory_kind") or _item_value(item, "kind", "memory")),
                item_id=str(item_id),
                session_id=metadata.get("session_id"),
                score=float(_item_value(item, "score", 0.0) or 0.0),
                summary=str(
                    _item_value(item, "summary", None)
                    or _item_value(item, "abstract", None)
                    or _item_value(item, "name", None)
                    or uri
                    or item_id
                ),
                metadata=dict(metadata),
            )
        )
    hits.sort(key=lambda item: (item.score, item.item_id), reverse=True)
    return tuple(hits)


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _memory_id_from_uri(uri: str) -> str | None:
    match = re.search(r"/([^/]+?)(?:\.md)?$", uri)
    if match is None:
        return None
    return match.group(1)


def _memory_summary(memory: OpenVikingMemoryRecord) -> str:
    payload = memory.payload
    if memory.memory_kind == "paper_memory":
        parts = [
            payload.get("problem"),
            payload.get("method"),
            " ".join(payload.get("key_results", ())),
            payload.get("novelty_claim"),
        ]
        return " | ".join(part for part in parts if part) or memory.memory_id
    if memory.memory_kind == "relation_memory":
        return str(payload.get("summary") or memory.memory_id)
    if memory.memory_kind == "open_question_memory":
        return str(payload.get("unresolved_question") or memory.memory_id)
    return memory.memory_id


def _memory_text(memory: OpenVikingMemoryRecord) -> str:
    return json.dumps(memory.payload, ensure_ascii=False).lower()


def _query_terms(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _match_score(text: str, query: str) -> int:
    return sum(1 for term in _query_terms(query) if term in text)


def _normalized_score(text: str, query: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    matches = sum(1 for term in terms if term in text)
    return min(1.0, matches / max(1, len(terms)))


__all__ = [
    "InMemoryOpenVikingMemorySurface",
    "InMemoryOpenVikingMessageSurface",
    "InMemoryOpenVikingSessionSurface",
    "OpenVikingSurfaceConfig",
    "SDKBackedOpenVikingMemorySurface",
    "SDKBackedOpenVikingMessageSurface",
    "SDKBackedOpenVikingSessionSurface",
    "build_inmemory_openviking_surface_bundle",
    "build_embedded_openviking_surface_bundle",
    "build_sdk_openviking_surface_bundle",
]
