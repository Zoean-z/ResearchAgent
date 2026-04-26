"""OpenViking ownership and deletion policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.domain.enums import MessageType


@dataclass(frozen=True, slots=True)
class OpenVikingOwnershipPlan:
    """Canonical ownership plan for OpenViking and SQLite."""

    canonical_message_store: str = "openviking"
    canonical_memory_store: str = "openviking"
    runtime_store: str = "sqlite"
    mirror_messages_to_openviking: bool = True
    mirror_memories_to_openviking: bool = True
    mirror_trace_to_openviking: bool = False
    mirror_timeline_to_openviking: bool = False
    sqlite_keeps_runtime_snapshots: bool = True


@dataclass(frozen=True, slots=True)
class OpenVikingDeletionPlan:
    """Deletion plan for dialogue and memory data."""

    delete_sqlite_messages: bool
    delete_sqlite_memories: bool
    tombstone_sqlite_session: bool
    delete_openviking_messages: bool
    delete_openviking_memories: bool
    delete_openviking_session: bool
    refresh_sqlite_snapshot: bool = True


def build_openviking_ownership_plan() -> OpenVikingOwnershipPlan:
    """Return the default ownership plan for the repository."""

    return OpenVikingOwnershipPlan()


def should_mirror_message_to_openviking(message_type: MessageType) -> bool:
    """Mirror all accepted intake messages to OpenViking."""

    return message_type in {
        MessageType.INGEST_ARXIV,
        MessageType.INGEST_PDF,
        MessageType.FOLLOWUP_QUERY,
    }


def should_mirror_memory_to_openviking(memory_kind: str) -> bool:
    """Mirror structured memory items to OpenViking."""

    return memory_kind in {"paper_memory", "relation_memory", "open_question_memory"}


def build_dialogue_deletion_plan() -> OpenVikingDeletionPlan:
    """Build the deletion plan for removing a dialogue thread."""

    return OpenVikingDeletionPlan(
        delete_sqlite_messages=True,
        delete_sqlite_memories=True,
        tombstone_sqlite_session=True,
        delete_openviking_messages=True,
        delete_openviking_memories=True,
        delete_openviking_session=True,
    )


def build_memory_deletion_plan() -> OpenVikingDeletionPlan:
    """Build the deletion plan for removing stored memories."""

    return OpenVikingDeletionPlan(
        delete_sqlite_messages=False,
        delete_sqlite_memories=True,
        tombstone_sqlite_session=False,
        delete_openviking_messages=False,
        delete_openviking_memories=True,
        delete_openviking_session=False,
    )
