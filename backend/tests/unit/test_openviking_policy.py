"""Tests for OpenViking ownership and deletion policy helpers."""

from __future__ import annotations

from research_agent.domain.enums import MessageType
from research_agent.domain.policies import (
    build_dialogue_deletion_plan,
    build_memory_deletion_plan,
    build_openviking_ownership_plan,
    should_mirror_memory_to_openviking,
    should_mirror_message_to_openviking,
)


def test_openviking_ownership_prefers_openviking_for_memory_and_sqlite_for_runtime() -> None:
    plan = build_openviking_ownership_plan()

    assert plan.canonical_message_store == "openviking"
    assert plan.canonical_memory_store == "openviking"
    assert plan.runtime_store == "sqlite"
    assert plan.mirror_messages_to_openviking is True
    assert plan.mirror_memories_to_openviking is True
    assert plan.sqlite_keeps_runtime_snapshots is True


def test_openviking_message_mirroring_covers_current_intake_types() -> None:
    assert should_mirror_message_to_openviking(MessageType.INGEST_ARXIV) is True
    assert should_mirror_message_to_openviking(MessageType.INGEST_PDF) is True
    assert should_mirror_message_to_openviking(MessageType.FOLLOWUP_QUERY) is True


def test_openviking_memory_mirroring_covers_three_memory_types() -> None:
    assert should_mirror_memory_to_openviking("paper_memory") is True
    assert should_mirror_memory_to_openviking("relation_memory") is True
    assert should_mirror_memory_to_openviking("open_question_memory") is True
    assert should_mirror_memory_to_openviking("other") is False


def test_dialogue_deletion_plan_removes_both_message_stores() -> None:
    plan = build_dialogue_deletion_plan()

    assert plan.delete_sqlite_messages is True
    assert plan.delete_sqlite_memories is True
    assert plan.tombstone_sqlite_session is True
    assert plan.delete_openviking_messages is True
    assert plan.delete_openviking_memories is True
    assert plan.delete_openviking_session is True
    assert plan.refresh_sqlite_snapshot is True


def test_memory_deletion_plan_only_removes_memories() -> None:
    plan = build_memory_deletion_plan()

    assert plan.delete_sqlite_messages is False
    assert plan.delete_sqlite_memories is True
    assert plan.tombstone_sqlite_session is False
    assert plan.delete_openviking_messages is False
    assert plan.delete_openviking_memories is True
    assert plan.delete_openviking_session is False
