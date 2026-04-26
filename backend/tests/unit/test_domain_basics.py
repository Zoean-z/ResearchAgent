"""Domain-level tests for enums and value objects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.domain.enums import MessageType, RelationType, TaskRunStatus
from research_agent.domain.policies import build_canonical_key
from research_agent.domain.value_objects import ConfidenceScore


def test_message_type_values_are_stable() -> None:
    assert MessageType.INGEST_ARXIV.value == "ingest_arxiv"
    assert MessageType.INGEST_PDF.value == "ingest_pdf"
    assert MessageType.FOLLOWUP_QUERY.value == "followup_query"


def test_task_run_status_values_are_stable() -> None:
    assert TaskRunStatus.PENDING.value == "pending"
    assert TaskRunStatus.STEP_LIMIT_REACHED.value == "step_limit_reached"


def test_build_canonical_key_prefers_arxiv_id() -> None:
    key = build_canonical_key(arxiv_id="ArXiv:2401.12345", pdf_checksum="deadbeef")
    assert key.value == "paper:arxiv:2401.12345"


def test_build_canonical_key_uses_pdf_checksum_without_arxiv() -> None:
    key = build_canonical_key(pdf_checksum="ABC123")
    assert key.value == "paper:pdf:abc123"


def test_confidence_score_must_stay_within_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        ConfidenceScore(value=1.1)

    with pytest.raises(ValidationError):
        ConfidenceScore(value=-0.1)

    assert ConfidenceScore(value=0.7).value == 0.7


def test_relation_type_enum_rejects_unknown_values() -> None:
    assert RelationType("compares_with") is RelationType.COMPARES_WITH

    with pytest.raises(ValueError):
        RelationType("unknown")
