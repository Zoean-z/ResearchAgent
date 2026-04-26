"""Trace API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from research_agent.domain.models import TraceNarrative, TraceStep


class TraceStepResponse(BaseModel):
    """Serialized raw trace step."""

    model_config = ConfigDict(extra="forbid")

    id: str
    run_id: str
    action: str
    input_payload: dict[str, Any]
    result_payload: dict[str, Any]
    status: str
    started_at: datetime
    finished_at: datetime | None

    @classmethod
    def from_domain(cls, trace_step: TraceStep) -> "TraceStepResponse":
        return cls(
            id=trace_step.id,
            run_id=trace_step.run_id,
            action=trace_step.action,
            input_payload=trace_step.input_payload,
            result_payload=trace_step.result_payload,
            status=trace_step.status,
            started_at=trace_step.started_at,
            finished_at=trace_step.finished_at,
        )


class TraceNarrativeResponse(BaseModel):
    """Serialized generated trace narrative."""

    model_config = ConfigDict(extra="forbid")

    trace_step_id: str
    reason_text: str
    impact_text: str

    @classmethod
    def from_domain(cls, narrative: TraceNarrative) -> "TraceNarrativeResponse":
        return cls(
            trace_step_id=narrative.trace_step_id,
            reason_text=narrative.reason_text,
            impact_text=narrative.impact_text,
        )


class TraceResponse(BaseModel):
    """Trace response grouped by raw steps and narratives."""

    model_config = ConfigDict(extra="forbid")

    steps: list[TraceStepResponse]
    narratives: list[TraceNarrativeResponse]
