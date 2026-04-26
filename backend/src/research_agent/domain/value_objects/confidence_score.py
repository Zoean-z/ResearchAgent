"""Confidence score value object."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceScore(BaseModel):
    """Normalized confidence score between 0 and 1 inclusive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(ge=0.0, le=1.0)

    def is_low(self, threshold: float = 0.6) -> bool:
        return self.value < threshold
