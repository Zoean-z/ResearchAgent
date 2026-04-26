"""Canonical paper identity value object."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CanonicalKey(BaseModel):
    """Stable identity key for a paper record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def validate_prefix(cls, value: str) -> str:
        if value.startswith("paper:arxiv:") or value.startswith("paper:pdf:"):
            return value
        raise ValueError("CanonicalKey must start with 'paper:arxiv:' or 'paper:pdf:'.")

    @classmethod
    def from_arxiv_id(cls, arxiv_id: str) -> "CanonicalKey":
        return cls(value=f"paper:arxiv:{arxiv_id.strip()}")

    @classmethod
    def from_pdf_checksum(cls, checksum: str) -> "CanonicalKey":
        return cls(value=f"paper:pdf:{checksum.strip().lower()}")

    def __str__(self) -> str:
        return self.value
