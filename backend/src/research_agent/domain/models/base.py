"""Shared model primitives for domain entities."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class DomainModel(BaseModel):
    """Base configuration for domain entities."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
