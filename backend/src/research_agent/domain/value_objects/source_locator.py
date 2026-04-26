"""Source location value object for pages, sections, and chunk ids."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SourceLocator(BaseModel):
    """Logical location inside a paper artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page: int | None = Field(default=None, ge=1)
    section: str | None = None
    chunk_id: str | None = None

    def is_empty(self) -> bool:
        return self.page is None and self.section is None and self.chunk_id is None
