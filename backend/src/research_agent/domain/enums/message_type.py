"""Message type enums used by the handwritten runtime and API boundaries."""

from __future__ import annotations

from enum import StrEnum


class MessageType(StrEnum):
    """Supported user message categories for the MVP."""

    INGEST_ARXIV = "ingest_arxiv"
    INGEST_PDF = "ingest_pdf"
    FOLLOWUP_QUERY = "followup_query"
