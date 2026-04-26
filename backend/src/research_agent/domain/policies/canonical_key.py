"""Canonical key policy rules."""

from __future__ import annotations

from research_agent.domain.value_objects import CanonicalKey


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize arXiv ids to a stable lowercase representation."""

    normalized = arxiv_id.strip().lower()
    if normalized.startswith("arxiv:"):
        normalized = normalized.removeprefix("arxiv:")
    if not normalized:
        raise ValueError("arXiv id cannot be empty.")
    return normalized


def build_canonical_key(arxiv_id: str | None = None, pdf_checksum: str | None = None) -> CanonicalKey:
    """Build a canonical key according to the project spec."""

    if arxiv_id:
        return CanonicalKey.from_arxiv_id(normalize_arxiv_id(arxiv_id))
    if pdf_checksum:
        return CanonicalKey.from_pdf_checksum(pdf_checksum)
    raise ValueError("Either arxiv_id or pdf_checksum must be provided.")
