"""Utilities for validating and normalizing arXiv references in query tools."""

from __future__ import annotations

import re

from research_agent.domain.policies import normalize_arxiv_id

_ARXIV_ID_BODY = r"\d{4}\.\d{4,5}(?:v\d+)?"
_DIRECT_ARXIV_ID_PATTERN = re.compile(rf"^(?:arxiv:)?(?P<arxiv_id>{_ARXIV_ID_BODY})$", re.IGNORECASE)
_DIRECT_ARXIV_URL_PATTERN = re.compile(
    rf"^https?://(?:www\.)?arxiv\.org/(?P<kind>abs|pdf)/(?P<arxiv_id>{_ARXIV_ID_BODY})(?:\.pdf)?/?$",
    re.IGNORECASE,
)
_EMBEDDED_ARXIV_URL_PATTERN = re.compile(
    rf"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/{_ARXIV_ID_BODY}(?:\.pdf)?/?",
    re.IGNORECASE,
)
_EMBEDDED_ARXIV_ID_PATTERN = re.compile(rf"(?<!\d)(?:arxiv:)?(?P<arxiv_id>{_ARXIV_ID_BODY})(?!\d)", re.IGNORECASE)
_ARXIV_IMPORT_INTENT_PATTERN = re.compile(
    r"(?:\bimport\b|\bingest\b|\bload\b|导入|读入|加入(?:到)?(?:当前)?(?:会话|session)?|添加(?:到)?(?:当前)?(?:会话|session)?)",
    re.IGNORECASE,
)


def normalize_arxiv_id_or_url(value: str) -> str:
    """Return a canonical arXiv abs URL from an id or supported arXiv URL."""

    text = value.strip()
    if not text:
        raise ValueError("arXiv id or URL cannot be empty.")

    url_match = _DIRECT_ARXIV_URL_PATTERN.fullmatch(text)
    if url_match is not None:
        return build_arxiv_abs_url(url_match.group("arxiv_id"))

    id_match = _DIRECT_ARXIV_ID_PATTERN.fullmatch(text)
    if id_match is not None:
        return build_arxiv_abs_url(id_match.group("arxiv_id"))

    raise ValueError("Expected an arXiv id or an arXiv abs/pdf URL.")


def extract_arxiv_id_or_url_from_text(text: str) -> str | None:
    """Extract and normalize an explicit arXiv reference from free text."""

    url_match = _EMBEDDED_ARXIV_URL_PATTERN.search(text)
    if url_match is not None:
        return normalize_arxiv_id_or_url(url_match.group(0))

    stripped = text.strip()
    if _DIRECT_ARXIV_ID_PATTERN.fullmatch(stripped) is not None:
        return normalize_arxiv_id_or_url(stripped)

    if _ARXIV_IMPORT_INTENT_PATTERN.search(text) is None:
        return None

    id_match = _EMBEDDED_ARXIV_ID_PATTERN.search(text)
    if id_match is None:
        return None
    return build_arxiv_abs_url(id_match.group("arxiv_id"))


def build_arxiv_abs_url(arxiv_id: str) -> str:
    """Return the canonical arXiv abs URL for a normalized id."""

    return f"https://arxiv.org/abs/{normalize_arxiv_id(arxiv_id.removesuffix('.pdf'))}"


__all__ = [
    "build_arxiv_abs_url",
    "extract_arxiv_id_or_url_from_text",
    "normalize_arxiv_id_or_url",
]
