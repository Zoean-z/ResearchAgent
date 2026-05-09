"""Thin arXiv search service for lightweight metadata discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import xml.etree.ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_API_TIMEOUT_SECONDS = 12.0
ARXIV_MAX_RESULTS_LIMIT = 50
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass(frozen=True, slots=True)
class ArxivSearchPaper:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str
    published: str
    updated: str
    categories: tuple[str, ...]
    abs_url: str
    pdf_url: str


@dataclass(frozen=True, slots=True)
class ArxivSearchError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ArxivSearchResult:
    success: bool
    query: str
    count: int
    papers: tuple[ArxivSearchPaper, ...]
    error: ArxivSearchError | None = None


@dataclass(frozen=True, slots=True)
class ArxivHttpResponse:
    status_code: int
    body: bytes


class ArxivHttpGet(Protocol):
    def __call__(self, url: str, timeout_seconds: float) -> ArxivHttpResponse:
        """Return an HTTP response snapshot for the arXiv API request."""


def _default_http_get(url: str, timeout_seconds: float) -> ArxivHttpResponse:
    request = urllib_request.Request(
        url=url,
        headers={
            "User-Agent": "research-agent/1.0 (+https://export.arxiv.org)",
            "Accept": "application/atom+xml",
        },
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            return ArxivHttpResponse(
                status_code=getattr(response, "status", 200),
                body=response.read(),
            )
    except urllib_error.HTTPError as error:
        return ArxivHttpResponse(status_code=error.code, body=error.read())


class ArxivSearchService:
    """Search the official arXiv API and return lightweight paper metadata only."""

    def __init__(
        self,
        *,
        http_get: ArxivHttpGet | None = None,
        timeout_seconds: float = ARXIV_API_TIMEOUT_SECONDS,
    ) -> None:
        self._http_get = http_get or _default_http_get
        self._timeout_seconds = timeout_seconds

    def search(
        self,
        *,
        query: str,
        max_results: int = 10,
        category: str | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        start: int = 0,
    ) -> ArxivSearchResult:
        normalized_query = query.strip()
        if not normalized_query:
            return self._failure(query="", code="invalid_query", message="Query cannot be empty.")

        if start < 0:
            return self._failure(query=normalized_query, code="invalid_start", message="start must be >= 0.")

        normalized_sort_by = self._normalize_sort_by(sort_by)
        if normalized_sort_by is None:
            return self._failure(
                query=normalized_query,
                code="invalid_sort_by",
                message="sort_by must be one of: relevance, lastUpdatedDate, submittedDate.",
            )
        normalized_sort_order = self._normalize_sort_order(sort_order)
        if normalized_sort_order is None:
            return self._failure(
                query=normalized_query,
                code="invalid_sort_order",
                message="sort_order must be either ascending or descending.",
            )

        clamped_max_results = min(max(1, int(max_results)), ARXIV_MAX_RESULTS_LIMIT)
        search_query = self.build_search_query(normalized_query, category=category)
        request_url = self.build_request_url(
            search_query=search_query,
            start=start,
            max_results=clamped_max_results,
            sort_by=normalized_sort_by,
            sort_order=normalized_sort_order,
        )

        try:
            response = self._http_get(request_url, self._timeout_seconds)
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            return self._failure(
                query=normalized_query,
                code="network_error",
                message=f"Failed to reach arXiv API: {exc}",
            )

        if response.status_code != 200:
            return self._failure(
                query=normalized_query,
                code="api_error",
                message=f"arXiv API returned HTTP {response.status_code}.",
            )

        try:
            papers = self.parse_atom_response(response.body)
        except ValueError as exc:
            return self._failure(
                query=normalized_query,
                code="parse_error",
                message=str(exc),
            )

        if not papers:
            return self._failure(
                query=normalized_query,
                code="no_results",
                message="No arXiv papers matched the query.",
            )

        return ArxivSearchResult(
            success=True,
            query=normalized_query,
            count=len(papers),
            papers=papers,
            error=None,
        )

    def build_search_query(self, query: str, *, category: str | None = None) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query cannot be empty.")
        search_query = f"all:{normalized_query}"
        normalized_category = (category or "").strip()
        if normalized_category:
            search_query = f"({search_query}) AND cat:{normalized_category}"
        return search_query

    def build_request_url(
        self,
        *,
        search_query: str,
        start: int,
        max_results: int,
        sort_by: str,
        sort_order: str,
    ) -> str:
        query_params = urllib_parse.urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": max_results,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
        )
        return f"{ARXIV_API_URL}?{query_params}"

    def parse_atom_response(self, payload: bytes) -> tuple[ArxivSearchPaper, ...]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ValueError(f"Failed to parse arXiv Atom response: {exc}") from exc

        papers: list[ArxivSearchPaper] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            abs_url = self._entry_text(entry, "atom:id")
            arxiv_id = self._arxiv_id_from_abs_url(abs_url)
            title = self._clean_text(self._entry_text(entry, "atom:title"))
            abstract = self._clean_text(self._entry_text(entry, "atom:summary"))
            authors = tuple(
                self._clean_text(author.findtext("atom:name", default="", namespaces=_ATOM_NS))
                for author in entry.findall("atom:author", _ATOM_NS)
                if self._clean_text(author.findtext("atom:name", default="", namespaces=_ATOM_NS))
            )
            published = self._entry_text(entry, "atom:published")
            updated = self._entry_text(entry, "atom:updated")
            categories = tuple(
                term
                for term in (
                    category.attrib.get("term", "").strip()
                    for category in entry.findall("atom:category", _ATOM_NS)
                )
                if term
            )
            pdf_url = self._pdf_url_for_entry(entry, abs_url, arxiv_id)
            papers.append(
                ArxivSearchPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published=published,
                    updated=updated,
                    categories=categories,
                    abs_url=abs_url.replace("http://", "https://"),
                    pdf_url=pdf_url,
                )
            )
        return tuple(papers)

    def _normalize_sort_by(self, value: str) -> str | None:
        normalized = value.strip()
        allowed = {"relevance", "lastUpdatedDate", "submittedDate"}
        if normalized in allowed:
            return normalized
        lowercase = normalized.lower()
        aliases = {
            "relevance": "relevance",
            "lastupdateddate": "lastUpdatedDate",
            "submitteddate": "submittedDate",
        }
        return aliases.get(lowercase)

    def _normalize_sort_order(self, value: str) -> str | None:
        normalized = value.strip().lower()
        if normalized in {"ascending", "descending"}:
            return normalized
        return None

    def _pdf_url_for_entry(self, entry: ET.Element, abs_url: str, arxiv_id: str) -> str:
        for link in entry.findall("atom:link", _ATOM_NS):
            href = (link.attrib.get("href") or "").strip()
            title = (link.attrib.get("title") or "").strip().lower()
            if not href:
                continue
            if title == "pdf" or "/pdf/" in href:
                normalized_href = href.replace("http://", "https://")
                if normalized_href.endswith(".pdf"):
                    return normalized_href
                return normalized_href.rstrip("/") + ".pdf"
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    def _entry_text(self, entry: ET.Element, path: str) -> str:
        text = entry.findtext(path, default="", namespaces=_ATOM_NS).strip()
        if not text:
            raise ValueError(f"arXiv Atom entry is missing required field: {path}")
        return text

    def _arxiv_id_from_abs_url(self, abs_url: str) -> str:
        normalized_url = abs_url.strip().rstrip("/")
        marker = "/abs/"
        if marker not in normalized_url:
            raise ValueError("arXiv Atom entry id is not a valid abs URL.")
        return normalized_url.split(marker, 1)[1]

    def _clean_text(self, value: str) -> str:
        return " ".join(part for part in value.replace("\n", " ").split() if part)

    def _failure(self, *, query: str, code: str, message: str) -> ArxivSearchResult:
        return ArxivSearchResult(
            success=False,
            query=query,
            count=0,
            papers=(),
            error=ArxivSearchError(code=code, message=message),
        )


__all__ = [
    "ARXIV_API_TIMEOUT_SECONDS",
    "ARXIV_API_URL",
    "ARXIV_MAX_RESULTS_LIMIT",
    "ArxivHttpResponse",
    "ArxivSearchError",
    "ArxivSearchPaper",
    "ArxivSearchResult",
    "ArxivSearchService",
]
