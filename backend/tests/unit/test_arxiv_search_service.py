from __future__ import annotations

from urllib import error as urllib_error

from research_agent.services import ArxivHttpResponse, ArxivSearchService
from research_agent.tools.protocol import ImportArxivPaperInput


_ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2026-01-15T00:00:00Z</updated>
    <published>2026-01-10T00:00:00Z</published>
    <title>  Memory Routed Paper Agents  </title>
    <summary>
      This paper studies memory-first paper agents.
    </summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2.pdf" rel="related" type="application/pdf" />
    <category term="cs.CL" />
    <category term="cs.AI" />
  </entry>
</feed>
"""


def test_arxiv_search_service_parses_mocked_atom_response() -> None:
    service = ArxivSearchService(
        http_get=lambda url, timeout: ArxivHttpResponse(status_code=200, body=_ATOM_FEED.encode("utf-8"))
    )

    result = service.search(query="memory routed paper agents")

    assert result.success is True
    assert result.count == 1
    paper = result.papers[0]
    assert paper.arxiv_id == "2401.12345v2"
    assert paper.title == "Memory Routed Paper Agents"
    assert paper.authors == ("Alice", "Bob")
    assert paper.abstract == "This paper studies memory-first paper agents."
    assert paper.categories == ("cs.CL", "cs.AI")
    assert paper.abs_url == "https://arxiv.org/abs/2401.12345v2"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345v2.pdf"


def test_arxiv_search_service_returns_structured_empty_result_error() -> None:
    empty_feed = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    service = ArxivSearchService(
        http_get=lambda url, timeout: ArxivHttpResponse(status_code=200, body=empty_feed.encode("utf-8"))
    )

    result = service.search(query="no hits please")

    assert result.success is False
    assert result.count == 0
    assert result.papers == ()
    assert result.error is not None
    assert result.error.code == "no_results"


def test_arxiv_search_service_clamps_max_results_and_builds_category_query() -> None:
    captured: dict[str, object] = {}

    def _http_get(url: str, timeout: float) -> ArxivHttpResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return ArxivHttpResponse(status_code=200, body=_ATOM_FEED.encode("utf-8"))

    service = ArxivSearchService(http_get=_http_get, timeout_seconds=7.5)

    result = service.search(
        query="memory routing",
        max_results=999,
        category="cs.LG",
        sort_by="relevance",
        sort_order="descending",
    )

    assert result.success is True
    request_url = str(captured["url"])
    assert "max_results=50" in request_url
    assert "cat%3Acs.LG" in request_url
    assert "all%3Amemory+routing" in request_url
    assert captured["timeout"] == 7.5


def test_arxiv_search_service_returns_structured_network_error() -> None:
    service = ArxivSearchService(
        http_get=lambda url, timeout: (_ for _ in ()).throw(urllib_error.URLError("down"))
    )

    result = service.search(query="memory routing")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "network_error"


def test_arxiv_search_service_returns_structured_api_error() -> None:
    service = ArxivSearchService(
        http_get=lambda url, timeout: ArxivHttpResponse(status_code=503, body=b"unavailable")
    )

    result = service.search(query="memory routing")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "api_error"


def test_arxiv_search_result_arxiv_id_is_compatible_with_import_tool() -> None:
    service = ArxivSearchService(
        http_get=lambda url, timeout: ArxivHttpResponse(status_code=200, body=_ATOM_FEED.encode("utf-8"))
    )

    result = service.search(query="memory routed paper agents")

    assert result.success is True
    imported = ImportArxivPaperInput(arxiv_id_or_url=result.papers[0].arxiv_id)
    assert imported.arxiv_id_or_url == "https://arxiv.org/abs/2401.12345v2"
