from __future__ import annotations

from typing import Callable

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.semantic_scholar import (
    SEMANTIC_SCHOLAR_API_URL,
    SEMANTIC_SCHOLAR_FIELDS,
    SemanticScholarAdapter,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _paper(paper_id: str = "abc123", **overrides: object) -> dict:
    paper: dict = {
        "paperId": paper_id,
        "title": "Housing <em>prices</em> and income",
        "abstract": "<p>Renters and owners respond differently.</p>",
        "year": 2024,
        "authors": [
            {"authorId": "1", "name": "Ada Lovelace"},
            {"authorId": "2", "name": "Grace Hopper"},
            {"authorId": "3", "name": "   "},
        ],
        "externalIds": {"DOI": "https://doi.org/10.1234/housing.2024"},
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        "publicationDate": "2024-05-01",
        "venue": "Journal of Housing",
        "fieldsOfStudy": ["Economics"],
    }
    paper.update(overrides)
    return paper


def _payload(*papers: dict) -> dict:
    return {"total": len(papers), "offset": 0, "data": list(papers)}


def test_search_maps_graph_fields_and_sends_expected_params() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_payload(_paper()))

    adapter = SemanticScholarAdapter(min_interval=0)
    assert adapter.name == "semantic_scholar"
    assert adapter.ttl_seconds == 24 * 60 * 60
    assert adapter.limited_mode is True

    with _client(handler) as client:
        records = adapter.search("housing prices", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith(f"{SEMANTIC_SCHOLAR_API_URL}?")
    assert url.params["query"] == "housing prices"
    assert url.params["limit"] == "5"
    assert url.params["fields"] == SEMANTIC_SCHOLAR_FIELDS
    assert captured["api_key"] is None

    assert len(records) == 1
    record = records[0]
    assert record.id == "semantic_scholar:abc123"
    assert record.source == "semantic_scholar"
    assert record.source_id == "abc123"
    assert record.title == "Housing prices and income"
    assert record.abstract == "Renters and owners respond differently."
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.year == 2024
    assert record.doi == "10.1234/housing.2024"
    assert record.url == "https://www.semanticscholar.org/paper/abc123"
    assert record.license is None
    assert record.query == "housing prices"
    assert record.retrieved_at


def test_search_sends_api_key_header_when_configured() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=_payload())

    adapter = SemanticScholarAdapter(api_key="s2-secret", min_interval=0)
    assert adapter.limited_mode is False

    with _client(handler) as client:
        assert adapter.search("housing", 5, client) == []

    assert seen["api_key"] == "s2-secret"


def test_parse_builds_doi_and_semanticscholar_url_fallbacks() -> None:
    paper_without_url = _paper(
        paper_id="doi-only",
        url=None,
        externalIds={"DOI": "10.5555/legacy"},
    )
    paper_without_doi = _paper(
        paper_id="s2-only",
        url=None,
        externalIds={"ArXiv": "2401.00001"},
        year=None,
        publicationDate="2019-11-02",
    )
    body = httpx.Response(
        200,
        json=_payload(
            paper_without_url,
            paper_without_doi,
            _paper(paper_id="", title="No id"),
            _paper(paper_id="no-title", title="   "),
        ),
    ).text
    adapter = SemanticScholarAdapter(min_interval=0)

    records = adapter.parse(body, query="housing")

    assert [record.source_id for record in records] == ["doi-only", "s2-only"]
    assert records[0].doi == "10.5555/legacy"
    assert records[0].url == "https://doi.org/10.5555/legacy"
    assert records[1].doi is None
    assert records[1].url == (
        "https://www.semanticscholar.org/paper/s2-only"
    )
    assert records[1].year == 2019


def test_parse_accepts_empty_data_list() -> None:
    body = httpx.Response(200, json=_payload()).text
    assert SemanticScholarAdapter(min_interval=0).parse(body, query="housing") == []


def test_rate_limit_honors_retry_after_then_succeeds() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                429,
                text="slow down",
                headers={"Retry-After": "2"},
            )
        return httpx.Response(200, json=_payload(_paper()))

    adapter = SemanticScholarAdapter(
        min_interval=0,
        sleep=sleeps.append,
    )
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    assert calls["count"] == 2
    assert sleeps == [2.0]
    assert adapter.rate_limited is True
    assert adapter.last_retry_after == 2.0
    assert [record.source_id for record in records] == ["abc123"]


def test_persistent_rate_limit_raises_source_error() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="slow down")

    adapter = SemanticScholarAdapter(
        min_interval=0,
        sleep=sleeps.append,
    )
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert calls["count"] == 3
    assert sleeps == [1.0, 4.0]
    assert exc_info.value.status_code == 429
    assert adapter.rate_limited is True


def test_auth_failure_raises_source_error_without_leaking_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="invalid api key")

    adapter = SemanticScholarAdapter(
        api_key="s2-secret",
        min_interval=0,
        sleep=lambda _: None,
    )
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.status_code == 403
    assert "s2-secret" not in str(exc_info.value)


def test_requests_are_throttled_to_the_configured_interval() -> None:
    clock = {"now": 100.0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload())

    adapter = SemanticScholarAdapter(
        min_interval=1.0,
        sleep=sleeps.append,
        clock=lambda: clock["now"],
    )
    with _client(handler) as client:
        adapter.search("first", 5, client)
        clock["now"] = 100.25
        adapter.search("second", 5, client)

    assert sleeps == [pytest.approx(0.75)]


def test_malformed_payload_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    adapter = SemanticScholarAdapter(min_interval=0)
    with _client(handler) as client:
        with pytest.raises(SourceError, match="data list"):
            adapter.search("housing", 5, client)
