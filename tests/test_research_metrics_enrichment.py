"""Tests for OpenAlex metrics enrichment of research records."""

from __future__ import annotations

from typing import Callable

import httpx

from whack26.config import Settings
from whack26.research.base import ResearchRecord
from whack26.research.cache import ResearchCache
from whack26.research.metrics import enrich_records

Handler = Callable[[httpx.Request], httpx.Response]

WORKS_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1234/a",
            "cited_by_count": 12,
            "authorships": [
                {"author": {"id": "https://openalex.org/A1"}},
                {"author": {"id": "https://openalex.org/A2"}},
            ],
            "primary_location": {
                "source": {
                    "type": "journal",
                    "host_organization_name": "Elsevier BV",
                }
            },
        }
    ]
}

AUTHORS_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/A1",
            "display_name": "Ada",
            "cited_by_count": 100,
            "summary_stats": {"h_index": 10},
        },
        {
            "id": "https://openalex.org/A2",
            "display_name": "Grace",
            "cited_by_count": 500,
            "summary_stats": {"h_index": 30},
        },
    ]
}

BATCH_AUTHORS_PAYLOAD = {
    "results": [
        {
            "id": f"https://openalex.org/A{index}",
            "display_name": f"Author {index}",
            "cited_by_count": index * 10,
            "summary_stats": {"h_index": index},
        }
        for index in (1, 2, 3)
    ]
}


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _record(
    *,
    record_id: str,
    source: str,
    source_id: str,
    doi: str | None = None,
    author_ids: list[str] | None = None,
    cited_by_count: int | None = None,
) -> ResearchRecord:
    return ResearchRecord(
        id=record_id,
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title=f"Record {source_id}",
        doi=doi,
        url=f"https://example.test/{source_id}",
        retrieved_at="2026-01-01T00:00:00Z",
        query="housing prices",
        author_ids=list(author_ids or []),
        cited_by_count=cited_by_count,
    )


def _enrichment_records() -> list[ResearchRecord]:
    return [
        _record(
            record_id="arxiv:a",
            source="arxiv",
            source_id="a",
            doi="10.1234/a",
        ),
        _record(
            record_id="openalex:W2",
            source="openalex",
            source_id="W2",
            author_ids=["openalex:A1", "openalex:A2"],
            cited_by_count=3,
        ),
    ]


def _forbidden_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request: {request.url}")


def _enrichment_handler(requests: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        if request.url.path == "/works":
            assert params["filter"].startswith("doi:")
            assert "10.1234/a" in params["filter"]
            assert "cited_by_count" in params["select"]
            assert "authorships" in params["select"]
            assert "primary_location" in params["select"]
            assert params["api_key"] == "test-key"
            return httpx.Response(200, json=WORKS_PAYLOAD)
        if request.url.path == "/authors":
            assert params["filter"].startswith("id:")
            assert "A1" in params["filter"]
            assert "A2" in params["filter"]
            assert "openalex:" not in params["filter"]
            assert "summary_stats" in params["select"]
            assert params["api_key"] == "test-key"
            return httpx.Response(200, json=AUTHORS_PAYLOAD)
        raise AssertionError(f"unexpected HTTP request: {request.url}")

    return handler


def test_skip_when_metrics_disabled() -> None:
    records = _enrichment_records()
    warnings: list[str] = []

    with _client(_forbidden_handler) as client:
        result = enrich_records(
            records,
            settings=Settings(research_metrics_enabled=False),
            client=client,
            warnings=warnings,
        )

    assert result == records
    assert any("research_metrics_disabled" in warning for warning in warnings)


def test_skip_when_openalex_key_missing() -> None:
    records = _enrichment_records()
    warnings: list[str] = []

    with _client(_forbidden_handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                research_metrics_enabled=True,
                openalex_api_key=None,
            ),
            client=client,
            warnings=warnings,
        )

    assert result == records
    joined = " ".join(warnings)
    assert "research_metrics_skipped" in joined
    assert "OPENALEX_API_KEY" in joined


def test_skip_when_offline() -> None:
    records = _enrichment_records()
    warnings: list[str] = []

    with _client(_forbidden_handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                research_metrics_enabled=True,
                openalex_api_key="test-key",
                offline=True,
            ),
            client=client,
            warnings=warnings,
        )

    assert result == records
    joined = " ".join(warnings).lower()
    assert "research_metrics_skipped" in joined
    assert "offline" in joined


def test_skip_when_mock_providers() -> None:
    records = _enrichment_records()
    warnings: list[str] = []

    with _client(_forbidden_handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                research_metrics_enabled=True,
                openalex_api_key="test-key",
                llm_provider="mock",
                embed_provider="mock",
            ),
            client=client,
            warnings=warnings,
        )

    assert result == records
    joined = " ".join(warnings).lower()
    assert "research_metrics_skipped" in joined
    assert "mock" in joined


def test_invalid_doi_is_not_requested() -> None:
    records = [
        _record(
            record_id="arxiv:bad-doi",
            source="arxiv",
            source_id="bad-doi",
            doi="not a doi",
        )
    ]
    warnings: list[str] = []

    with _client(_forbidden_handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                openalex_api_key="test-key",
                research_metrics_enabled=True,
            ),
            client=client,
            warnings=warnings,
        )

    assert result == records
    assert result[0].cited_by_count is None
    assert warnings == []


def test_enriches_works_and_authors_from_openalex() -> None:
    records = _enrichment_records()
    requests: list[httpx.Request] = []
    warnings: list[str] = []

    with _client(_enrichment_handler(requests)) as client:
        result = enrich_records(
            records,
            settings=Settings(
                openalex_api_key="test-key",
                research_metrics_enabled=True,
            ),
            client=client,
            warnings=warnings,
        )

    assert len(result) == 2
    enriched_a, enriched_b = result

    assert enriched_a.cited_by_count == 12
    assert enriched_a.author_ids == ["openalex:A1", "openalex:A2"]
    assert enriched_a.venue_type == "journal"
    assert enriched_a.publisher == "Elsevier BV"

    for enriched in result:
        assert enriched.author_h_index == 30
        assert enriched.author_citation_count == 500
        assert enriched.author_known is True
        assert enriched.metrics_source == "openalex"

    paths = [request.url.path for request in requests]
    assert paths.count("/works") == 1
    assert paths.count("/authors") == 1

    # Input records must not be mutated in place.
    assert records[0].cited_by_count is None
    assert records[0].author_ids == []
    assert records[1].cited_by_count == 3


def test_cache_prevents_repeat_http_calls() -> None:
    calls: list[httpx.Request] = []
    settings = Settings(
        openalex_api_key="test-key",
        research_metrics_enabled=True,
    )
    cache = ResearchCache(":memory:")

    try:
        with _client(_enrichment_handler(calls)) as client:
            first = enrich_records(
                _enrichment_records(),
                settings=settings,
                client=client,
                cache=cache,
                dataset_id="ds_metrics",
                warnings=[],
            )
            calls_after_first = len(calls)
            assert calls_after_first > 0

            second = enrich_records(
                _enrichment_records(),
                settings=settings,
                client=client,
                cache=cache,
                dataset_id="ds_metrics",
                warnings=[],
            )
    finally:
        cache.close()

    assert len(calls) == calls_after_first
    assert second == first
    assert first[0].cited_by_count == 12


def test_works_http_error_warns_and_preserves_original_values() -> None:
    records = _enrichment_records()
    warnings: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/works":
            return httpx.Response(500, json={"error": "upstream failure"})
        if request.url.path == "/authors":
            return httpx.Response(200, json=AUTHORS_PAYLOAD)
        raise AssertionError(f"unexpected HTTP request: {request.url}")

    with _client(handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                openalex_api_key="test-key",
                research_metrics_enabled=True,
            ),
            client=client,
            warnings=warnings,
        )

    assert len(result) == 2
    assert any("research_metrics_failed" in warning for warning in warnings)
    assert result[0].cited_by_count is None
    assert result[0].author_ids == []
    assert result[1].cited_by_count == 3
    assert records[0].cited_by_count is None


def test_author_lookup_batches_ids() -> None:
    records = [
        _record(
            record_id=f"openalex:W{index}",
            source="openalex",
            source_id=f"W{index}",
            author_ids=[f"openalex:A{index}"],
        )
        for index in (1, 2, 3)
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/authors"
        params = request.url.params
        filter_value = params["filter"]
        assert filter_value.startswith("id:")
        assert "|" in filter_value
        assert "openalex:" not in filter_value
        for author_id in ("A1", "A2", "A3"):
            assert author_id in filter_value
        assert int(params["per_page"]) >= 3
        assert "summary_stats" in params["select"]
        assert params["api_key"] == "test-key"
        requests.append(request)
        return httpx.Response(200, json=BATCH_AUTHORS_PAYLOAD)

    with _client(handler) as client:
        result = enrich_records(
            records,
            settings=Settings(
                openalex_api_key="test-key",
                research_metrics_enabled=True,
            ),
            client=client,
            warnings=[],
        )

    assert len(requests) == 1
    assert len(result) == 3
    assert all(record.author_known is True for record in result)
