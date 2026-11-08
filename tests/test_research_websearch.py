from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.cache import ResearchCache
from whack26.research.websearch import (
    BRAVE_ENDPOINT,
    GOOGLE_ENDPOINT,
    WebSearchAdapter,
)

GOOGLE_RESULT = {
    "kind": "customsearch#search",
    "items": [
        {
            "title": "<b>Housing</b> prices &amp; rents",
            "link": (
                "https://Example.com/report/?utm_source=news&utm_medium=email"
                "&id=7#section"
            ),
            "snippet": "<p>Prices   and rents.</p>",
            "pagemap": {
                "metatags": [
                    {"article:published_time": "2021-05-01T00:00:00Z"},
                ]
            },
        },
        {
            "title": "Housing without a date",
            "link": "https://example.org/no-date",
            "snippet": "No date is available.",
        },
    ],
}

BRAVE_RESULT = {
    "type": "search",
    "web": {
        "type": "search",
        "results": [
            {
                "title": "Housing report",
                "url": "https://example.com/report?utm_medium=social",
                "description": "A <strong>report</strong> on rents.",
                "page_age": "2022-03-04T00:00:00",
            }
        ],
    },
}


def _no_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP request to {request.url}")


def test_google_maps_items_params_and_canonical_urls() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url.copy_with(query=None))
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=GOOGLE_RESULT)

    adapter = WebSearchAdapter(
        env={
            "WEBSEARCH_PROVIDER": "google",
            "WEBSEARCH_API_KEY": "google_key",
            "WEBSEARCH_CX": "engine_id",
        }
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("  Housing   PRICES ", 5, client)

    assert seen["url"] == GOOGLE_ENDPOINT
    assert seen["params"] == {
        "key": "google_key",
        "cx": "engine_id",
        "q": "housing prices",
        "num": "5",
    }
    assert len(records) == 2

    first = records[0]
    assert first.id == "websearch:https://example.com/report?id=7"
    assert first.source == "websearch"
    assert first.source_id == "https://example.com/report?id=7"
    assert first.title == "Housing prices & rents"
    assert first.abstract == "Prices and rents."
    assert first.url == "https://example.com/report?id=7"
    assert first.year == 2021
    assert first.authors == []
    assert first.doi is None
    assert first.license is None

    second = records[1]
    assert second.url == "https://example.org/no-date"
    assert second.abstract == "No date is available."
    assert second.year is None
    client.close()


def test_google_caps_num_at_provider_limit() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"items": []})

    adapter = WebSearchAdapter(
        provider="google", api_key="k", cx="c"
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert adapter.search("housing", 50, client) == []
    assert seen["num"] == "10"
    client.close()


def test_brave_maps_results_and_sends_subscription_header() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url.copy_with(query=None))
        seen["params"] = dict(request.url.params)
        seen["token"] = request.headers.get("x-subscription-token")
        return httpx.Response(200, json=BRAVE_RESULT)

    adapter = WebSearchAdapter(
        env={"WEBSEARCH_PROVIDER": "brave", "WEBSEARCH_API_KEY": "brave_key"}
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("Housing Prices", 5, client)

    assert adapter.provider == "brave"
    assert seen["url"] == BRAVE_ENDPOINT
    assert seen["params"] == {"q": "housing prices", "count": "5"}
    assert seen["token"] == "brave_key"
    assert len(records) == 1
    record = records[0]
    assert record.id == "websearch:https://example.com/report"
    assert record.source == "websearch"
    assert record.title == "Housing report"
    assert record.abstract == "A report on rents."
    assert record.url == "https://example.com/report"
    assert record.year == 2022
    client.close()


def test_brave_missing_web_block_is_empty_not_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "search"})

    adapter = WebSearchAdapter(provider="brave", api_key="k")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert adapter.search("housing", 5, client) == []
    client.close()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "WEBSEARCH_PROVIDER"),
        ({"WEBSEARCH_PROVIDER": "none", "WEBSEARCH_API_KEY": "k"}, "not set"),
        ({"WEBSEARCH_PROVIDER": "google", "WEBSEARCH_CX": "c"}, "API_KEY"),
        ({"WEBSEARCH_PROVIDER": "google", "WEBSEARCH_API_KEY": "k"}, "CX"),
        (
            {"WEBSEARCH_PROVIDER": "bing", "WEBSEARCH_API_KEY": "k"},
            "unsupported",
        ),
    ],
)
def test_unconfigured_provider_skips_without_http(
    env: dict[str, str], expected: str
) -> None:
    adapter = WebSearchAdapter(env=env)
    client = httpx.Client(transport=httpx.MockTransport(_no_http))
    assert adapter.search("housing", 5, client) == []
    assert adapter.warnings
    assert expected in adapter.warnings[0]
    client.close()


def test_provider_none_is_disabled_even_with_a_key() -> None:
    adapter = WebSearchAdapter(
        env={"WEBSEARCH_PROVIDER": "none", "WEBSEARCH_API_KEY": "k"}
    )
    assert adapter.enabled is False
    assert adapter.limited_mode is False
    client = httpx.Client(transport=httpx.MockTransport(_no_http))
    assert adapter.search("housing", 5, client) == []
    client.close()


def test_named_provider_without_key_marks_limited_mode() -> None:
    adapter = WebSearchAdapter(env={"WEBSEARCH_PROVIDER": "brave"})
    assert adapter.enabled is False
    assert adapter.limited_mode is True


def test_google_dedupes_canonical_urls_and_skips_bad_schemes() -> None:
    payload = {
        "items": [
            {
                "title": "Housing report",
                "link": "https://example.com/report?utm_source=a",
            },
            {
                "title": "Duplicate housing report",
                "link": "https://example.com/report/?utm_medium=b#top",
            },
            {"title": "FTP result", "link": "ftp://example.com/file"},
            {"title": "Hostless", "link": "not-a-url"},
            {"title": "   ", "link": "https://example.com/blank"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    adapter = WebSearchAdapter(provider="google", api_key="k", cx="c")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 10, client)

    assert [record.url for record in records] == ["https://example.com/report"]
    client.close()


def test_google_quota_payload_raises_source_error_and_warns() -> None:
    payload = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for quota metric 'Queries'",
            "errors": [{"reason": "rateLimitExceeded"}],
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    adapter = WebSearchAdapter(provider="google", api_key="k", cx="c")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="Quota exceeded"):
        adapter.search("housing", 5, client)

    assert adapter.limited_mode is True
    assert any("quota" in warning.lower() for warning in adapter.warnings)
    client.close()


def test_google_http_403_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    adapter = WebSearchAdapter(provider="google", api_key="k", cx="c")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as excinfo:
        adapter.search("housing", 5, client)

    assert excinfo.value.status_code == 403
    assert adapter.limited_mode is True
    client.close()


def test_brave_rate_limit_retries_then_raises_source_error() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="rate limited")

    adapter = WebSearchAdapter(
        provider="brave", api_key="k", sleep=lambda _: None
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError) as excinfo:
        adapter.search("housing", 5, client)

    assert excinfo.value.status_code == 429
    assert calls["count"] == 3
    assert adapter.limited_mode is True
    assert any("429" in warning for warning in adapter.warnings)
    client.close()


def test_query_normalization_shares_the_cache_entry(tmp_path: Path) -> None:
    calls = {"count": 0}
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        queries.append(request.url.params["q"])
        return httpx.Response(200, json=GOOGLE_RESULT)

    cache = ResearchCache(tmp_path / "research.db")
    adapter = WebSearchAdapter(
        provider="google",
        api_key="k",
        cx="c",
        cache=cache,
        dataset_id="ds_a",
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = adapter.search("  Housing   PRICES ", 5, client)
    second = adapter.search("housing prices", 5, client)

    assert queries == ["housing prices"]
    assert calls["count"] == 1
    assert [record.id for record in first] == [record.id for record in second]
    assert cache.stats()["hits"] == 1

    client.close()
    cache.close()


def test_cache_key_includes_provider(tmp_path: Path) -> None:
    calls = {"count": 0}

    def google_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"items": []})

    def brave_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"web": {"results": []}})

    cache = ResearchCache(tmp_path / "research.db")
    common = {"cache": cache, "dataset_id": "ds_a"}
    google = WebSearchAdapter(provider="google", api_key="k", cx="c", **common)
    brave = WebSearchAdapter(provider="brave", api_key="k", **common)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (
                google_handler(request)
                if request.url.host == "www.googleapis.com"
                else brave_handler(request)
            )
        )
    )

    assert google.search("housing", 5, client) == []
    assert brave.search("housing", 5, client) == []
    assert calls["count"] == 2

    client.close()
    cache.close()


def test_offline_cache_miss_raises_without_http(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db", offline=True)
    adapter = WebSearchAdapter(
        provider="google",
        api_key="k",
        cx="c",
        cache=cache,
        dataset_id="ds_a",
        offline=True,
    )
    client = httpx.Client(transport=httpx.MockTransport(_no_http))
    with pytest.raises(SourceError, match="offline cache miss"):
        adapter.search("housing", 5, client)
    client.close()
    cache.close()


def test_malformed_json_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{not json")

    adapter = WebSearchAdapter(provider="google", api_key="k", cx="c")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="could not parse response"):
        adapter.search("housing", 5, client)
    client.close()


def test_wrong_items_shape_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": {"unexpected": True}})

    adapter = WebSearchAdapter(provider="google", api_key="k", cx="c")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="items is not a list"):
        adapter.search("housing", 5, client)
    client.close()


def test_zero_limit_and_blank_query_make_no_request() -> None:
    adapter = WebSearchAdapter(provider="brave", api_key="k")
    client = httpx.Client(transport=httpx.MockTransport(_no_http))
    assert adapter.search("housing", 0, client) == []
    assert adapter.search("   ", 5, client) == []
    client.close()
