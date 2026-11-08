from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.cache import ResearchCache, cache_key
from whack26.research.wikipedia import WikipediaAdapter

SEARCH_RESULT = {
    "query": {
        "search": [
            {"title": "Housing market", "pageid": 123},
            {"title": "Real estate", "pageid": 456},
        ]
    }
}


def _summary(title: str, pageid: int, *, url: str | None = None) -> dict:
    payload = {
        "title": title,
        "pageid": pageid,
        "timestamp": "2024-02-03T00:00:00Z",
        "extract": f"{title} is described here.",
        "description": "a short description that is not stored",
    }
    if url is not None:
        payload["content_urls"] = {"desktop": {"page": url}}
    return payload


def test_wikipedia_search_routes_two_steps_and_maps_summaries() -> None:
    search_params: list[dict[str, str]] = []
    summary_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            search_params.append(dict(request.url.params))
            return httpx.Response(200, json=SEARCH_RESULT)
        if request.url.path.startswith("/api/rest_v1/page/summary/"):
            summary_paths.append(request.url.raw_path.decode("ascii"))
            title = request.url.path.rsplit("/", 1)[-1]
            if title == "Housing market":
                return httpx.Response(
                    200,
                    json=_summary(
                        "Housing market",
                        123,
                        url="https://en.wikipedia.org/wiki/Housing_market",
                    ),
                )
            return httpx.Response(200, json=_summary("Real estate", 456))
        return httpx.Response(404)

    adapter = WikipediaAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing market", 3, client)

    assert adapter.is_fallback is True
    assert search_params == [
        {
            "action": "query",
            "list": "search",
            "srsearch": "housing market",
            "format": "json",
            "utf8": "1",
            "srlimit": "3",
        }
    ]
    assert summary_paths == [
        "/api/rest_v1/page/summary/Housing%20market",
        "/api/rest_v1/page/summary/Real%20estate",
    ]
    assert len(records) == 2
    first = records[0]
    assert first.id == "wikipedia:123"
    assert first.source == "wikipedia"
    assert first.source_id == "123"
    assert first.title == "Housing market"
    assert first.abstract == "Housing market is described here."
    assert first.authors == []
    assert first.year == 2024
    assert first.doi is None
    assert first.url == "https://en.wikipedia.org/wiki/Housing_market"
    assert first.license == "CC BY-SA 4.0"
    second = records[1]
    assert second.source_id == "456"
    assert second.url == "https://en.wikipedia.org/wiki/Real%20estate"
    client.close()


def test_wikipedia_cache_replays_both_steps(tmp_path: Path) -> None:
    calls = {"search": 0, "summary": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            calls["search"] += 1
            return httpx.Response(200, json=SEARCH_RESULT)
        calls["summary"] += 1
        title = request.url.path.rsplit("/", 1)[-1]
        pageid = 123 if title == "Housing market" else 456
        return httpx.Response(200, json=_summary(title, pageid))

    cache = ResearchCache(tmp_path / "research.db")
    adapter = WikipediaAdapter(cache=cache, dataset_id="ds_a")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = adapter.search("housing market", 2, client)
    second = adapter.search("housing market", 2, client)

    assert [record.id for record in first] == [record.id for record in second]
    assert calls == {"search": 1, "summary": 2}
    assert cache.stats()["hits"] == 3
    client.close()
    cache.close()


def test_wikipedia_offline_cache_miss_raises(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, text="should not be called")

    cache = ResearchCache(tmp_path / "research.db", offline=True)
    adapter = WikipediaAdapter(
        cache=cache, dataset_id="ds_a", offline=True
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="offline cache miss"):
        adapter.search("housing market", 2, client)
    assert calls["count"] == 0
    client.close()
    cache.close()


def test_wikipedia_skips_404_summaries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/w/api.php":
            payload = {
                "query": {
                    "search": [
                        {"title": "Missing page", "pageid": 1},
                        {"title": "Existing page", "pageid": 2},
                    ]
                }
            }
            return httpx.Response(200, json=payload)
        if request.url.path.endswith("/Missing page"):
            return httpx.Response(404, text="not found")
        return httpx.Response(200, json=_summary("Existing page", 2))

    adapter = WikipediaAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 2, client)

    assert [record.source_id for record in records] == ["2"]
    assert records[0].title == "Existing page"
    client.close()


def test_wikipedia_malformed_search_json_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{not json")

    adapter = WikipediaAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="could not parse response"):
        adapter.search("housing", 2, client)
    client.close()


def test_wikipedia_offline_partial_cache_skips_missing_summaries(
    tmp_path: Path,
) -> None:
    cache = ResearchCache(tmp_path / "research.db", offline=True)
    search_key = cache_key(
        "ds_a", "wikipedia", "housing market", {"step": "search", "limit": 2}
    )
    cache.put(
        search_key,
        dataset_id="ds_a",
        source="wikipedia",
        query="housing market",
        status=200,
        body=json.dumps(SEARCH_RESULT),
        ttl_seconds=7 * 24 * 60 * 60,
    )
    summary_key = cache_key(
        "ds_a",
        "wikipedia",
        "housing market",
        {"step": "summary", "title": "Housing market"},
    )
    cache.put(
        summary_key,
        dataset_id="ds_a",
        source="wikipedia",
        query="housing market",
        status=200,
        body=json.dumps(_summary("Housing market", 123)),
        ttl_seconds=7 * 24 * 60 * 60,
    )
    adapter = WikipediaAdapter(cache=cache, dataset_id="ds_a", offline=True)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, text="should not be called")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing market", 2, client)

    assert [record.source_id for record in records] == ["123"]
    assert records[0].title == "Housing market"
    assert calls["count"] == 0
    assert any(
        "'Real estate'" in warning and "offline cache miss" in warning
        for warning in adapter.warnings
    )
    client.close()
    cache.close()


def test_wikipedia_offline_cached_error_summary_is_skipped_with_warning(
    tmp_path: Path,
) -> None:
    cache = ResearchCache(tmp_path / "research.db", offline=True)
    cache.put(
        cache_key(
            "ds_a", "wikipedia", "housing market", {"step": "search", "limit": 2}
        ),
        dataset_id="ds_a",
        source="wikipedia",
        query="housing market",
        status=200,
        body=json.dumps(SEARCH_RESULT),
        ttl_seconds=7 * 24 * 60 * 60,
    )
    cache.put(
        cache_key(
            "ds_a",
            "wikipedia",
            "housing market",
            {"step": "summary", "title": "Housing market"},
        ),
        dataset_id="ds_a",
        source="wikipedia",
        query="housing market",
        status=500,
        body=json.dumps(_summary("Housing market", 123)),
        ttl_seconds=7 * 24 * 60 * 60,
    )
    cache.put(
        cache_key(
            "ds_a",
            "wikipedia",
            "housing market",
            {"step": "summary", "title": "Real estate"},
        ),
        dataset_id="ds_a",
        source="wikipedia",
        query="housing market",
        status=200,
        body=json.dumps(_summary("Real estate", 456)),
        ttl_seconds=7 * 24 * 60 * 60,
    )
    adapter = WikipediaAdapter(cache=cache, dataset_id="ds_a", offline=True)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="should not be called")
        )
    )
    records = adapter.search("housing market", 2, client)

    assert [record.source_id for record in records] == ["456"]
    assert any(
        "'Housing market'" in warning and "HTTP status 500" in warning
        for warning in adapter.warnings
    )
    client.close()
    cache.close()
