from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from whack26.research.base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    clean_text,
    reconstruct_abstract,
    request_with_retries,
)
from whack26.research.cache import ResearchCache, cache_key


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class DemoAdapter(BaseHttpAdapter):
    name = "openalex"
    ttl_seconds = 3600

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.requests = 0

    def fetch(self, query, limit, client) -> FetchResult:
        self.requests += 1
        response = client.get("https://example.test/search", params={"q": query})
        response.raise_for_status()
        return FetchResult(body=response.text, headers={"x-openalex-cost-usd": "0.001"})

    def parse(self, body, *, query, retrieved_at=None):
        payload = httpx.Response(200, text=body).json()
        return [
            self._record(
                source_id=item["id"],
                title=item["title"],
                url=f"https://example.test/{item['id']}",
                query=query,
                retrieved_at=retrieved_at,
            )
            for item in payload["results"]
        ]


def test_cache_key_normalizes_query_case_and_whitespace() -> None:
    first = cache_key("ds_a", "openalex", "Housing  Prices", {"limit": 5})
    second = cache_key("ds_a", "openalex", "housing prices", {"limit": 5})
    third = cache_key("ds_a", "openalex", "housing prices", {"limit": 6})
    assert first == second
    assert first != third


def test_cache_round_trip_and_ttl(tmp_path: Path) -> None:
    clock = Clock()
    cache = ResearchCache(tmp_path / "research.db", now=clock)
    key = cache_key("ds_a", "openalex", "housing", {"limit": 5})
    cache.put(
        key,
        dataset_id="ds_a",
        source="openalex",
        query="housing",
        status=200,
        body='{"ok": true}',
        ttl_seconds=60,
        headers={"x-openalex-cost-usd": "0.001"},
    )
    cached = cache.get(key)
    assert cached is not None
    assert cached.body == '{"ok": true}'
    assert cached.headers["x-openalex-cost-usd"] == "0.001"
    assert cache.stats()["hits"] == 1

    clock.advance(61)
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1
    cache.close()


def test_offline_cache_returns_stale_rows(tmp_path: Path) -> None:
    clock = Clock()
    cache = ResearchCache(tmp_path / "research.db", offline=True, now=clock)
    key = cache_key("ds_a", "openalex", "housing")
    cache.put(
        key,
        dataset_id="ds_a",
        source="openalex",
        query="housing",
        status=200,
        body="body",
        ttl_seconds=1,
    )
    clock.advance(3600)
    cached = cache.get(key)
    assert cached is not None
    assert cached.stale is True
    assert cache.stats()["stale_hits"] == 1
    cache.close()


def test_query_cache_round_trip_and_refresh(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    key = cache_key("ds_a", "queries", "", {"summary_sha256": "abc"})
    queries = [
        {
            "source": "openalex",
            "query": "housing prices",
            "rationale": "papers",
        }
    ]
    cache.put_queries(key, dataset_id="ds_a", queries=queries)
    assert cache.get_queries(key) == queries
    cache.refresh = True
    assert cache.get_queries(key) is None
    cache.close()


def test_query_cache_ignores_corrupt_rows(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    key = cache_key("ds_a", "queries", "", {"summary_sha256": "abc"})
    cache._connection.execute(
        "INSERT INTO research_queries (cache_key, dataset_id, created_at, queries_json) "
        "VALUES (?, ?, ?, ?)",
        (key, "ds_a", "2026-01-01T00:00:00Z", "{not json"),
    )
    cache._connection.commit()
    assert cache.get_queries(key) is None
    cache.close()


def test_research_cache_round_trips_bodies_larger_than_512k(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    key = cache_key("ds_a", "openalex", "housing")
    body = json.dumps(
        {
            "results": [
                {
                    "id": "W1",
                    "title": "Housing prices",
                    "padding": "x" * (600 * 1024),
                }
            ]
        }
    )
    assert len(body) > 512 * 1024
    cache.put(
        key,
        dataset_id="ds_a",
        source="openalex",
        query="housing",
        status=200,
        body=body,
        ttl_seconds=60,
    )
    cached = cache.get(key)
    assert cached is not None
    assert cached.body == body
    assert len(cached.body) == len(body)
    assert json.loads(cached.body)["results"][0]["id"] == "W1"
    cache.close()


def test_adapter_caches_and_replays(tmp_path: Path) -> None:
    clock = Clock()
    cache = ResearchCache(tmp_path / "research.db", now=clock)
    adapter = DemoAdapter(cache=cache, dataset_id="ds_a")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json={"results": [{"id": "W1", "title": "Housing prices"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = adapter.search("housing", 5, client)
    second = adapter.search("housing", 5, client)
    assert len(first) == len(second) == 1
    assert first[0].title == "Housing prices"
    assert first[0].id == "openalex:W1"
    assert calls["count"] == 1
    assert cache.stats()["hits"] == 1
    client.close()
    cache.close()


def test_adapter_offline_miss_raises(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db", offline=True)
    adapter = DemoAdapter(cache=cache, dataset_id="ds_a", offline=True)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="should not be called")
        )
    )
    with pytest.raises(SourceError, match="offline cache miss"):
        adapter.search("housing", 5, client)
    client.close()
    cache.close()


def test_adapter_refetches_instead_of_parsing_a_cached_error(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    key = cache_key("ds_a", "openalex", "housing", {"limit": 5})
    cache.put(
        key,
        dataset_id="ds_a",
        source="openalex",
        query="housing",
        status=500,
        body=json.dumps(
            {"results": [{"id": "CACHED", "title": "Cached error body"}]}
        ),
        ttl_seconds=3600,
    )
    adapter = DemoAdapter(cache=cache, dataset_id="ds_a")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json={"results": [{"id": "FRESH", "title": "Fresh result"}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 5, client)
    assert [record.id for record in records] == ["openalex:FRESH"]
    assert calls["count"] == 1
    client.close()
    cache.close()


def test_adapter_offline_cached_error_raises_instead_of_parsing(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db", offline=True)
    key = cache_key("ds_a", "openalex", "housing", {"limit": 5})
    cache.put(
        key,
        dataset_id="ds_a",
        source="openalex",
        query="housing",
        status=500,
        body=json.dumps(
            {"results": [{"id": "CACHED", "title": "Cached error body"}]}
        ),
        ttl_seconds=3600,
    )
    adapter = DemoAdapter(cache=cache, dataset_id="ds_a", offline=True)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, text="should not be called")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="HTTP status 500"):
        adapter.search("housing", 5, client)
    assert calls["count"] == 0
    client.close()
    cache.close()


def test_request_with_retries_retries_retryable_status() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = request_with_retries(
        client,
        "https://example.test",
        source="openalex",
        query="housing",
        sleep=lambda _: None,
    )
    assert response.status_code == 200
    assert calls["count"] == 2
    client.close()


def test_request_with_retries_gives_up() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(502, text="bad gateway")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SourceError, match="HTTP 502"):
        request_with_retries(
            client,
            "https://example.test",
            source="openalex",
            query="housing",
            sleep=lambda _: None,
        )
    assert calls["count"] == 3
    client.close()


def test_text_and_abstract_helpers() -> None:
    assert clean_text("<p>Hello&nbsp; <b>world</b></p>") == "Hello world"
    index = {"Housing": [0, 2], "prices": [1, 3], "rise": [4]}
    assert reconstruct_abstract(index) == "Housing prices Housing prices rise"
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({"bad": "shape"}) is None
