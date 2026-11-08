from __future__ import annotations

from pathlib import Path

import httpx

from whack26.research.cache import ResearchCache
from whack26.research.huggingface import HuggingFaceAdapter


def test_huggingface_search_maps_fields_and_sends_auth() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json=[
                {
                    "id": "owner/housing",
                    "author": "owner",
                    "description": "<p>California <b>housing</b> data</p>",
                    "createdAt": "2023-04-01T12:00:00.000Z",
                    "tags": ["task:regression", "license:mit"],
                    "private": False,
                    "gated": False,
                }
            ],
        )

    adapter = HuggingFaceAdapter(token="hf_test")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 5, client)

    assert seen["host"] == "huggingface.co"
    assert seen["path"] == "/api/datasets"
    assert seen["params"] == {"search": "housing", "limit": "5"}
    assert seen["authorization"] == "Bearer hf_test"
    assert len(records) == 1
    record = records[0]
    assert record.id == "huggingface:owner/housing"
    assert record.source == "huggingface"
    assert record.source_id == "owner/housing"
    assert record.title == "owner/housing"
    assert record.abstract == "California housing data"
    assert record.authors == ["owner"]
    assert record.year == 2023
    assert record.doi is None
    assert record.url == "https://huggingface.co/datasets/owner/housing"
    assert record.license == "mit"
    client.close()


def test_huggingface_skips_private_and_keeps_gated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") is None
        return httpx.Response(
            200,
            json=[
                {"id": "secret/data", "private": True, "gated": False},
                {"id": "org/gated", "private": False, "gated": "auto"},
            ],
        )

    adapter = HuggingFaceAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 5, client)

    assert [record.source_id for record in records] == ["org/gated"]
    assert records[0].license is None
    assert records[0].year is None
    client.close()


def test_huggingface_cache_replays_without_http(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json=[
                {
                    "id": "owner/housing",
                    "author": "owner",
                    "description": "Housing data",
                    "createdAt": "2022-01-01T00:00:00Z",
                    "tags": ["license:apache-2.0"],
                }
            ],
        )

    cache = ResearchCache(tmp_path / "research.db")
    adapter = HuggingFaceAdapter(cache=cache, dataset_id="ds_a")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = adapter.search("housing", 5, client)
    second = adapter.search("housing", 5, client)

    assert [record.id for record in first] == [record.id for record in second]
    assert first[0].title == "owner/housing"
    assert calls["count"] == 1
    assert cache.stats()["hits"] == 1
    client.close()
    cache.close()
