from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import httpx
import pytest

from whack26.research import openml as openml_module
from whack26.research.base import DATASET_SOURCES, SourceError, SourceName
from whack26.research.cache import ResearchCache
from whack26.research.normalize import deduplicate, record_identity_keys
from whack26.research.openml import OpenMLAdapter


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    real = openml_module.request_with_retries

    def wrapper(*args, **kwargs):
        kwargs.setdefault("sleep", lambda _seconds: None)
        return real(*args, **kwargs)

    monkeypatch.setattr(openml_module, "request_with_retries", wrapper)


def _list_payload(*entries: dict) -> dict:
    return {"data": {"dataset": list(entries)}}


def _description(**overrides: object) -> dict:
    description = {
        "id": "61",
        "name": "iris",
        "description": "<p>Iris flowers</p>",
        "creator": "Ada Lovelace, Grace Hopper",
        "upload_date": "2014-04-06T23:19:20",
        "licence": "Public",
        "doi": "https://doi.org/10.1234/iris",
        "url": "https://openml.org/data/v1/download/61/iris.arff",
    }
    description.update(overrides)
    return description


def test_openml_search_maps_list_and_detail_fields() -> None:
    seen: dict[str, object] = {"detail_paths": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/data/61"):
            seen["detail_paths"].append(request.url.path)  # type: ignore[union-attr]
            return httpx.Response(
                200, json={"data_set_description": _description()}
            )
        seen["list_path"] = request.url.path
        return httpx.Response(
            200,
            json=_list_payload(
                {"did": 61, "name": "iris", "status": "active"}
            ),
        )

    adapter = OpenMLAdapter()
    assert adapter.name == "openml"
    assert adapter.ttl_seconds == 24 * 60 * 60
    assert adapter.limited_mode is False

    with _client(handler) as client:
        records = adapter.search("California Housing", 5, client)

    assert seen["list_path"] == (
        "/api/v1/json/data/list/data_name/california_housing/limit/5"
    )
    assert seen["detail_paths"] == ["/api/v1/json/data/61"]

    assert len(records) == 1
    record = records[0]
    assert record.id == "openml:61"
    assert record.source == "openml"
    assert record.source_id == "61"
    assert record.title == "iris"
    assert record.abstract == "Iris flowers"
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.year == 2014
    assert record.doi == "10.1234/iris"
    assert record.url == "https://www.openml.org/d/61"
    assert record.license == "Public"
    assert record.query == "California Housing"


def test_openml_no_results_returns_empty_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            412, json={"error": {"code": "372", "message": "No results"}}
        )

    adapter = OpenMLAdapter()
    with _client(handler) as client:
        assert adapter.search("missing dataset", 5, client) == []


def test_openml_detail_failure_keeps_list_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/data/61"):
            return httpx.Response(400, text="detail unavailable")
        return httpx.Response(
            200,
            json=_list_payload(
                {"did": 61, "name": "iris", "status": "active"}
            ),
        )

    adapter = OpenMLAdapter()
    with _client(handler) as client:
        records = adapter.search("iris", 5, client)

    assert len(records) == 1
    record = records[0]
    assert record.id == "openml:61"
    assert record.title == "iris"
    assert record.url == "https://www.openml.org/d/61"
    assert record.abstract is None
    assert record.authors == []
    assert record.year is None
    assert record.license is None


def test_openml_skips_deactivated_datasets() -> None:
    body = json.dumps(
        _list_payload(
            {
                "did": 61,
                "name": "anneal",
                "status": "deactivated",
                "description": "Inactive",
            },
            {
                "did": 62,
                "name": "iris",
                "status": "active",
                "description": "Active",
            },
        )
    )
    adapter = OpenMLAdapter()

    records = adapter.parse(body, query="iris")

    assert [record.source_id for record in records] == ["62"]


def test_openml_duplicate_datasets_are_deduplicated() -> None:
    body = json.dumps(
        _list_payload(
            {
                "did": 61,
                "name": "iris",
                "status": "active",
                "description": "Iris flowers",
                "creator": "Ada Lovelace",
                "upload_date": "2014-01-01",
                "licence": "Public",
            },
            {
                "did": 62,
                "name": "iris",
                "status": "active",
                "description": "Iris flowers, second version",
                "creator": "Ada Lovelace",
                "upload_date": "2014-06-01",
                "licence": "Public",
            },
        )
    )
    adapter = OpenMLAdapter()

    records = adapter.parse(body, query="iris")

    assert len(records) == 2
    keys = record_identity_keys(records[0])
    assert any(key.startswith("title:") for key in keys)

    merged = deduplicate(records)
    assert len(merged) == 1
    assert merged[0].title == "iris"


def test_openml_malformed_list_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": {"dataset": "not a list"}}
        )

    adapter = OpenMLAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="missing data.dataset"):
            adapter.search("iris", 5, client)


def test_openml_rate_limit_raises_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retries(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    adapter = OpenMLAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("iris", 5, client)

    assert exc_info.value.status_code == 429


def test_openml_cache_replays_without_http(tmp_path: Path) -> None:
    calls = {"list": 0, "detail": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/data/61"):
            calls["detail"] += 1
            return httpx.Response(
                200, json={"data_set_description": _description()}
            )
        calls["list"] += 1
        return httpx.Response(
            200,
            json=_list_payload(
                {"did": 61, "name": "iris", "status": "active"}
            ),
        )

    cache = ResearchCache(tmp_path / "research.db")
    adapter = OpenMLAdapter(cache=cache, dataset_id="ds_a")
    with _client(handler) as client:
        first = adapter.search("iris", 5, client)
        second = adapter.search("iris", 5, client)

    assert [record.id for record in first] == [record.id for record in second]
    assert first[0].abstract == "Iris flowers"
    assert calls == {"list": 1, "detail": 1}
    assert cache.stats()["hits"] == 1
    cache.close()


def test_openml_source_name_is_registered() -> None:
    assert OpenMLAdapter.name == "openml"
    assert "openml" in get_args(SourceName)
    assert "openml" in DATASET_SOURCES
