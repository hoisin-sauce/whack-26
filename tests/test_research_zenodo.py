from __future__ import annotations

from pathlib import Path

import httpx

from whack26.research.cache import ResearchCache
from whack26.research.zenodo import ZenodoAdapter


def test_zenodo_search_maps_fields_and_strips_doi() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "id": 12345,
                            "doi": "https://doi.org/10.5281/zenodo.12345",
                            "metadata": {
                                "title": "<b>Housing</b> prices",
                                "description": "<p>Prices and income.</p>",
                                "publication_date": "2021-05-01",
                                "creators": [
                                    {"name": "Ada Lovelace"},
                                    {"name": "Grace Hopper"},
                                ],
                                "license": {
                                    "id": "cc-by-4.0",
                                    "title": "Creative Commons Attribution 4.0",
                                },
                            },
                            "links": {
                                "self_html": "https://zenodo.org/records/12345"
                            },
                        }
                    ]
                }
            },
        )

    adapter = ZenodoAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 5, client)

    assert seen["host"] == "zenodo.org"
    assert seen["path"] == "/api/records"
    assert seen["params"] == {
        "q": "housing",
        "size": "5",
        "sort": "bestmatch",
    }
    assert seen["authorization"] is None
    assert len(records) == 1
    record = records[0]
    assert record.id == "zenodo:12345"
    assert record.source == "zenodo"
    assert record.source_id == "12345"
    assert record.title == "Housing prices"
    assert record.abstract == "Prices and income."
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.year == 2021
    assert record.doi == "10.5281/zenodo.12345"
    assert record.url == "https://zenodo.org/records/12345"
    assert record.license == "cc-by-4.0"
    client.close()


def test_zenodo_skips_restricted_embargoed_and_empty_titles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "id": 1,
                            "metadata": {
                                "title": "Restricted data",
                                "access_right": "restricted",
                            },
                        },
                        {
                            "id": 2,
                            "metadata": {
                                "title": "Embargoed data",
                                "access_right": "embargoed",
                            },
                        },
                        {
                            "id": 3,
                            "metadata": {"title": "   ", "access_right": "open"},
                        },
                        {
                            "id": 4,
                            "doi": "10.5281/zenodo.4",
                            "metadata": {
                                "title": "Open data",
                                "access_right": "open",
                                "license": "cc-by-4.0",
                            },
                            "links": {
                                "self_html": "https://zenodo.org/records/4"
                            },
                        },
                    ]
                }
            },
        )

    adapter = ZenodoAdapter()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = adapter.search("housing", 5, client)

    assert [record.source_id for record in records] == ["4"]
    assert records[0].license == "cc-by-4.0"
    client.close()


def test_zenodo_sends_bearer_token() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"hits": {"hits": []}})

    adapter = ZenodoAdapter(token="zenodo_test")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert adapter.search("housing", 5, client) == []
    assert seen["authorization"] == "Bearer zenodo_test"
    client.close()


def test_zenodo_cache_replays_without_http(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "id": 99,
                            "doi": "10.5281/zenodo.99",
                            "metadata": {
                                "title": "Housing data",
                                "publication_date": "2020-01-01",
                                "creators": [{"name": "Owner"}],
                            },
                            "links": {
                                "self_html": "https://zenodo.org/records/99"
                            },
                        }
                    ]
                }
            },
        )

    cache = ResearchCache(tmp_path / "research.db")
    adapter = ZenodoAdapter(cache=cache, dataset_id="ds_a")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = adapter.search("housing", 5, client)
    second = adapter.search("housing", 5, client)

    assert [record.id for record in first] == [record.id for record in second]
    assert calls["count"] == 1
    assert cache.stats()["hits"] == 1
    client.close()
    cache.close()
