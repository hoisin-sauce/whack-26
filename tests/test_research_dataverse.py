from __future__ import annotations

from typing import get_args

import httpx
import pytest

from whack26.research import dataverse as dataverse_module
from whack26.research.base import DATASET_SOURCES, SourceError, SourceName
from whack26.research.dataverse import (
    DATAVERSE_SEARCH_URL,
    DataverseAdapter,
)
from whack26.research.normalize import deduplicate, record_identity_keys


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    real = dataverse_module.request_with_retries

    def wrapper(*args, **kwargs):
        kwargs.setdefault("sleep", lambda _seconds: None)
        return real(*args, **kwargs)

    monkeypatch.setattr(dataverse_module, "request_with_retries", wrapper)


def _payload(*items: dict) -> dict:
    return {
        "status": "OK",
        "data": {
            "q": "housing",
            "total_count": len(items),
            "start": 0,
            "items": list(items),
        },
    }


def test_dataverse_search_maps_fields_and_requests_public_datasets() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(
            200,
            json=_payload(
                {
                    "name": "Housing Prices",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/HOUSING",
                    "global_id": "doi:10.7910/DVN/HOUSING",
                    "description": "<p>Housing and income.</p>",
                    "published_at": "2020-05-01T00:00:00Z",
                    "authors": ["Ada Lovelace", "Grace Hopper"],
                    "license": {
                        "name": "CC0 1.0",
                        "uri": "https://creativecommons.org/publicdomain/zero/1.0/",
                    },
                }
            ),
        )

    adapter = DataverseAdapter()
    assert adapter.name == "dataverse"
    assert adapter.ttl_seconds == 24 * 60 * 60
    assert adapter.limited_mode is False

    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    url = seen["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith(DATAVERSE_SEARCH_URL)
    assert url.params["q"] == "housing"
    assert url.params["type"] == "dataset"
    assert url.params["per_page"] == "5"

    assert len(records) == 1
    record = records[0]
    assert record.id == "dataverse:10.7910/DVN/HOUSING"
    assert record.source == "dataverse"
    assert record.source_id == "10.7910/DVN/HOUSING"
    assert record.title == "Housing Prices"
    assert record.abstract == "Housing and income."
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.year == 2020
    assert record.doi == "10.7910/DVN/HOUSING"
    assert record.url == "https://doi.org/10.7910/DVN/HOUSING"
    assert record.license == "CC0 1.0"


def test_dataverse_skips_non_datasets_drafts_and_empty_titles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_payload(
                {
                    "name": "A Dataverse",
                    "type": "dataverse",
                    "url": "https://dataverse.harvard.edu/dataverse/example",
                },
                {
                    "name": "Draft dataset",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/DRAFT",
                    "global_id": "doi:10.7910/DVN/DRAFT",
                    "versionState": "DRAFT",
                },
                {
                    "name": "Unpublished dataset",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/UNPUBLISHED",
                    "global_id": "doi:10.7910/DVN/UNPUBLISHED",
                    "publicationStatuses": ["Unpublished"],
                },
                {
                    "name": "   ",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/EMPTY",
                    "global_id": "doi:10.7910/DVN/EMPTY",
                },
                {
                    "name": "Published dataset",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/PUBLISHED",
                    "global_id": "doi:10.7910/DVN/PUBLISHED",
                    "publicationStatuses": ["Published"],
                },
            ),
        )

    adapter = DataverseAdapter()
    with _client(handler) as client:
        records = adapter.search("housing", 10, client)

    assert [record.source_id for record in records] == [
        "10.7910/DVN/PUBLISHED"
    ]


def test_dataverse_duplicate_dois_are_deduplicated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_payload(
                {
                    "name": "Housing Prices",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/HOUSING",
                    "global_id": "doi:10.7910/DVN/HOUSING",
                    "description": "Housing and income.",
                    "published_at": "2020-05-01T00:00:00Z",
                    "authors": ["Ada Lovelace"],
                },
                {
                    "name": "Housing Prices (copy)",
                    "type": "dataset",
                    "url": "https://doi.org/10.7910/DVN/HOUSING",
                    "global_id": "doi:10.7910/DVN/HOUSING",
                    "published_at": "2020-06-01T00:00:00Z",
                    "authors": ["Grace Hopper"],
                },
            ),
        )

    adapter = DataverseAdapter()
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    assert len(records) == 2
    keys = record_identity_keys(records[0])
    assert "doi:10.7910/dvn/housing" in keys

    merged = deduplicate(records)
    assert len(merged) == 1
    assert merged[0].title == "Housing Prices"
    assert merged[0].doi == "10.7910/dvn/housing"


def test_dataverse_custom_base_url_is_used() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["path"] = request.url.path
        return httpx.Response(200, json=_payload())

    adapter = DataverseAdapter(base_url="https://demo.dataverse.org/")
    with _client(handler) as client:
        assert adapter.search("housing", 5, client) == []

    assert seen == {"host": "demo.dataverse.org", "path": "/api/search"}


def test_dataverse_error_status_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": "ERROR", "message": "search unavailable"}
        )

    adapter = DataverseAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="Dataverse search failed"):
            adapter.search("housing", 5, client)


def test_dataverse_malformed_payload_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "OK", "data": {}})

    adapter = DataverseAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="missing data.items"):
            adapter.search("housing", 5, client)


def test_dataverse_rate_limit_raises_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fast_retries(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    adapter = DataverseAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.status_code == 429


def test_dataverse_source_name_is_registered() -> None:
    assert DataverseAdapter.name == "dataverse"
    assert "dataverse" in get_args(SourceName)
    assert "dataverse" in DATASET_SOURCES
