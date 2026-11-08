from __future__ import annotations

from typing import Callable

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.datagov import (
    DATAGOV_API_URL,
    DATAGOV_DATASET_URL,
    DataGovAdapter,
)


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _package(**overrides: object) -> dict:
    package: dict = {
        "id": "11111111-2222-3333-4444-555555555555",
        "name": "housing-affordability",
        "title": "<b>Housing</b> affordability",
        "notes": "<p>Income and rent burden by county.</p>",
        "metadata_created": "2019-03-04T12:00:00",
        "metadata_modified": "2024-01-01T00:00:00",
        "license_id": "cc-by",
        "license_title": "Creative Commons Attribution",
        "organization": {"title": "Department of Housing"},
        "extras": [
            {"key": "doi", "value": "https://doi.org/10.5555/housing.2019"}
        ],
    }
    package.update(overrides)
    return package


def _payload(*packages: dict) -> dict:
    return {
        "success": True,
        "result": {"count": len(packages), "results": list(packages)},
    }


def test_search_maps_ckan_fields_and_sends_expected_params() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["api_key"] = request.headers.get("x-ckan-api-key")
        return httpx.Response(200, json=_payload(_package()))

    adapter = DataGovAdapter()
    assert adapter.name == "datagov"
    assert adapter.ttl_seconds == 24 * 60 * 60

    with _client(handler) as client:
        records = adapter.search("housing affordability", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith(f"{DATAGOV_API_URL}?")
    assert url.params["q"] == "housing affordability"
    assert url.params["rows"] == "5"
    assert captured["api_key"] is None

    assert len(records) == 1
    record = records[0]
    assert record.id == "datagov:housing-affordability"
    assert record.source == "datagov"
    assert record.source_id == "housing-affordability"
    assert record.title == "Housing affordability"
    assert record.abstract == "Income and rent burden by county."
    assert record.authors == ["Department of Housing"]
    assert record.year == 2019
    assert record.doi == "10.5555/housing.2019"
    assert record.url == f"{DATAGOV_DATASET_URL}housing-affordability"
    assert record.license == "cc-by"
    assert record.query == "housing affordability"
    assert record.retrieved_at


def test_search_sends_optional_ckan_api_key() -> None:
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("x-ckan-api-key")
        return httpx.Response(200, json=_payload())

    adapter = DataGovAdapter(api_key="ckan-secret")
    with _client(handler) as client:
        assert adapter.search("housing", 5, client) == []

    assert seen["api_key"] == "ckan-secret"


def test_parse_skips_packages_without_usable_identity() -> None:
    body = httpx.Response(
        200,
        json=_payload(
            _package(name=None, id=None),
            _package(name="", title="   "),
            _package(name="kept", title="Kept dataset", extras=None),
            {"not": "a package"},
        ),
    ).text
    adapter = DataGovAdapter()

    records = adapter.parse(body, query="housing")

    assert [record.source_id for record in records] == ["kept"]
    assert records[0].doi is None
    assert records[0].license == "cc-by"


def test_parse_accepts_empty_result_list() -> None:
    body = httpx.Response(200, json=_payload()).text
    assert DataGovAdapter().parse(body, query="housing") == []


def test_ckan_error_payload_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": False,
                "error": {"message": "Invalid search query"},
            },
        )

    adapter = DataGovAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="CKAN error") as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.source == "datagov"
    assert exc_info.value.query == "housing"


def test_auth_failure_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    adapter = DataGovAdapter(api_key="ckan-secret")
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.status_code == 403
    assert "ckan-secret" not in str(exc_info.value)


def test_rate_limit_retries_then_raises_source_error() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="slow down")

    adapter = DataGovAdapter(sleep=sleeps.append)
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert calls["count"] == 3
    assert sleeps == [1.0, 4.0]
    assert exc_info.value.status_code == 429


def test_malformed_payload_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "result": {}})

    adapter = DataGovAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="result.results"):
            adapter.search("housing", 5, client)
