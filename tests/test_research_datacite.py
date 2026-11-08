from __future__ import annotations

from typing import Callable

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.datacite import DATACITE_DOIS_URL, DataCiteAdapter


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _entry(doi: str = "10.5281/zenodo.999", **overrides: object) -> dict:
    attributes: dict = {
        "doi": f"https://doi.org/{doi}",
        "url": "https://zenodo.org/records/999",
        "titles": [{"title": "Housing <i>survey</i> microdata"}],
        "descriptions": [
            {"descriptionType": "Other", "description": "Other note"},
            {
                "descriptionType": "Abstract",
                "description": "<p>Household survey of renters.</p>",
            },
        ],
        "creators": [
            {"name": "Lovelace, Ada"},
            {"givenName": "Grace", "familyName": "Hopper"},
            {"name": "   "},
        ],
        "publicationYear": 2021,
        "types": {"resourceTypeGeneral": "Dataset"},
        "rightsList": [
            {
                "rights": "Creative Commons Attribution 4.0",
                "rightsIdentifier": "cc-by-4.0",
            }
        ],
    }
    entry: dict = {
        "id": f"https://doi.org/{doi}",
        "type": "dois",
        "attributes": attributes,
    }
    attributes.update(overrides)
    return entry


def _payload(*entries: dict) -> dict:
    return {"data": list(entries), "meta": {"total": len(entries)}}


def test_search_maps_datacite_fields_and_sends_expected_params() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json=_payload(_entry()))

    adapter = DataCiteAdapter()
    assert adapter.name == "datacite"
    assert adapter.ttl_seconds == 24 * 60 * 60

    with _client(handler) as client:
        records = adapter.search("housing survey", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith(f"{DATACITE_DOIS_URL}?")
    assert url.params["query"] == "housing survey"
    assert url.params["page[size]"] == "5"
    assert url.params["resource-type-id"] == "dataset"
    assert captured["authorization"] is None

    assert len(records) == 1
    record = records[0]
    assert record.id == "datacite:10.5281/zenodo.999"
    assert record.source == "datacite"
    assert record.source_id == "10.5281/zenodo.999"
    assert record.title == "Housing survey microdata"
    # The abstract wins over other description types.
    assert record.abstract == "Household survey of renters."
    assert record.authors == ["Lovelace, Ada", "Grace Hopper"]
    assert record.year == 2021
    assert record.doi == "10.5281/zenodo.999"
    assert record.url == "https://zenodo.org/records/999"
    assert record.license == "cc-by-4.0"
    assert record.query == "housing survey"
    assert record.retrieved_at


def test_parse_uses_doi_url_and_skips_unusable_entries() -> None:
    body = httpx.Response(
        200,
        json=_payload(
            _entry(
                doi="10.5555/fallback",
                url=None,
                creators=None,
                publicationYear=None,
                descriptions=None,
                rightsList=None,
            ),
            _entry(doi="10.5555/no-title", titles=[]),
            {"id": "https://doi.org/10.5555/no-attributes"},
            _entry(doi=""),
        ),
    ).text
    adapter = DataCiteAdapter()

    records = adapter.parse(body, query="housing")

    assert [record.source_id for record in records] == ["10.5555/fallback"]
    record = records[0]
    assert record.doi == "10.5555/fallback"
    assert record.url == "https://doi.org/10.5555/fallback"
    assert record.authors == []
    assert record.year is None
    assert record.abstract is None
    assert record.license is None


def test_parse_accepts_empty_data_list() -> None:
    body = httpx.Response(200, json=_payload()).text
    assert DataCiteAdapter().parse(body, query="housing") == []


def test_rate_limit_retries_then_raises_source_error() -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="rate limited")

    adapter = DataCiteAdapter(sleep=sleeps.append)
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert calls["count"] == 3
    assert sleeps == [1.0, 4.0]
    assert exc_info.value.status_code == 429


def test_auth_failure_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    adapter = DataCiteAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.status_code == 401


def test_malformed_payload_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {}})

    adapter = DataCiteAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="data list"):
            adapter.search("housing", 5, client)


def test_invalid_json_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="{not json")

    adapter = DataCiteAdapter()
    with _client(handler) as client:
        with pytest.raises(SourceError, match="could not parse response"):
            adapter.search("housing", 5, client)
