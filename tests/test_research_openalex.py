from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import httpx
import pytest

from whack26.research.base import SourceError
from whack26.research.cache import ResearchCache
from whack26.research.openalex import OPENALEX_SELECT, OpenAlexAdapter


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _work(work_id: str = "W123", **overrides: object) -> dict:
    work: dict = {
        "id": f"https://openalex.org/{work_id}",
        "doi": "https://doi.org/10.1234/housing.2024",
        "title": "Housing <b>prices</b> and income",
        "publication_year": 2024,
        "authorships": [
            {"author": {"display_name": "Ada Lovelace"}},
            {"author": {"display_name": "Grace Hopper"}},
            {"author": {}},
            {"not": "an author"},
        ],
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "license": "cc-by",
        },
        "open_access": {"oa_status": "gold"},
        "cited_by_count": 12,
        "abstract_inverted_index": {
            "Housing": [0, 2],
            "prices": [1],
            "rise": [3],
        },
        "is_retracted": False,
        "topics": [],
        "keywords": [],
    }
    work.update(overrides)
    return work


def _payload(*works: dict, cost: float | None = None) -> dict:
    payload: dict = {"results": list(works)}
    if cost is not None:
        payload["meta"] = {"cost_usd": cost}
    return payload


def test_search_sends_expected_params_and_maps_fields() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        return httpx.Response(200, json=_payload(_work(), cost=0.001))

    adapter = OpenAlexAdapter()
    assert adapter.name == "openalex"
    assert adapter.ttl_seconds == 24 * 60 * 60
    assert adapter.limited_mode is True

    with _client(handler) as client:
        records = adapter.search("Housing Prices", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    assert str(url).startswith("https://api.openalex.org/works?")
    assert url.params["search"] == "Housing Prices"
    assert url.params["filter"] == "is_retracted:false"
    assert url.params["per_page"] == "5"
    assert url.params["select"] == OPENALEX_SELECT
    assert url.params.get("api_key") is None

    assert len(records) == 1
    record = records[0]
    assert record.id == "openalex:W123"
    assert record.source == "openalex"
    assert record.source_id == "W123"
    assert record.title == "Housing prices and income"
    assert record.abstract == "Housing prices Housing rise"
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.year == 2024
    assert record.doi == "10.1234/housing.2024"
    assert record.url == "https://example.org/paper"
    assert record.license == "cc-by"
    assert record.query == "Housing Prices"
    assert record.retrieved_at


def test_fetch_exposes_openalex_cost_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == "secret-key"
        return httpx.Response(
            200,
            json={"results": [], "meta": {"cost_usd": 0.0025}},
        )

    adapter = OpenAlexAdapter(api_key="secret-key")
    assert adapter.limited_mode is False

    with _client(handler) as client:
        result = adapter.fetch("housing", 5, client)

    assert result.status == 200
    assert result.headers["x-openalex-cost-usd"] == "0.0025"
    assert json.loads(result.body)["meta"]["cost_usd"] == 0.0025


def test_parse_skips_retracted_and_untitled_works() -> None:
    body = json.dumps(
        _payload(
            _work(work_id="W1", title="Retracted paper", is_retracted=True),
            _work(work_id="W2", title="   "),
            _work(work_id="W3", title="Kept paper"),
        )
    )
    adapter = OpenAlexAdapter()

    records = adapter.parse(body, query="housing")

    assert [record.source_id for record in records] == ["W3"]


def test_parse_strips_doi_prefixes_and_builds_fallbacks() -> None:
    body = json.dumps(
        _payload(
            _work(
                work_id="W1",
                doi="doi:10.5555/legacy",
                primary_location=None,
                abstract_inverted_index=None,
            ),
            _work(
                work_id="W2",
                doi=None,
                primary_location={"landing_page_url": None, "license": None},
                abstract_inverted_index=None,
            ),
        )
    )
    adapter = OpenAlexAdapter()

    records = adapter.parse(body, query="housing")

    assert records[0].doi == "10.5555/legacy"
    assert records[0].url == "https://doi.org/10.5555/legacy"
    assert records[0].license is None
    assert records[0].abstract is None
    # OpenAlex's open_access.oa_status must not stand in for the license.
    assert records[1].doi is None
    assert records[1].url == "https://openalex.org/W2"
    assert records[1].license is None


def test_parse_returns_at_most_ten_records() -> None:
    body = json.dumps(
        _payload(
            *[
                _work(work_id=f"W{index}", title=f"Paper {index}")
                for index in range(12)
            ]
        )
    )
    adapter = OpenAlexAdapter()

    records = adapter.parse(body, query="housing")

    assert len(records) == 10
    assert records[0].source_id == "W0"
    assert records[-1].source_id == "W9"


def test_search_replays_from_cache(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "research.db")
    adapter = OpenAlexAdapter(cache=cache, dataset_id="ds_a")
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_payload(_work()))

    with _client(handler) as client:
        first = adapter.search("housing", 5, client)
        second = adapter.search("housing", 5, client)

    assert calls["count"] == 1
    assert len(first) == len(second) == 1
    assert first[0].id == "openalex:W123"
    assert cache.stats()["hits"] == 1
    cache.close()


def test_keyed_403_falls_back_to_keyless_and_sets_limited_mode() -> None:
    seen_keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        api_key = request.url.params.get("api_key")
        seen_keys.append(api_key)
        if api_key:
            return httpx.Response(403, json={"error": "key_not_activated"})
        return httpx.Response(200, json=_payload(_work()))

    adapter = OpenAlexAdapter(api_key="not-activated")
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    assert adapter.limited_mode is True
    assert seen_keys == ["not-activated", None]
    assert len(records) == 1


def test_non_403_http_error_raises_source_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    adapter = OpenAlexAdapter(api_key="secret-key")
    with _client(handler) as client:
        with pytest.raises(SourceError) as exc_info:
            adapter.search("housing", 5, client)

    assert exc_info.value.source == "openalex"
    assert exc_info.value.query == "housing"
    assert exc_info.value.status_code == 400
