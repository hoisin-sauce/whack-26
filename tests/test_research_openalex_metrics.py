from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from whack26.research.base import ResearchRecord
from whack26.research.openalex import OpenAlexAdapter

_MISSING = object()


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
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada Lovelace",
                }
            },
            {
                "author": {
                    "id": "https://openalex.org/A2",
                    "display_name": "Grace Hopper",
                }
            },
            {"author": {"display_name": "No Id"}},
            {"author": {}},
            {"not": "an author"},
        ],
        "primary_location": {
            "landing_page_url": "https://example.org/paper",
            "license": "cc-by",
            "source": {
                "type": "journal",
                "host_organization_name": "Elsevier BV",
            },
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


def _payload(*works: dict) -> dict:
    return {"results": list(works)}


def test_research_record_exposes_metric_fields_with_defaults() -> None:
    record = ResearchRecord(
        id="openalex:W1",
        source="openalex",
        source_id="W1",
        title="Housing prices",
        url="https://openalex.org/W1",
        retrieved_at="2026-01-01T00:00:00Z",
        query="housing",
    )

    assert record.cited_by_count is None
    assert record.author_ids == []
    assert record.author_h_index is None
    assert record.author_citation_count is None
    assert record.author_known is None
    assert record.venue_type is None
    assert record.publisher is None
    assert record.metrics_source is None


def test_parse_maps_citation_count_and_author_ids() -> None:
    body = json.dumps(_payload(_work()))
    records = OpenAlexAdapter().parse(body, query="housing")

    record = records[0]
    assert record.cited_by_count == 12
    assert record.author_ids == ["openalex:A1", "openalex:A2"]
    # Author display names keep their existing shape and ordering.
    assert record.authors == ["Ada Lovelace", "Grace Hopper", "No Id"]
    assert record.metrics_source == "openalex"


def test_parse_maps_venue_type_and_publisher() -> None:
    body = json.dumps(_payload(_work()))
    records = OpenAlexAdapter().parse(body, query="housing")

    record = records[0]
    assert record.venue_type == "journal"
    assert record.publisher == "Elsevier BV"
    assert record.metrics_source == "openalex"


@pytest.mark.parametrize("value", [True, False, "12", None, _MISSING])
def test_parse_ignores_non_integer_cited_by_count(value: object) -> None:
    work = _work()
    if value is _MISSING:
        work.pop("cited_by_count")
    else:
        work["cited_by_count"] = value

    body = json.dumps(_payload(work))
    records = OpenAlexAdapter().parse(body, query="housing")

    assert records[0].cited_by_count is None


def test_parse_skips_missing_and_blank_author_ids() -> None:
    work = _work(
        authorships=[
            {
                "author": {
                    "id": "https://openalex.org/A10",
                    "display_name": "First",
                }
            },
            {"author": {"display_name": "No id"}},
            {"author": {"id": "", "display_name": "Empty id"}},
            {"author": {"id": "   ", "display_name": "Blank id"}},
            {
                "author": {
                    "id": "https://openalex.org/A20",
                    "display_name": "Second",
                }
            },
        ]
    )
    body = json.dumps(_payload(work))
    records = OpenAlexAdapter().parse(body, query="housing")

    record = records[0]
    assert record.author_ids == ["openalex:A10", "openalex:A20"]
    assert record.authors == [
        "First",
        "No id",
        "Empty id",
        "Blank id",
        "Second",
    ]


def test_parse_missing_location_or_source_has_no_venue_metrics() -> None:
    body = json.dumps(
        _payload(
            _work(
                work_id="W1",
                primary_location=None,
                cited_by_count=None,
                authorships=[],
            ),
            _work(
                work_id="W2",
                primary_location={"landing_page_url": "https://example.org"},
                cited_by_count=None,
                authorships=[],
            ),
        )
    )
    records = OpenAlexAdapter().parse(body, query="housing")

    assert [
        (record.venue_type, record.publisher, record.metrics_source)
        for record in records
    ] == [(None, None, None), (None, None, None)]


def test_metrics_source_is_set_when_only_some_metrics_are_present() -> None:
    body = json.dumps(
        _payload(
            _work(
                work_id="W1",
                cited_by_count=None,
                authorships=[
                    {"author": {"id": "https://openalex.org/A1"}},
                ],
                primary_location=None,
            ),
            _work(
                work_id="W2",
                cited_by_count=7,
                authorships=[],
                primary_location=None,
            ),
            _work(
                work_id="W3",
                cited_by_count=None,
                authorships=[],
                primary_location={"source": {"type": "journal"}},
            ),
        )
    )
    records = OpenAlexAdapter().parse(body, query="housing")

    assert [record.metrics_source for record in records] == [
        "openalex",
        "openalex",
        "openalex",
    ]


def test_search_maps_metrics_through_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload(_work()))

    adapter = OpenAlexAdapter()
    with _client(handler) as client:
        records = adapter.search("housing", 5, client)

    record = records[0]
    assert record.cited_by_count == 12
    assert record.author_ids == ["openalex:A1", "openalex:A2"]
    assert record.venue_type == "journal"
    assert record.publisher == "Elsevier BV"
    assert record.metrics_source == "openalex"
