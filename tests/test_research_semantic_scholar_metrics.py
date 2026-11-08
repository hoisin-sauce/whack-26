from __future__ import annotations

from typing import Callable

import httpx
import pytest

from whack26.research.semantic_scholar import (
    SEMANTIC_SCHOLAR_FIELDS,
    SemanticScholarAdapter,
)

_MISSING = object()


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _paper(paper_id: str = "abc123", **overrides: object) -> dict:
    paper: dict = {
        "paperId": paper_id,
        "title": "Housing <em>prices</em> and income",
        "abstract": "<p>Renters and owners respond differently.</p>",
        "year": 2024,
        "authors": [
            {"authorId": "1", "name": "Ada Lovelace"},
            {"authorId": "2", "name": "Grace Hopper"},
            {"authorId": "3", "name": "   "},
        ],
        "externalIds": {"DOI": "https://doi.org/10.1234/housing.2024"},
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        "publicationDate": "2024-05-01",
        "venue": "Journal of Housing",
        "fieldsOfStudy": ["Economics"],
        "citationCount": 42,
        "publicationTypes": ["JournalArticle", "Review"],
    }
    paper.update(overrides)
    return paper


def _payload(*papers: dict) -> dict:
    return {"total": len(papers), "offset": 0, "data": list(papers)}


def _parse(*papers: dict):
    body = httpx.Response(200, json=_payload(*papers)).text
    return SemanticScholarAdapter(min_interval=0).parse(body, query="housing")


def test_fields_constant_includes_citation_and_publication_types() -> None:
    fields = set(SEMANTIC_SCHOLAR_FIELDS.split(","))

    assert "citationCount" in fields
    assert "publicationTypes" in fields


def test_search_requests_citation_and_publication_type_fields() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        return httpx.Response(200, json=_payload(_paper()))

    adapter = SemanticScholarAdapter(min_interval=0)
    with _client(handler) as client:
        adapter.search("housing", 5, client)

    url = captured["url"]
    assert isinstance(url, httpx.URL)
    fields = url.params["fields"].split(",")
    assert "citationCount" in fields
    assert "publicationTypes" in fields


def test_parse_maps_citation_count_and_author_ids() -> None:
    record = _parse(_paper())[0]

    assert record.cited_by_count == 42
    assert record.author_ids == ["semantic_scholar:1", "semantic_scholar:2"]
    # Author display names keep their existing shape and ordering.
    assert record.authors == ["Ada Lovelace", "Grace Hopper"]
    assert record.metrics_source == "semantic_scholar"


@pytest.mark.parametrize(
    ("publication_types", "expected"),
    [
        (["JournalArticle"], "journal"),
        (["Review", "JournalArticle"], "journal"),
        (["Conference"], "conference"),
        (["Review"], None),
        ([], None),
        (None, None),
    ],
)
def test_parse_maps_publication_types_to_venue_type(
    publication_types: object,
    expected: str | None,
) -> None:
    record = _parse(_paper(publicationTypes=publication_types))[0]

    assert record.venue_type == expected


@pytest.mark.parametrize("value", [True, False, "42", None, _MISSING])
def test_parse_ignores_non_integer_citation_count(value: object) -> None:
    paper = _paper()
    if value is _MISSING:
        paper.pop("citationCount")
    else:
        paper["citationCount"] = value

    record = _parse(paper)[0]

    assert record.cited_by_count is None


def test_parse_skips_missing_and_blank_author_ids() -> None:
    paper = _paper(
        authors=[
            {"authorId": "10", "name": "First"},
            {"name": "No id"},
            {"authorId": "", "name": "Empty id"},
            {"authorId": "   ", "name": "Blank id"},
            {"authorId": "20", "name": "Second"},
        ]
    )
    record = _parse(paper)[0]

    assert record.author_ids == ["semantic_scholar:10", "semantic_scholar:20"]
    assert record.authors == [
        "First",
        "No id",
        "Empty id",
        "Blank id",
        "Second",
    ]


def test_parse_without_metrics_leaves_source_none() -> None:
    missing = _paper(paper_id="missing", authors=[{"name": "Ada Lovelace"}])
    missing.pop("citationCount")
    missing.pop("publicationTypes")
    explicit_none = _paper(
        paper_id="none",
        citationCount=None,
        authors=[{"name": "Grace Hopper"}],
        publicationTypes=None,
    )

    records = _parse(missing, explicit_none)

    assert [
        (
            record.cited_by_count,
            record.author_ids,
            record.venue_type,
            record.metrics_source,
        )
        for record in records
    ] == [
        (None, [], None, None),
        (None, [], None, None),
    ]


def test_metrics_source_is_set_when_only_some_metrics_are_present() -> None:
    records = _parse(
        _paper(
            paper_id="citation-only",
            citationCount=5,
            authors=[{"name": "No id"}],
            publicationTypes=None,
        ),
        _paper(
            paper_id="author-only",
            citationCount=None,
            authors=[{"authorId": "9", "name": "Ada Lovelace"}],
            publicationTypes=None,
        ),
        _paper(
            paper_id="venue-only",
            citationCount=None,
            authors=[{"name": "No id"}],
            publicationTypes=["Conference"],
        ),
    )

    assert [record.metrics_source for record in records] == [
        "semantic_scholar",
        "semantic_scholar",
        "semantic_scholar",
    ]
