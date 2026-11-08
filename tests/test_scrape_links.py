"""Tests for link extraction and candidate-to-document conversion."""

from __future__ import annotations

from whack26.research.scrape.identity import canonicalize_url, url_hash
from whack26.research.scrape.links import (
    LinkCandidate,
    candidate_to_document,
    extract_links,
)
from whack26.research.scrape.models import Document


def make_document(metadata: dict) -> Document:
    return Document(
        id="pap_seed",
        kind="paper",
        source="openalex",
        title="Seed paper",
        metadata=metadata,
    )


def test_extract_links_supports_all_identifier_shapes_and_keys() -> None:
    document = make_document(
        {
            "links": [
                "https://Example.com/Paper/?utm_source=feed&utm_medium=email#section",
                {
                    "url": "https://example.org/thing",
                    "title": "Thing",
                    "authors": ["Ada", {"display_name": "Bob"}],
                    "year": "2021",
                    "abstract": "An abstract",
                    "source": "zenodo",
                    "source_id": "z1",
                    "extra": "kept",
                },
            ],
            "references": [
                {"href": "https://doi.org/10.1234/ABC", "title": "DOI Paper"}
            ],
            "referenced_works": [
                "https://openalex.org/W12345",
                "W6789",
            ],
            "urls": [
                "arxiv:2101.00001v2",
                "https://arxiv.org/abs/2101.00002",
            ],
            "citations": ["10.5555/XyZ"],
        }
    )

    candidates = extract_links(document)

    urls = [candidate.url for candidate in candidates]
    assert urls == [
        "https://example.com/Paper",
        "https://example.org/thing",
        "https://doi.org/10.1234/abc",
        "https://openalex.org/W12345",
        "https://openalex.org/W6789",
        "https://arxiv.org/abs/2101.00001",
        "https://arxiv.org/abs/2101.00002",
        "https://doi.org/10.5555/xyz",
    ]

    thing = candidates[1]
    assert thing.title == "Thing"
    assert thing.authors == ["Ada", "Bob"]
    assert thing.year == 2021
    assert thing.abstract == "An abstract"
    assert thing.source == "zenodo"
    assert thing.source_id == "z1"
    assert thing.metadata == {"extra": "kept"}

    doi_candidate = candidates[2]
    assert doi_candidate.doi == "10.1234/abc"
    assert candidates[-1].doi == "10.5555/xyz"


def test_extract_links_deduplicates_by_canonical_url_and_doi() -> None:
    document = make_document(
        {
            "links": [
                "https://example.com/a?utm_source=newsletter",
                "https://example.com/a",
            ],
            "references": [
                {"url": "https://doi.org/10.1234/ABC", "title": "First"},
                {"url": "https://doi.org/10.1234/abc", "abstract": "Second"},
            ],
        }
    )

    candidates = extract_links(document)

    assert [candidate.url for candidate in candidates] == [
        "https://example.com/a",
        "https://doi.org/10.1234/abc",
    ]
    # A later, richer duplicate fills gaps instead of adding a second row.
    assert candidates[1].title == "First"
    assert candidates[1].abstract == "Second"


def test_extract_links_caps_after_deduplication() -> None:
    document = make_document(
        {
            "links": [
                "https://example.com/1",
                "https://example.com/1?utm_source=x",
                "https://example.com/2",
                "https://example.com/3",
            ]
        }
    )

    candidates = extract_links(document, max_links=2)

    assert [candidate.url for candidate in candidates] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert extract_links(document, max_links=0) == []


def test_extract_links_skips_unsupported_entries() -> None:
    document = make_document(
        {
            "links": [
                "mailto:someone@example.com",
                "ftp://example.com/file",
                "javascript:alert(1)",
                "not a url",
                "",
                {"title": "No link here"},
                "https://example.com/ok",
            ]
        }
    )

    assert [candidate.url for candidate in extract_links(document)] == [
        "https://example.com/ok"
    ]


def test_extract_links_does_not_mutate_the_document() -> None:
    metadata = {
        "links": [
            {"url": "https://example.com/a?utm_source=x", "title": "A"},
        ],
        "references": ["https://openalex.org/W1"],
    }
    document = make_document(metadata)
    before = document.model_dump()

    extract_links(document)

    assert document.model_dump() == before


def test_candidate_to_document_uses_url_hash_and_preserves_metadata() -> None:
    url = "https://example.com/paper?utm_source=feed"
    canonical = canonicalize_url(url)
    candidate = LinkCandidate(
        url=url,
        title="A Paper",
        relation="cites",
        source="openalex",
        source_id="W42",
        abstract="Abstract",
        authors=["Ada"],
        year=2019,
        doi="10.1234/abc",
        metadata={"custom": {"nested": True}},
    )

    document = candidate_to_document(candidate)

    assert document.id == f"pap_{url_hash(canonical)}"
    assert document.kind == "paper"
    assert document.source == "openalex"
    assert document.source_id == "W42"
    assert document.title == "A Paper"
    assert document.summary == ""
    assert document.status == "new"
    assert document.canonical_url == canonical
    assert document.url_hash == url_hash(canonical)
    assert document.authors == ["Ada"]
    assert document.metadata["custom"] == {"nested": True}
    assert document.metadata["doi"] == "10.1234/abc"
    assert document.metadata["year"] == 2019
    assert document.metadata["abstract"] == "Abstract"


def test_candidate_to_document_supports_other_kinds_and_defaults() -> None:
    candidate = LinkCandidate(url="https://example.org/dataset")

    dataset = candidate_to_document(candidate, kind="dataset")

    assert dataset.id.startswith("ds_")
    assert dataset.kind == "dataset"
    assert dataset.source == "web"
    assert dataset.title == "https://example.org/dataset"
    assert dataset.metadata == {}


def test_candidate_to_document_does_not_mutate_candidate() -> None:
    candidate = LinkCandidate(
        url="https://example.com/a",
        authors=["Ada"],
        metadata={"tags": ["x"]},
    )

    document = candidate_to_document(candidate)
    document.authors.append("Bob")
    document.metadata["tags"].append("y")

    assert candidate.authors == ["Ada"]
    assert candidate.metadata == {"tags": ["x"]}
