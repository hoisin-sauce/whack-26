"""Plumbing tests for bibliometric metrics across research and scraping."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from whack26.artifact import ResearchRecordSection
from whack26.config import Settings
from whack26.research.base import ResearchRecord
from whack26.research.metrics import merge_document_metrics
from whack26.research.normalize import deduplicate
from whack26.research.scrape import pipeline, preload
from whack26.research.scrape.credibility import score_credibility
from whack26.research.scrape.models import Document

METRIC_KEYS = (
    "cited_by_count",
    "author_ids",
    "author_h_index",
    "author_citation_count",
    "author_known",
    "venue_type",
    "metrics_source",
)

METRICS = {
    "cited_by_count": 123,
    "author_ids": [
        "https://openalex.org/A111",
        "https://openalex.org/A222",
    ],
    "author_h_index": 17,
    "author_citation_count": 2500,
    "author_known": True,
    "venue_type": "journal",
    "publisher": "Journal of Examples",
    "metrics_source": "openalex",
}


def make_record(
    *,
    record_id: str = "openalex:W1",
    source: str = "openalex",
    source_id: str = "W1",
    title: str = "A study of housing prices",
    abstract: str | None = None,
    authors: list[str] | None = None,
    year: int | None = 2024,
    doi: str | None = None,
    url: str = "https://example.test/paper",
    cited_by_count: int | None = None,
    author_ids: list[str] | None = None,
    author_h_index: int | None = None,
    author_citation_count: int | None = None,
    author_known: bool | None = None,
    venue_type: str | None = None,
    publisher: str | None = None,
    metrics_source: str | None = None,
) -> ResearchRecord:
    """Build a record with every required field and optional metrics."""
    return ResearchRecord(
        id=record_id,
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title=title,
        abstract=abstract,
        authors=list(authors or []),
        year=year,
        doi=doi,
        url=url,
        retrieved_at="2026-01-01T00:00:00+00:00",
        query="housing prices",
        cited_by_count=cited_by_count,
        author_ids=list(author_ids or []),
        author_h_index=author_h_index,
        author_citation_count=author_citation_count,
        author_known=author_known,
        venue_type=venue_type,
        publisher=publisher,
        metrics_source=metrics_source,
    )


def test_deduplicate_merges_metrics_from_shorter_duplicate() -> None:
    winner = make_record(
        record_id="arxiv:2401.00001",
        source="arxiv",
        source_id="2401.00001",
        abstract="A detailed abstract. " * 20,
        doi="10.1000/metrics",
        authors=["Ada Lovelace"],
    )
    loser = make_record(
        record_id="openalex:W1",
        source="openalex",
        source_id="W1",
        abstract="A short abstract.",
        doi="https://doi.org/10.1000/METRICS",
        authors=["Ada Lovelace"],
        cited_by_count=321,
        author_ids=[
            "https://openalex.org/A111",
            "https://openalex.org/A222",
        ],
        author_h_index=42,
        author_citation_count=9000,
        author_known=True,
        venue_type="journal",
        publisher="Journal of Examples",
        metrics_source="openalex",
    )

    merged = deduplicate([winner, loser])

    assert len(merged) == 1
    record = merged[0]
    # The longer abstract still picks the winner; metrics come from the loser.
    assert record.id == "arxiv:2401.00001"
    assert record.cited_by_count == 321
    assert record.author_ids == [
        "https://openalex.org/A111",
        "https://openalex.org/A222",
    ]
    assert record.author_h_index == 42
    assert record.author_citation_count == 9000
    assert record.author_known is True
    assert record.venue_type == "journal"
    assert record.publisher == "Journal of Examples"
    assert record.metrics_source == "openalex"


def test_deduplicate_merges_metric_maxima_and_author_id_order() -> None:
    winner = make_record(
        record_id="arxiv:2401.00002",
        source="arxiv",
        source_id="2401.00002",
        abstract="A much longer abstract. " * 20,
        doi="10.1000/maxima",
        authors=["Ada Lovelace"],
        cited_by_count=42,
        author_ids=["A1", "A2"],
        author_h_index=12,
        author_citation_count=1500,
        author_known=False,
        venue_type="preprint",
        publisher="arXiv",
        metrics_source="arxiv",
    )
    loser = make_record(
        record_id="openalex:W2",
        source="openalex",
        source_id="W2",
        abstract="Short.",
        doi="10.1000/maxima",
        authors=["Ada Lovelace"],
        cited_by_count=100,
        author_ids=["A2", "A3"],
        author_h_index=33,
        author_citation_count=5000,
        author_known=None,
        venue_type="journal",
        publisher="Journal of Examples",
        metrics_source="openalex",
    )

    merged = deduplicate([winner, loser])

    assert len(merged) == 1
    record = merged[0]
    assert record.id == "arxiv:2401.00002"
    assert record.cited_by_count == 100
    assert record.author_ids == ["A1", "A2", "A3"]
    assert record.author_h_index == 33
    assert record.author_citation_count == 5000
    assert record.author_known is False
    assert record.venue_type == "preprint"
    assert record.publisher == "arXiv"
    assert record.metrics_source == "arxiv"


def test_pipeline_record_to_document_copies_metrics() -> None:
    record = make_record(doi="10.1000/copy", **METRICS)

    document = pipeline._record_to_document(record)

    assert document.metadata["cited_by_count"] == METRICS["cited_by_count"]
    assert document.metadata["author_ids"] == METRICS["author_ids"]
    assert document.metadata["author_h_index"] == METRICS["author_h_index"]
    assert (
        document.metadata["author_citation_count"]
        == METRICS["author_citation_count"]
    )
    assert document.metadata["author_known"] is True
    assert document.metadata["venue_type"] == METRICS["venue_type"]
    assert document.metadata["metrics_source"] == METRICS["metrics_source"]
    assert document.publisher == METRICS["publisher"]


def test_pipeline_record_to_document_omits_absent_metrics() -> None:
    document = pipeline._record_to_document(make_record())

    for key in METRIC_KEYS:
        assert key not in document.metadata
    assert document.publisher is None


def test_preload_record_to_document_copies_metrics() -> None:
    record = make_record(doi="10.1000/preload-copy", **METRICS)

    document = preload._record_to_document(record, preloaded=True)

    assert document.preloaded is True
    assert document.metadata["cited_by_count"] == METRICS["cited_by_count"]
    assert document.metadata["author_ids"] == METRICS["author_ids"]
    assert document.metadata["author_h_index"] == METRICS["author_h_index"]
    assert (
        document.metadata["author_citation_count"]
        == METRICS["author_citation_count"]
    )
    assert document.metadata["author_known"] is True
    assert document.metadata["venue_type"] == METRICS["venue_type"]
    assert document.metadata["metrics_source"] == METRICS["metrics_source"]
    assert document.publisher == METRICS["publisher"]


def test_preload_record_to_document_omits_absent_metrics() -> None:
    document = preload._record_to_document(make_record(), preloaded=True)

    for key in METRIC_KEYS:
        assert key not in document.metadata
    assert document.publisher is None


def test_merge_document_metrics_upgrades_a_reused_document() -> None:
    existing = Document(
        id="doc_reused",
        kind="paper",
        source="openalex",
        title="A study",
        metadata={},
    )
    incoming = Document(
        id="doc_reused",
        kind="paper",
        source="openalex",
        title="A study",
        publisher="Elsevier BV",
        metadata={
            "cited_by_count": 42,
            "author_ids": ["openalex:A1"],
            "author_h_index": 20,
            "author_known": True,
            "venue_type": "journal",
            "metrics_source": "openalex",
        },
    )

    merged = merge_document_metrics(existing, incoming)

    assert merged.metadata["cited_by_count"] == 42
    assert merged.metadata["author_ids"] == ["openalex:A1"]
    assert merged.metadata["author_h_index"] == 20
    assert merged.metadata["author_known"] is True
    assert merged.metadata["venue_type"] == "journal"
    assert merged.metadata["metrics_source"] == "openalex"
    assert merged.publisher == "Elsevier BV"
    # The original document is not mutated.
    assert existing.metadata == {}
    assert existing.publisher is None


def test_merge_document_metrics_keeps_maximums_and_unions_authors() -> None:
    existing = Document(
        id="doc_reused",
        kind="paper",
        source="openalex",
        title="A study",
        metadata={
            "cited_by_count": 10,
            "author_ids": ["openalex:A1"],
            "author_h_index": 30,
        },
    )
    incoming = Document(
        id="doc_reused",
        kind="paper",
        source="openalex",
        title="A study",
        metadata={
            "cited_by_count": 7,
            "author_ids": ["openalex:A1", "openalex:A2"],
            "author_h_index": 20,
        },
    )

    merged = merge_document_metrics(existing, incoming)

    assert merged.metadata["cited_by_count"] == 10
    assert merged.metadata["author_ids"] == ["openalex:A1", "openalex:A2"]
    assert merged.metadata["author_h_index"] == 30


class _FakeDocumentStore:
    def __init__(self, existing: Document) -> None:
        self.existing = existing
        self.upserts: list[Document] = []

    def find_document_by_url(self, url_hash: str) -> Document | None:
        return self.existing if self.existing.url_hash == url_hash else None

    def find_document_by_content(self, content_sha512: str) -> Document | None:
        return (
            self.existing
            if self.existing.content_sha512 == content_sha512
            else None
        )

    def upsert_document(self, document: Document) -> Document:
        self.upserts.append(document)
        self.existing = document
        return document


def test_dedupe_seeds_persists_metrics_on_a_reused_document() -> None:
    existing = Document(
        id="doc_reuse",
        kind="paper",
        source="openalex",
        title="A study",
        summary="An existing summary.",
        summary_embedding=[1.0],
        url_hash="url-hash",
        content_sha512="content-hash",
        metadata={},
    )
    incoming = Document(
        id="doc_reuse",
        kind="paper",
        source="openalex",
        title="A study",
        url_hash="url-hash",
        content_sha512="content-hash",
        publisher="Elsevier BV",
        metadata={
            "cited_by_count": 42,
            "author_ids": ["openalex:A1"],
            "author_h_index": 20,
        },
    )
    store = _FakeDocumentStore(existing)

    result = pipeline._dedupe_seeds(
        [incoming],
        target="",
        store=store,
        dataset_id=None,
        llm=None,
        embeddings=None,
        settings=Settings(),
        warnings=[],
    )

    assert result[0].metadata["cited_by_count"] == 42
    assert result[0].metadata["author_ids"] == ["openalex:A1"]
    assert result[0].publisher == "Elsevier BV"
    assert store.upserts, "merged metrics must be written back to the store"


def test_credibility_uses_document_metric_metadata() -> None:
    document = Document(
        id="doc_openalex_metrics",
        kind="paper",
        source="openalex",
        source_id="W1",
        title="A study",
        authors=["Ada Lovelace"],
        published_at=datetime.now(timezone.utc).date().isoformat(),
        metadata={
            "cited_by_count": 100,
            "author_h_index": 20,
            "author_known": True,
            "venue_type": "journal",
            "peer_reviewed": True,
        },
    )

    result = score_credibility(
        document, Settings(scrape_credibility_threshold=0.5)
    )

    assert result.components["author"] == pytest.approx(0.75)
    assert result.components["citation"] == pytest.approx(
        math.log1p(100) / math.log1p(1000)
    )
    assert result.credible is True


def test_artifact_record_section_round_trips_metric_fields() -> None:
    record = make_record(
        doi="10.1000/artifact",
        cited_by_count=77,
        author_ids=[
            "https://openalex.org/A111",
            "https://openalex.org/A222",
        ],
        author_h_index=31,
        author_citation_count=4321,
        author_known=True,
        venue_type="journal",
        publisher="Journal of Examples",
        metrics_source="openalex",
    )

    section = ResearchRecordSection.model_validate(record.model_dump(mode="json"))

    assert section.cited_by_count == 77
    assert section.author_ids == [
        "https://openalex.org/A111",
        "https://openalex.org/A222",
    ]
    assert section.author_h_index == 31
    assert section.author_citation_count == 4321
    assert section.author_known is True
    assert section.venue_type == "journal"
    assert section.publisher == "Journal of Examples"
    assert section.metrics_source == "openalex"
