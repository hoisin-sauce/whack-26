from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone

import pytest

from whack26.embeddings import EmbeddingError
from whack26.research.base import ResearchRecord
from whack26.research.normalize import (
    deduplicate,
    normalize_doi,
    rank_records,
    record_identity_keys,
    record_kind,
)


def make_record(
    *,
    source: str = "openalex",
    source_id: str = "W1",
    record_id: str | None = None,
    title: str = "A title",
    abstract: str | None = None,
    authors: list[str] | None = None,
    year: int | None = 2024,
    doi: str | None = None,
    url: str | None = None,
    license: str | None = None,
    also_seen_in: list[str] | None = None,
    relevance: float = 0.0,
) -> ResearchRecord:
    return ResearchRecord(
        id=record_id or f"{source}:{source_id}",
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title=title,
        abstract=abstract,
        authors=list(authors or []),
        year=year,
        doi=doi,
        url=url or f"https://example.test/{source}/{source_id}",
        license=license,
        retrieved_at="2026-01-01T00:00:00+00:00",
        query="housing prices",
        relevance=relevance,
        also_seen_in=list(also_seen_in or []),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1000/ABC. ", "10.1000/abc"),
        (" http://dx.doi.org/10.1000/XyZ ", "10.1000/xyz"),
        ("doi: 10.1000/Test.", "10.1000/test"),
        ("DOI: 10.1000/Q;", "10.1000/q"),
        ("10.1000/Plain", "10.1000/plain"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_normalize_doi_strips_prefixes_whitespace_and_punctuation(
    raw: str | None, expected: str | None
) -> None:
    assert normalize_doi(raw) == expected


def test_record_identity_keys_priority_and_forms() -> None:
    record = make_record(
        source="openalex",
        source_id="https://openalex.org/W123",
        doi="https://doi.org/10.1000/ABC.",
        title="Housing Prices",
        authors=["Alice"],
        year=2020,
        url="https://arxiv.org/abs/2401.01234v2",
    )
    keys = record_identity_keys(record)
    expected_title_key = "title:" + hashlib.sha256(
        "housing prices|alice|2020".encode("utf-8")
    ).hexdigest()
    assert keys == [
        "doi:10.1000/abc",
        "arxiv:2401.01234",
        "openalex:W123",
        expected_title_key,
    ]

    hf = make_record(source="huggingface", source_id="owner/dataset")
    assert record_identity_keys(hf)[0] == "hf:owner/dataset"
    zenodo = make_record(source="zenodo", source_id="12345")
    assert record_identity_keys(zenodo)[0] == "zenodo:12345"
    arxiv = make_record(source="arxiv", source_id="hep-th/9901001v2")
    assert record_identity_keys(arxiv)[0] == "arxiv:hep-th/9901001"
    arxiv_doi = make_record(
        source="openalex", source_id="W9", doi="10.48550/arXiv.2401.99999"
    )
    assert "arxiv:2401.99999" in record_identity_keys(arxiv_doi)

    normalized_title = record.model_copy(
        update={
            "title": "  housing   PRICES ",
            "authors": [" ALICE "],
            "doi": None,
            "url": "https://example.test/no-arxiv",
        }
    )
    assert record_identity_keys(normalized_title)[-1] == expected_title_key


def test_deduplicate_same_doi_ranks_richer_metadata_before_source() -> None:
    arxiv = make_record(
        source="arxiv",
        source_id="2401.00001",
        title="An arXiv title",
        abstract="The arXiv abstract",
        authors=["Bob"],
        year=2021,
        doi="https://doi.org/10.1000/ABC",
        license="CC-BY-4.0",
    )
    openalex = make_record(
        source="openalex",
        source_id="W1",
        title="An OpenAlex title",
        abstract=None,
        authors=[],
        year=None,
        doi="10.1000/abc.",
        license=None,
    )

    merged = deduplicate([arxiv, openalex])

    assert len(merged) == 1
    record = merged[0]
    # The richer record wins even though OpenAlex has the higher source rank.
    assert record.id == "arxiv:2401.00001"
    assert record.title == "An arXiv title"
    assert record.abstract == "The arXiv abstract"
    assert record.authors == ["Bob"]
    assert record.year == 2021
    assert record.doi == "10.1000/abc"
    assert record.license == "CC-BY-4.0"
    assert record.also_seen_in == ["openalex:W1"]

    assert arxiv.abstract == "The arXiv abstract"
    assert arxiv.also_seen_in == []
    assert openalex.abstract is None
    assert openalex.authors == []
    assert openalex.year is None
    assert openalex.also_seen_in == []


def test_deduplicate_keeps_longer_abstract_from_lower_ranked_source() -> None:
    long_abstract = "A detailed arXiv abstract. " * 20
    short_abstract = "A short OpenAlex abstract."
    openalex = make_record(
        source="openalex",
        source_id="W1",
        title="Housing prices",
        abstract=short_abstract,
        authors=["Ada", "Bob", "Cara"],
        year=2024,
        doi="10.1000/housing",
    )
    arxiv = make_record(
        source="arxiv",
        source_id="2401.00001",
        title="Housing prices",
        abstract=long_abstract,
        authors=[],
        year=None,
        doi="10.1000/housing",
    )

    merged = deduplicate([openalex, arxiv])

    assert len(merged) == 1
    assert merged[0].abstract == long_abstract
    assert merged[0].authors == ["Ada", "Bob", "Cara"]


def test_deduplicate_bridges_doi_and_arxiv_keys() -> None:
    bridge = make_record(
        source="openalex",
        source_id="W1",
        title="Bridge A",
        abstract="Abstract A",
        doi="10.1000/abc",
        url="https://arxiv.org/abs/2401.01234",
    )
    arxiv_only = make_record(
        source="arxiv",
        source_id="2401.01234v2",
        title="Bridge B",
        abstract="Abstract B",
    )
    doi_only = make_record(
        source="zenodo",
        source_id="999",
        title="Bridge C",
        abstract="Abstract C",
        doi="https://doi.org/10.1000/ABC",
    )

    merged = deduplicate([bridge, arxiv_only, doi_only])

    assert len(merged) == 1
    assert merged[0].id == "openalex:W1"
    assert set(merged[0].also_seen_in) == {
        "arxiv:2401.01234v2",
        "zenodo:999",
    }


def test_deduplicate_title_author_year_fallback() -> None:
    wikipedia = make_record(
        source="wikipedia",
        source_id="Housing_prices",
        title="Housing Prices",
        authors=["Alice Smith"],
        year=2020,
        abstract="Wikipedia text",
    )
    paper = make_record(
        source="openalex",
        source_id="W7",
        title="  housing   prices ",
        authors=[" alice   smith "],
        year=2020,
        abstract="Paper abstract",
    )
    other_year = make_record(
        source="arxiv",
        source_id="2401.00002",
        title="Housing Prices",
        authors=["Alice Smith"],
        year=2021,
    )

    merged = deduplicate([wikipedia, paper])
    assert len(merged) == 1
    assert merged[0].id == "openalex:W7"
    assert merged[0].abstract == "Paper abstract"
    assert merged[0].also_seen_in == ["wikipedia:Housing_prices"]

    assert len(deduplicate([wikipedia, other_year])) == 2


def test_deduplicate_preserves_group_order_and_carries_seen_ids() -> None:
    first = make_record(
        source="openalex",
        source_id="W1",
        title="First paper",
        doi="10.1000/one",
        also_seen_in=["openalex:W0"],
    )
    second = make_record(source="arxiv", source_id="2401.00002", title="Second")
    third = make_record(source="wikipedia", source_id="Third", title="Third")
    duplicate = make_record(
        source="zenodo",
        source_id="42",
        title="Dataset mirror",
        doi="10.1000/one",
        also_seen_in=["huggingface:mirror"],
    )

    merged = deduplicate([first, second, third, duplicate])

    assert [record.id for record in merged] == [
        "openalex:W1",
        "arxiv:2401.00002",
        "wikipedia:Third",
    ]
    assert merged[0].also_seen_in == [
        "openalex:W0",
        "zenodo:42",
        "huggingface:mirror",
    ]


def test_record_kind_maps_sources() -> None:
    assert record_kind(make_record(source="openalex")) == "paper"
    assert record_kind(make_record(source="arxiv")) == "paper"
    assert record_kind(make_record(source="huggingface")) == "dataset"
    assert record_kind(make_record(source="zenodo")) == "dataset"
    assert record_kind(make_record(source="wikipedia")) == "web"


class FixedEmbeddings:
    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        default: tuple[float, ...] = (1.0, 0.0),
    ) -> None:
        self.mapping = mapping or {}
        self.default = list(default)
        self.calls: list[list[str]] = []

    def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.mapping.get(text, self.default)) for text in texts]


class FailingEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        self.calls += 1
        raise EmbeddingError("embedding provider unavailable")


def test_rank_records_uses_weighted_formula_and_single_embed_call() -> None:
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    openalex = make_record(source="openalex", source_id="W1", title="OpenAlex", year=2026)
    wikipedia = make_record(
        source="wikipedia", source_id="Housing", title="Housing", year=2026
    )
    arxiv = make_record(
        source="arxiv", source_id="2401.00001", title="arXiv", year=2021
    )
    client = FixedEmbeddings()

    ranked = rank_records(
        [arxiv, wikipedia, openalex],
        summary="dataset summary",
        embeddings=client,
        top_k=10,
        now=now,
    )

    assert len(client.calls) == 1
    assert client.calls[0] == [
        "dataset summary",
        "arXiv",
        "Housing",
        "OpenAlex",
    ]
    by_id = {record.id: record for record in ranked}
    assert by_id["openalex:W1"].relevance == pytest.approx(
        round(0.55 + 0.25 * 0.8 + 0.20 * 1.0, 6)
    )
    assert by_id["wikipedia:Housing"].relevance == pytest.approx(
        round(0.55 + 0.25 * 0.5 + 0.20 * 1.0, 6)
    )
    assert by_id["arxiv:2401.00001"].relevance == pytest.approx(
        round(0.55 + 0.25 * 0.7 + 0.20 * math.exp(-1.0), 6)
    )
    assert [record.id for record in ranked] == [
        "openalex:W1",
        "wikipedia:Housing",
        "arxiv:2401.00001",
    ]
    assert openalex.relevance == 0.0
    assert wikipedia.relevance == 0.0
    assert arxiv.relevance == 0.0


def test_rank_records_top_k_keeps_a_paper_and_a_dataset() -> None:
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    wikipedia = [
        make_record(
            source="wikipedia",
            source_id=f"Wiki_{index}",
            title=f"Wiki {index}",
            year=2026,
        )
        for index in range(5)
    ]
    paper = make_record(source="arxiv", source_id="2401.00001", title="Paper", year=2026)
    dataset = make_record(
        source="huggingface", source_id="owner/data", title="Dataset", year=2026
    )
    mapping = {record.title: [1.0, 0.0] for record in wikipedia}
    mapping["Paper"] = [0.0, 1.0]
    mapping["Dataset"] = [0.0, 1.0]
    client = FixedEmbeddings(mapping=mapping)

    ranked = rank_records(
        [*wikipedia, paper, dataset],
        summary="dataset summary",
        embeddings=client,
        top_k=3,
        now=now,
    )

    assert len(ranked) == 3
    assert len({record.id for record in ranked}) == 3
    kinds = {record_kind(record) for record in ranked}
    assert "paper" in kinds
    assert "dataset" in kinds
    assert [record.id for record in ranked] == [
        "wikipedia:Wiki_0",
        "arxiv:2401.00001",
        "huggingface:owner/data",
    ]
    assert ranked == sorted(ranked, key=lambda record: (-record.relevance, record.id))


def test_rank_records_breaks_ties_by_id() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = make_record(source="wikipedia", source_id="b", title="B", year=2026)
    second = make_record(source="wikipedia", source_id="a", title="A", year=2026)
    client = FixedEmbeddings()

    ranked = rank_records(
        [first, second],
        summary="summary",
        embeddings=client,
        top_k=10,
        now=now,
    )

    assert [record.id for record in ranked] == ["wikipedia:a", "wikipedia:b"]


def test_rank_records_missing_year_uses_five_year_recency() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    missing = make_record(source="openalex", source_id="W1", title="Missing", year=None)
    known = make_record(source="openalex", source_id="W2", title="Known", year=2021)
    client = FixedEmbeddings()

    ranked = rank_records(
        [known, missing],
        summary="summary",
        embeddings=client,
        top_k=10,
        now=now,
    )

    expected = round(0.55 + 0.25 * 0.8 + 0.20 * math.exp(-1.0), 6)
    assert [record.id for record in ranked] == ["openalex:W1", "openalex:W2"]
    assert all(record.relevance == pytest.approx(expected) for record in ranked)


def test_rank_records_embedding_error_surfaces_warning_and_falls_back() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    openalex = make_record(source="openalex", source_id="W1", title="OpenAlex", year=2026)
    wikipedia = make_record(
        source="wikipedia", source_id="Housing", title="Housing", year=2026
    )
    arxiv = make_record(source="arxiv", source_id="2401.00001", title="arXiv", year=None)
    client = FailingEmbeddings()
    warnings: list[str] = []

    ranked = rank_records(
        [wikipedia, arxiv, openalex],
        summary="summary",
        embeddings=client,
        top_k=10,
        now=now,
        warnings=warnings,
    )

    assert client.calls == 1
    assert len(warnings) == 1
    assert "research_ranking_embeddings_failed" in warnings[0]
    assert "embedding provider unavailable" in warnings[0]
    by_id = {record.id: record for record in ranked}
    assert by_id["openalex:W1"].relevance == pytest.approx(round(0.25 * 0.8 + 0.20, 6))
    assert by_id["wikipedia:Housing"].relevance == pytest.approx(
        round(0.25 * 0.5 + 0.20, 6)
    )
    assert by_id["arxiv:2401.00001"].relevance == pytest.approx(
        round(0.25 * 0.7 + 0.20 * math.exp(-1.0), 6)
    )
    assert [record.id for record in ranked] == [
        "openalex:W1",
        "wikipedia:Housing",
        "arxiv:2401.00001",
    ]


def test_rank_records_embedding_failure_uses_lexical_overlap() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    matching = make_record(
        source="openalex",
        source_id="W1",
        title="Housing prices",
        abstract="Housing prices and income across block groups.",
        year=2026,
    )
    unrelated = make_record(
        source="openalex",
        source_id="W2",
        title="Ocean currents",
        abstract="Circulation patterns in the North Atlantic.",
        year=2026,
    )
    warnings: list[str] = []

    ranked = rank_records(
        [unrelated, matching],
        summary="housing prices",
        embeddings=FailingEmbeddings(),
        top_k=10,
        now=now,
        warnings=warnings,
    )

    assert len(warnings) == 1
    by_id = {record.id: record for record in ranked}
    assert by_id["openalex:W1"].relevance == pytest.approx(
        round(0.55 * 1.0 + 0.25 * 0.8 + 0.20, 6)
    )
    assert by_id["openalex:W2"].relevance == pytest.approx(round(0.25 * 0.8 + 0.20, 6))
    assert [record.id for record in ranked] == ["openalex:W1", "openalex:W2"]


def test_rank_records_empty_input_returns_empty_without_embedding() -> None:
    client = FixedEmbeddings()

    assert (
        rank_records([], summary="summary", embeddings=client, top_k=20) == []
    )
    assert client.calls == []
