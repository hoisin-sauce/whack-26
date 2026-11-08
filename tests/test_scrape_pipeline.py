from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.base import ResearchRecord
from whack26.research.scrape import pipeline
from whack26.research.scrape.identity import canonicalize_url, metadata_identity, url_hash
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

ROOT_ID = "fld_statistics"
DATASET_ID = "ds_local_dataset"
SEED_ID = "pap_seed"
CHILD_ID = "pap_child"
CHILD_URL = "https://example.test/child"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": "mock",
        "embed_provider": "mock",
        "embed_dims": 768,
        "base_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "corpus_path": tmp_path / "missing-corpus.md",
        "scrape_enabled": True,
        "top_k": 5,
        "scrape_summary_threshold": 0.99,
        "scrape_link_threshold": 0.0,
        "scrape_max_depth": 2,
        "scrape_max_resources": 10,
        "scrape_max_links_per_resource": 5,
        "scrape_max_fetches": 10,
    }
    values.update(overrides)
    return Settings(**values)


def make_store(tmp_path: Path) -> FieldStore:
    return FieldStore(tmp_path / "fields.db", dims=768)


def make_root(store: FieldStore, embeddings: MockEmbeddingClient):
    return store.create_field(
        parent_id=None,
        name="Research Methods",
        description="Research methods and statistical validity.",
        description_embedding=embeddings.embed(["Research methods."])[0],
        field_id=ROOT_ID,
    )


def make_llm(root_id: str) -> MockLLMClient:
    return MockLLMClient(
        fixtures={
            "document_summary": {
                "summary": "A study of housing prices and income.",
                "claims": ["Income predicts housing prices."],
                "methods": ["Regression on survey data."],
                "population": None,
                "time_period": None,
                "links": [],
                "credibility_signals": {
                    "peer_reviewed": True,
                    "publisher": "Journal of Housing",
                    "retracted": False,
                },
            },
            "document_classification": {
                "action": "assign",
                "field_id": root_id,
                "reason": "fixture",
            },
            "document_comparison": {
                "verdict": "supports",
                "rationale": "The document studies the same housing population.",
                "evidence": ["housing prices and income"],
            },
            "critic": {
                "claim_type": "description",
                "verdict": "pass",
                "issues": [],
                "missing_information": [],
                "summary": "No method gap was identified by the mock critic.",
                "retrieved_chunks": ["S0-how-to-use"],
            },
        }
    )


def make_dataset(embeddings: MockEmbeddingClient) -> Document:
    summary = "Housing prices by block group."
    return Document(
        id=DATASET_ID,
        kind="dataset",
        source="user",
        source_id=DATASET_ID,
        title="Local housing dataset",
        summary=summary,
        summary_embedding=embeddings.embed([summary])[0],
        embedding_model="mock",
        metadata={"is_target_dataset": True},
        status="classified",
    )


def make_seed() -> Document:
    return Document(
        id=SEED_ID,
        kind="paper",
        source="openalex",
        source_id="W-seed",
        title="Housing prices and income",
        summary="",
        canonical_url="https://example.test/seed",
        authors=["Ada Lovelace"],
        metadata={
            "abstract": "We study housing prices and income.",
            "year": 2024,
            "links": [CHILD_URL],
        },
        status="new",
    )


def make_child() -> Document:
    abstract = "A second study of housing prices."
    canonical = canonicalize_url(CHILD_URL)
    return Document(
        id=CHILD_ID,
        kind="paper",
        source="openalex",
        source_id="W-child",
        title="A second housing study",
        summary="",
        canonical_url=canonical,
        url_hash=url_hash(canonical),
        content_sha512=metadata_identity(
            title="A second housing study",
            abstract=abstract,
            authors=["Grace Hopper"],
            year=2023,
            doi="10.1234/child",
            canonical_url=canonical,
        ),
        authors=["Grace Hopper"],
        metadata={"abstract": abstract, "year": 2023, "doi": "10.1234/child"},
        status="new",
    )


def write_corpus(path: Path) -> Path:
    path.write_text(
        "---\nversion: \"test-1\"\n---\n\n"
        "## How to use\n"
        "**Chunk ID:** `S0-how-to-use`\n"
        "Use this corpus to review methods.\n\n"
        "## Assumptions\n"
        "**Chunk ID:** `S1-assumptions`\n"
        "Check the assumptions behind a method.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def store(tmp_path: Path):
    with make_store(tmp_path) as instance:
        root = make_root(instance, MockEmbeddingClient())
        assert root.id == ROOT_ID
        yield instance


def test_disabled_scraping_makes_no_calls(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("semantic scraping must not run when disabled")

    monkeypatch.setattr("whack26.research.scrape.frontier.expand_frontier", explode)
    monkeypatch.setattr(
        "whack26.research.scrape.summarize.embed_summary", explode
    )

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed()],
        target="housing prices",
        store=store,
        llm=make_llm(ROOT_ID),
        embeddings=MockEmbeddingClient(),
        settings=make_settings(tmp_path, scrape_enabled=False),
    )

    assert result.warnings == ["scrape_disabled"]
    assert result.documents == []
    assert result.links == []
    assert result.comparisons == []
    assert result.critiques == []
    assert result.stats["documents"] == 0


def test_two_document_graph_is_processed_classified_and_compared(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm(ROOT_ID)
    fetched: list[str] = []

    def fake_fetch(candidate: Any) -> Document | None:
        fetched.append(candidate.url)
        if candidate.url == CHILD_URL:
            return make_child()
        return None

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed(), make_dataset(embeddings)],
        target="Housing prices by block group.",
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
        fetch=fake_fetch,
    )

    assert fetched == [CHILD_URL]
    assert {document.id for document in result.documents} == {SEED_ID, CHILD_ID}
    assert len(result.links) == 1
    assert result.links[0].from_id == SEED_ID
    assert result.links[0].to_id == CHILD_ID
    assert result.links[0].relation == "references"

    assert len(result.comparisons) == 2
    assert {comparison.right_id for comparison in result.comparisons} == {
        SEED_ID,
        CHILD_ID,
    }
    assert all(comparison.left_id == DATASET_ID for comparison in result.comparisons)
    assert all(comparison.kind == "dataset_relevance" for comparison in result.comparisons)
    assert all(comparison.verdict == "supports" for comparison in result.comparisons)

    for document in result.documents:
        stored = store.get_document(document.id)
        assert stored is not None
        assert stored.credible is True
        assert document.status == "compared"
        path = store.document_path(document.id)
        assert path and path[0].id == ROOT_ID

    assert result.stats["documents"] == 2
    assert result.stats["credible"] == 2
    assert result.stats["links"] == 1
    assert result.stats["comparisons"] == 2
    assert result.stats["critiques"] == 0
    assert result.stats["cache_hits"] == 0
    assert llm.calls.count("document_summary") == 2
    assert llm.calls.count("document_comparison") == 2


def test_second_run_reuses_summaries_and_comparisons(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm(ROOT_ID)

    def fake_fetch(candidate: Any) -> Document | None:
        return make_child() if candidate.url == CHILD_URL else None

    seeds = [make_seed(), make_dataset(embeddings)]
    args = {
        "seeds": seeds,
        "target": "Housing prices by block group.",
        "store": store,
        "llm": llm,
        "embeddings": embeddings,
        "settings": make_settings(tmp_path),
        "fetch": fake_fetch,
    }
    first = pipeline.run_semantic_scraping(**args)
    assert first.stats["comparisons"] == 2
    summary_calls = llm.calls.count("document_summary")
    comparison_calls = llm.calls.count("document_comparison")

    second = pipeline.run_semantic_scraping(**args)

    assert llm.calls.count("document_summary") == summary_calls
    assert llm.calls.count("document_comparison") == comparison_calls
    assert second.stats["cache_hits"] >= 1


def test_fetch_failure_is_a_warning_not_a_crash(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm(ROOT_ID)

    def missing(candidate: Any) -> Document | None:
        return None

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed(), make_dataset(embeddings)],
        target="Housing prices by block group.",
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
        fetch=missing,
    )

    assert {document.id for document in result.documents} == {SEED_ID}
    assert any(
        warning.startswith("scrape_fetch_failed") for warning in result.warnings
    )
    assert result.stats["comparisons"] == 1


def test_record_seeds_are_converted_by_source(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm(ROOT_ID)
    record = ResearchRecord(
        id="zenodo:123",
        source="zenodo",
        source_id="123",
        title="Housing survey dataset",
        abstract="A survey of housing and income.",
        authors=["Alan Turing"],
        year=2022,
        doi="10.5281/zenodo.123",
        url="https://zenodo.org/records/123",
        license="cc-by-4.0",
        retrieved_at="2026-01-01T00:00:00Z",
        query="housing dataset",
        relevance=0.8,
    )

    result = pipeline.run_semantic_scraping(
        seeds=[record],
        target="Housing prices by block group.",
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.kind == "dataset"
    assert document.source == "zenodo"
    assert document.source_id == "123"
    assert document.metadata["doi"] == "10.5281/zenodo.123"
    assert store.get_document(document.id) is not None


def test_document_critic_runs_when_a_corpus_is_available(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm(ROOT_ID)
    corpus_path = write_corpus(tmp_path / "corpus.md")

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed(), make_dataset(embeddings)],
        target="Housing prices by block group.",
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path, corpus_path=corpus_path),
        fetch=lambda candidate: make_child() if candidate.url == CHILD_URL else None,
    )

    assert len(result.critiques) == 2
    assert {critique.document_id for critique in result.critiques} == {
        SEED_ID,
        CHILD_ID,
    }
    assert all(critique.verdict == "pass" for critique in result.critiques)
    assert result.stats["critiques"] == 2
    assert store.get_critique(SEED_ID, result.critiques[0].corpus_version) is not None


def test_target_may_be_a_precomputed_embedding(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    target = embeddings.embed(["Housing prices by block group."])[0]

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed()],
        target=target,
        store=store,
        llm=make_llm(ROOT_ID),
        embeddings=embeddings,
        settings=make_settings(tmp_path),
        fetch=lambda candidate: make_child() if candidate.url == CHILD_URL else None,
    )

    assert {document.id for document in result.documents} == {SEED_ID, CHILD_ID}
    assert "scrape_target_missing" not in result.warnings


def test_missing_target_is_a_warning_not_a_crash(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    result = pipeline.run_semantic_scraping(
        seeds=[make_seed()],
        target="",
        store=store,
        llm=make_llm(ROOT_ID),
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert result.documents == []
    assert "scrape_target_missing" in result.warnings


def test_default_mock_fixtures_still_produce_documents(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()

    result = pipeline.run_semantic_scraping(
        seeds=[make_seed()],
        target="housing prices",
        store=store,
        llm=MockLLMClient(),
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.summary
    assert store.document_path(document.id)


def test_empty_seed_list_is_a_clean_no_op(
    tmp_path: Path, store: FieldStore
) -> None:
    result = pipeline.run_semantic_scraping(
        seeds=[],
        target="housing prices",
        store=store,
        llm=make_llm(ROOT_ID),
        embeddings=MockEmbeddingClient(),
        settings=make_settings(tmp_path),
    )

    assert result.documents == []
    assert result.warnings == ["scrape_no_seeds"]
