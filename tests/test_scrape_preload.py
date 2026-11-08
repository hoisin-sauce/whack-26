from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.base import ResearchRecord, SourceError
from whack26.research.scrape import preload
from whack26.research.scrape.identity import metadata_identity, url_hash
from whack26.research.scrape.models import Document
from whack26.store import FieldStore


def make_record(
    *,
    source: str = "openalex",
    source_id: str = "W1",
    title: str = "Statistical validity in practice",
    url: str = "https://example.test/paper-1",
    abstract: str = "A study of validity and reproducibility.",
    year: int = 2024,
    doi: str | None = "10.1234/paper-1",
    query: str = "statistical validity and reproducibility",
) -> ResearchRecord:
    return ResearchRecord(
        id=f"{source}:{source_id}",
        source=source,  # type: ignore[arg-type]
        source_id=source_id,
        title=title,
        abstract=abstract,
        authors=["Ada Lovelace"],
        year=year,
        doi=doi,
        url=url,
        license="cc-by",
        retrieved_at="2026-01-01T00:00:00Z",
        query=query,
        relevance=0.9,
    )


class FakeAdapter:
    name = "openalex"
    ttl_seconds = 60

    def __init__(
        self,
        records: list[ResearchRecord] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.records = records or []
        self.error = error
        self.warnings: list[str] = []

    def search(self, query: str, limit: int, client: Any) -> list[ResearchRecord]:
        if self.error is not None:
            raise self.error
        return [
            record.model_copy(update={"query": query})
            for record in self.records[:limit]
        ]


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_provider": "mock",
        "embed_provider": "mock",
        "embed_dims": 768,
        "base_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "corpus_path": tmp_path / "missing-corpus.md",
    }
    values.update(overrides)
    return Settings(**values)


def make_store(tmp_path: Path) -> FieldStore:
    return FieldStore(tmp_path / "fields.db", dims=768)


def make_root(store: FieldStore, embeddings: MockEmbeddingClient, name: str | None = None):
    vector = embeddings.embed([name or preload.PRELOAD_ROOT_NAME])[0]
    return store.create_field(
        parent_id=None,
        name=name or preload.PRELOAD_ROOT_NAME,
        description=f"{name or preload.PRELOAD_ROOT_NAME} description.",
        description_embedding=vector,
    )


def make_llm(root_id: str | None = None) -> MockLLMClient:
    fixtures = {
        "document_summary": {
            "summary": "A study of statistical validity and reproducibility.",
            "claims": ["Validity matters."],
            "methods": ["Review."],
            "population": None,
            "time_period": None,
            "links": [],
            "credibility_signals": {
                "peer_reviewed": True,
                "publisher": "Journal of Methods",
                "retracted": False,
            },
        }
    }
    if root_id is not None:
        fixtures["document_classification"] = {
            "action": "assign",
            "field_id": root_id,
            "reason": "fixture",
        }
    return MockLLMClient(fixtures=fixtures)


@pytest.fixture()
def store(tmp_path: Path):
    with make_store(tmp_path) as instance:
        yield instance


def test_statistics_queries_are_the_thirteen_plan_seeds() -> None:
    assert len(preload.STATISTICS_QUERIES) == 13
    assert preload.PRELOAD_ROOT_NAME == "Statistics and Validity"
    assert preload.STATISTICS_QUERIES[0] == "statistical validity and reproducibility"
    assert preload.STATISTICS_QUERIES[-1] == "pre-registration and reporting standards"


def test_preload_creates_the_statistics_root(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    llm = make_llm()
    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter([make_record()])
    )
    monkeypatch.setattr(preload, "arXivAdapter", lambda **kwargs: FakeAdapter([]))

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    root = store.find_sibling(None, preload.PRELOAD_ROOT_NAME)
    assert root is not None
    assert len(result.documents) == 1
    path = store.document_path(result.documents[0].id)
    assert path and path[0].id == root.id


def test_preload_processes_deduplicates_and_marks_root(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)
    first = make_record()
    second = make_record(
        source="arxiv",
        source_id="2401.00001",
        title="Causal identification assumptions",
        url="https://arxiv.org/abs/2401.00001",
        doi="10.48550/arxiv.2401.00001",
    )

    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter([first, first])
    )
    monkeypatch.setattr(
        preload, "arXivAdapter", lambda **kwargs: FakeAdapter([second])
    )

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert result.warnings == []
    assert len(result.documents) == 2
    assert {document.title for document in result.documents} == {
        first.title,
        second.title,
    }
    assert all(document.preloaded for document in result.documents)
    for document in result.documents:
        path = store.document_path(document.id)
        assert path and path[0].id == root.id
        assert store.get_document(document.id) is not None

    assert result.stats["documents"] == 2
    assert result.stats["new"] == 2
    assert result.stats["reused"] == 0
    assert result.stats["llm_calls"] >= 3
    assert llm.calls.count("document_summary") == 2


def test_preload_deduplicates_by_title_when_urls_differ(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)
    first = make_record()
    same_work = make_record(
        source="arxiv",
        source_id="2401.00002",
        url="https://arxiv.org/abs/2401.00002",
        doi=first.doi,
        title=first.title,
        abstract=first.abstract,
        year=first.year,
    )

    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter([first])
    )
    monkeypatch.setattr(
        preload, "arXivAdapter", lambda **kwargs: FakeAdapter([same_work])
    )

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert len(result.documents) == 1
    assert llm.calls.count("document_summary") == 1


def test_preload_caps_at_limit(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)
    records = [
        make_record(
            source_id=f"W{index}",
            title=f"Validity paper {index}",
            url=f"https://example.test/paper-{index}",
            doi=f"10.1234/paper-{index}",
        )
        for index in range(5)
    ]

    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter(records)
    )
    monkeypatch.setattr(
        preload, "arXivAdapter", lambda **kwargs: FakeAdapter(error=SourceError(
            "arxiv", "q", "offline cache miss"
        ))
    )

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
        limit=2,
    )

    assert len(result.documents) == 2
    assert result.stats["documents"] == 2
    assert sum(1 for document in result.documents if document.preloaded) == 2


def test_preload_reuses_an_existing_document_without_a_summary_call(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)
    record = make_record()
    existing = Document(
        id="pap_existing",
        kind="paper",
        source="openalex",
        source_id=record.source_id,
        title=record.title,
        summary="Already summarized.",
        summary_embedding=embeddings.embed(["Already summarized."])[0],
        embedding_model="mock",
        canonical_url=record.url,
        url_hash=url_hash(record.url),
        content_sha512=metadata_identity(
            title=record.title,
            abstract=record.abstract,
            authors=record.authors,
            year=record.year,
            doi=record.doi,
            canonical_url=record.url,
        ),
        authors=list(record.authors),
        metadata={"abstract": record.abstract, "year": record.year},
        status="classified",
    )
    store.upsert_document(existing)

    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter([record])
    )
    monkeypatch.setattr(preload, "arXivAdapter", lambda **kwargs: FakeAdapter([]))

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert len(result.documents) == 1
    assert result.stats["reused"] == 1
    assert result.stats["new"] == 0
    assert llm.calls.count("document_summary") == 0
    reused = store.get_document("pap_existing")
    assert reused is not None and reused.preloaded is True
    path = store.document_path("pap_existing")
    assert path and path[0].id == root.id


def test_preload_keeps_documents_under_the_statistics_root(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    other = make_root(store, embeddings, name="Other Root")
    llm = make_llm(other.id)

    monkeypatch.setattr(
        preload, "OpenAlexAdapter", lambda **kwargs: FakeAdapter([make_record()])
    )
    monkeypatch.setattr(preload, "arXivAdapter", lambda **kwargs: FakeAdapter([]))

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert len(result.documents) == 1
    path = store.document_path(result.documents[0].id)
    assert path and path[0].id == root.id
    assert any(
        "preload_classification_outside_root" in warning for warning in result.warnings
    )


def test_preload_returns_warnings_when_sources_fail(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)

    def fail(**kwargs: Any) -> FakeAdapter:
        return FakeAdapter(error=SourceError("openalex", "q", "offline cache miss"))

    monkeypatch.setattr(preload, "OpenAlexAdapter", fail)
    monkeypatch.setattr(preload, "arXivAdapter", fail)

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path, offline=True),
    )

    assert result.documents == []
    assert result.stats["documents"] == 0
    assert result.warnings
    assert all(
        "preload_source_failed" in warning or "preload" in warning
        for warning in result.warnings
    )


def test_preload_returns_a_warning_when_the_client_cannot_be_created(
    tmp_path: Path, store: FieldStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    embeddings = MockEmbeddingClient()
    root = make_root(store, embeddings)
    llm = make_llm(root.id)

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(preload.httpx, "Client", unavailable)

    result = preload.preload_statistics(
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=make_settings(tmp_path),
    )

    assert result.documents == []
    assert any("preload_unavailable" in warning for warning in result.warnings)


def test_preload_limit_zero_is_a_clean_no_op(
    tmp_path: Path, store: FieldStore
) -> None:
    embeddings = MockEmbeddingClient()
    result = preload.preload_statistics(
        store=store,
        llm=make_llm(),
        embeddings=embeddings,
        settings=make_settings(tmp_path),
        limit=0,
    )

    assert result.documents == []
    assert result.warnings == ["preload_limit_zero"]
