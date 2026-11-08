from __future__ import annotations

from typing import Any

import pytest

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.scrape.models import Document, SummaryResult
from whack26.research.scrape.process import process_document

FIXTURE: dict[str, Any] = {
    "summary": "A panel study of housing prices and income.",
    "claims": ["Income predicts prices."],
    "methods": ["Fixed-effects regression"],
    "population": "Homeowners",
    "time_period": "2000-2020",
    "links": ["https://example.test/new-reference"],
    "credibility_signals": {
        "peer_reviewed": True,
        "publisher": "Journal of Housing",
        "retracted": False,
    },
}


def make_settings(**overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "llm_provider": "mock",
        "embed_provider": "mock",
        "embed_dims": 8,
    }
    data.update(overrides)
    return Settings(**data)


def make_document(**overrides: Any) -> Document:
    data: dict[str, Any] = {
        "id": "doc_abc123",
        "kind": "paper",
        "source": "openalex",
        "source_id": "W123",
        "title": "Housing prices and income",
        "authors": ["A. Author"],
        "publisher": "Journal of Housing",
        "published_at": "2021-05-01",
        "license": "CC-BY-4.0",
        "canonical_url": "https://example.test/paper",
        "url_hash": "url-hash",
        "content_sha512": "content-hash",
        "metadata": {
            "abstract": "We estimate the relationship between income and prices.",
            "links": ["https://example.test/old-reference"],
            "source_query": "housing",
        },
        "credibility_score": 0.8,
        "credible": True,
        "status": "fetched",
        "created_at": "2026-11-01T00:00:00Z",
    }
    data.update(overrides)
    return Document(**data)


def make_summary(**overrides: Any) -> SummaryResult:
    data = dict(FIXTURE)
    data.update(overrides)
    return SummaryResult(**data)


def test_process_document_summarizes_and_embeds() -> None:
    document = make_document()
    llm = MockLLMClient(fixtures={"document_summary": FIXTURE})
    embeddings = MockEmbeddingClient(dims=8)

    result = process_document(document, "target question", llm, embeddings, make_settings())

    assert llm.calls == ["document_summary"]
    assert embeddings.calls == [[FIXTURE["summary"]]]
    assert result.summary == FIXTURE["summary"]
    assert len(result.summary_embedding) == 8
    assert result.embedding_model == "mock"
    assert result.status == "summarized"
    assert result.metadata["links"] == [
        "https://example.test/old-reference",
        "https://example.test/new-reference",
    ]


def test_process_document_accepts_a_precomputed_summary_without_llm() -> None:
    llm = MockLLMClient()
    summary = make_summary(links=["https://example.test/precomputed"])

    result = process_document(
        make_document(),
        "target",
        llm,
        MockEmbeddingClient(dims=8),
        make_settings(),
        summary=summary,
    )

    assert llm.calls == []
    assert result.summary == summary.summary
    assert "https://example.test/precomputed" in result.metadata["links"]


def test_process_document_accepts_a_summary_mapping() -> None:
    result = process_document(
        make_document(),
        "target",
        MockLLMClient(),
        MockEmbeddingClient(dims=8),
        make_settings(),
        summary=dict(FIXTURE),
    )
    assert result.summary == FIXTURE["summary"]


def test_process_document_does_not_mutate_the_input() -> None:
    document = make_document()
    before = document.model_dump()

    result = process_document(
        document,
        "target",
        MockLLMClient(fixtures={"document_summary": FIXTURE}),
        MockEmbeddingClient(dims=8),
        make_settings(),
    )

    assert document.model_dump() == before
    assert result is not document
    assert result != document

    result.metadata["links"].append("https://example.test/later")
    result.metadata["abstract"] = "changed"
    assert document.metadata["abstract"] == before["metadata"]["abstract"]
    assert document.metadata["links"] == ["https://example.test/old-reference"]


def test_process_document_preserves_links_and_other_fields() -> None:
    document = make_document()
    summary = make_summary(
        links=[
            "https://example.test/old-reference",
            "https://example.test/new-reference",
            "https://example.test/new-reference",
        ]
    )

    result = process_document(
        document,
        "target",
        MockLLMClient(),
        MockEmbeddingClient(dims=8),
        make_settings(),
        summary=summary,
    )

    assert result.metadata["links"] == [
        "https://example.test/old-reference",
        "https://example.test/new-reference",
    ]
    assert result.metadata["abstract"] == document.metadata["abstract"]
    assert result.metadata["source_query"] == "housing"
    for field in (
        "id",
        "kind",
        "source",
        "source_id",
        "title",
        "authors",
        "publisher",
        "published_at",
        "license",
        "canonical_url",
        "url_hash",
        "content_sha512",
        "credibility_score",
        "credible",
        "created_at",
    ):
        assert getattr(result, field) == getattr(document, field)


def test_embedding_model_tracks_the_provider() -> None:
    summary = make_summary()

    gemini = process_document(
        make_document(),
        "target",
        MockLLMClient(),
        MockEmbeddingClient(dims=8),
        make_settings(
            llm_provider="gemini",
            embed_provider="gemini",
            gemini_embed_model="custom-embedding-model",
        ),
        summary=summary,
    )
    assert gemini.embedding_model == "custom-embedding-model"

    local = process_document(
        make_document(),
        "target",
        MockLLMClient(),
        MockEmbeddingClient(dims=8),
        make_settings(embed_provider="local"),
        summary=summary,
    )
    assert local.embedding_model == "local"


def test_process_document_rejects_bad_summary_type() -> None:
    with pytest.raises(TypeError):
        process_document(
            make_document(),
            "target",
            MockLLMClient(),
            MockEmbeddingClient(dims=8),
            make_settings(),
            summary="not a summary",  # type: ignore[arg-type]
        )
