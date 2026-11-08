from __future__ import annotations

import math
from typing import Any

import pytest

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.scrape.models import Document, SummaryResult
from whack26.research.scrape.summarize import (
    MAX_SUMMARY_CHARS,
    SUMMARY_SCHEMA,
    SUMMARY_SYSTEM_PROMPT,
    embed_summary,
    summarize_document,
)

FIXTURE: dict[str, Any] = {
    "summary": "The paper studies housing prices using a panel model.",
    "claims": ["Prices rise with income."],
    "methods": ["Fixed-effects regression"],
    "population": "Homeowners",
    "time_period": "2000-2020",
    "links": ["https://example.test/reference"],
    "credibility_signals": {
        "peer_reviewed": True,
        "publisher": "Journal of Housing",
        "retracted": False,
    },
}


def make_settings() -> Settings:
    return Settings(llm_provider="mock", embed_provider="mock", embed_dims=8)


def make_document(**overrides: Any) -> Document:
    data: dict[str, Any] = {
        "id": "doc_abc123",
        "kind": "paper",
        "source": "openalex",
        "source_id": "W123",
        "title": "Housing prices and income",
        "authors": ["A. Author", "B. Author"],
        "publisher": "Journal of Housing",
        "published_at": "2021-05-01",
        "license": "CC-BY-4.0",
        "canonical_url": "https://example.test/paper",
        "metadata": {
            "abstract": "We estimate the relationship between income and prices.",
            "doi": "10.1234/example",
        },
    }
    data.update(overrides)
    return Document(**data)


class RecordingMockLLM(MockLLMClient):
    """Mock client that also records the prompts for redaction assertions."""

    def __init__(self, fixtures: dict[str, Any] | None = None) -> None:
        super().__init__(fixtures)
        self.user_prompts: list[str] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        self.user_prompts.append(user)
        return super().generate_json(
            system=system,
            user=user,
            schema=schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
        )


def test_summary_schema_and_prompt_are_exposed() -> None:
    assert SUMMARY_SCHEMA["type"] == "object"
    assert set(SUMMARY_SCHEMA["required"]) == {
        "summary",
        "claims",
        "methods",
        "population",
        "time_period",
        "links",
        "credibility_signals",
    }
    assert "summary" in SUMMARY_SYSTEM_PROMPT.lower()


def test_summarize_document_returns_validated_summary() -> None:
    llm = MockLLMClient(fixtures={"document_summary": FIXTURE})
    result = summarize_document(
        make_document(), "housing price determinants", llm, make_settings()
    )

    assert isinstance(result, SummaryResult)
    assert result.summary == FIXTURE["summary"]
    assert result.claims == FIXTURE["claims"]
    assert result.methods == FIXTURE["methods"]
    assert result.population == "Homeowners"
    assert result.time_period == "2000-2020"
    assert result.links == FIXTURE["links"]
    assert result.credibility_signals["peer_reviewed"] is True
    assert llm.calls == ["document_summary"]


def test_prompt_contains_title_abstract_target_and_bounded_public_text() -> None:
    llm = RecordingMockLLM(fixtures={"document_summary": FIXTURE})
    document = make_document(
        title="A very specific title",
        metadata={
            "abstract": "A short abstract.",
            "text": "x" * 5000,
        },
    )

    summarize_document(
        document,
        "the target question",
        llm,
        make_settings(),
        max_input_chars=2000,
    )

    prompt = llm.user_prompts[0]
    assert "A very specific title" in prompt
    assert "A short abstract." in prompt
    assert "the target question" in prompt
    assert "x" * 2000 in prompt
    assert "x" * 2001 not in prompt


def test_custom_max_input_chars_truncates_public_text() -> None:
    llm = RecordingMockLLM(fixtures={"document_summary": FIXTURE})
    summarize_document(
        make_document(metadata={"text": "abcdefghij"}),
        "target",
        llm,
        make_settings(),
        max_input_chars=4,
    )
    prompt = llm.user_prompts[0]
    assert "abcd" in prompt
    assert "abcde" not in prompt


def test_empty_summary_is_rejected() -> None:
    for fixture in ({"summary": ""}, {"summary": "   "}, {"claims": []}):
        llm = MockLLMClient(fixtures={"document_summary": fixture})
        with pytest.raises(ValueError, match="empty"):
            summarize_document(make_document(), "target", llm, make_settings())


def test_oversized_summary_is_rejected() -> None:
    llm = MockLLMClient(
        fixtures={"document_summary": {**FIXTURE, "summary": "y" * 1201}}
    )
    with pytest.raises(ValueError, match="1200"):
        summarize_document(make_document(), "target", llm, make_settings())


def test_summary_at_the_limit_is_accepted() -> None:
    boundary = "y" * MAX_SUMMARY_CHARS
    llm = MockLLMClient(fixtures={"document_summary": {**FIXTURE, "summary": boundary}})
    result = summarize_document(make_document(), "target", llm, make_settings())
    assert result.summary == boundary


def test_secrets_are_redacted_before_the_llm_call() -> None:
    llm = RecordingMockLLM(fixtures={"document_summary": FIXTURE})
    document = make_document(
        metadata={
            "abstract": "Contact the author: Authorization: Bearer sk-live-token",
            "text": "Fetch https://example.test/?api_key=super-secret-value for data.",
        }
    )

    summarize_document(document, "target", llm, make_settings())

    prompt = llm.user_prompts[0]
    assert "sk-live-token" not in prompt
    assert "super-secret-value" not in prompt
    assert "[redacted]" in prompt


def test_embed_summary_returns_one_normalized_vector() -> None:
    embeddings = MockEmbeddingClient(dims=16)
    vector = embed_summary("a concise summary", embeddings)

    assert len(vector) == 16
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
    assert embeddings.calls == [["a concise summary"]]


class BadEmbeddings:
    def __init__(self, batch: Any) -> None:
        self.batch = batch

    def embed(self, texts: list[str], *, model: str | None = None) -> Any:
        return self.batch


def test_embed_summary_validates_vector_count() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        embed_summary("text", BadEmbeddings([]))
    with pytest.raises(ValueError, match="exactly one"):
        embed_summary("text", BadEmbeddings([[1.0], [2.0]]))


def test_embed_summary_rejects_empty_or_non_finite_vectors() -> None:
    with pytest.raises(ValueError, match="empty"):
        embed_summary("text", BadEmbeddings([[]]))
    with pytest.raises(ValueError, match="finite"):
        embed_summary("text", BadEmbeddings([[1.0, float("nan")]]))
    with pytest.raises(ValueError, match="empty"):
        embed_summary("   ", MockEmbeddingClient(dims=8))
