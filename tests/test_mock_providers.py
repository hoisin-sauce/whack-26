from __future__ import annotations

import math

import pytest

from whack26.embeddings import MockEmbeddingClient, cosine, l2_normalize
from whack26.llm import MockLLMClient


def test_mock_embeddings_are_deterministic_and_normalized() -> None:
    client = MockEmbeddingClient(dims=16)
    first = client.embed(["alpha", "beta"])
    second = client.embed(["alpha", "beta"])
    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 16 for vector in first)
    assert all(abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-9 for vector in first)
    assert first[0] != first[1]


def test_empty_embedding_batch() -> None:
    assert MockEmbeddingClient(dims=8).embed([]) == []


def test_cosine_and_normalization() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="mismatch"):
        cosine([1.0], [1.0, 2.0])
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_mock_llm_fixtures_are_deterministic_and_isolated() -> None:
    client = MockLLMClient()
    first = client.generate_json(
        system="s", user="u", schema={}, purpose="classification"
    )
    first["name"] = "mutated"
    second = client.generate_json(
        system="s", user="u", schema={}, purpose="classification"
    )
    assert second["name"] == "Mock Domain"
    assert client.calls == ["classification", "classification"]

    queries = client.generate_json(system="s", user="u", schema={}, purpose="queries")
    assert len(queries["queries"]) == 5
    assert {item["source"] for item in queries["queries"]} >= {"openalex", "huggingface"}

    factors = client.generate_json(system="s", user="u", schema={}, purpose="factors")
    assert len(factors["factors"]) == 5
    assert all(item["evidence_ids"] for item in factors["factors"])

    hypothesis = client.generate_json(
        system="s", user="u", schema={}, purpose="hypothesis:linear_model"
    )
    assert "def applies" in hypothesis["code"]
    assert "def run" in hypothesis["code"]

    critic = client.generate_json(system="s", user="u", schema={}, purpose="critic:hyp_1")
    assert critic["verdict"] == "pass"
