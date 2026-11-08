"""Local (model2vec) embedding provider tests."""

from __future__ import annotations

import math
import sys
import types

import pytest

from whack26.config import Settings
from whack26.embeddings import EmbeddingError, build_embedding_client
from whack26.embeddings import LocalEmbeddingClient


class FakeStaticModel:
    loads: list[str] = []
    dim = 3

    @classmethod
    def from_pretrained(cls, name: str):
        cls.loads.append(name)
        return cls()

    def encode(self, texts: list[str]):
        return [[1.0, 0.0, 0.0] for _ in texts]


def install_model2vec(monkeypatch, model_cls=FakeStaticModel) -> None:
    module = types.SimpleNamespace(StaticModel=model_cls)
    monkeypatch.setitem(sys.modules, "model2vec", module)


def test_build_embedding_client_selects_local(monkeypatch) -> None:
    FakeStaticModel.loads = []
    install_model2vec(monkeypatch)
    settings = Settings(
        embed_provider="local",
        embed_dims=3,
        local_embed_model="fake-model",
    )

    client = build_embedding_client(settings)
    vectors = client.embed(["alpha", "beta"])

    assert isinstance(client, LocalEmbeddingClient)
    assert len(vectors) == 2
    assert all(len(vector) == 3 for vector in vectors)
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
        for vector in vectors
    )
    assert FakeStaticModel.loads == ["fake-model"]

    client.embed(["gamma"])
    assert FakeStaticModel.loads == ["fake-model"]


def test_local_embeddings_empty_input_does_not_load(monkeypatch) -> None:
    FakeStaticModel.loads = []
    install_model2vec(monkeypatch)
    client = LocalEmbeddingClient(model="fake-model", dims=3)

    assert client.embed([]) == []
    assert FakeStaticModel.loads == []


def test_local_embeddings_dimension_mismatch_is_clear(monkeypatch) -> None:
    class TwoDimModel(FakeStaticModel):
        dim = 2

    install_model2vec(monkeypatch, TwoDimModel)
    client = LocalEmbeddingClient(model="fake-model", dims=3)

    with pytest.raises(EmbeddingError, match="set EMBED_DIMS=2"):
        client.embed(["alpha"])


def test_local_embeddings_missing_package_is_clear(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "model2vec", None)
    client = LocalEmbeddingClient(model="fake-model", dims=3)

    with pytest.raises(EmbeddingError, match="requirements-local-embeddings"):
        client.embed(["alpha"])


def test_local_embeddings_model_load_failure_is_wrapped(monkeypatch) -> None:
    class BrokenModel(FakeStaticModel):
        @classmethod
        def from_pretrained(cls, name: str):
            raise RuntimeError("no such model")

    install_model2vec(monkeypatch, BrokenModel)
    client = LocalEmbeddingClient(model="missing-model", dims=3)

    with pytest.raises(EmbeddingError, match="could not load local embedding model"):
        client.embed(["alpha"])


def test_local_embeddings_encode_failure_is_wrapped(monkeypatch) -> None:
    class FailingEncodeModel(FakeStaticModel):
        def encode(self, texts: list[str]):
            raise ValueError("encode failed")

    install_model2vec(monkeypatch, FailingEncodeModel)
    client = LocalEmbeddingClient(model="fake-model", dims=3)

    with pytest.raises(EmbeddingError, match="local embedding failed"):
        client.embed(["alpha"])


def test_local_embedding_settings_parse_from_env() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_EMBED_PROVIDER": "local",
            "LOCAL_EMBED_MODEL": "fake-model",
            "EMBED_DIMS": "3",
        }
    )

    assert settings.embed_provider == "local"
    assert settings.local_embed_model == "fake-model"
    assert settings.embed_dims == 3
