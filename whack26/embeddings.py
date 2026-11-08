"""Embedding clients and vector helpers."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable, Protocol

import httpx

from .common import redact_secrets
from .config import Settings

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = (1.0, 4.0)
MAX_BATCH = 50


class EmbeddingError(RuntimeError):
    """A provider-neutral embedding failure."""


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...


def l2_normalize(values: list[float]) -> list[float]:
    """Return a unit-length copy. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return list(values)
    return [value / norm for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity with a safe zero-vector result."""
    if len(left) != len(right):
        raise ValueError(f"vector length mismatch: {len(left)} != {len(right)}")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _hash_vector(text: str, dims: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dims:
        digest = hashlib.sha256(f"{text}\x00{counter}".encode("utf-8")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return l2_normalize(values[:dims])


class MockEmbeddingClient:
    """Deterministic hash embeddings; plumbing only, not semantic quality."""

    def __init__(self, dims: int = 768) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_hash_vector(text, self.dims) for text in texts]


class GeminiEmbeddingClient:
    """Native Gemini embeddings with batch splitting and bounded retries."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-embedding-2",
        dims: int = 768,
        base_url: str = GEMINI_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
        batch_size: int = MAX_BATCH,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        if dims <= 0:
            raise ValueError("dims must be positive")
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH}")
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiEmbeddingClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        attempts = self.max_retries + 1
        last_error = "unknown error"
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise EmbeddingError(
                    redact_secrets(f"Gemini embedding request failed: {last_error}")
                ) from exc

            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise EmbeddingError(
                    redact_secrets(f"Gemini embedding request failed: {last_error}")
                )

            if response.status_code >= 400:
                raise EmbeddingError(
                    redact_secrets(
                        f"Gemini embedding HTTP {response.status_code}: {response.text[:500]}"
                    )
                )

            try:
                body = response.json()
            except ValueError as exc:
                last_error = f"invalid response JSON: {exc}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise EmbeddingError(
                    redact_secrets(f"Gemini embedding request failed: {last_error}")
                ) from exc
            if not isinstance(body, dict):
                raise EmbeddingError("Gemini embedding returned a non-object response")
            return body

        raise EmbeddingError(
            redact_secrets(f"Gemini embedding request failed: {last_error}")
        )

    @staticmethod
    def _values(entry: dict[str, Any]) -> list[float]:
        values = entry.get("values")
        if values is None and isinstance(entry.get("embedding"), dict):
            values = entry["embedding"].get("values")
        if not isinstance(values, list):
            raise EmbeddingError("Gemini embedding response has no values array")
        return [float(value) for value in values]

    def _embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        url = f"{self.base_url}/models/{model}:batchEmbedContents"
        payload = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": text or " "}]},
                    "outputDimensionality": self.dims,
                }
                for text in texts
            ]
        }
        body = self._post(url, payload)
        entries = body.get("embeddings")
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise EmbeddingError(
                f"Gemini embedding batch returned {len(entries or [])} vectors for {len(texts)} inputs"
            )
        vectors = [self._checked_vector(self._values(entry)) for entry in entries]
        return vectors

    def _embed_one(self, text: str, model: str) -> list[float]:
        url = f"{self.base_url}/models/{model}:embedContent"
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text or " "}]},
            "outputDimensionality": self.dims,
        }
        body = self._post(url, payload)
        entry = body.get("embedding")
        if not isinstance(entry, dict):
            entry = body
        return self._checked_vector(self._values(entry))

    def _checked_vector(self, values: list[float]) -> list[float]:
        if len(values) != self.dims:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {self.dims}, got {len(values)}"
            )
        return l2_normalize(values)

    def _embed_batch_with_fallback(self, texts: list[str], model: str) -> list[list[float]]:
        if len(texts) == 1:
            return [self._embed_one(texts[0], model)]
        try:
            return self._embed_batch(texts, model)
        except EmbeddingError:
            midpoint = len(texts) // 2
            return self._embed_batch_with_fallback(
                texts[:midpoint], model
            ) + self._embed_batch_with_fallback(texts[midpoint:], model)

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if not texts:
            return []
        selected_model = model or self.model
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch_with_fallback(batch, selected_model))
        return vectors


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embed_provider == "mock":
        return MockEmbeddingClient(settings.embed_dims)
    if settings.embed_provider == "gemini":
        return GeminiEmbeddingClient(
            settings.gemini_api_key or "",
            model=settings.gemini_embed_model,
            dims=settings.embed_dims,
            timeout=settings.embed_timeout,
        )
    if settings.embed_provider == "local":
        raise EmbeddingError(
            "local embeddings are configured but model2vec is not installed yet; "
            "use WHACK26_EMBED_PROVIDER=gemini or mock"
        )
    raise EmbeddingError(f"unknown embedding provider {settings.embed_provider!r}")
