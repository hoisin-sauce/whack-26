from __future__ import annotations

import json

import httpx
import pytest

from whack26.embeddings import EmbeddingError, GeminiEmbeddingClient
from whack26.llm import GeminiLLMClient, LLMError


def _gemini_text_response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "candidates": [
                {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
            ]
        },
    )


def test_gemini_llm_sends_structured_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return _gemini_text_response('{"ok": true}')

    client = GeminiLLMClient(
        "test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        result = client.generate_json(
            system="system prompt",
            user="user prompt",
            schema={"type": "object"},
            max_tokens=64,
            purpose="doctor",
        )
    finally:
        client.close()

    assert result == {"ok": True}
    assert captured["url"].endswith("/models/test-model:generateContent")
    headers = captured["headers"]
    assert headers["x-goog-api-key"] == "test-key"
    body = captured["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "system prompt"
    assert body["contents"][0]["parts"][0]["text"] == "user prompt"
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == {"type": "object"}
    assert config["thinkingConfig"]["thinkingBudget"] == 0
    assert config["maxOutputTokens"] == 64


def test_gemini_llm_drops_thinking_config_when_model_rejects_it() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "status": "INVALID_ARGUMENT"}},
            )
        return _gemini_text_response('{"ok": true}')

    client = GeminiLLMClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        result = client.generate_json(system="", user="u", schema={})
    finally:
        client.close()

    assert result == {"ok": True}
    assert len(calls) == 2
    assert calls[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "thinkingConfig" not in calls[1]["generationConfig"]
    assert client.disable_thinking is False


def test_gemini_llm_retries_retryable_status() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return _gemini_text_response('{"ok": true}')

    client = GeminiLLMClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        assert client.generate_json(system="", user="u", schema={}) == {"ok": True}
    finally:
        client.close()
    assert calls["count"] == 2


def test_gemini_llm_raises_after_empty_content() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            200,
            json={"candidates": [{"content": {}, "finishReason": "MAX_TOKENS"}]},
        )

    client = GeminiLLMClient(
        "test-key",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(LLMError, match="empty text"):
        client.generate_json(system="", user="u", schema={}, purpose="classification")
    client.close()
    assert calls["count"] == 3


def test_gemini_embeddings_batch_and_normalize() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [3.0, 4.0]}, {"values": [0.0, 2.0]}]},
        )

    client = GeminiEmbeddingClient(
        "test-key",
        dims=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        vectors = client.embed(["a", "b"])
    finally:
        client.close()

    assert captured["url"].endswith(":batchEmbedContents")
    assert len(captured["body"]["requests"]) == 2
    assert captured["body"]["requests"][0]["outputDimensionality"] == 2
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_gemini_embeddings_split_failed_batch() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith(":batchEmbedContents"):
            body = json.loads(request.content)
            if len(body["requests"]) > 1:
                return httpx.Response(400, text="batch too large for this account")
        body = json.loads(request.content)
        text = body["content"]["parts"][0]["text"]
        value = 1.0 if text == "a" else 2.0
        return httpx.Response(200, json={"embedding": {"values": [value, 0.0]}})

    client = GeminiEmbeddingClient(
        "test-key",
        dims=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        vectors = client.embed(["a", "b"])
    finally:
        client.close()

    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert sum(url.endswith(":embedContent") for url in calls) == 2


def test_gemini_embeddings_dimension_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": {"values": [1.0]}})

    client = GeminiEmbeddingClient(
        "test-key",
        dims=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        client.embed(["a"])
    client.close()
