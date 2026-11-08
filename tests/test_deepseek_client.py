from __future__ import annotations

import json

import httpx
import pytest

from whack26.config import Settings
from whack26.llm import DeepSeekLLMClient, LLMError, build_llm_client


def _chat_response(content: str, *, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"completion_tokens": 12},
        },
    )


def test_deepseek_sends_json_chat_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return _chat_response('{"ok": true}')

    client = DeepSeekLLMClient(
        "test-key",
        model="test-model",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        result = client.generate_json(
            system="system prompt",
            user="user prompt",
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            max_tokens=64,
            purpose="doctor",
        )
    finally:
        client.close()

    assert result == {"ok": True}
    assert captured["url"].endswith("/chat/completions")
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "test-model"
    assert body["messages"][0]["role"] == "system"
    assert "system prompt" in body["messages"][0]["content"]
    assert "JSON Schema" in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "user prompt"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0


def test_deepseek_drops_thinking_when_model_rejects_it() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "thinking" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "thinking is not supported"}},
            )
        return _chat_response('{"ok": true}')

    client = DeepSeekLLMClient(
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
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert "thinking" not in calls[1]
    assert client.thinking_supported is False


def test_deepseek_retries_retryable_status() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return _chat_response('{"ok": true}')

    client = DeepSeekLLMClient(
        "test-key",
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    try:
        assert client.generate_json(system="", user="u", schema={}) == {"ok": True}
    finally:
        client.close()
    assert calls["count"] == 2


def test_deepseek_raises_after_empty_content() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return _chat_response("", finish_reason="length")

    client = DeepSeekLLMClient(
        "test-key",
        max_retries=2,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(LLMError, match="empty text"):
        client.generate_json(system="", user="u", schema={}, purpose="classification")
    client.close()
    assert calls["count"] == 3


def test_deepseek_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekLLMClient("")


def test_build_llm_client_selects_deepseek() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
        }
    )
    client = build_llm_client(settings)
    try:
        assert isinstance(client, DeepSeekLLMClient)
        assert client.model == "deepseek-flash"
    finally:
        client.close()
