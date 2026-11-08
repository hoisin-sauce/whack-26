from __future__ import annotations

from whack26.common import redact_secrets


def test_redacts_authorization_headers() -> None:
    text = "Authorization: Bearer sk-secret-token"
    assert "sk-secret-token" not in redact_secrets(text)
    assert "[redacted]" in redact_secrets(text)


def test_redacts_google_api_key_header() -> None:
    text = "x-goog-api-key: AIzaSySecret"
    assert "AIzaSySecret" not in redact_secrets(text)


def test_redacts_query_parameter() -> None:
    text = "GET https://api.openalex.org/works?search=x&api_key=abc123&per_page=1"
    redacted = redact_secrets(text)
    assert "abc123" not in redacted
    assert "api_key=[redacted]" in redacted
