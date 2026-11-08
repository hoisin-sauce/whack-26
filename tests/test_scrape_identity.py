from __future__ import annotations

import hashlib

import pytest

from whack26.research.scrape.identity import (
    canonicalize_url,
    content_sha256,
    content_sha512,
    metadata_identity,
    title_identity,
    url_hash,
)


def test_canonicalize_url_removes_tracking_and_sorts_query() -> None:
    result = canonicalize_url(
        "HTTPS://Example.COM:443/path/?utm_source=x&b=2&a=1&from=feed#section"
    )

    assert result == "https://example.com/path?a=1&b=2"


def test_canonicalize_url_keeps_identity_parameters() -> None:
    result = canonicalize_url(
        "https://example.test/record?id=42&doi=10.1000%2FABC&ref=home"
    )

    assert "id=42" in result
    assert "doi=10.1000%2FABC" in result
    assert "ref=" not in result


def test_canonicalize_url_strips_trailing_slash_but_keeps_root() -> None:
    assert canonicalize_url("https://example.test/a/") == "https://example.test/a"
    assert canonicalize_url("https://example.test/") == "https://example.test/"


def test_canonicalize_url_rejects_unsupported_schemes() -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        canonicalize_url("ftp://example.test/file")
    with pytest.raises(ValueError, match="host"):
        canonicalize_url("https:///path")


def test_url_hash_is_stable_across_tracking_variants() -> None:
    first = url_hash("https://example.test/a?utm_source=x&id=1")
    second = url_hash("https://EXAMPLE.test/a/?id=1&ref=feed")

    assert first == second
    assert len(first) == 64


def test_content_hashes_match_hashlib() -> None:
    payload = "hello world"

    assert content_sha256(payload) == hashlib.sha256(payload.encode()).hexdigest()
    assert content_sha512(payload) == hashlib.sha512(payload.encode()).hexdigest()
    assert len(content_sha512(payload)) == 128


def test_metadata_identity_is_stable_and_sensitive_to_title() -> None:
    first = metadata_identity(
        title="A Study",
        abstract="An abstract.",
        authors=["A. Author"],
        year=2024,
        doi="10.1000/ABC",
    )
    second = metadata_identity(
        title="  a   study ",
        abstract="an abstract.",
        authors=["a. author"],
        year=2024,
        doi="10.1000/abc",
    )
    different = metadata_identity(title="Another Study", year=2024)

    assert first == second
    assert first != different


def test_title_identity_uses_first_author_and_year() -> None:
    first = title_identity("A Study", ["A. Author", "B. Author"], 2024)
    second = title_identity("a study", ["a. author"], 2024)

    assert first == second
    assert first != title_identity("A Study", ["B. Author"], 2024)
