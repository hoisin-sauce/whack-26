"""Resource identity: canonical URLs and content hashes.

Deduplication is intentionally conservative. The same content at a different
URL should merge; two different resources with similar titles should not.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "ref",
        "referrer",
        "from",
        "source",
        "via",
        "si",
        "s",
        "t",
        "start",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
    }
)

_KEEP_QUERY_PARAMS = frozenset(
    {
        "id",
        "ids",
        "doi",
        "q",
        "query",
        "search",
        "page",
        "p",
        "v",
        "isbn",
        "issn",
        "pmid",
        "arxiv",
    }
)

_WHITESPACE = re.compile(r"\s+")


def canonicalize_url(url: str) -> str:
    """Return a stable identity URL.

    Removes fragments, tracking/access parameters, default ports, and a
    trailing slash. Identity-bearing query parameters are kept and sorted.
    Raises ``ValueError`` for a non-http(s) or hostless URL.
    """
    raw = (url or "").strip()
    if not raw:
        raise ValueError("url must not be empty")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {parts.scheme!r}")
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("url must include a host")
    port = parts.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"

    kept: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered in TRACKING_PARAMS:
            continue
        if lowered in _KEEP_QUERY_PARAMS or not lowered.startswith("utm_"):
            kept.append((key, value))
    kept.sort()

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, urlencode(kept, doseq=True), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode("utf-8")).hexdigest()


def content_sha256(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def content_sha512(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha512(payload).hexdigest()


def metadata_identity(
    *,
    title: str,
    abstract: str | None = None,
    authors: list[str] | tuple[str, ...] = (),
    year: int | None = None,
    doi: str | None = None,
    canonical_url: str | None = None,
) -> str:
    """Hash the stable metadata when raw bytes are unavailable."""
    payload = {
        "title": _WHITESPACE.sub(" ", (title or "").strip().lower()),
        "abstract": _WHITESPACE.sub(" ", (abstract or "").strip().lower()),
        "authors": [_WHITESPACE.sub(" ", str(author).strip().lower()) for author in authors],
        "year": year,
        "doi": (doi or "").strip().lower() or None,
        "url": canonical_url,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha512(encoded).hexdigest()


def title_identity(title: str, authors: list[str] | tuple[str, ...] = (), year: int | None = None) -> str:
    normalized = _WHITESPACE.sub(" ", (title or "").strip().lower())
    first_author = (
        _WHITESPACE.sub(" ", str(authors[0]).strip().lower()) if authors else ""
    )
    payload = f"{normalized}|{first_author}|{year if year is not None else ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
