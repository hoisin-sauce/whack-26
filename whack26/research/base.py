"""Shared research types, text helpers, and cached HTTP plumbing."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import httpx
from pydantic import BaseModel, Field

from ..common import redact_secrets
from ..common import utc_now_iso
from .cache import CachedResponse, ResearchCache, cache_key

SourceName = Literal["openalex", "arxiv", "huggingface", "zenodo", "wikipedia"]

PAPER_SOURCES: frozenset[str] = frozenset({"openalex", "arxiv"})
DATASET_SOURCES: frozenset[str] = frozenset({"huggingface", "zenodo"})

SOURCE_PRIOR: dict[str, float] = {
    "openalex": 0.8,
    "arxiv": 0.7,
    "zenodo": 0.7,
    "huggingface": 0.6,
    "wikipedia": 0.5,
}

# Keep the best record per type, not per source.
SOURCE_RANK: dict[str, int] = {
    "openalex": 5,
    "arxiv": 4,
    "zenodo": 3,
    "huggingface": 2,
    "wikipedia": 1,
}

RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0)
# Wikimedia rejects user agents without contact information (HTTP 403).
DEFAULT_USER_AGENT = "whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)"
MAX_ABSTRACT_CHARS = 4000

_TAG_RE = re.compile(r"<[^>]+>")

logger = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """A recoverable failure from one research source."""

    def __init__(
        self,
        source: str,
        query: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{source} query {query!r} failed: {redact_secrets(message)}")
        self.source = source
        self.query = query
        self.status_code = status_code


class ResearchRecord(BaseModel):
    id: str
    source: SourceName
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str
    license: str | None = None
    retrieved_at: str
    query: str
    relevance: float = 0.0
    also_seen_in: list[str] = Field(default_factory=list)


class SourceAdapter(Protocol):
    name: str
    ttl_seconds: int

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]: ...


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    return html.unescape(_TAG_RE.sub(" ", value))


def collapse_whitespace(value: str | None) -> str:
    return " ".join((value or "").split())


def clean_text(value: str | None) -> str:
    return collapse_whitespace(strip_html(value))


def excerpt(value: str | None, limit: int = MAX_ABSTRACT_CHARS) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def reconstruct_abstract(index: Any) -> str | None:
    """Rebuild OpenAlex's inverted abstract index into readable text."""
    if not isinstance(index, dict) or not index:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and not isinstance(position, bool):
                positioned.append((position, word))
    if not positioned:
        return None
    positioned.sort(key=lambda item: item[0])
    return excerpt(" ".join(word for _, word in positioned))


@dataclass
class FetchResult:
    body: str
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


def cached_status_is_success(status: int) -> bool:
    """Return True only for a 2xx cached HTTP status.

    A cached error body must never be parsed as a successful payload. Callers
    should refetch online or raise a SourceError offline.
    """
    return 200 <= int(status) < 300


def request_with_retries(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    source: str,
    query: str,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = 2,
) -> httpx.Response:
    """GET with bounded retries on transport errors and retryable statuses."""
    attempts = retries + 1
    last_error = "unknown error"
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                continue
            raise SourceError(source, query, last_error) from exc
        if response.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if attempt < retries:
                sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                continue
            raise SourceError(
                source, query, last_error, status_code=response.status_code
            )
        return response
    raise SourceError(source, query, last_error)


def raise_for_status(response: httpx.Response, *, source: str, query: str) -> None:
    if response.status_code >= 400:
        raise SourceError(
            source,
            query,
            f"HTTP {response.status_code}: {response.text[:300]}",
            status_code=response.status_code,
        )


class BaseHttpAdapter:
    """Base adapter with per-source TTL caching and offline handling."""

    name: str = "base"
    ttl_seconds: int = 24 * 60 * 60
    user_agent: str = DEFAULT_USER_AGENT

    def __init__(
        self,
        *,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        self.cache = cache
        self.dataset_id = dataset_id
        self.offline = offline
        self.limited_mode = False
        self.last_headers: dict[str, str] = {}

    @property
    def source_key(self) -> str:
        return self.name

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0:
            return []
        key = None
        if self.cache is not None and self.dataset_id:
            key = cache_key(
                self.dataset_id, self.source_key, query, {"limit": int(limit)}
            )
            cached = self.cache.get(key)
            if cached is not None:
                if not cached_status_is_success(cached.status):
                    message = (
                        f"cached response has HTTP status {cached.status}; "
                        "refusing to parse it"
                    )
                    if self.offline:
                        raise SourceError(
                            self.name,
                            query,
                            message,
                            status_code=cached.status,
                        )
                    logger.warning(
                        "%s query %r: %s; refetching online",
                        self.name,
                        query,
                        message,
                    )
                else:
                    try:
                        return self.parse(
                            cached.body, query=query, retrieved_at=cached.fetched_at
                        )
                    except Exception as exc:  # noqa: BLE001 - cache may be truncated
                        if self.offline:
                            raise SourceError(
                                self.name,
                                query,
                                f"cached response could not be parsed: {exc}",
                            ) from exc
                        logger.warning(
                            "%s query %r: cached response could not be parsed "
                            "(%s); refetching online",
                            self.name,
                            query,
                            exc,
                        )
        if self.offline:
            raise SourceError(
                self.name, query, "offline cache miss; source skipped"
            )
        result = self.fetch(query, limit, client)
        if key is not None and self.cache is not None:
            self.cache.put(
                key,
                dataset_id=self.dataset_id,
                source=self.source_key,
                query=query,
                status=result.status,
                body=result.body,
                ttl_seconds=self.ttl_seconds,
                headers=result.headers,
            )
        try:
            return self.parse(result.body, query=query)
        except SourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed upstream payload
            raise SourceError(
                self.name, query, f"could not parse response: {exc}"
            ) from exc

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        raise NotImplementedError

    def parse(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        raise NotImplementedError

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        headers.update(extra or {})
        return headers

    def _record(
        self,
        *,
        source_id: str,
        title: str,
        url: str,
        query: str,
        retrieved_at: str | None = None,
        abstract: str | None = None,
        authors: Sequence[str] = (),
        year: int | None = None,
        doi: str | None = None,
        license: str | None = None,
    ) -> ResearchRecord:
        clean_title = clean_text(title)
        if not clean_title:
            raise ValueError("research record is missing a title")
        return ResearchRecord(
            id=f"{self.name}:{source_id}",
            source=self.name,  # type: ignore[arg-type]
            source_id=source_id,
            title=clean_title,
            abstract=excerpt(abstract),
            authors=[clean_text(author) for author in authors if clean_text(author)],
            year=year,
            doi=doi.strip() if doi else None,
            url=url,
            license=license,
            retrieved_at=retrieved_at or utc_now_iso(),
            query=query,
        )

    @staticmethod
    def _json(body: str) -> Any:
        return json.loads(body)

    @staticmethod
    def _header_subset(response: httpx.Response) -> dict[str, str]:
        wanted = {
            "x-ratelimit-remaining",
            "x-ratelimit-limit",
            "retry-after",
            "x-openalex-cost-usd",
        }
        return {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in wanted
        }
