"""General web-search adapter with a graceful no-key fallback.

Supported providers, selected with ``WEBSEARCH_PROVIDER``:

- ``google``: Google Programmable Search JSON API. Requires
  ``WEBSEARCH_API_KEY`` and ``WEBSEARCH_CX`` (search-engine id).
- ``brave``: Brave Web Search API. Requires ``WEBSEARCH_API_KEY``.
- ``none`` (default): the source is skipped.

A missing provider, key, or engine id is a configuration gap, not a failure:
``search`` records a warning and returns an empty list. Transport, HTTP, parse,
and quota failures raise :class:`SourceError` so the research pipeline can skip
this source and keep the run alive.

Result records use ``source="websearch"`` and deduplicate within one response by
the canonical URL from :func:`whack26.research.scrape.identity.canonicalize_url`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    cached_status_is_success,
    normalize_query,
    raise_for_status,
    request_with_retries,
)
from .cache import ResearchCache, cache_key
from .scrape.identity import canonicalize_url

logger = logging.getLogger(__name__)

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"google", "brave"})
DISABLED_PROVIDER = "none"

# Provider-side result caps: Google CSE returns at most 10 per request, Brave
# accepts up to 20 web results per request.
GOOGLE_MAX_RESULTS = 10
BRAVE_MAX_RESULTS = 20

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_GOOGLE_DATE_KEYS: tuple[str, ...] = (
    "article:published_time",
    "article:modified_time",
    "datepublished",
    "datecreated",
    "date",
    "pubdate",
    "dc.date",
    "og:updated_time",
)
_RATE_LIMIT_STATUS: frozenset[int] = frozenset({403, 429})


def _year_from_value(value: Any) -> int | None:
    """Extract a plausible publication year from an explicit date field."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 1900 <= value <= 2100 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    prefix = text[:4]
    if prefix.isdigit() and 1900 <= int(prefix) <= 2100:
        return int(prefix)
    match = _YEAR_RE.search(text)
    return int(match.group(1)) if match else None


def _google_year(item: Mapping[str, Any]) -> int | None:
    pagemap = item.get("pagemap")
    if not isinstance(pagemap, dict):
        return None
    metatags = pagemap.get("metatags")
    if not isinstance(metatags, list):
        return None
    for entry in metatags:
        if not isinstance(entry, dict):
            continue
        lowered = {str(key).lower(): value for key, value in entry.items()}
        for key in _GOOGLE_DATE_KEYS:
            year = _year_from_value(lowered.get(key))
            if year is not None:
                return year
    return None


class WebSearchAdapter(BaseHttpAdapter):
    """Search Google Programmable Search or Brave, skipping when unconfigured."""

    name = "websearch"
    ttl_seconds = 24 * 60 * 60
    # Declared pacing hint for callers that throttle sources; the adapter does
    # not sleep on its own because the search pipeline runs sources serially.
    rate_limit_seconds = 1.0

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        cx: str | None = None,
        env: Mapping[str, str] | None = None,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        source = os.environ if env is None else env
        raw_provider = (
            provider if provider is not None else source.get("WEBSEARCH_PROVIDER")
        )
        self.provider = (raw_provider or DISABLED_PROVIDER).strip().lower()
        raw_key = api_key if api_key is not None else source.get("WEBSEARCH_API_KEY")
        self.api_key = raw_key.strip() if isinstance(raw_key, str) else None
        raw_cx = cx if cx is not None else source.get("WEBSEARCH_CX")
        self.cx = raw_cx.strip() if isinstance(raw_cx, str) else None
        self.sleep = sleep
        self.warnings: list[str] = []
        # A named provider that cannot run is a degraded configuration; a
        # provider explicitly set to "none" is a normal, expected skip.
        if self.provider in SUPPORTED_PROVIDERS and self.disabled_reason:
            self.limited_mode = True

    @property
    def disabled_reason(self) -> str | None:
        if self.provider == DISABLED_PROVIDER:
            return "WEBSEARCH_PROVIDER is not set to google or brave"
        if self.provider not in SUPPORTED_PROVIDERS:
            return f"unsupported WEBSEARCH_PROVIDER {self.provider!r}"
        if not self.api_key:
            return "WEBSEARCH_API_KEY is not set"
        if self.provider == "google" and not self.cx:
            return "WEBSEARCH_CX is not set"
        return None

    @property
    def enabled(self) -> bool:
        return self.disabled_reason is None

    def _warn(self, reason: str) -> None:
        message = f"source_warning: websearch {reason}"
        if message in self.warnings:
            return
        self.warnings.append(message)
        logger.warning("%s", message)

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0:
            return []
        normalized = normalize_query(query)
        if not normalized:
            return []
        reason = self.disabled_reason
        if reason is not None:
            self._warn(f"skipped: {reason}")
            return []

        key = self._cache_key(normalized, limit)
        if key is not None and self.cache is not None:
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
                            normalized,
                            message,
                            status_code=cached.status,
                        )
                    logger.warning(
                        "%s query %r: %s; refetching online",
                        self.name,
                        normalized,
                        message,
                    )
                else:
                    try:
                        return self.parse(
                            cached.body,
                            query=normalized,
                            retrieved_at=cached.fetched_at,
                        )[:limit]
                    except Exception as exc:  # noqa: BLE001 - cache may be corrupt
                        if self.offline:
                            raise SourceError(
                                self.name,
                                normalized,
                                f"cached response could not be parsed: {exc}",
                            ) from exc
                        logger.warning(
                            "%s query %r: cached response could not be parsed "
                            "(%s); refetching online",
                            self.name,
                            normalized,
                            exc,
                        )

        if self.offline:
            raise SourceError(
                self.name, normalized, "offline cache miss; source skipped"
            )

        result = self.fetch(normalized, limit, client)
        if key is not None and self.cache is not None:
            self.cache.put(
                key,
                dataset_id=self.dataset_id,
                source=self.source_key,
                query=normalized,
                status=result.status,
                body=result.body,
                ttl_seconds=self.ttl_seconds,
                headers=result.headers,
            )
        try:
            return self.parse(result.body, query=normalized)[:limit]
        except SourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed upstream payload
            raise SourceError(
                self.name, normalized, f"could not parse response: {exc}"
            ) from exc

    def _cache_key(self, query: str, limit: int) -> str | None:
        if self.cache is None or not self.dataset_id:
            return None
        return cache_key(
            self.dataset_id,
            self.source_key,
            query,
            {"provider": self.provider, "limit": int(limit)},
        )

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        try:
            if self.provider == "google":
                return self._fetch_google(query, limit, client)
            return self._fetch_brave(query, limit, client)
        except SourceError as exc:
            if exc.status_code in _RATE_LIMIT_STATUS:
                self.limited_mode = True
                self._warn(
                    f"request rejected with HTTP {exc.status_code}; "
                    "quota or credentials may be exhausted"
                )
            raise

    def _fetch_google(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        response = request_with_retries(
            client,
            GOOGLE_ENDPOINT,
            params={
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": min(int(limit), GOOGLE_MAX_RESULTS),
            },
            headers=self._headers(),
            source=self.name,
            query=query,
            sleep=self.sleep,
        )
        raise_for_status(response, source=self.name, query=query)
        # Google can return HTTP 200 with an error object for an exhausted
        # daily quota; reject it here so it is never cached as success.
        self._raise_provider_error(response.text, query)
        return FetchResult(
            body=response.text,
            status=response.status_code,
            headers=self._header_subset(response),
        )

    def _fetch_brave(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        headers = self._headers()
        headers["X-Subscription-Token"] = self.api_key or ""
        response = request_with_retries(
            client,
            BRAVE_ENDPOINT,
            params={
                "q": query,
                "count": min(int(limit), BRAVE_MAX_RESULTS),
            },
            headers=headers,
            source=self.name,
            query=query,
            sleep=self.sleep,
        )
        raise_for_status(response, source=self.name, query=query)
        self._raise_provider_error(response.text, query)
        return FetchResult(
            body=response.text,
            status=response.status_code,
            headers=self._header_subset(response),
        )

    def _raise_provider_error(self, body: str, query: str) -> None:
        try:
            payload = json.loads(body)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        error = payload.get("error")
        if not isinstance(error, dict):
            return
        status = error.get("code")
        status_code = status if isinstance(status, int) else None
        message = error.get("message") or "provider reported an error"
        if status_code in _RATE_LIMIT_STATUS or "quota" in str(message).lower():
            self.limited_mode = True
            self._warn(f"quota or rate limit exhausted: {message}")
        raise SourceError(
            self.name,
            query,
            f"provider error: {message}",
            status_code=status_code,
        )

    def parse(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        if self.provider == "google":
            return self._parse_google(body, query=query, retrieved_at=retrieved_at)
        if self.provider == "brave":
            return self._parse_brave(body, query=query, retrieved_at=retrieved_at)
        raise SourceError(
            self.name, query, f"provider {self.provider!r} is not configured"
        )

    def _parse_google(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        payload = self._json(body)
        if not isinstance(payload, dict):
            raise SourceError(self.name, query, "expected a JSON object")
        self._raise_provider_error(body, query)
        items = payload.get("items")
        if items is None:
            return []
        if not isinstance(items, list):
            raise SourceError(self.name, query, "items is not a list")

        records: list[ResearchRecord] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not isinstance(link, str) or not link.strip():
                continue
            try:
                canonical = canonicalize_url(link)
            except ValueError:
                continue
            if canonical in seen:
                continue
            title = item.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            snippet = item.get("snippet")
            try:
                record = self._record(
                    source_id=canonical,
                    title=title,
                    url=canonical,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=snippet if isinstance(snippet, str) else None,
                    year=_google_year(item),
                )
            except ValueError:
                continue
            seen.add(canonical)
            records.append(record)
        return records

    def _parse_brave(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        payload = self._json(body)
        if not isinstance(payload, dict):
            raise SourceError(self.name, query, "expected a JSON object")
        self._raise_provider_error(body, query)
        web = payload.get("web")
        if web is None:
            return []
        if not isinstance(web, dict):
            raise SourceError(self.name, query, "web is not an object")
        results = web.get("results")
        if results is None:
            return []
        if not isinstance(results, list):
            raise SourceError(self.name, query, "web.results is not a list")

        records: list[ResearchRecord] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            try:
                canonical = canonicalize_url(url)
            except ValueError:
                continue
            if canonical in seen:
                continue
            title = item.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            description = item.get("description")
            try:
                record = self._record(
                    source_id=canonical,
                    title=title,
                    url=canonical,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=(
                        description if isinstance(description, str) else None
                    ),
                    year=_year_from_value(item.get("page_age")),
                )
            except ValueError:
                continue
            seen.add(canonical)
            records.append(record)
        return records
