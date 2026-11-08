"""Wikipedia search and summary adapter (encyclopedic fallback)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    cached_status_is_success,
    raise_for_status,
    request_with_retries,
)
from .cache import cache_key

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


class WikipediaAdapter(BaseHttpAdapter):
    """Two-step Wikipedia search: find titles, then fetch page summaries."""

    name = "wikipedia"
    ttl_seconds = 7 * 24 * 60 * 60
    is_fallback = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.warnings: list[str] = []

    def _warn(self, title: str, reason: str) -> None:
        message = f"source_warning: wikipedia summary {title!r}: {reason}"
        self.warnings.append(message)
        logger.warning("%s", message)

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0:
            return []
        pages = self._search_pages(query, limit, client)
        records: list[ResearchRecord] = []
        for title, pageid in pages[:limit]:
            record = self._summary_record(title, pageid, query, client)
            if record is not None:
                records.append(record)
        return records

    def _cache_key(
        self, query: str, params: dict[str, object]
    ) -> str | None:
        if self.cache is None or not self.dataset_id:
            return None
        return cache_key(self.dataset_id, self.source_key, query, params)

    def _search_pages(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[tuple[str, Any]]:
        key = self._cache_key(
            query, {"step": "search", "limit": int(limit)}
        )
        if key is not None and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                if not cached_status_is_success(cached.status):
                    if self.offline:
                        raise SourceError(
                            self.name,
                            query,
                            f"cached search response has HTTP status "
                            f"{cached.status}; refusing to parse it",
                            status_code=cached.status,
                        )
                    logger.warning(
                        "%s query %r: cached search response has HTTP status "
                        "%s; refetching online",
                        self.name,
                        query,
                        cached.status,
                    )
                else:
                    try:
                        return self._parse_search(cached.body, query)
                    except Exception as exc:  # noqa: BLE001 - cache may be truncated
                        if self.offline:
                            raise SourceError(
                                self.name,
                                query,
                                f"cached response could not be parsed: {exc}",
                            ) from exc
                        logger.warning(
                            "%s query %r: cached search response could not be "
                            "parsed (%s); refetching online",
                            self.name,
                            query,
                            exc,
                        )
        if self.offline:
            raise SourceError(
                self.name, query, "offline cache miss; source skipped"
            )
        result = self._fetch_search(query, limit, client)
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
            return self._parse_search(result.body, query)
        except SourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - malformed upstream payload
            raise SourceError(
                self.name, query, f"could not parse response: {exc}"
            ) from exc

    def _fetch_search(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        response = request_with_retries(
            client,
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": limit,
            },
            headers=self._headers(),
            source=self.name,
            query=query,
        )
        raise_for_status(response, source=self.name, query=query)
        return FetchResult(
            body=response.text,
            status=response.status_code,
            headers=self._header_subset(response),
        )

    def _summary_record(
        self, title: str, pageid: Any, query: str, client: httpx.Client
    ) -> ResearchRecord | None:
        key = self._cache_key(query, {"step": "summary", "title": title})
        if key is not None and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                if not cached_status_is_success(cached.status):
                    if self.offline:
                        self._warn(
                            title,
                            f"cached summary has HTTP status {cached.status}; "
                            "skipped",
                        )
                        return None
                    logger.warning(
                        "%s summary %r: cached response has HTTP status %s; "
                        "refetching online",
                        self.name,
                        title,
                        cached.status,
                    )
                else:
                    try:
                        return self._parse_summary(
                            cached.body,
                            query=query,
                            fallback_title=title,
                            pageid=pageid,
                            retrieved_at=cached.fetched_at,
                        )
                    except Exception as exc:  # noqa: BLE001 - skip an unusable summary
                        if self.offline:
                            self._warn(
                                title,
                                f"cached summary could not be parsed ({exc}); "
                                "skipped",
                            )
                            return None
                        logger.warning(
                            "%s summary %r: cached response could not be parsed "
                            "(%s); refetching online",
                            self.name,
                            title,
                            exc,
                        )
        if self.offline:
            self._warn(title, "offline cache miss; skipped")
            return None
        response = request_with_retries(
            client,
            WIKIPEDIA_SUMMARY_URL + quote(title, safe=""),
            headers=self._headers(),
            source=self.name,
            query=query,
        )
        if response.status_code == 404:
            return None
        raise_for_status(response, source=self.name, query=query)
        if key is not None and self.cache is not None:
            self.cache.put(
                key,
                dataset_id=self.dataset_id,
                source=self.source_key,
                query=query,
                status=response.status_code,
                body=response.text,
                ttl_seconds=self.ttl_seconds,
                headers=self._header_subset(response),
            )
        try:
            return self._parse_summary(
                response.text,
                query=query,
                fallback_title=title,
                pageid=pageid,
            )
        except Exception:  # noqa: BLE001 - skip an unusable summary
            return None

    def _parse_search(self, body: str, query: str) -> list[tuple[str, Any]]:
        payload = self._json(body)
        if not isinstance(payload, dict):
            raise ValueError("search response is not an object")
        query_block = payload.get("query")
        if not isinstance(query_block, dict):
            raise ValueError("search response is missing query")
        entries = query_block.get("search")
        if not isinstance(entries, list):
            raise ValueError("search response is missing query.search")
        pages: list[tuple[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            pages.append((title, entry.get("pageid")))
        return pages

    def _parse_summary(
        self,
        body: str,
        *,
        query: str,
        fallback_title: str,
        pageid: Any,
        retrieved_at: str | None = None,
    ) -> ResearchRecord:
        payload = self._json(body)
        if not isinstance(payload, dict):
            raise ValueError("summary response is not an object")
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            title = fallback_title
        summary_pageid = payload.get("pageid")
        if summary_pageid in (None, ""):
            summary_pageid = pageid
        source_id = (
            str(summary_pageid)
            if summary_pageid not in (None, "")
            else title
        )
        url = None
        content_urls = payload.get("content_urls")
        if isinstance(content_urls, dict):
            desktop = content_urls.get("desktop")
            if isinstance(desktop, dict):
                candidate = desktop.get("page")
                if isinstance(candidate, str) and candidate:
                    url = candidate
        if not url:
            url = f"https://en.wikipedia.org/wiki/{quote(title, safe='')}"
        extract = payload.get("extract")
        return self._record(
            source_id=source_id,
            title=title,
            url=url,
            query=query,
            retrieved_at=retrieved_at,
            abstract=extract if isinstance(extract, str) else None,
            authors=[],
            year=_year_from(payload.get("timestamp")),
            doi=None,
            license=WIKIPEDIA_LICENSE,
        )
