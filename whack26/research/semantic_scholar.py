"""Semantic Scholar Graph API adapter for paper metadata."""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from .base import (
    BACKOFF_SECONDS,
    RETRYABLE_STATUS,
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    raise_for_status,
)
from .cache import ResearchCache

SEMANTIC_SCHOLAR_API_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
SEMANTIC_SCHOLAR_FIELDS = (
    "paperId,title,abstract,year,authors,externalIds,url,openAccessPdf,"
    "publicationDate,venue,fieldsOfStudy"
)
SEMANTIC_SCHOLAR_API_KEY_HEADER = "x-api-key"
# Semantic Scholar asks callers to stay around one request per second. The
# limit is shared and heavily enforced for keyless traffic.
DEFAULT_MIN_INTERVAL_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    doi = value.strip()
    for prefix in _DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi or None


def _year_from(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and len(value) >= 4:
        prefix = value[:4]
        if prefix.isdigit():
            return int(prefix)
    return None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


def _doi_from_external_ids(external_ids: Any) -> str | None:
    if not isinstance(external_ids, dict):
        return None
    for key, value in external_ids.items():
        if isinstance(key, str) and key.strip().lower() == "doi":
            return _normalize_doi(value)
    return None


def _author_names(authors: Any) -> list[str]:
    if not isinstance(authors, list):
        return []
    names: list[str] = []
    for author in authors:
        if not isinstance(author, dict):
            continue
        name = _optional_str(author.get("name"))
        if name:
            names.append(name)
    return names


class SemanticScholarAdapter(BaseHttpAdapter):
    """Search Semantic Scholar papers with optional API key and throttling."""

    name = "semantic_scholar"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        self.api_key = api_key
        self.min_interval = max(0.0, float(min_interval))
        self.retries = max(0, int(retries))
        self.sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self.limited_mode = not bool(api_key)
        self.rate_limited = False
        self.last_retry_after: float | None = None

    def _throttle(self) -> None:
        """Space out requests to stay inside Semantic Scholar's rate limit."""
        if self.min_interval <= 0:
            self._last_request_at = self._clock()
            return
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._last_request_at + self.min_interval - now
            if wait > 0:
                self.sleep(wait)
        self._last_request_at = self._clock()

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        headers = self._headers()
        if self.api_key:
            headers[SEMANTIC_SCHOLAR_API_KEY_HEADER] = self.api_key
        params = {
            "query": query,
            "limit": limit,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        attempts = self.retries + 1
        last_error = "unknown error"
        for attempt in range(attempts):
            self._throttle()
            try:
                response = client.get(
                    SEMANTIC_SCHOLAR_API_URL,
                    params=params,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    self.sleep(
                        BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                    )
                    continue
                raise SourceError(self.name, query, last_error) from exc

            if response.status_code == 429:
                self.rate_limited = True
                retry_after = _retry_after_seconds(response)
                self.last_retry_after = retry_after
                last_error = f"HTTP 429: {response.text[:300]}"
                if attempt < attempts - 1:
                    self.sleep(
                        retry_after
                        if retry_after is not None
                        else BACKOFF_SECONDS[
                            min(attempt, len(BACKOFF_SECONDS) - 1)
                        ]
                    )
                    continue
                raise SourceError(
                    self.name,
                    query,
                    last_error,
                    status_code=response.status_code,
                )
            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if attempt < attempts - 1:
                    self.sleep(
                        BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                    )
                    continue
                raise SourceError(
                    self.name,
                    query,
                    last_error,
                    status_code=response.status_code,
                )
            raise_for_status(response, source=self.name, query=query)
            return FetchResult(
                body=response.text,
                status=response.status_code,
                headers=self._header_subset(response),
            )
        raise SourceError(self.name, query, last_error)

    def parse(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        try:
            payload = self._json(body)
        except ValueError as exc:
            raise SourceError(
                self.name, query, f"could not parse response: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SourceError(
                self.name, query, "expected a Semantic Scholar JSON object"
            )
        papers = payload.get("data")
        if not isinstance(papers, list):
            raise SourceError(self.name, query, "missing data list")

        records: list[ResearchRecord] = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = _optional_str(paper.get("paperId"))
            title = _optional_str(paper.get("title"))
            if not paper_id or not title:
                continue
            doi = _doi_from_external_ids(paper.get("externalIds"))
            url = _optional_str(paper.get("url"))
            if not url:
                url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://www.semanticscholar.org/paper/{paper_id}"
                )
            year = _year_from(paper.get("year"))
            if year is None:
                year = _year_from(paper.get("publicationDate"))
            try:
                record = self._record(
                    source_id=paper_id,
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=paper.get("abstract")
                    if isinstance(paper.get("abstract"), str)
                    else None,
                    authors=_author_names(paper.get("authors")),
                    year=year,
                    doi=doi,
                    license=None,
                )
            except ValueError:
                continue
            records.append(record)
        return records


__all__ = ["SemanticScholarAdapter"]
