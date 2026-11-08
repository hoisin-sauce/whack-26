"""arXiv Atom adapter for secondary paper research."""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from typing import Callable

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    raise_for_status,
    request_with_retries,
)
from .cache import ResearchCache

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT: float | None = None


def _reset_rate_limit_state() -> None:
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        _LAST_REQUEST_AT = None


def reset_rate_limit() -> None:
    """Reset the shared arXiv request clock (used to isolate tests)."""
    _reset_rate_limit_state()


def _throttle(
    min_interval: float, sleep: Callable[[float], None]
) -> None:
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        if _LAST_REQUEST_AT is not None and now < _LAST_REQUEST_AT + min_interval:
            sleep(min_interval)
            # Account for injected sleeps that do not advance monotonic time.
            now = _LAST_REQUEST_AT + min_interval
        _LAST_REQUEST_AT = now


def _element_text(
    element: ET.Element, namespace: str, tag: str
) -> str | None:
    value = element.findtext(f"{{{namespace}}}{tag}")
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _source_id(entry_id: str) -> str:
    """Keep the full old-style identifier after ``/abs/`` when present."""
    value = entry_id.strip()
    marker = "/abs/"
    if marker in value:
        return value.split(marker, 1)[1].strip("/")
    return value.rstrip("/").rsplit("/", 1)[-1]


class arXivAdapter(BaseHttpAdapter):
    """Search arXiv Atom feeds with a process-wide 3-second throttle."""

    name = "arxiv"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        *,
        min_interval: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        self.min_interval = max(0.0, float(min_interval))
        self.sleep = sleep

    @staticmethod
    def reset_rate_limit() -> None:
        _reset_rate_limit_state()

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        _throttle(self.min_interval, self.sleep)
        response = request_with_retries(
            client,
            ARXIV_API_URL,
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
            },
            headers=self._headers({"Accept": "application/atom+xml"}),
            source=self.name,
            query=query,
            sleep=self.sleep,
        )
        raise_for_status(response, source=self.name, query=query)
        return FetchResult(
            body=response.text,
            status=response.status_code,
            headers=self._header_subset(response),
        )

    def parse(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise SourceError(
                self.name, query, f"could not parse arXiv Atom: {exc}"
            ) from exc

        records: list[ResearchRecord] = []
        for entry in root.findall(f"{{{ATOM_NAMESPACE}}}entry"):
            title = _element_text(entry, ATOM_NAMESPACE, "title")
            if not title:
                continue
            entry_id = _element_text(entry, ATOM_NAMESPACE, "id")
            if not entry_id:
                continue
            source_id = _source_id(entry_id)
            if not source_id:
                continue

            authors = [
                name
                for author in entry.findall(f"{{{ATOM_NAMESPACE}}}author")
                if (name := _element_text(author, ATOM_NAMESPACE, "name"))
            ]

            year: int | None = None
            published = _element_text(entry, ATOM_NAMESPACE, "published")
            if published and len(published) >= 4:
                try:
                    year = int(published[:4])
                except ValueError:
                    year = None

            url = entry_id
            for link in entry.findall(f"{{{ATOM_NAMESPACE}}}link"):
                if link.get("rel") == "alternate" and link.get("href"):
                    url = link.get("href", entry_id)
                    break

            records.append(
                self._record(
                    source_id=source_id,
                    title=title,
                    abstract=_element_text(
                        entry, ATOM_NAMESPACE, "summary"
                    ),
                    authors=authors,
                    year=year,
                    doi=_element_text(entry, ARXIV_NAMESPACE, "doi"),
                    url=url,
                    license=None,
                    query=query,
                    retrieved_at=retrieved_at,
                )
            )
        return records


__all__ = ["arXivAdapter", "reset_rate_limit"]
