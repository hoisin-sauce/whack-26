"""Zenodo DOI-backed dataset search adapter."""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    raise_for_status,
    request_with_retries,
)

ZENODO_RECORDS_URL = "https://zenodo.org/api/records"
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)
_SKIPPED_ACCESS_RIGHTS = frozenset({"restricted", "embargoed"})


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def _normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    doi = value.strip()
    for prefix in _DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi or None


def _license_from_metadata(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("license")
    if isinstance(value, dict):
        identifier = value.get("id")
        return str(identifier) if identifier else None
    if isinstance(value, str):
        return value or None
    return None


class ZenodoAdapter(BaseHttpAdapter):
    """Search public Zenodo records for datasets and other research outputs."""

    name = "zenodo"
    ttl_seconds = 24 * 60 * 60

    def __init__(self, token: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.token = token

    def fetch(self, query: str, limit: int, client: httpx.Client) -> FetchResult:
        headers = self._headers()
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = request_with_retries(
            client,
            ZENODO_RECORDS_URL,
            # `mostrecent` returns unrelated recent deposits for multi-word
            # queries; `bestmatch` keeps the ranking tied to the query.
            params={"q": query, "size": limit, "sort": "bestmatch"},
            headers=headers,
            source=self.name,
            query=query,
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
        payload = self._json(body)
        if not isinstance(payload, dict):
            raise SourceError(self.name, query, "expected a JSON object")
        hits = payload.get("hits")
        if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
            raise SourceError(self.name, query, "missing hits.hits list")
        records: list[ResearchRecord] = []
        for entry in hits["hits"]:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("access_right") in _SKIPPED_ACCESS_RIGHTS:
                continue
            title = metadata.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            source_id = entry.get("id")
            if source_id is None:
                continue
            doi = _normalize_doi(entry.get("doi"))
            links = entry.get("links")
            url = None
            if isinstance(links, dict):
                candidate = links.get("self_html")
                if isinstance(candidate, str) and candidate:
                    url = candidate
            if not url:
                url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://zenodo.org/records/{source_id}"
                )
            creators = metadata.get("creators")
            authors = [
                creator["name"]
                for creator in (creators if isinstance(creators, list) else [])
                if isinstance(creator, dict)
                and isinstance(creator.get("name"), str)
                and creator["name"].strip()
            ]
            description = metadata.get("description")
            try:
                record = self._record(
                    source_id=str(source_id),
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=description if isinstance(description, str) else None,
                    authors=authors,
                    year=_year_from(metadata.get("publication_date")),
                    doi=doi,
                    license=_license_from_metadata(metadata),
                )
            except ValueError:
                continue
            records.append(record)
        return records
