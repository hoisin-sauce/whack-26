"""DataCite REST API adapter for DOI-backed datasets."""

from __future__ import annotations

import time
from typing import Any, Callable

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

DATACITE_DOIS_URL = "https://api.datacite.org/dois"
# data.gov, Zenodo, and most repositories expose datasets through DataCite.
DATACITE_RESOURCE_TYPE = "dataset"

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


def _optional_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _title_from(titles: Any) -> str | None:
    if not isinstance(titles, list):
        return None
    for item in titles:
        if not isinstance(item, dict):
            continue
        title = _optional_str(item.get("title"))
        if title:
            return title
    return None


def _description_from(descriptions: Any) -> str | None:
    if not isinstance(descriptions, list):
        return None
    fallback: str | None = None
    for item in descriptions:
        if not isinstance(item, dict):
            continue
        text = _optional_str(item.get("description"))
        if not text:
            continue
        if fallback is None:
            fallback = text
        description_type = item.get("descriptionType")
        if (
            isinstance(description_type, str)
            and description_type.strip().lower() == "abstract"
        ):
            return text
    return fallback


def _creator_name(creator: dict[str, Any]) -> str | None:
    name = _optional_str(creator.get("name"))
    if name:
        return name
    parts = [
        _optional_str(creator.get("givenName")),
        _optional_str(creator.get("familyName")),
    ]
    combined = " ".join(part for part in parts if part)
    return combined or None


def _authors_from(creators: Any) -> list[str]:
    if not isinstance(creators, list):
        return []
    names: list[str] = []
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        name = _creator_name(creator)
        if name:
            names.append(name)
    return names


def _license_from(rights: Any) -> str | None:
    if not isinstance(rights, list):
        return None
    for item in rights:
        if not isinstance(item, dict):
            continue
        for key in ("rightsIdentifier", "rights", "rightsUri"):
            value = _optional_str(item.get(key))
            if value:
                return value
    return None


class DataCiteAdapter(BaseHttpAdapter):
    """Search DataCite DOIs for dataset metadata."""

    name = "datacite"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        self.sleep = sleep

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        response = request_with_retries(
            client,
            DATACITE_DOIS_URL,
            params={
                "query": query,
                "page[size]": limit,
                "resource-type-id": DATACITE_RESOURCE_TYPE,
            },
            headers=self._headers(),
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
            payload = self._json(body)
        except ValueError as exc:
            raise SourceError(
                self.name, query, f"could not parse response: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SourceError(self.name, query, "expected a DataCite JSON object")
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise SourceError(self.name, query, "missing data list")

        records: list[ResearchRecord] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            attributes = entry.get("attributes")
            if not isinstance(attributes, dict):
                continue
            doi = _normalize_doi(attributes.get("doi") or entry.get("id"))
            if not doi:
                continue
            title = _title_from(attributes.get("titles"))
            if not title:
                continue
            url = _optional_str(attributes.get("url")) or f"https://doi.org/{doi}"
            try:
                record = self._record(
                    source_id=doi,
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=_description_from(attributes.get("descriptions")),
                    authors=_authors_from(attributes.get("creators")),
                    year=_optional_int(attributes.get("publicationYear")),
                    doi=doi,
                    license=_license_from(attributes.get("rightsList")),
                )
            except ValueError:
                continue
            records.append(record)
        return records


__all__ = ["DataCiteAdapter"]
