"""Dataverse dataset search adapter (keyless public search API)."""

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

DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
DATAVERSE_SEARCH_PATH = "/api/search"
DATAVERSE_SEARCH_URL = f"{DATAVERSE_BASE_URL}{DATAVERSE_SEARCH_PATH}"
_DOI_PREFIX = "doi:"
_DRAFT_STATUSES = frozenset({"draft", "unpublished"})


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _doi_from_global_id(global_id: str | None) -> str | None:
    if not global_id:
        return None
    lowered = global_id.lower()
    if lowered.startswith(_DOI_PREFIX):
        value = global_id[len(_DOI_PREFIX) :].strip()
        return value or None
    return None


def _license_from(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("name", "uri", "id", "title"):
            candidate = _optional_str(value.get(key))
            if candidate:
                return candidate
    return None


def _is_unpublished(item: dict[str, Any]) -> bool:
    state = _optional_str(item.get("versionState"))
    if state is not None and state.lower() in _DRAFT_STATUSES:
        return True
    statuses = item.get("publicationStatuses")
    if isinstance(statuses, list):
        lowered = {
            status.strip().lower()
            for status in statuses
            if isinstance(status, str) and status.strip()
        }
        if lowered & _DRAFT_STATUSES:
            return True
    return False


def _authors(item: dict[str, Any]) -> list[str]:
    authors = _string_list(item.get("authors"))
    if authors:
        return authors
    contacts = item.get("contacts")
    if not isinstance(contacts, list):
        return []
    return [
        contact["name"].strip()
        for contact in contacts
        if isinstance(contact, dict)
        and isinstance(contact.get("name"), str)
        and contact["name"].strip()
    ]


class DataverseAdapter(BaseHttpAdapter):
    """Search dataset metadata in a public Dataverse installation."""

    name = "dataverse"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_url = (base_url or DATAVERSE_BASE_URL).rstrip("/")

    @property
    def search_url(self) -> str:
        return f"{self.base_url}{DATAVERSE_SEARCH_PATH}"

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0 or not query.strip():
            return []
        return super().search(query, limit, client)

    def fetch(self, query: str, limit: int, client: httpx.Client) -> FetchResult:
        response = request_with_retries(
            client,
            self.search_url,
            params={
                "q": query,
                "type": "dataset",
                "per_page": limit,
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

        status = payload.get("status")
        if status not in (None, "OK"):
            message = (
                payload.get("message")
                or payload.get("error")
                or f"status {status}"
            )
            raise SourceError(
                self.name, query, f"Dataverse search failed: {message}"
            )

        data = payload.get("data")
        if not isinstance(data, dict):
            raise SourceError(self.name, query, "missing data object")
        items = data.get("items")
        if not isinstance(items, list):
            raise SourceError(self.name, query, "missing data.items list")

        records: list[ResearchRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "dataset":
                continue
            if _is_unpublished(item):
                continue

            title = _optional_str(item.get("name"))
            if not title:
                continue

            global_id = _optional_str(item.get("global_id"))
            doi = _doi_from_global_id(global_id)
            url = _optional_str(item.get("url"))
            if not url and doi:
                url = f"https://doi.org/{doi}"
            if not url:
                continue

            if doi:
                source_id = doi
            elif global_id:
                source_id = global_id
            else:
                source_id = url

            try:
                record = self._record(
                    source_id=source_id,
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=_optional_str(item.get("description")),
                    authors=_authors(item),
                    year=_year_from(item.get("published_at"))
                    or _year_from(item.get("createdAt")),
                    doi=doi,
                    license=_license_from(item.get("license")),
                )
            except ValueError:
                continue
            records.append(record)
        return records
