"""data.gov CKAN package search adapter."""

from __future__ import annotations

import time
from typing import Any, Callable
from urllib.parse import quote

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    clean_text,
    raise_for_status,
    request_with_retries,
)
from .cache import ResearchCache

DATAGOV_API_URL = "https://catalog.data.gov/api/3/action/package_search"
DATAGOV_DATASET_URL = "https://catalog.data.gov/dataset/"
DATAGOV_API_KEY_HEADER = "X-CKAN-API-Key"

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_PLACEHOLDER_LICENSES = frozenset(
    {
        "license not specified",
        "no license provided",
        "not specified",
        "notspecified",
    }
)
_DOI_EXTRA_KEYS = frozenset({"doi", "identifier"})


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


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


def _license_from(package: dict[str, Any]) -> str | None:
    for key in ("license_id", "license_title"):
        value = _optional_str(package.get(key))
        if value and value.lower() not in _PLACEHOLDER_LICENSES:
            return value
    return None


def _doi_from_extras(extras: Any) -> str | None:
    if not isinstance(extras, list):
        return None
    for item in extras:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str) or key.strip().lower() not in _DOI_EXTRA_KEYS:
            continue
        doi = _normalize_doi(item.get("value"))
        if doi and doi.lower().startswith("10."):
            return doi
    return None


def _publisher(package: dict[str, Any]) -> str | None:
    organization = package.get("organization")
    if isinstance(organization, dict):
        value = _optional_str(organization.get("title")) or _optional_str(
            organization.get("name")
        )
        if value:
            return value
    return _optional_str(package.get("publisher"))


class DataGovAdapter(BaseHttpAdapter):
    """Search data.gov's CKAN catalogue for dataset metadata.

    data.gov allows anonymous use; an optional key raises the per-IP limits.
    """

    name = "datagov"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        api_key: str | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        self.api_key = api_key
        self.sleep = sleep

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        headers = self._headers()
        if self.api_key:
            headers[DATAGOV_API_KEY_HEADER] = self.api_key
        response = request_with_retries(
            client,
            DATAGOV_API_URL,
            params={"q": query, "rows": limit},
            headers=headers,
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
            raise SourceError(self.name, query, "expected a CKAN JSON object")
        if payload.get("success") is False:
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or error
            else:
                detail = error or "success=false"
            raise SourceError(
                self.name, query, f"CKAN error: {clean_text(str(detail))}"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise SourceError(self.name, query, "missing result object")
        packages = result.get("results")
        if not isinstance(packages, list):
            raise SourceError(self.name, query, "missing result.results list")

        records: list[ResearchRecord] = []
        for package in packages:
            if not isinstance(package, dict):
                continue
            name = _optional_str(package.get("name"))
            package_id = _optional_str(package.get("id"))
            source_id = name or package_id
            if not source_id:
                continue
            title = _optional_str(package.get("title")) or name
            if not title:
                continue
            slug = name or source_id
            publisher = _publisher(package)
            try:
                record = self._record(
                    source_id=source_id,
                    title=title,
                    url=f"{DATAGOV_DATASET_URL}{quote(slug, safe='')}",
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=package.get("notes")
                    if isinstance(package.get("notes"), str)
                    else None,
                    authors=[publisher] if publisher else [],
                    year=_year_from(package.get("metadata_created"))
                    or _year_from(package.get("metadata_modified")),
                    doi=_doi_from_extras(package.get("extras")),
                    license=_license_from(package),
                )
            except ValueError:
                continue
            records.append(record)
        return records


__all__ = ["DataGovAdapter"]
