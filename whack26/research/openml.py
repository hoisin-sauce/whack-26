"""OpenML dataset metadata adapter (keyless public REST API)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    SourceError,
    raise_for_status,
    request_with_retries,
)

OPENML_API_URL = "https://www.openml.org/api/v1/json"
OPENML_DATA_LIST_URL = f"{OPENML_API_URL}/data/list"
OPENML_NO_RESULTS_CODE = "372"
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def _normalize_doi(value: Any) -> str | None:
    doi = _optional_str(value)
    if doi is None:
        return None
    lowered = doi.lower()
    for prefix in _DOI_PREFIXES:
        if lowered.startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    return doi or None


def _creator_names(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
    if isinstance(value, str):
        return [
            part.strip() for part in value.split(",") if part.strip()
        ]
    return []


def _dataset_name(query: str) -> str:
    """OpenML's list endpoint filters by exact dataset name, not full text."""
    return "_".join(query.lower().split())


def _is_no_results(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    return str(error.get("code")) == OPENML_NO_RESULTS_CODE


def _dataset_entries(payload: Any, *, source: str, query: str) -> list[dict]:
    if not isinstance(payload, dict):
        raise SourceError(source, query, "expected a JSON object")
    if _is_no_results(payload):
        return []
    if "error" in payload:
        error = payload.get("error")
        raise SourceError(source, query, f"OpenML search failed: {error}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise SourceError(source, query, "missing data object")
    entries = data.get("dataset")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise SourceError(source, query, "missing data.dataset list")
    return [entry for entry in entries if isinstance(entry, dict)]


class OpenMLAdapter(BaseHttpAdapter):
    """Search public OpenML datasets.

    The list endpoint only accepts an exact dataset name, so a query is
    normalized to OpenML's underscore naming convention. Dataset details are
    fetched for the bounded result page to expose description, creator, date,
    licence, and DOI metadata.
    """

    name = "openml"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_url = (base_url or OPENML_API_URL).rstrip("/")

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0 or not query.strip():
            return []
        return super().search(query, limit, client)

    def fetch(self, query: str, limit: int, client: httpx.Client) -> FetchResult:
        name = _dataset_name(query)
        list_url = (
            f"{self.base_url}/data/list/data_name/"
            f"{quote(name, safe='')}/limit/{int(limit)}"
        )
        response = request_with_retries(
            client,
            list_url,
            headers=self._headers(),
            source=self.name,
            query=query,
        )
        if response.status_code >= 400:
            if response.status_code == 412:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if _is_no_results(payload):
                    empty = json.dumps({"data": {"dataset": []}})
                    return FetchResult(
                        body=empty,
                        status=200,
                        headers=self._header_subset(response),
                    )
            raise_for_status(response, source=self.name, query=query)

        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceError(
                self.name, query, f"could not parse JSON list: {exc}"
            ) from exc
        entries = _dataset_entries(payload, source=self.name, query=query)
        enriched = [
            self._detail(entry, query=query, client=client)
            for entry in entries[: max(0, int(limit))]
        ]
        body = json.dumps({"data": {"dataset": enriched}})
        return FetchResult(
            body=body,
            status=response.status_code,
            headers=self._header_subset(response),
        )

    def _detail(
        self, entry: dict, *, query: str, client: httpx.Client
    ) -> dict:
        did = entry.get("did")
        if did is None or did == "":
            return entry
        detail_url = f"{self.base_url}/data/{quote(str(did), safe='')}"
        response = request_with_retries(
            client,
            detail_url,
            headers=self._headers(),
            source=self.name,
            query=query,
        )
        if response.status_code >= 400:
            # A missing or forbidden detail page does not discard the list
            # result; the list already carries the stable id and name.
            return entry
        try:
            payload = response.json()
        except ValueError:
            return entry
        if not isinstance(payload, dict) or _is_no_results(payload):
            return entry
        description = payload.get("data_set_description")
        if not isinstance(description, dict):
            return entry
        merged = dict(entry)
        merged.update(description)
        merged["did"] = did
        return merged

    def parse(
        self,
        body: str,
        *,
        query: str,
        retrieved_at: str | None = None,
    ) -> list[ResearchRecord]:
        payload = self._json(body)
        entries = _dataset_entries(payload, source=self.name, query=query)

        records: list[ResearchRecord] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            description = entry.get("data_set_description")
            if isinstance(description, dict):
                item = dict(entry)
                item.update(description)
            else:
                item = entry

            did = item.get("did")
            if did is None or did == "":
                did = item.get("id")
            if did is None or did == "":
                continue
            status = _optional_str(item.get("status"))
            if status is not None and status.lower() == "deactivated":
                continue

            title = _optional_str(item.get("name"))
            if not title:
                continue

            source_id = str(did)
            url = f"https://www.openml.org/d/{quote(source_id, safe='')}"
            try:
                record = self._record(
                    source_id=source_id,
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=_optional_str(item.get("description")),
                    authors=_creator_names(item.get("creator")),
                    year=_year_from(item.get("upload_date"))
                    or _year_from(item.get("collection_date"))
                    or _year_from(item.get("processing_date")),
                    doi=_normalize_doi(item.get("doi")),
                    license=_optional_str(item.get("licence"))
                    or _optional_str(item.get("license")),
                )
            except ValueError:
                continue
            records.append(record)
        return records
