"""Hugging Face Hub dataset search adapter."""

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

HF_DATASETS_URL = "https://huggingface.co/api/datasets"


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def _license_from_tags(tags: Any) -> str | None:
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("license:"):
            value = tag[len("license:") :].strip()
            if value:
                return value
    return None


class HuggingFaceAdapter(BaseHttpAdapter):
    """Search public dataset metadata on the Hugging Face Hub."""

    name = "huggingface"
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
            HF_DATASETS_URL,
            params={"search": query, "limit": limit},
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
        if not isinstance(payload, list):
            raise SourceError(self.name, query, "expected a JSON list of datasets")
        records: list[ResearchRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            # Private repositories are never exposed; gated repositories stay
            # metadata-only and are treated like any other public dataset.
            if item.get("private") is True:
                continue
            source_id = item.get("id")
            if not isinstance(source_id, str) or not source_id.strip():
                continue
            author = item.get("author")
            description = item.get("description")
            try:
                record = self._record(
                    source_id=source_id,
                    title=source_id,
                    url=f"https://huggingface.co/datasets/{source_id}",
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=description if isinstance(description, str) else None,
                    authors=[author]
                    if isinstance(author, str) and author.strip()
                    else [],
                    year=_year_from(item.get("createdAt")),
                    doi=None,
                    license=_license_from_tags(item.get("tags")),
                )
            except ValueError:
                continue
            records.append(record)
        return records
