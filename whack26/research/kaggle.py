"""Kaggle dataset metadata adapter."""

from __future__ import annotations

import base64
import logging
import os
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

KAGGLE_DATASETS_URL = "https://www.kaggle.com/api/v1/datasets/list"
KAGGLE_LIMITED_MESSAGE = (
    "source_warning: kaggle requires KAGGLE_API_TOKEN; source skipped "
    "(limited mode)"
)

logger = logging.getLogger(__name__)


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _year_from(value: Any) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    prefix = value[:4]
    return int(prefix) if prefix.isdigit() else None


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


class KaggleAdapter(BaseHttpAdapter):
    """Search public Kaggle dataset metadata.

    Kaggle's public API requires a ``KGAT_`` bearer token. Without one the
    adapter stays in limited mode and returns no records rather than making an
    unauthenticated request that would fail.
    """

    name = "kaggle"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        token: str | None = None,
        username: str | None = None,
        key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.token = (
            token.strip() if isinstance(token, str) and token.strip() else None
        ) or _env_value("KAGGLE_API_TOKEN")
        self.username = (
            username.strip()
            if isinstance(username, str) and username.strip()
            else None
        ) or _env_value("KAGGLE_USERNAME")
        # Legacy Kaggle credentials use a username plus an API key. Keep the
        # fallback so an existing kaggle.json still works when copied to env.
        self.key = (
            key.strip() if isinstance(key, str) and key.strip() else None
        ) or _env_value("KAGGLE_KEY")
        self.warnings: list[str] = []
        if self._authorization() is None:
            self.limited_mode = True

    def _authorization(self) -> str | None:
        if self.token:
            return f"Bearer {self.token}"
        if self.username and self.key:
            credentials = f"{self.username}:{self.key}".encode("utf-8")
            return "Basic " + base64.b64encode(credentials).decode("ascii")
        return None

    def _warn_limited(self) -> None:
        if KAGGLE_LIMITED_MESSAGE not in self.warnings:
            self.warnings.append(KAGGLE_LIMITED_MESSAGE)
            logger.warning("%s", KAGGLE_LIMITED_MESSAGE)

    def search(
        self, query: str, limit: int, client: httpx.Client
    ) -> list[ResearchRecord]:
        if limit <= 0 or not query.strip():
            return []
        if self._authorization() is None:
            self.limited_mode = True
            self._warn_limited()
            return []
        # The list endpoint returns a fixed page (currently 20 records) even
        # when a page size is requested, so trim to the caller's limit.
        return super().search(query, limit, client)[:limit]

    def fetch(self, query: str, limit: int, client: httpx.Client) -> FetchResult:
        headers = self._headers()
        authorization = self._authorization()
        if authorization is not None:
            headers["Authorization"] = authorization
        response = request_with_retries(
            client,
            KAGGLE_DATASETS_URL,
            params={"search": query, "page": 1, "pageSize": limit},
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
        if isinstance(payload, dict):
            payload = payload.get("datasets")
        if not isinstance(payload, list):
            raise SourceError(
                self.name, query, "expected a JSON list of datasets"
            )

        records: list[ResearchRecord] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if item.get("isPrivate") is True:
                continue

            ref = _optional_str(item.get("ref"))
            source_id = ref or _optional_str(item.get("id"))
            if source_id is None and item.get("id") is not None:
                source_id = str(item["id"])
            if not source_id:
                continue

            raw_title = item.get("title")
            title = _optional_str(raw_title)
            if title is None:
                # A blank title is unusable; a missing one may fall back to
                # the repository ref, which is still a meaningful name.
                if raw_title is not None:
                    continue
                title = ref
            if not title:
                continue
            description = _optional_str(item.get("description"))
            subtitle = _optional_str(item.get("subtitle"))
            owner = _optional_str(item.get("ownerName")) or _optional_str(
                item.get("creatorName")
            )
            url = _optional_str(item.get("url")) or (
                f"https://www.kaggle.com/datasets/{ref or source_id}"
            )
            try:
                record = self._record(
                    source_id=source_id,
                    title=title,
                    url=url,
                    query=query,
                    retrieved_at=retrieved_at,
                    abstract=description or subtitle,
                    authors=[owner] if owner else [],
                    year=_year_from(item.get("lastUpdated"))
                    or _year_from(item.get("creationDate")),
                    doi=_optional_str(item.get("doi")),
                    license=_optional_str(item.get("licenseName")),
                )
            except ValueError:
                continue
            records.append(record)
        return records
