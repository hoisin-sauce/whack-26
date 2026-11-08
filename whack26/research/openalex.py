"""OpenAlex works adapter for primary paper research."""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    BaseHttpAdapter,
    FetchResult,
    ResearchRecord,
    raise_for_status,
    reconstruct_abstract,
    request_with_retries,
)
from .cache import ResearchCache

OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_ID_PREFIX = "https://openalex.org/"
OPENALEX_SELECT = (
    "id,doi,title,publication_year,authorships,primary_location,open_access,"
    "cited_by_count,abstract_inverted_index,topics,keywords,is_retracted"
)
MAX_RESULTS = 10


def _strip_prefix(value: Any, *prefixes: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :]
    return text


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _author_names(authorships: Any) -> list[str]:
    if not isinstance(authorships, list):
        return []
    names: list[str] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        name = _optional_str(author.get("display_name"))
        if name:
            names.append(name)
    return names


def _author_ids(authorships: Any) -> list[str]:
    """Return source-prefixed author IDs; an id can exist without a name."""
    if not isinstance(authorships, list):
        return []
    ids: list[str] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        author_id = _strip_prefix(author.get("id"), OPENALEX_ID_PREFIX).strip().strip("/")
        if author_id:
            prefixed = f"openalex:{author_id}"
            if prefixed not in ids:
                ids.append(prefixed)
    return ids


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _venue_metrics(primary: Any) -> tuple[str | None, str | None]:
    if not isinstance(primary, dict):
        return None, None
    source = primary.get("source")
    if not isinstance(source, dict):
        return None, None
    return (
        _optional_str(source.get("type")),
        _optional_str(source.get("host_organization_name")),
    )


class OpenAlexAdapter(BaseHttpAdapter):
    """Search OpenAlex works, falling back to keyless limited mode."""

    name = "openalex"
    ttl_seconds = 24 * 60 * 60

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache: ResearchCache | None = None,
        dataset_id: str = "",
        offline: bool = False,
    ) -> None:
        super().__init__(cache=cache, dataset_id=dataset_id, offline=offline)
        self.api_key = api_key
        if not api_key:
            self.limited_mode = True

    def fetch(
        self, query: str, limit: int, client: httpx.Client
    ) -> FetchResult:
        params: dict[str, Any] = {
            "search": query,
            "filter": "is_retracted:false",
            "per_page": limit,
            "select": OPENALEX_SELECT,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        response = request_with_retries(
            client,
            OPENALEX_API_URL,
            params=params,
            headers=self._headers(),
            source=self.name,
            query=query,
        )
        if response.status_code == 403 and self.api_key:
            # A key can exist but not be activated yet. Retry keyless once and
            # mark this adapter as limited for the rest of the run.
            self.limited_mode = True
            params.pop("api_key", None)
            response = request_with_retries(
                client,
                OPENALEX_API_URL,
                params=params,
                headers=self._headers(),
                source=self.name,
                query=query,
            )
        raise_for_status(response, source=self.name, query=query)

        headers = self._header_subset(response)
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            meta = payload.get("meta")
            if isinstance(meta, dict) and "cost_usd" in meta:
                headers["x-openalex-cost-usd"] = str(meta["cost_usd"])
        return FetchResult(
            body=response.text,
            status=response.status_code,
            headers=headers,
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
            return []
        works = payload.get("results")
        if not isinstance(works, list):
            return []

        records: list[ResearchRecord] = []
        for work in works:
            if len(records) >= MAX_RESULTS:
                break
            if not isinstance(work, dict) or work.get("is_retracted"):
                continue

            title = _optional_str(work.get("title"))
            if not title:
                continue

            source_id = _strip_prefix(work.get("id"), OPENALEX_ID_PREFIX)
            if not source_id:
                continue

            primary = work.get("primary_location")
            if not isinstance(primary, dict):
                primary = {}
            doi = _strip_prefix(
                work.get("doi"), "https://doi.org/", "doi:"
            ) or None
            url = _optional_str(primary.get("landing_page_url"))
            if not url:
                url = (
                    f"https://doi.org/{doi}"
                    if doi
                    else f"https://openalex.org/{source_id}"
                )

            year = work.get("publication_year")
            if not isinstance(year, int) or isinstance(year, bool):
                year = None

            cited_by_count = _optional_int(work.get("cited_by_count"))
            author_ids = _author_ids(work.get("authorships"))
            venue_type, publisher = _venue_metrics(primary)
            metrics_source = (
                "openalex"
                if cited_by_count is not None
                or author_ids
                or venue_type is not None
                or publisher is not None
                else None
            )

            records.append(
                self._record(
                    source_id=source_id,
                    title=title,
                    abstract=reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    ),
                    authors=_author_names(work.get("authorships")),
                    year=year,
                    doi=doi,
                    url=url,
                    license=_optional_str(primary.get("license")),
                    query=query,
                    retrieved_at=retrieved_at,
                    cited_by_count=cited_by_count,
                    author_ids=author_ids,
                    venue_type=venue_type,
                    publisher=publisher,
                    metrics_source=metrics_source,
                )
            )
        return records
