"""Batched bibliometric enrichment from OpenAlex.

The research adapters populate the fields that arrive with a search response
(for example ``cited_by_count`` and author IDs).  Author-level h-index and
citation totals need a second lookup, so they are fetched here in batches
instead of once per adapter or once per scraped document.

The pass is deliberately optional and key-gated: without
``OPENALEX_API_KEY`` this module is a no-op.  The OpenAlex free key is enough;
no paid plan is required.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import httpx

from ..config import Settings
from .base import DEFAULT_USER_AGENT, ResearchRecord, request_with_retries
from .cache import ResearchCache, cache_key
from .normalize import normalize_doi

if TYPE_CHECKING:
    from .scrape.models import Document

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
WORKS_SELECT = "id,doi,cited_by_count,authorships,primary_location"
AUTHORS_SELECT = "id,display_name,cited_by_count,summary_stats"
METRICS_BATCH_SIZE = 50
METRICS_TTL_SECONDS = 7 * 24 * 60 * 60
OPENALEX_PREFIX = "openalex:"
_DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s|]+$")

__all__ = [
    "AUTHORS_SELECT",
    "METRICS_BATCH_SIZE",
    "METRICS_TTL_SECONDS",
    "OPENALEX_AUTHORS_URL",
    "OPENALEX_WORKS_URL",
    "WORKS_SELECT",
    "enrich_records",
    "merge_document_metrics",
    "metrics_metadata",
]


def metrics_metadata(record: ResearchRecord) -> dict[str, Any]:
    """Return the document-metadata keys consumed by the credibility scorer."""
    values: dict[str, Any] = {}
    if record.cited_by_count is not None:
        values["cited_by_count"] = record.cited_by_count
    if record.author_ids:
        values["author_ids"] = list(record.author_ids)
    if record.author_h_index is not None:
        values["author_h_index"] = record.author_h_index
    if record.author_citation_count is not None:
        values["author_citation_count"] = record.author_citation_count
    if record.author_known is not None:
        values["author_known"] = record.author_known
    if record.venue_type is not None:
        values["venue_type"] = record.venue_type
    if record.metrics_source is not None:
        values["metrics_source"] = record.metrics_source
    return values


METRIC_METADATA_KEYS = (
    "cited_by_count",
    "author_ids",
    "author_h_index",
    "author_citation_count",
    "author_known",
    "venue_type",
    "metrics_source",
)


def _merge_metric_value(current: Any, incoming: Any, key: str) -> Any:
    if key == "author_ids":
        merged = list(current or [])
        for item in incoming or []:
            if item not in merged:
                merged.append(item)
        return merged
    if key in {"cited_by_count", "author_h_index", "author_citation_count"}:
        if incoming is None:
            return current
        if current is None:
            return incoming
        return max(current, incoming)
    if key == "author_known":
        return True if current is True or incoming is True else current
    return incoming if current is None else current


def merge_document_metrics(existing: "Document", incoming: "Document") -> "Document":
    """Merge incoming metric metadata into an already-catalogued document.

    Reused documents are not reprocessed, so a fresh research result is the
    only chance to upgrade an older row that predates metrics support.
    """
    existing_metadata = dict(existing.metadata or {})
    incoming_metadata = incoming.metadata or {}
    metadata = dict(existing_metadata)
    for key in METRIC_METADATA_KEYS:
        if key not in incoming_metadata:
            continue
        metadata[key] = _merge_metric_value(
            metadata.get(key), incoming_metadata[key], key
        )

    updates: dict[str, Any] = {}
    if metadata != existing_metadata:
        updates["metadata"] = metadata
    if not existing.publisher and incoming.publisher:
        updates["publisher"] = incoming.publisher
    return existing.model_copy(update=updates) if updates else existing


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _valid_doi(value: str | None) -> bool:
    return bool(value and _DOI_RE.fullmatch(value))


def _bare_openalex_author_id(value: Any) -> str | None:
    """Return the bare ``A...`` id from a prefixed id or OpenAlex URL."""
    text = _optional_str(value)
    if not text or text.startswith("semantic_scholar:"):
        return None
    if text.startswith(OPENALEX_PREFIX):
        text = text[len(OPENALEX_PREFIX) :]
    else:
        for prefix in ("https://openalex.org/", "http://openalex.org/"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
    text = text.strip().strip("/")
    if not text.startswith("A") or not text[1:].isdigit():
        return None
    return text


def _author_ids_from_work(work: dict[str, Any]) -> list[str]:
    authorships = work.get("authorships")
    if not isinstance(authorships, list):
        return []
    ids: list[str] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if not isinstance(author, dict):
            continue
        author_id = _bare_openalex_author_id(author.get("id"))
        if author_id:
            prefixed = f"{OPENALEX_PREFIX}{author_id}"
            if prefixed not in ids:
                ids.append(prefixed)
    return ids


def _venue_metrics(work: dict[str, Any]) -> tuple[str | None, str | None]:
    primary = work.get("primary_location")
    if not isinstance(primary, dict):
        return None, None
    source = primary.get("source")
    if not isinstance(source, dict):
        return None, None
    return (
        _optional_str(source.get("type")),
        _optional_str(source.get("host_organization_name")),
    )


def _work_metrics(work: dict[str, Any]) -> dict[str, Any]:
    venue_type, publisher = _venue_metrics(work)
    return {
        "cited_by_count": _optional_int(work.get("cited_by_count")),
        "author_ids": _author_ids_from_work(work),
        "venue_type": venue_type,
        "publisher": publisher,
    }


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _request_json(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
    cache: ResearchCache | None,
    dataset_id: str,
    query: str,
    warnings: list[str],
) -> Any | None:
    cache_params = {k: v for k, v in params.items() if k != "api_key"}
    key = cache_key(
        dataset_id,
        "openalex_metrics",
        query,
        {"url": url, **cache_params},
    )
    if cache is not None:
        try:
            cached = cache.get(key)
        except Exception as exc:  # noqa: BLE001 - a broken cache is advisory
            warnings.append(
                f"research_metrics_failed: cache read: {type(exc).__name__}: {exc}"
            )
            cached = None
        if cached is not None and 200 <= cached.status < 300:
            try:
                return json.loads(cached.body)
            except json.JSONDecodeError:
                warnings.append("research_metrics_failed: cached body was not JSON")

    try:
        response = request_with_retries(
            client,
            url,
            params=params,
            headers={"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"},
            source="openalex_metrics",
            query=query,
            retries=1,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment is advisory
        warnings.append(
            f"research_metrics_failed: {url}: {type(exc).__name__}: {exc}"
        )
        return None

    if not 200 <= response.status_code < 300:
        warnings.append(
            f"research_metrics_failed: {url}: HTTP {response.status_code}"
        )
        return None

    body = response.text
    if cache is not None:
        try:
            cache.put(
                key,
                dataset_id=dataset_id,
                source="openalex_metrics",
                query=query,
                status=response.status_code,
                body=body,
                ttl_seconds=METRICS_TTL_SECONDS,
                headers={"content-type": response.headers.get("content-type", "")},
            )
        except Exception as exc:  # noqa: BLE001 - a broken cache is advisory
            warnings.append(
                f"research_metrics_failed: cache write: {type(exc).__name__}: {exc}"
            )
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        warnings.append(f"research_metrics_failed: {url}: response was not JSON")
        return None


def _fetch_works_by_doi(
    dois: Sequence[str],
    *,
    client: httpx.Client,
    settings: Settings,
    cache: ResearchCache | None,
    dataset_id: str,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(dois, METRICS_BATCH_SIZE):
        filter_value = "doi:" + "|".join(chunk)
        params: dict[str, Any] = {
            "filter": filter_value,
            "select": WORKS_SELECT,
            "per_page": len(chunk),
        }
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        payload = _request_json(
            client,
            OPENALEX_WORKS_URL,
            params=params,
            cache=cache,
            dataset_id=dataset_id,
            query=filter_value,
            warnings=warnings,
        )
        if not isinstance(payload, dict):
            continue
        for work in payload.get("results") or []:
            if not isinstance(work, dict):
                continue
            doi = normalize_doi(work.get("doi"))
            if doi:
                results[doi] = work
    return results


def _fetch_author_metrics(
    author_ids: Sequence[str],
    *,
    client: httpx.Client,
    settings: Settings,
    cache: ResearchCache | None,
    dataset_id: str,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(author_ids, METRICS_BATCH_SIZE):
        filter_value = "id:" + "|".join(chunk)
        params: dict[str, Any] = {
            "filter": filter_value,
            "select": AUTHORS_SELECT,
            "per_page": len(chunk),
        }
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        payload = _request_json(
            client,
            OPENALEX_AUTHORS_URL,
            params=params,
            cache=cache,
            dataset_id=dataset_id,
            query=filter_value,
            warnings=warnings,
        )
        if not isinstance(payload, dict):
            continue
        for author in payload.get("results") or []:
            if not isinstance(author, dict):
                continue
            author_id = _bare_openalex_author_id(author.get("id"))
            if author_id:
                results[author_id] = author
    return results


def _author_ids(record: ResearchRecord) -> list[str]:
    """Return bare OpenAlex author IDs; other namespaces are ignored."""
    ids: list[str] = []
    for value in record.author_ids:
        author_id = _bare_openalex_author_id(value)
        if author_id and author_id not in ids:
            ids.append(author_id)
    return ids


def _apply_author_metrics(
    record: ResearchRecord,
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Use the strongest resolved author as the paper's author signal.

    Taking the maximum h-index (falling back to maximum author citations) is
    deterministic and avoids penalising a paper for having one early-career
    author.  The trade-off is that a highly cited co-author can lift the score.
    """
    updates: dict[str, Any] = {}
    best_h_index: int | None = None
    best_citations: int | None = None
    resolved = False
    for author_id in _author_ids(record):
        author = metrics.get(author_id)
        if author is None:
            continue
        resolved = True
        summary = author.get("summary_stats")
        h_index = None
        if isinstance(summary, dict):
            h_index = _optional_int(summary.get("h_index"))
        citations = _optional_int(author.get("cited_by_count"))
        if h_index is not None and (best_h_index is None or h_index > best_h_index):
            best_h_index = h_index
        if citations is not None and (
            best_citations is None or citations > best_citations
        ):
            best_citations = citations

    if not resolved:
        return updates

    if best_h_index is not None:
        current = record.author_h_index
        updates["author_h_index"] = (
            best_h_index if current is None else max(current, best_h_index)
        )
    if best_citations is not None:
        current = record.author_citation_count
        updates["author_citation_count"] = (
            best_citations if current is None else max(current, best_citations)
        )
    updates["author_known"] = True
    updates["metrics_source"] = "openalex"
    return updates


def enrich_records(
    records: Sequence[ResearchRecord],
    *,
    settings: Settings,
    client: httpx.Client,
    cache: ResearchCache | None = None,
    dataset_id: str = "",
    warnings: list[str] | None = None,
) -> list[ResearchRecord]:
    """Return records enriched with OpenAlex citation and author metrics.

    The function never raises for provider failures.  It is a no-op unless
    ``research_metrics_enabled`` is true and an OpenAlex key is configured.
    """
    warnings = warnings if warnings is not None else []
    items = list(records)

    if not settings.research_metrics_enabled:
        warnings.append("research_metrics_disabled")
        return items
    if not settings.openalex_api_key:
        warnings.append("research_metrics_skipped: no OPENALEX_API_KEY")
        return items
    if settings.offline:
        warnings.append("research_metrics_skipped: offline")
        return items
    if settings.llm_provider == "mock" or settings.embed_provider == "mock":
        warnings.append("research_metrics_skipped: mock")
        return items
    if not items:
        return items

    doi_by_record: dict[str, str] = {}
    for record in items:
        doi = normalize_doi(record.doi)
        if (
            doi is not None
            and _valid_doi(doi)
            and (not _author_ids(record) or record.cited_by_count is None)
        ):
            doi_by_record.setdefault(doi, record.id)

    works: dict[str, dict[str, Any]] = {}
    if doi_by_record:
        works = _fetch_works_by_doi(
            list(doi_by_record),
            client=client,
            settings=settings,
            cache=cache,
            dataset_id=dataset_id,
            warnings=warnings,
        )

    updates_by_id: dict[str, dict[str, Any]] = {
        record.id: {} for record in items
    }

    for record in items:
        doi = normalize_doi(record.doi)
        work = works.get(doi) if doi else None
        if not work:
            continue
        metrics = _work_metrics(work)
        updates = updates_by_id[record.id]
        if record.cited_by_count is None and metrics["cited_by_count"] is not None:
            updates["cited_by_count"] = metrics["cited_by_count"]
        if not _author_ids(record) and metrics["author_ids"]:
            merged_ids = list(record.author_ids)
            for author_id in metrics["author_ids"]:
                if author_id not in merged_ids:
                    merged_ids.append(author_id)
            updates["author_ids"] = merged_ids
        if record.venue_type is None and metrics["venue_type"] is not None:
            updates["venue_type"] = metrics["venue_type"]
        if record.publisher is None and metrics["publisher"] is not None:
            updates["publisher"] = metrics["publisher"]
        if updates:
            updates["metrics_source"] = "openalex"

    author_ids: list[str] = []
    for record in items:
        for author_id in _author_ids(record):
            if author_id not in author_ids:
                author_ids.append(author_id)
        resolved = updates_by_id[record.id].get("author_ids") or []
        for value in resolved:
            author_id = _bare_openalex_author_id(value)
            if author_id and author_id not in author_ids:
                author_ids.append(author_id)

    author_metrics: dict[str, dict[str, Any]] = {}
    if author_ids:
        author_metrics = _fetch_author_metrics(
            author_ids,
            client=client,
            settings=settings,
            cache=cache,
            dataset_id=dataset_id,
            warnings=warnings,
        )

    enriched: list[ResearchRecord] = []
    for record in items:
        updates = dict(updates_by_id[record.id])
        # Author IDs resolved from a work are not yet on the record model, so
        # apply them to a temporary copy before selecting author metrics.
        probe = record.model_copy(update=updates) if updates else record
        author_updates = _apply_author_metrics(probe, author_metrics)
        updates.update(author_updates)
        enriched.append(record.model_copy(update=updates) if updates else record)
    return enriched
