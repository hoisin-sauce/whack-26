"""Field-scoped pitfall search over the existing research sources.

The stage is deliberately opt-in: it adds one short query per field (leaf
first) and reuses the normal adapter/cache/credibility machinery.  It never
creates fields, never mutates the store, and never blocks a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from ..common import utc_now_iso
from ..config import Settings
from ..profile import DatasetProfile
from ..research.base import DEFAULT_USER_AGENT, ResearchRecord, SourceError
from ..research.cache import ResearchCache
from ..research.pipeline import build_adapters
from .extract import build_signals, extract_field_pitfalls
from .queries import build_field_queries, field_name
from .types import FieldPitfall, FieldPitfallQuery, FieldPitfallReport

_PITFALL_SOURCES: tuple[str, ...] = (
    "openalex",
    "arxiv",
    "semantic_scholar",
    "websearch",
)


def _dedupe_warnings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _search_query(
    query: FieldPitfallQuery,
    *,
    adapters: dict[str, Any],
    client: httpx.Client,
    settings: Settings,
    warnings: list[str],
) -> list[ResearchRecord]:
    """Try the requested source, then configured fallbacks, for one query."""
    order = [query.source]
    for source in settings.field_pitfalls_sources:
        if source not in order:
            order.append(source)
    for source in order:
        if source not in _PITFALL_SOURCES:
            continue
        adapter = adapters.get(source)
        if adapter is None:
            continue
        try:
            found = adapter.search(
                query.query,
                settings.field_pitfalls_records_per_query,
                client,
            )
        except SourceError as exc:
            warnings.append(f"field_pitfalls_source_failed: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one source must not kill the run
            warnings.append(
                f"field_pitfalls_source_failed: {source} query "
                f"{query.query!r}: {type(exc).__name__}: {exc}"
            )
            continue
        for adapter_warning in getattr(adapter, "warnings", []) or []:
            warnings.append(f"field_pitfalls_adapter_warning: {adapter_warning}")
        if found:
            return list(found)
    return []


def run_field_pitfalls(
    *,
    profile: DatasetProfile,
    summary: str,
    field_path: Sequence[Any],
    settings: Settings,
    existing_records: Sequence[ResearchRecord] = (),
    frame: Any | None = None,
    client: httpx.Client | None = None,
    cache_path: Path | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> FieldPitfallReport:
    """Search and extract field-scoped pitfalls for every field in the path.

    ``summary`` is accepted for call-site symmetry with research; it is not
    used because the field names and descriptions are the query inputs.
    """
    del summary  # the field path is the query surface
    fields = list(field_path)
    offline = settings.offline if offline is None else bool(offline)
    warnings: list[str] = []
    queries: list[FieldPitfallQuery] = []
    pitfalls: list[FieldPitfall] = []
    cache_stats: dict[str, Any] = {}

    if not fields:
        return FieldPitfallReport(
            generated_at=utc_now_iso(),
            field_path=[],
            warnings=["field_pitfalls_no_fields"],
            degraded=True,
            offline=offline,
        )

    signals = build_signals(profile, frame=frame)
    owned_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(settings.http_timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )

    cache = ResearchCache(
        cache_path or settings.research_db,
        offline=offline,
        refresh=refresh,
    )
    adapters: dict[str, Any] = {}
    if not settings.mock:
        try:
            adapters = build_adapters(
                settings,
                cache=cache,
                dataset_id=profile.dataset_id,
                offline=offline,
            )
        except Exception as exc:  # noqa: BLE001 - stage is advisory
            warnings.append(
                f"field_pitfalls_adapters_failed: {type(exc).__name__}: {exc}"
            )

    used_evidence: set[str] = set()
    try:
        for index, field in enumerate(reversed(fields)):
            field_queries = build_field_queries(
                [field],
                per_field=settings.field_pitfalls_max_queries_per_field,
                sources=settings.field_pitfalls_sources,
            )
            queries.extend(field_queries)
            name = field_name(field)
            records: list[ResearchRecord] = []
            if settings.mock:
                warnings.append(
                    "field_pitfalls_mock_records_only: external searches are "
                    "disabled in mock mode"
                )
            else:
                for query in field_queries:
                    records.extend(
                        _search_query(
                            query,
                            adapters=adapters,
                            client=client,
                            settings=settings,
                            warnings=warnings,
                        )
                    )
            if not records and existing_records:
                records = list(existing_records)
                if not settings.mock:
                    warnings.append(
                        f"field_pitfalls_existing_research_fallback: {name}"
                    )

            unique: list[ResearchRecord] = []
            seen: set[str] = set()
            for record in records:
                record_id = str(getattr(record, "id", "") or "")
                if not record_id or record_id in seen or record_id in used_evidence:
                    continue
                seen.add(record_id)
                unique.append(record)
            if not unique:
                warnings.append(f"field_pitfalls_no_records: {name}")
                continue

            extracted = extract_field_pitfalls(
                field,
                unique,
                profile,
                frame=frame,
                signals=signals,
                max_items=settings.field_pitfalls_max_per_field,
                scope="leaf" if index == 0 else "ancestor",
            )
            if not extracted:
                warnings.append(f"field_pitfalls_no_pitfalls: {name}")
                continue
            kept = 0
            for pitfall in extracted:
                # A pitfall the profile rules out is not useful context and
                # would only add noise to the critic and preview.
                if pitfall.applies_to_dataset == "no":
                    continue
                if pitfall.evidence_id in used_evidence:
                    continue
                used_evidence.add(pitfall.evidence_id)
                pitfalls.append(pitfall)
                kept += 1
            if kept == 0:
                warnings.append(f"field_pitfalls_no_applicable_pitfalls: {name}")
    finally:
        cache_stats = cache.stats()
        cache.close()
        if owned_client and client is not None:
            client.close()

    pitfalls.sort(
        key=lambda item: (
            0 if item.scope == "leaf" else 1,
            -item.field_level,
            -item.confidence,
            item.id,
        )
    )
    pitfalls = pitfalls[: settings.field_pitfalls_max_total]
    # Re-number after the cap so citation ids are contiguous and stable.
    for index, pitfall in enumerate(pitfalls, start=1):
        pitfall.id = f"F:{_citation_field(pitfall)}:{index}"

    degraded = bool(
        [
            warning
            for warning in warnings
            if warning.startswith(
                (
                    "field_pitfalls_source_failed",
                    "field_pitfalls_adapters_failed",
                )
            )
        ]
    )
    return FieldPitfallReport(
        generated_at=utc_now_iso(),
        field_path=[field_name(field) for field in fields],
        queries=queries,
        pitfalls=pitfalls,
        warnings=_dedupe_warnings(warnings),
        degraded=degraded,
        offline=offline,
        cache=cache_stats,
    )


def _citation_field(pitfall: Any) -> str:
    """Return the field component used in a pitfall citation id."""
    field = getattr(pitfall, "field_id", None)
    if field:
        return str(field)
    name = str(getattr(pitfall, "field_name", "") or "field")
    slug = "".join(
        character if character.isalnum() else "-" for character in name.lower()
    ).strip("-")
    return f"field-{slug[:40] or 'field'}"


__all__ = ["run_field_pitfalls"]
