"""Parent-field, statistics, and research context for hypothesis generation.

The generator should not have to rediscover the dataset summary, the field
hierarchy, sibling/ancestor research, the descriptive screen, or the existing
factor/model-search output.  :func:`build_context_pack` assembles one bounded,
deterministic bundle of those inputs.

The pack is deliberately defensive about two things:

* **Privacy** - every free-text field passes through the same PII/secret
  redaction as :mod:`whack26.hypotheses.variants`, so a summary or retrieved
  chunk that contains an email, phone number, IP address, or bearer token never
  reaches an LLM prompt.
* **Size** - rendering is capped by :data:`CONTEXT_MAX_CHARS`.  Truncation is
  deterministic, so repeated runs with the same inputs produce byte-identical
  packs.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..critic.corpus import Corpus, CorpusChunk
from ..embeddings import cosine
from ..factors import Factor
from ..model_search.types import ModelCandidate, ModelSearchResult
from ..profile import DatasetProfile
from ..research.scrape.models import Document
from ..screen import ScreenSummary
from ..store import Field as StoreField
from .variants import redact_text

__all__ = [
    "CONTEXT_MAX_CHARS",
    "ContextDocument",
    "ContextPack",
    "ResearchModelRef",
    "build_context_pack",
    "documents_for_fields",
    "render_context_pack",
]

CONTEXT_MAX_CHARS = 20_000

_MAX_SUMMARY_CHARS = 6_000
_MAX_ANCESTOR_SUMMARY_CHARS = 1_200
_MAX_ANCESTOR_DOCUMENTS = 6
_MAX_FACTORS = 20
_MAX_RESEARCH_MODELS = 10
_MAX_MODEL_CANDIDATES = 5
_MAX_CRITIC_CHUNKS = 8
_MAX_CHUNK_CHARS = 1_200
_MAX_FIELD_PITFALLS = 8
_MAX_FIELD_PITFALL_CHARS = 500

_FIELD_PITFALL_SCOPES = ("leaf", "ancestor")
_FIELD_PITFALL_APPLICABILITY = ("yes", "unknown", "no")

_RESEARCH_MODEL_FIELDS = (
    "document_id",
    "family",
    "outcome",
    "predictors",
    "target_transform",
    "feature_transforms",
    "interactions",
    "assumptions",
    "reported_metrics",
    "evidence",
    "mapped_columns",
    "confidence",
    "applied",
)


class ContextDocument(BaseModel):
    """One sanitized ancestor-document summary."""

    document_id: str
    title: str
    summary: str = ""
    source: str = ""
    field_id: str | None = None
    field_name: str | None = None
    credibility: float = 0.0
    year: int | None = None
    similarity: float | None = None
    authors: list[str] = Field(default_factory=list)
    canonical_url: str | None = None


class ResearchModelRef(BaseModel):
    """Duck-typed mirror of ``research.to_model.ResearchModelSpec``.

    The extraction module may not exist yet, so the context pack accepts any
    object or mapping with these attributes and normalizes it here.  Every field
    is optional with a safe default.
    """

    document_id: str = ""
    family: str = ""
    outcome: str = ""
    predictors: list[str] = Field(default_factory=list)
    target_transform: str = "none"
    feature_transforms: dict[str, str] = Field(default_factory=dict)
    interactions: list[tuple[str, str]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    reported_metrics: dict[str, float] = Field(default_factory=dict)
    evidence: str = ""
    mapped_columns: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    applied: bool = False


class ContextPack(BaseModel):
    summary: str = ""
    field_path: list[str] = Field(default_factory=list)
    ancestor_documents: list[ContextDocument] = Field(default_factory=list)
    ancestor_stats: dict[str, Any] = Field(default_factory=dict)
    factors: list[Factor] = Field(default_factory=list)
    research_models: list[ResearchModelRef] = Field(default_factory=list)
    model_candidates: list[ModelCandidate] = Field(default_factory=list)
    screen: dict[str, Any] = Field(default_factory=dict)
    critic_chunks: list[str] = Field(default_factory=list)
    field_pitfalls: list[dict[str, Any]] = Field(default_factory=list)
    token_estimate: int = 0


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _clean(text: str | None, *, limit: int | None = None) -> str:
    value = redact_text(text or "").strip()
    if limit is not None and len(value) > limit:
        value = value[: max(0, limit - 3)].rstrip() + "..."
    return value


def _redact_object(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_object(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_object(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _redact_object(item) for key, item in value.items()}
    return value


def _fallback_summary(profile: DatasetProfile) -> str:
    target = profile.target.name if profile.target is not None else "none"
    return (
        f"Dataset: {profile.source_path}; rows={profile.rows}; "
        f"columns={len(profile.columns)}; target={target}"
    )


def _field_entry(field: StoreField) -> dict[str, Any]:
    return {
        "field_id": field.id,
        "name": field.name,
        "level": field.level,
        "member_count": field.member_count,
    }


def _resolve_field_path(
    store: Any,
    dataset_id: str | None,
    field_path: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if field_path is not None:
        parent_id: str | None = None
        for item in field_path:
            if isinstance(item, StoreField):
                entries.append(_field_entry(item))
                parent_id = item.id
            elif isinstance(item, str):
                resolved = store.get_field(item) if store is not None else None
                if resolved is None and store is not None:
                    try:
                        resolved = store.find_sibling(parent_id, item)
                    except Exception:  # noqa: BLE001 - fake stores may differ
                        resolved = None
                if resolved is not None:
                    entries.append(_field_entry(resolved))
                    parent_id = resolved.id
                else:
                    entries.append(
                        {
                            "field_id": None,
                            "name": item,
                            "level": None,
                            "member_count": 0,
                        }
                    )
                    parent_id = None
            elif isinstance(item, Mapping):
                entry = {
                    "field_id": item.get("field_id") or item.get("id"),
                    "name": str(item.get("name") or ""),
                    "level": item.get("level"),
                    "member_count": int(item.get("member_count") or 0),
                }
                entries.append(entry)
                parent_id = entry["field_id"]
        return entries

    if store is None or not dataset_id:
        return entries

    try:
        fields = store.dataset_path(dataset_id)
    except Exception:  # noqa: BLE001 - a broken path must not block context
        return entries
    return [_field_entry(field) for field in fields]


def _dataset_record(store: Any, dataset_id: str | None) -> dict[str, Any] | None:
    if store is None or not dataset_id:
        return None
    try:
        return store.get_dataset(dataset_id)
    except Exception:  # noqa: BLE001
        return None


def _dataset_embedding(record: Mapping[str, Any] | None) -> list[float] | None:
    if not record:
        return None
    embedding = record.get("embedding")
    if isinstance(embedding, list) and embedding:
        return embedding
    return None


def _document_year(document: Document) -> int | None:
    raw = document.published_at
    if raw:
        match = re.search(r"(?:19|20)\d{2}", str(raw))
        if match:
            return int(match.group(0))
    for key in ("year", "publication_year", "published_year"):
        value = document.metadata.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def documents_for_fields(
    store: Any,
    field_ids: Sequence[str],
    *,
    exclude_document_id: str | None = None,
) -> list[tuple[Document, str]]:
    """Return ``(document, matched_field_id)`` for documents in these fields.

    The store has no public "documents by field" query yet, so this helper uses
    the existing read-only SQLite connection.  Dataset rows are excluded; the
    ancestor context is about research documents.
    """
    ids = [str(field_id) for field_id in field_ids if field_id]
    if not ids:
        return []
    try:
        conn = store._connection()  # noqa: SLF001 - read-only catalog query
    except Exception:  # noqa: BLE001 - fake stores may not expose a connection
        return []

    placeholders = ", ".join("?" for _ in ids)
    params: list[object] = list(ids)
    exclude = ""
    if exclude_document_id:
        exclude = " AND d.id != ?"
        params.append(exclude_document_id)
    rows = conn.execute(
        "SELECT df.field_id AS matched_field_id, d.* "
        "FROM document_fields AS df "
        "JOIN documents AS d ON d.id = df.document_id "
        f"WHERE df.field_id IN ({placeholders}) AND d.kind != 'dataset'{exclude} "
        "ORDER BY d.id, df.ordinal",
        params,
    ).fetchall()

    field_order = {field_id: index for index, field_id in enumerate(ids)}
    selected: dict[str, tuple[Document, str]] = {}
    for row in rows:
        try:
            document = store._row_to_document(row)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            continue
        matched = str(row["matched_field_id"])
        current = selected.get(document.id)
        if current is None:
            selected[document.id] = (document, matched)
        else:
            current_order = field_order.get(current[1], len(field_order))
            new_order = field_order.get(matched, len(field_order))
            if new_order < current_order:
                selected[document.id] = (document, matched)
    return list(selected.values())


def _field_document_counts(store: Any, field_ids: Sequence[str]) -> dict[str, int]:
    ids = [str(field_id) for field_id in field_ids if field_id]
    if not ids:
        return {}
    try:
        conn = store._connection()  # noqa: SLF001 - read-only catalog query
    except Exception:  # noqa: BLE001
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT df.field_id, COUNT(DISTINCT df.document_id) AS n "
        "FROM document_fields AS df "
        "JOIN documents AS d ON d.id = df.document_id "
        f"WHERE df.field_id IN ({placeholders}) AND d.kind != 'dataset' "
        "GROUP BY df.field_id",
        list(ids),
    ).fetchall()
    return {str(row["field_id"]): int(row["n"]) for row in rows}


def _context_document(
    document: Document,
    *,
    field_id: str | None,
    field_name: str | None,
    similarity: float | None,
) -> ContextDocument:
    return ContextDocument(
        document_id=document.id,
        title=_clean(document.title),
        summary=_clean(document.summary, limit=_MAX_ANCESTOR_SUMMARY_CHARS),
        source=str(document.source or ""),
        field_id=field_id,
        field_name=field_name,
        credibility=float(document.credibility_score or 0.0),
        year=_document_year(document),
        similarity=None if similarity is None else round(float(similarity), 6),
        authors=[_clean(author) for author in document.authors[:6]],
        canonical_url=_clean(document.canonical_url) or None,
    )


def _rank_ancestor_documents(
    entries: Sequence[tuple[Document, str | None]],
    *,
    field_names: Mapping[str, str],
    dataset_embedding: list[float] | None,
    limit: int,
    exclude_document_id: str | None,
) -> list[ContextDocument]:
    scored: list[ContextDocument] = []
    seen: set[str] = set()
    for document, field_id in entries:
        if document.id in seen or document.id == exclude_document_id:
            continue
        seen.add(document.id)
        similarity: float | None = None
        if dataset_embedding and document.summary_embedding:
            try:
                similarity = cosine(dataset_embedding, document.summary_embedding)
            except (TypeError, ValueError):
                similarity = None
        scored.append(
            _context_document(
                document,
                field_id=field_id,
                field_name=field_names.get(field_id) if field_id else None,
                similarity=similarity,
            )
        )

    scored.sort(
        key=lambda document: (
            -(document.similarity if document.similarity is not None else -1.0),
            -document.credibility,
            -(document.year or 0),
            document.document_id,
        )
    )
    return scored[: max(0, limit)]


def _target_stats(
    profile: DatasetProfile,
    screen: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if screen:
        target = screen.get("target")
        if isinstance(target, Mapping):
            return _redact_object(dict(target))
    target_name = profile.target.name if profile.target is not None else None
    column = next(
        (item for item in profile.columns if item.name == target_name),
        None,
    )
    if target_name is None:
        return {"kind": "none", "name": None}
    kind = "numeric" if column is not None and column.numeric is not None else "categorical"
    stats: dict[str, Any] = {
        "kind": kind,
        "name": target_name,
        "rows": profile.rows,
    }
    if column is not None:
        stats["missing_pct"] = column.missing_pct
        stats["cardinality"] = column.cardinality
        if column.numeric is not None:
            stats["numeric"] = dict(column.numeric)
        if column.top_values:
            stats["top_values"] = [
                [_clean(str(label)), int(count)]
                for label, count in column.top_values[:10]
            ]
    return stats


def _sanitize_factor(factor: Factor) -> Factor:
    cleaned = factor.model_copy(deep=True)
    cleaned.name = _clean(cleaned.name)
    cleaned.description = _clean(cleaned.description)
    cleaned.rationale = _clean(cleaned.rationale)
    return cleaned


def _factors_by_field(
    documents: Sequence[tuple[Document, str | None]],
    factors: Sequence[Factor],
) -> dict[str, list[str]]:
    """Map each ancestor field to the factors supported by its documents."""
    document_fields = {
        document.id: field_id
        for document, field_id in documents
        if field_id is not None
    }
    by_field: dict[str, list[str]] = {}
    for factor in factors:
        matched = sorted(
            {
                document_fields[evidence_id]
                for evidence_id in factor.evidence_ids
                if evidence_id in document_fields
            }
        )
        for field_id in matched:
            names = by_field.setdefault(field_id, [])
            if factor.name not in names and len(names) < 5:
                names.append(factor.name)
    return {field_id: by_field[field_id] for field_id in sorted(by_field)}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        except Exception:  # noqa: BLE001
            pass
    return {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return []


def _interactions(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not isinstance(value, (list, tuple)):
        return pairs
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def _coerce_research_model(value: Any) -> ResearchModelRef:
    mapping = _as_mapping(value)
    # Stored rows wrap the spec in ``spec`` and may also carry mapping/evidence.
    spec = mapping.get("spec")
    if isinstance(spec, Mapping):
        merged = dict(spec)
        for key in ("document_id", "mapping", "evidence", "confidence", "applied"):
            if key in mapping:
                merged.setdefault(key, mapping[key])
        mapping = merged

    def attribute(name: str, default: Any) -> Any:
        if name in mapping:
            return mapping[name]
        return getattr(value, name, default)

    feature_transforms: dict[str, str] = {}
    raw_transforms = attribute("feature_transforms", {})
    if isinstance(raw_transforms, Mapping):
        feature_transforms = {
            str(key): _clean(str(item)) for key, item in raw_transforms.items()
        }

    reported_metrics: dict[str, float] = {}
    raw_metrics = attribute("reported_metrics", {})
    if isinstance(raw_metrics, Mapping):
        for key, item in raw_metrics.items():
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                reported_metrics[str(key)] = number

    try:
        confidence = float(attribute("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0

    return ResearchModelRef(
        document_id=str(attribute("document_id", "") or ""),
        family=str(attribute("family", "") or ""),
        outcome=str(attribute("outcome", "") or ""),
        predictors=[_clean(item) for item in _string_list(attribute("predictors", []))],
        target_transform=str(attribute("target_transform", "none") or "none"),
        feature_transforms=feature_transforms,
        interactions=_interactions(attribute("interactions", [])),
        assumptions=[_clean(item) for item in _string_list(attribute("assumptions", []))],
        reported_metrics=reported_metrics,
        evidence=_clean(str(attribute("evidence", "") or ""), limit=1_200),
        mapped_columns=[
            _clean(item) for item in _string_list(attribute("mapped_columns", []))
        ],
        confidence=round(confidence, 6),
        applied=bool(attribute("applied", False)),
    )


def _research_models(values: Sequence[Any]) -> list[ResearchModelRef]:
    models = [_coerce_research_model(value) for value in values]
    models.sort(
        key=lambda model: (
            -model.confidence,
            model.document_id,
            model.family,
            model.outcome,
        )
    )
    return models[:_MAX_RESEARCH_MODELS]


def _coerce_field_pitfall(value: Any) -> dict[str, Any] | None:
    """Normalize one pitfall record from a model, mapping, or object.

    Free text is redacted and bounded here so the rendered pack never depends on
    the caller having sanitized its records.  Records without pitfall text are
    dropped: they carry no context for the generator.
    """
    mapping = _as_mapping(value)

    def attribute(name: str, default: Any = None) -> Any:
        if name in mapping:
            return mapping[name]
        return getattr(value, name, default)

    def text(name: str, *, limit: int = _MAX_FIELD_PITFALL_CHARS) -> str:
        raw = attribute(name, "")
        return _clean(None if raw is None else str(raw), limit=limit)

    pitfall = text("pitfall")
    if not pitfall:
        return None

    scope = text("scope", limit=40).casefold()
    if scope not in _FIELD_PITFALL_SCOPES:
        scope = "ancestor"
    applies = text("applies_to_dataset", limit=40).casefold()
    if applies not in _FIELD_PITFALL_APPLICABILITY:
        applies = "unknown"

    try:
        field_level = int(float(attribute("field_level", 1)))
    except (TypeError, ValueError, OverflowError):
        field_level = 1
    try:
        confidence = float(attribute("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0

    return {
        "id": text("id", limit=120),
        "field_id": text("field_id", limit=120) or None,
        "field_name": text("field_name", limit=160),
        "field_level": field_level,
        "scope": scope,
        "category": text("category", limit=40).casefold() or "other",
        "pitfall": pitfall,
        "applies_to_dataset": applies,
        "applicability_reason": text("applicability_reason"),
        "check": text("check"),
        "evidence_id": text("evidence_id", limit=160),
        "source": text("source", limit=160),
        "url": text("url"),
        "confidence": round(confidence, 6),
    }


def _field_pitfalls(values: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Return bounded, leaf-first field pitfalls for the context pack."""
    records: list[dict[str, Any]] = []
    for value in values or ():
        record = _coerce_field_pitfall(value)
        if record is None or record["applies_to_dataset"] == "no":
            continue
        records.append(record)
    records.sort(
        key=lambda record: (
            0 if record["scope"] == "leaf" else 1,
            -record["field_level"],
            -record["confidence"],
            record["id"],
        )
    )
    return records[:_MAX_FIELD_PITFALLS]


def _field_pitfall_text(pitfall: Mapping[str, Any], name: str) -> str:
    return _clean(str(pitfall.get(name) or ""), limit=_MAX_FIELD_PITFALL_CHARS)


def _coerce_candidates(
    value: ModelSearchResult | Sequence[ModelCandidate] | None,
) -> list[ModelCandidate]:
    if value is None:
        return []
    if isinstance(value, ModelSearchResult):
        candidates = list(value.candidates)
        if value.selected is not None and all(
            candidate is not value.selected for candidate in candidates
        ):
            candidates.append(value.selected)
    elif isinstance(value, Mapping):
        raw = value.get("candidates") or []
        candidates = [
            item
            if isinstance(item, ModelCandidate)
            else ModelCandidate.model_validate(item)
            for item in raw
        ]
    else:
        candidates = list(value)
    return candidates


def _sanitize_candidate(candidate: ModelCandidate) -> ModelCandidate:
    cleaned = candidate.model_copy(deep=True)
    cleaned.warnings = [_clean(warning) for warning in cleaned.warnings]
    return cleaned


def _model_candidates(
    value: ModelSearchResult | Sequence[ModelCandidate] | None,
    *,
    limit: int,
) -> list[ModelCandidate]:
    candidates = [_sanitize_candidate(item) for item in _coerce_candidates(value)]

    def key(candidate: ModelCandidate) -> tuple[Any, ...]:
        rank = candidate.rank
        if rank is None:
            rank = 0 if candidate.selected else 10**9
        return (
            int(rank),
            str(candidate.spec.family),
            tuple(candidate.spec.features),
            candidate.model_dump_json(),
        )

    candidates.sort(key=key)
    return candidates[: max(0, limit)]


def _screen_dict(screen: ScreenSummary | Mapping[str, Any] | None) -> dict[str, Any]:
    if screen is None:
        return {}
    if isinstance(screen, ScreenSummary):
        data = screen.model_dump(mode="json")
    elif isinstance(screen, Mapping):
        data = dict(screen)
    else:
        return {}
    redacted = _redact_object(data)
    return redacted if isinstance(redacted, dict) else {}


def _chunk_payload(value: Any) -> tuple[str, str]:
    if isinstance(value, CorpusChunk):
        return value.id, f"### {value.id} — {value.title}\n{value.text}"
    if isinstance(value, Mapping):
        identifier = str(value.get("id") or value.get("chunk_id") or "")
        title = str(value.get("title") or "")
        text = str(value.get("text") or value.get("body") or "")
        return identifier, f"### {identifier} — {title}\n{text}".strip()
    text = str(value)
    identifier = ""
    match = re.search(r"^###\s+(\S+)", text)
    if match:
        identifier = match.group(1)
    return identifier, text


def _preloaded_statistics_chunks(
    store: Any,
    *,
    dataset_id: str | None,
    embeddings: Any,
    limit: int = 3,
) -> list[str]:
    """Return summaries from the preloaded statistics/validity catalog.

    The static pitfalls corpus remains the fallback; this adds the layer-1
    documents that ``preload-statistics`` stored in the catalog.
    """
    if store is None or limit <= 0:
        return []
    try:
        from ..research.catalog import CatalogQuery, search_documents

        documents = search_documents(
            store,
            CatalogQuery(
                preloaded=True,
                credible_only=True,
                similar_to=dataset_id,
                limit=limit,
            ),
            embeddings=embeddings,
        )
    except Exception:  # noqa: BLE001 - preloaded context is advisory
        return []
    chunks: list[str] = []
    for document in documents:
        summary = _clean(document.summary, limit=_MAX_CHUNK_CHARS)
        if not summary:
            continue
        chunks.append(
            f"### {document.id} — {_clean(document.title)}\n"
            f"Statistics corpus document: {summary}"
        )
    return chunks


def _critic_chunks(
    *,
    corpus: Corpus | None,
    embeddings: Any,
    query: str,
    supplied: Sequence[Any],
    how_to_chunk: Any,
    top_k: int,
) -> list[str]:
    chunks: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        identifier, text = _chunk_payload(value)
        text = _clean(text, limit=_MAX_CHUNK_CHARS)
        if not text:
            return
        dedupe_key = identifier or text
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        chunks.append(text)

    retrieved: list[Any] = []
    if corpus is not None and embeddings is not None:
        try:
            retrieved, _warnings = corpus.retrieve(
                query, embeddings, top_k=top_k
            )
        except Exception:  # noqa: BLE001 - retrieval must not block a run
            retrieved = []

    how_to = how_to_chunk
    if how_to is None and corpus is not None:
        how_to = corpus.get("S0-how-to-use")
    add(how_to)
    for chunk in retrieved:
        add(chunk)
    for chunk in supplied:
        add(chunk)
    return chunks[:_MAX_CRITIC_CHUNKS]


# ---------------------------------------------------------------------------
# Rendering and truncation
# ---------------------------------------------------------------------------


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _render_pack(pack: ContextPack) -> str:
    blocks: list[str] = []
    blocks.append("## Dataset summary\n" + (pack.summary or "(none)"))
    blocks.append(
        "## Field path\n" + (" > ".join(pack.field_path) or "(none)")
    )

    if pack.ancestor_documents:
        lines = ["## Ancestor documents"]
        for document in pack.ancestor_documents:
            similarity = (
                "n/a"
                if document.similarity is None
                else f"{document.similarity:.3f}"
            )
            lines.append(
                f"- [{document.document_id}] {document.title} "
                f"(source={document.source or 'n/a'}, year={document.year}, "
                f"credibility={document.credibility:.2f}, "
                f"similarity={similarity}): {document.summary}"
            )
        blocks.append("\n".join(lines))
    else:
        blocks.append("## Ancestor documents\n(none)")

    blocks.append("## Ancestor statistics\n" + _json(pack.ancestor_stats))

    if pack.field_pitfalls:
        lines = [
            "## Field pitfalls",
            "Context only: `applies=unknown` is not a finding; only `yes` is a "
            "profile-established precondition.",
        ]
        for pitfall in pack.field_pitfalls:
            identifier = _field_pitfall_text(pitfall, "id") or "(none)"
            field_name = _field_pitfall_text(pitfall, "field_name") or "(unnamed)"
            level = pitfall.get("field_level", 1)
            scope = _field_pitfall_text(pitfall, "scope") or "ancestor"
            category = _field_pitfall_text(pitfall, "category") or "other"
            applies = (
                _field_pitfall_text(pitfall, "applies_to_dataset") or "unknown"
            )
            pitfall_text = _field_pitfall_text(pitfall, "pitfall") or "(none)"
            reason = _field_pitfall_text(pitfall, "applicability_reason")
            check = _field_pitfall_text(pitfall, "check") or "(none)"
            evidence = _field_pitfall_text(pitfall, "evidence_id") or "(none)"
            url = _field_pitfall_text(pitfall, "url")
            evidence_line = f"  evidence: {evidence}"
            if url:
                evidence_line += f" | url: {url}"
            reason_line = f"  why: {reason}\n" if reason else ""
            lines.append(
                f"- [{identifier}] {field_name} | level={level} | scope={scope} | "
                f"category={category} | applies={applies}\n"
                f"  pitfall: {pitfall_text}\n"
                f"{reason_line}"
                f"  check: {check}\n"
                f"{evidence_line}"
            )
        blocks.append("\n".join(lines))

    if pack.factors:
        lines = ["## Factors"]
        for factor in pack.factors:
            lines.append(
                f"- [{factor.id or '(none)'}] {factor.name} | "
                f"direction={factor.direction} | claim={factor.claim_type} | "
                f"mapped={factor.mapped_column or '(unmapped)'} | "
                f"evidence={', '.join(factor.evidence_ids) or '(none)'}\n"
                f"  {factor.description}"
            )
        blocks.append("\n".join(lines))
    else:
        blocks.append("## Factors\n(none)")

    if pack.research_models:
        lines = ["## Research models"]
        for model in pack.research_models:
            lines.append(
                f"- [{model.document_id}] family={model.family} | "
                f"outcome={model.outcome} | "
                f"predictors={', '.join(model.predictors) or '(none)'} | "
                f"confidence={model.confidence:.3f} | applied={model.applied} | "
                f"mapped={', '.join(model.mapped_columns) or '(none)'}\n"
                f"  evidence: {model.evidence or '(none)'}"
            )
        blocks.append("\n".join(lines))
    else:
        blocks.append("## Research models\n(none)")

    if pack.model_candidates:
        lines = ["## Model candidates"]
        for candidate in pack.model_candidates:
            diagnostics = _json(
                [diagnostic.model_dump(mode="json") for diagnostic in candidate.diagnostics]
            )
            lines.append(
                f"- rank={candidate.rank} family={candidate.spec.family} "
                f"selected={candidate.selected} "
                f"features={', '.join(candidate.spec.features) or '(none)'} "
                f"metrics={_json(candidate.metrics)} diagnostics={diagnostics}"
            )
        blocks.append("\n".join(lines))
    else:
        blocks.append("## Model candidates\n(none)")

    blocks.append("## Descriptive screen\n" + _json(pack.screen))

    if pack.critic_chunks:
        blocks.append(
            "## Critic chunks\n" + "\n\n".join(pack.critic_chunks)
        )
    else:
        blocks.append("## Critic chunks\n(none)")

    return "\n\n".join(blocks)


def render_context_pack(
    pack: ContextPack,
    *,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """Render the pack as labeled prompt text, never exceeding ``max_chars``."""
    text = _render_pack(pack)
    limit = max(1, int(max_chars))
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _shrink_mapping(value: dict[str, Any]) -> None:
    """Drop the least important key of a stats/screen mapping."""
    for key in (
        "top_factors",
        "factors_by_field",
        "fields",
        "documents_per_field",
        "group_means",
        "top_correlations",
        "spearman",
        "pearson",
        "missingness",
        "categorical_columns",
        "numeric_columns",
        "target",
        "counts",
    ):
        if key in value:
            value.pop(key)
            return
    if value:
        value.pop(next(iter(value)))


def _fit_to_cap(pack: ContextPack, max_chars: int) -> None:
    limit = max(1, int(max_chars))
    summary_budget = max(120, int(limit * 0.35))
    if len(pack.summary) > summary_budget:
        pack.summary = _clean(pack.summary, limit=summary_budget)

    guard = 0
    while len(_render_pack(pack)) > limit and guard < 1_000:
        guard += 1
        if pack.critic_chunks:
            pack.critic_chunks.pop()
        elif pack.field_pitfalls:
            pack.field_pitfalls.pop()
        elif pack.model_candidates:
            pack.model_candidates.pop()
        elif pack.research_models:
            pack.research_models.pop()
        elif pack.ancestor_documents:
            pack.ancestor_documents.pop()
        elif pack.factors:
            pack.factors.pop()
        elif pack.ancestor_stats and len(pack.ancestor_stats) > 0:
            _shrink_mapping(pack.ancestor_stats)
        elif pack.screen:
            _shrink_mapping(pack.screen)
        elif len(pack.summary) > 0:
            pack.summary = (
                _clean(pack.summary, limit=max(40, len(pack.summary) * 3 // 4))
                if len(pack.summary) > 80
                else ""
            )
        else:
            break


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_context_pack(
    profile: DatasetProfile,
    summary: str | None = None,
    store: Any = None,
    dataset_id: str | None = None,
    field_path: Sequence[Any] | None = None,
    ancestor_documents: Sequence[Document] | None = None,
    factors: Sequence[Factor] = (),
    research_models: Sequence[Any] = (),
    model_candidates: (
        ModelSearchResult | Sequence[ModelCandidate] | Mapping[str, Any] | None
    ) = None,
    screen: ScreenSummary | Mapping[str, Any] | None = None,
    critic_chunks: Sequence[Any] = (),
    how_to_chunk: Any = None,
    corpus: Corpus | None = None,
    embeddings: Any = None,
    max_chars: int = CONTEXT_MAX_CHARS,
    ancestor_limit: int = _MAX_ANCESTOR_DOCUMENTS,
    critic_top_k: int = 4,
    model_candidate_limit: int = _MAX_MODEL_CANDIDATES,
    path_names: Sequence[str] | None = None,
    ancestor_stats: Mapping[str, Any] | None = None,
    field_pitfalls: Sequence[Any] = (),
) -> ContextPack:
    """Build the bounded, deterministic context pack for the generator.

    ``store``/``dataset_id`` are optional so unit tests and the agent loop can
    pass pre-fetched ``ancestor_documents`` instead.  When both are supplied,
    the dataset summary, field path, and ancestor documents are read from the
    store; the caller-provided values win.

    ``path_names`` is accepted as a convenience alias for a name-only
    ``field_path``; ``ancestor_stats`` overrides the computed statistics.
    ``field_pitfalls`` accepts field-scoped records (models, mappings, or
    objects) and renders the bounded advisory section; missing or invalid
    records are ignored.
    """
    if field_path is None and path_names is not None:
        field_path = list(path_names)

    record = _dataset_record(store, dataset_id)
    if summary is None and record is not None:
        summary = str(record.get("summary") or "")
    if summary is None:
        summary = _fallback_summary(profile)

    entries = _resolve_field_path(store, dataset_id, field_path)
    path_ids = [
        str(entry["field_id"]) for entry in entries if entry.get("field_id")
    ]
    path_names = [str(entry.get("name") or "") for entry in entries]
    field_names = {
        str(entry["field_id"]): str(entry.get("name") or "")
        for entry in entries
        if entry.get("field_id")
    }

    dataset_embedding = _dataset_embedding(record)
    raw_documents: list[tuple[Document, str | None]]
    field_counts: dict[str, int] = {}
    if ancestor_documents is not None:
        raw_documents = [
            (
                document,
                str(document.metadata.get("field_id"))
                if document.metadata.get("field_id")
                else None,
            )
            for document in ancestor_documents
        ]
    elif store is not None and path_ids:
        fetched = documents_for_fields(
            store,
            path_ids,
            exclude_document_id=dataset_id,
        )
        raw_documents = [(document, field_id) for document, field_id in fetched]
        field_counts = _field_document_counts(store, path_ids)
    else:
        raw_documents = []

    ranked_documents = _rank_ancestor_documents(
        raw_documents,
        field_names=field_names,
        dataset_embedding=dataset_embedding,
        limit=ancestor_limit,
        exclude_document_id=dataset_id,
    )

    cleaned_factors = [_sanitize_factor(factor) for factor in factors]
    cleaned_factors.sort(
        key=lambda factor: (
            factor.mapped_column or "",
            factor.name.casefold(),
            factor.id or "",
        )
    )
    cleaned_factors = cleaned_factors[:_MAX_FACTORS]

    top_factors = [
        {
            "id": factor.id,
            "name": factor.name,
            "direction": factor.direction,
            "claim_type": factor.claim_type,
            "mapped_column": factor.mapped_column,
            "evidence_ids": list(factor.evidence_ids),
        }
        for factor in cleaned_factors[:10]
    ]

    field_stats = []
    for entry in entries:
        field_id = entry.get("field_id")
        documents = field_counts.get(str(field_id), 0) if field_id else 0
        field_stats.append(
            {
                "field_id": field_id,
                "name": entry.get("name"),
                "level": entry.get("level"),
                "documents": int(documents),
                "member_count": int(entry.get("member_count") or 0),
            }
        )

    screen_data = _screen_dict(screen)
    computed_stats: dict[str, Any] = {
        "counts": {
            "fields": len(entries),
            "documents": len(ranked_documents),
        },
        "fields": field_stats,
        "documents_per_field": {
            str(entry["field_id"]): int(field_counts.get(str(entry["field_id"]), 0))
            for entry in entries
            if entry.get("field_id")
        },
        "target": _target_stats(profile, screen_data),
        "top_factors": top_factors,
        "factors_by_field": _factors_by_field(raw_documents, cleaned_factors),
    }
    if ancestor_stats is not None:
        computed_stats = {**computed_stats, **dict(ancestor_stats)}
    final_stats = _redact_object(computed_stats)

    context_chunks = _critic_chunks(
        corpus=corpus,
        embeddings=embeddings,
        query=summary,
        supplied=critic_chunks,
        how_to_chunk=how_to_chunk,
        top_k=critic_top_k,
    )
    context_chunks.extend(
        _preloaded_statistics_chunks(
            store,
            dataset_id=dataset_id,
            embeddings=embeddings,
            limit=3,
        )
    )
    pack = ContextPack(
        summary=_clean(summary, limit=_MAX_SUMMARY_CHARS),
        field_path=path_names,
        ancestor_documents=ranked_documents,
        ancestor_stats=final_stats,
        factors=cleaned_factors,
        research_models=_research_models(research_models),
        model_candidates=_model_candidates(
            model_candidates, limit=model_candidate_limit
        ),
        screen=screen_data,
        critic_chunks=context_chunks,
        field_pitfalls=_field_pitfalls(field_pitfalls),
    )
    _fit_to_cap(pack, max_chars)
    rendered = render_context_pack(pack, max_chars=max_chars)
    pack.token_estimate = int(math.ceil(len(rendered) / 4))
    return pack
