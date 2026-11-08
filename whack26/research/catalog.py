"""Filter and similarity queries over the document catalog.

The catalog lives in the SQLite schema owned by :class:`whack26.store.FieldStore`
(``documents``, ``document_fields``, and ``fields``).  This module keeps the
query surface small: build a :class:`CatalogQuery`, call
:func:`search_documents`, and get a deterministic page of ``Document`` objects.

Filters run as SQL predicates wherever the schema supports them.  Year and
vector ordering are evaluated in Python because ``published_at`` is a free-form
text column and summary vectors are stored as compact BLOBs.

``similar_to`` may be either a document id (its stored summary embedding is the
reference) or free text (``embeddings`` is then required).  The reference
document itself is excluded from the result, matching ``FieldStore.neighbors``.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from pydantic import BaseModel, Field

from ..common import redact_secrets
from ..embeddings import cosine
from .scrape.models import Document

__all__ = ["CatalogError", "CatalogQuery", "search_documents"]

_YEAR_RE = re.compile(r"\s*(-?\d{4})")


class CatalogError(RuntimeError):
    """Raised when a catalog query cannot be evaluated."""


class CatalogQuery(BaseModel):
    """A declarative document-catalog query."""

    kinds: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    field_id: str | None = None
    include_descendants: bool = True
    year_min: int | None = None
    year_max: int | None = None
    credible_only: bool = False
    preloaded: bool | None = None
    text: str | None = None
    similar_to: str | None = None
    limit: int = Field(default=20, ge=0)
    offset: int = Field(default=0, ge=0)


def _normalized_values(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value).strip().casefold()
        if item and item not in result:
            result.append(item)
    return result


def _candidate_documents(store: Any, query: CatalogQuery) -> list[Document]:
    """Return SQL-filtered documents, ordered by id for stability."""
    conn_getter = getattr(store, "_connection", None)
    row_converter = getattr(store, "_row_to_document", None)
    if not callable(conn_getter) or not callable(row_converter):
        raise CatalogError(
            "search_documents requires a FieldStore-compatible document store"
        )

    params: list[Any] = []
    clauses: list[str] = []
    cte = ""

    if query.field_id:
        if query.include_descendants:
            cte = (
                "WITH RECURSIVE subtree(id) AS ("
                "SELECT id FROM fields WHERE id = ? "
                "UNION ALL "
                "SELECT f.id FROM fields AS f "
                "JOIN subtree AS s ON f.parent_id = s.id"
                ") "
            )
            params.append(query.field_id)
            clauses.append(
                "d.id IN (SELECT df.document_id FROM document_fields AS df "
                "WHERE df.field_id IN (SELECT id FROM subtree))"
            )
        else:
            params.append(query.field_id)
            clauses.append(
                "d.id IN (SELECT document_id FROM document_fields "
                "WHERE field_id = ?)"
            )

    kinds = _normalized_values(query.kinds)
    if kinds:
        placeholders = ", ".join("?" for _ in kinds)
        clauses.append(f"d.kind IN ({placeholders})")
        params.extend(kinds)

    sources = _normalized_values(query.sources)
    if sources:
        placeholders = ", ".join("?" for _ in sources)
        clauses.append(f"d.source IN ({placeholders})")
        params.extend(sources)

    if query.credible_only:
        clauses.append("d.credible = 1")

    if query.preloaded is not None:
        clauses.append("d.preloaded = ?")
        params.append(int(bool(query.preloaded)))

    if query.text is not None and query.text.strip():
        needle = query.text.strip()
        clauses.append(
            "(instr(lower(d.title), lower(?)) > 0 "
            "OR instr(lower(d.summary), lower(?)) > 0)"
        )
        params.extend((needle, needle))

    sql = cte + "SELECT DISTINCT d.* FROM documents AS d"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY d.id"

    rows = conn_getter().execute(sql, params).fetchall()
    return [row_converter(row) for row in rows]


def _document_year(document: Document) -> int | None:
    """Return the publication year from ``published_at`` or metadata."""
    if document.published_at:
        match = _YEAR_RE.match(str(document.published_at))
        if match:
            return int(match.group(1))
    metadata = document.metadata or {}
    raw = metadata.get("year")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    if isinstance(raw, str):
        match = _YEAR_RE.match(raw)
        if match:
            return int(match.group(1))
    return None


def _reference_embedding(
    store: Any,
    query: CatalogQuery,
    embeddings: Any | None,
) -> tuple[str | None, list[float]]:
    """Resolve ``similar_to`` to a reference id (when it is a document) and vector."""
    assert query.similar_to is not None
    getter = getattr(store, "get_document", None)
    if callable(getter):
        target = getter(query.similar_to)
        if target is not None:
            vector = list(target.summary_embedding or [])
            if not vector:
                raise CatalogError(
                    redact_secrets(
                        "similar_to document has no summary embedding: "
                        f"{query.similar_to}"
                    )
                )
            return target.id, vector

    if embeddings is None:
        raise CatalogError(
            redact_secrets(
                f"similar_to {query.similar_to!r} is not a stored document and "
                "no embedding client was supplied"
            )
        )
    try:
        vectors = embeddings.embed([query.similar_to])
    except Exception as exc:  # noqa: BLE001 - surface a catalog error
        raise CatalogError(
            redact_secrets(
                f"could not embed similar_to text: {type(exc).__name__}: {exc}"
            )
        ) from exc
    if not vectors or not vectors[0]:
        raise CatalogError("embedding client returned no vector for similar_to")
    return None, list(vectors[0])


def _sort_key(
    document: Document,
    *,
    similarity: float,
) -> tuple[float, float, float, str]:
    year = _document_year(document)
    return (
        -similarity,
        -float(document.credibility_score),
        -(float(year) if year is not None else -1e12),
        document.id,
    )


def search_documents(
    store: Any,
    query: CatalogQuery,
    embeddings: Any | None = None,
) -> list[Document]:
    """Return the documents matching ``query`` in deterministic order.

    Ordering is similarity (descending, only when ``similar_to`` is supplied),
    credibility score (descending), publication year (descending, unknown last),
    and finally document id (ascending).  Pagination is applied after ordering.
    """
    if query.limit == 0:
        return []

    documents = _candidate_documents(store, query)

    if query.year_min is not None or query.year_max is not None:
        filtered: list[Document] = []
        for document in documents:
            year = _document_year(document)
            if year is None:
                continue
            if query.year_min is not None and year < query.year_min:
                continue
            if query.year_max is not None and year > query.year_max:
                continue
            filtered.append(document)
        documents = filtered

    reference_id: str | None = None
    similarities: dict[str, float] = {}
    if query.similar_to is not None:
        reference_id, reference = _reference_embedding(store, query, embeddings)
        for document in documents:
            if document.id == reference_id:
                similarities[document.id] = 1.0
                continue
            if not document.summary_embedding:
                similarities[document.id] = 0.0
                continue
            try:
                similarities[document.id] = cosine(
                    reference, document.summary_embedding
                )
            except (ValueError, TypeError):
                similarities[document.id] = 0.0
        documents = [
            document for document in documents if document.id != reference_id
        ]

    documents.sort(
        key=lambda document: _sort_key(
            document,
            similarity=similarities.get(document.id, 0.0),
        )
    )
    return documents[query.offset : query.offset + query.limit]
