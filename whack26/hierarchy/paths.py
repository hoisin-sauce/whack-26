"""Recompute and verify document paths through the field hierarchy.

The database stores a document's ancestry twice: as ``document_fields`` rows
(ordered root to leaf) and, for datasets, as ``datasets.path_json``.  Both must
always describe the same chain.  This module is the single place that rebuilds
those rows from ``documents.leaf_field_id`` and checks that they still agree.

``recompute_paths`` validates every affected document before it writes, so a
cycle or a dangling field reference leaves the stored paths untouched.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..store import Field, FieldStore
from .types import HierarchyIntegrityError

__all__ = ["recompute_paths", "verify_paths"]

_CHUNK_SIZE = 500

_PUNCTUATION = re.compile(r"[^\w\s]|_", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


@dataclass(frozen=True)
class _DocumentPlan:
    """A validated path rebuild, ready to be written."""

    document_id: str
    kind: str
    leaf_id: str
    path_ids: tuple[str, ...]
    confidence: float
    needs_leaf_write: bool


def _normalize_name(name: str) -> str:
    """Mirror the store's case/punctuation-insensitive sibling key."""
    spaced = _PUNCTUATION.sub(" ", name.casefold())
    return _WHITESPACE.sub(" ", spaced).strip()


def _chunks(values: Sequence[str], size: int = _CHUNK_SIZE) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _placeholders(values: Sequence[str]) -> str:
    return ", ".join("?" for _ in values)


def _fetch_documents(
    conn: sqlite3.Connection, document_ids: Sequence[str]
) -> dict[str, sqlite3.Row]:
    rows: dict[str, sqlite3.Row] = {}
    for chunk in _chunks(document_ids):
        for row in conn.execute(
            "SELECT id, kind, leaf_field_id FROM documents "
            f"WHERE id IN ({_placeholders(chunk)})",
            chunk,
        ).fetchall():
            rows[row["id"]] = row
    return rows


def _fetch_memberships(
    conn: sqlite3.Connection, document_ids: Sequence[str]
) -> dict[str, list[tuple[str, float, int]]]:
    rows: dict[str, list[tuple[str, float, int]]] = {}
    for chunk in _chunks(document_ids):
        for row in conn.execute(
            "SELECT document_id, field_id, confidence, ordinal FROM document_fields "
            f"WHERE document_id IN ({_placeholders(chunk)}) "
            "ORDER BY document_id, ordinal, field_id",
            chunk,
        ).fetchall():
            rows.setdefault(row["document_id"], []).append(
                (row["field_id"], float(row["confidence"]), int(row["ordinal"]))
            )
    return rows


def _fetch_datasets(
    conn: sqlite3.Connection, document_ids: Sequence[str]
) -> dict[str, tuple[str, float]]:
    rows: dict[str, tuple[str, float]] = {}
    for chunk in _chunks(document_ids):
        for row in conn.execute(
            "SELECT document_id, path_json, confidence FROM datasets "
            f"WHERE document_id IN ({_placeholders(chunk)})",
            chunk,
        ).fetchall():
            rows[row["document_id"]] = (row["path_json"], float(row["confidence"]))
    return rows


def _path_json_leaf(raw: str | None) -> str | None:
    """Return the last field id from a dataset path, or ``None`` if unusable."""
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, list) or not decoded:
        return None
    last = decoded[-1]
    if isinstance(last, str):
        return last
    if isinstance(last, dict):
        identifier = last.get("field_id") or last.get("id")
        if isinstance(identifier, str):
            return identifier
    return None


def _resolve_chain(
    field_map: dict[str, Field],
    leaf_id: str,
    *,
    cache: dict[str, tuple[Field, ...]],
) -> tuple[Field, ...]:
    """Walk ``parent_id`` from ``leaf_id`` to a root, rejecting bad trees."""
    cached = cache.get(leaf_id)
    if cached is not None:
        return cached

    chain: list[Field] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise HierarchyIntegrityError(
                f"cycle in field ancestry at {current!r} (leaf {leaf_id!r})"
            )
        field = field_map.get(current)
        if field is None:
            if not chain:
                raise HierarchyIntegrityError(
                    f"missing leaf field: {current!r}"
                )
            raise HierarchyIntegrityError(
                f"missing parent field: {current!r} "
                f"(parent of {chain[-1].id!r})"
            )
        seen.add(current)
        chain.append(field)
        current = field.parent_id

    resolved = tuple(reversed(chain))
    cache[leaf_id] = resolved
    return resolved


def _select_document_ids(
    store: FieldStore,
    *,
    field_ids: Iterable[str] | None,
    document_ids: Iterable[str] | None,
) -> list[str]:
    """Resolve the documents named directly or reachable from ``field_ids``."""
    conn = store._connection()

    if field_ids is None and document_ids is None:
        return [
            row["id"]
            for row in conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
        ]

    selected: set[str] = set()

    if document_ids is not None:
        requested = list(dict.fromkeys(document_ids))
        for chunk in _chunks(requested):
            rows = conn.execute(
                "SELECT id FROM documents "
                f"WHERE id IN ({_placeholders(chunk)})",
                chunk,
            ).fetchall()
            found = {row["id"] for row in rows}
            missing = [identifier for identifier in chunk if identifier not in found]
            if missing:
                raise HierarchyIntegrityError(
                    "unknown document id(s): " + ", ".join(sorted(missing))
                )
            selected.update(found)

    if field_ids is not None:
        requested_fields = set(field_ids)
        # A structural change names the fields that moved; documents attached
        # anywhere below them are affected too.
        children: dict[str | None, list[str]] = {}
        for field in store.all_fields():
            children.setdefault(field.parent_id, []).append(field.id)
        affected_fields = set(requested_fields)
        pending = list(requested_fields)
        while pending:
            current = pending.pop()
            for child_id in children.get(current, []):
                if child_id not in affected_fields:
                    affected_fields.add(child_id)
                    pending.append(child_id)

        ids = sorted(affected_fields)
        for chunk in _chunks(ids):
            placeholders = _placeholders(chunk)
            rows = conn.execute(
                "SELECT DISTINCT document_id AS id FROM document_fields "
                f"WHERE field_id IN ({placeholders}) "
                "UNION "
                "SELECT id FROM documents "
                f"WHERE leaf_field_id IN ({placeholders})",
                (*chunk, *chunk),
            ).fetchall()
            selected.update(row["id"] for row in rows)

    return sorted(selected)


def _confidence_for(
    leaf_id: str,
    memberships: Sequence[tuple[str, float, int]],
    dataset_confidence: float | None,
) -> float:
    for field_id, confidence, _ordinal in memberships:
        if field_id == leaf_id:
            return confidence
    if memberships:
        return memberships[-1][1]
    if dataset_confidence is not None:
        return dataset_confidence
    return 1.0


def recompute_paths(
    store: FieldStore,
    *,
    field_ids: Iterable[str] | None = None,
    document_ids: Iterable[str] | None = None,
    run_id: str | None = None,
) -> int:
    """Rebuild document paths from each document's leaf field.

    ``field_ids`` selects every document attached to one of those fields or to
    a descendant of one.  ``document_ids`` selects documents explicitly; when
    both are supplied their union is recomputed.  With neither, every document
    is considered.

    The leaf is ``documents.leaf_field_id``; when that is null it is derived
    from the highest ``document_fields.ordinal`` and then from the last
    ``datasets.path_json`` entry.  Documents with no resolvable leaf are left
    alone.  ``document_fields`` is replaced in root-to-leaf order, dataset
    ``path_json`` is rewritten, ``leaf_field_id`` is stored, and centroids for
    the old and new paths are recomputed.

    Raises :class:`HierarchyIntegrityError` if any affected leaf is missing,
    has a missing parent, or sits in a cycle.  All paths are validated before
    the first write.

    ``run_id`` is accepted for callers that audit a hierarchy operation; the
    operation log is written by the caller, not by this function.

    Returns the number of documents whose paths were rebuilt.
    """

    del run_id  # reserved for the caller's audit record

    selected = _select_document_ids(
        store, field_ids=field_ids, document_ids=document_ids
    )
    if not selected:
        return 0

    conn = store._connection()
    documents = _fetch_documents(conn, selected)
    memberships = _fetch_memberships(conn, selected)
    datasets = _fetch_datasets(conn, selected)

    field_map = {field.id: field for field in store.all_fields()}
    chain_cache: dict[str, tuple[Field, ...]] = {}
    plans: list[_DocumentPlan] = []

    for document_id in sorted(documents):
        row = documents[document_id]
        document_memberships = memberships.get(document_id, [])
        dataset_row = datasets.get(document_id)
        stored_leaf = row["leaf_field_id"]

        leaf_id = stored_leaf
        if leaf_id is None and document_memberships:
            leaf_id = document_memberships[-1][0]
        if leaf_id is None and dataset_row is not None:
            leaf_id = _path_json_leaf(dataset_row[0])
        if leaf_id is None:
            # A document that was never classified has no path to rebuild.
            continue

        chain = _resolve_chain(field_map, leaf_id, cache=chain_cache)
        path_ids = tuple(field.id for field in chain)
        dataset_confidence = None if dataset_row is None else dataset_row[1]
        plans.append(
            _DocumentPlan(
                document_id=document_id,
                kind=row["kind"],
                leaf_id=leaf_id,
                path_ids=path_ids,
                confidence=_confidence_for(
                    leaf_id, document_memberships, dataset_confidence
                ),
                needs_leaf_write=stored_leaf != leaf_id,
            )
        )

    affected: set[str] = set()
    refresh_datasets: list[_DocumentPlan] = []
    for plan in plans:
        affected.update(
            store.assign_document_path(
                plan.document_id,
                list(plan.path_ids),
                confidence=plan.confidence,
            )
        )
        if plan.needs_leaf_write:
            store.set_leaf(plan.document_id, plan.leaf_id)
        if plan.kind == "dataset":
            refresh_datasets.append(plan)
        affected.update(plan.path_ids)

    if refresh_datasets:
        # FieldStore has no public path_json setter; commit_path would also
        # rewrite document identity fields, so update the JSON column directly.
        with store._transaction():
            conn.executemany(
                "UPDATE datasets SET path_json = ? WHERE document_id = ?",
                [
                    (json.dumps(list(plan.path_ids)), plan.document_id)
                    for plan in refresh_datasets
                ],
            )

    store.recompute_centroids(sorted(affected))
    return len(plans)


def _cycle_problem(field_map: dict[str, Field], field_id: str) -> str | None:
    seen: set[str] = set()
    current: str | None = field_id
    while current is not None:
        if current in seen:
            if current == field_id:
                return (
                    f"field {field_id!r} is its own ancestor "
                    f"(cycle at {current!r})"
                )
            return (
                f"field {field_id!r} ancestry contains a cycle at "
                f"{current!r}"
            )
        seen.add(current)
        field = field_map.get(current)
        if field is None:
            return None
        current = field.parent_id
    return None


def _membership_problems(
    document_id: str,
    expected_ids: Sequence[str],
    memberships: Sequence[tuple[str, float, int]],
) -> list[str]:
    problems: list[str] = []
    actual_ids = [item[0] for item in memberships]
    if actual_ids != list(expected_ids):
        problems.append(
            f"document {document_id!r} document_fields {actual_ids!r} "
            f"does not match ancestor chain {list(expected_ids)!r}"
        )
    ordinals = sorted(item[2] for item in memberships)
    if ordinals != list(range(len(ordinals))):
        problems.append(
            f"document {document_id!r} document_fields ordinals {ordinals!r} "
            "are not contiguous from zero"
        )
    return problems


def verify_paths(store: FieldStore) -> list[str]:
    """Return a list of hierarchy and path problems (empty means clean).

    Checks every field's parent, level, cycles, and sibling-name uniqueness,
    then every document's leaf reachability, ``document_fields`` order and
    content, and dataset ``path_json`` equality.  This function is read-only
    and never raises for corrupt data; it reports it.
    """

    problems: list[str] = []
    fields = store.all_fields()
    field_map = {field.id: field for field in fields}

    for field in fields:
        if field.level < 1 or field.level > store.max_depth:
            problems.append(
                f"field {field.id!r} level {field.level} is outside "
                f"1..{store.max_depth}"
            )
        if field.parent_id is None:
            if field.level != 1:
                problems.append(
                    f"root field {field.id!r} has level {field.level}, expected 1"
                )
        elif field.parent_id == field.id:
            problems.append(f"field {field.id!r} is its own parent")
        else:
            parent = field_map.get(field.parent_id)
            if parent is None:
                problems.append(
                    f"field {field.id!r} references missing parent "
                    f"{field.parent_id!r}"
                )
            elif field.level != parent.level + 1:
                problems.append(
                    f"field {field.id!r} level {field.level} does not equal "
                    f"parent {parent.id!r} level + 1 ({parent.level + 1})"
                )
        cycle = _cycle_problem(field_map, field.id)
        if cycle is not None:
            problems.append(cycle)

    siblings: dict[tuple[str, str], list[str]] = {}
    for field in fields:
        key = (field.parent_id or "", _normalize_name(field.name))
        siblings.setdefault(key, []).append(field.id)
    for (parent_id, normalized), field_ids in sorted(siblings.items()):
        if len(field_ids) > 1:
            problems.append(
                f"duplicate sibling name {normalized!r} under parent "
                f"{parent_id or '<root>'!r}: {sorted(field_ids)!r}"
            )

    conn = store._connection()
    document_rows = conn.execute(
        "SELECT id, kind, leaf_field_id FROM documents ORDER BY id"
    ).fetchall()
    document_ids = [row["id"] for row in document_rows]
    memberships = _fetch_memberships(conn, document_ids)
    datasets = _fetch_datasets(conn, document_ids) if document_ids else {}
    chain_cache: dict[str, tuple[Field, ...]] = {}

    for row in document_rows:
        document_id = row["id"]
        document_memberships = memberships.get(document_id, [])
        dataset_row = datasets.get(document_id)
        leaf_id = row["leaf_field_id"]

        if leaf_id is None:
            if document_memberships:
                problems.append(
                    f"document {document_id!r} has document_fields "
                    f"{[item[0] for item in document_memberships]!r} "
                    "but no leaf_field_id"
                )
            if dataset_row is not None and _path_json_leaf(dataset_row[0]):
                problems.append(
                    f"dataset {document_id!r} has path_json but no leaf_field_id"
                )
            continue

        if leaf_id not in field_map:
            problems.append(
                f"document {document_id!r} leaf {leaf_id!r} does not exist"
            )
            continue

        try:
            chain = _resolve_chain(field_map, leaf_id, cache=chain_cache)
        except HierarchyIntegrityError as exc:
            problems.append(f"document {document_id!r} leaf is unreachable: {exc}")
            continue

        expected_ids = [field.id for field in chain]
        problems.extend(
            _membership_problems(document_id, expected_ids, document_memberships)
        )

        if row["kind"] != "dataset":
            continue
        if dataset_row is None:
            problems.append(f"dataset {document_id!r} has no datasets row")
            continue
        try:
            stored_path = json.loads(dataset_row[0])
        except (TypeError, ValueError, json.JSONDecodeError):
            problems.append(f"dataset {document_id!r} has invalid path_json")
            continue
        if not isinstance(stored_path, list) or stored_path != expected_ids:
            problems.append(
                f"dataset {document_id!r} path_json {stored_path!r} "
                f"does not match ancestor chain {expected_ids!r}"
            )

    return problems
