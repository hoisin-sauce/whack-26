"""SQLite-backed field hierarchy and dataset membership store.

The store owns the field tree, the dataset rows, and the membership links
between them.  Vectors are persisted as compact little-endian ``float32``
BLOBs; caller-facing values are plain Python lists.

The public surface is intentionally small: the classifier asks for siblings,
creates or reuses fields, and commits a dataset path in one transaction.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .common import utc_now_iso
from .embeddings import l2_normalize

__all__ = [
    "DuplicateFieldError",
    "Field",
    "FieldStore",
    "FieldStoreError",
]

MAX_NAME_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 500
MAX_FIELD_LEVEL = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fields (
  id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES fields(id) ON DELETE CASCADE,
  level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 4),
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  desc_embedding BLOB NOT NULL,
  centroid BLOB,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS fields_sibling_name
  ON fields(COALESCE(parent_id, ''), lower(name));

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  profile_json TEXT NOT NULL,
  summary TEXT NOT NULL,
  embedding BLOB NOT NULL,
  path_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  refined_after_research INTEGER NOT NULL DEFAULT 0,
  cache_key TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS field_members (
  field_id TEXT NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  PRIMARY KEY (field_id, dataset_id)
);

CREATE INDEX IF NOT EXISTS field_members_dataset
  ON field_members(dataset_id);
"""

_SELECT_FIELDS = """
SELECT f.id, f.parent_id, f.level, f.name, f.description, f.desc_embedding,
       f.centroid, f.created_at, f.updated_at,
       (SELECT COUNT(*) FROM field_members AS m WHERE m.field_id = f.id)
           AS member_count
FROM fields AS f
"""

_PUNCTUATION = re.compile(r"[^\w\s]|_", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)


def _normalize_name(name: str) -> str:
    """Return the case/punctuation-insensitive key used for sibling uniqueness."""
    spaced = _PUNCTUATION.sub(" ", name.casefold())
    return _WHITESPACE.sub(" ", spaced).strip()


@dataclass(frozen=True)
class Field:
    """A node in the dynamic topic hierarchy."""

    id: str
    parent_id: str | None
    level: int
    name: str
    description: str
    description_embedding: list[float]
    centroid: list[float] | None
    member_count: int
    created_at: str
    updated_at: str


class FieldStoreError(RuntimeError):
    """Raised when a field-store operation cannot be completed."""


class DuplicateFieldError(FieldStoreError):
    """Raised when a field name or id collides with an existing field."""


class FieldStore:
    """Persistent hierarchy of fields plus dataset membership rows."""

    def __init__(
        self, path: str | Path, *, dims: int = 768, max_depth: int = 4
    ) -> None:
        if dims <= 0:
            raise ValueError("dims must be positive")
        if not 1 <= max_depth <= MAX_FIELD_LEVEL:
            raise ValueError(f"max_depth must be between 1 and {MAX_FIELD_LEVEL}")

        self.dims = dims
        self.max_depth = max_depth

        if str(path) == ":memory:":
            target: str | Path = ":memory:"
        else:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = sqlite3.connect(target)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after a database was first created."""
        conn = self._connection()
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        if "cache_key" not in columns:
            conn.execute("ALTER TABLE datasets ADD COLUMN cache_key TEXT")
            conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the connection.  Safe to call more than once."""
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def __enter__(self) -> "FieldStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise FieldStoreError("field store is closed")
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        if conn.in_transaction:
            raise FieldStoreError("cannot start a nested transaction")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # Vector encoding
    # ------------------------------------------------------------------
    def _encode_vector(
        self, values: Sequence[float], *, label: str, normalize: bool
    ) -> bytes:
        if len(values) != self.dims:
            raise FieldStoreError(
                f"{label} dimension mismatch: expected {self.dims}, "
                f"got {len(values)}"
            )
        try:
            floats = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise FieldStoreError(f"{label} must contain only numbers") from exc
        if normalize:
            floats = l2_normalize(floats)
        return struct.pack(f"<{self.dims}f", *floats)

    def _decode_vector(self, blob: bytes, *, label: str) -> list[float]:
        expected_bytes = self.dims * 4
        if len(blob) != expected_bytes:
            raise FieldStoreError(
                f"stored {label} dimension mismatch: expected {self.dims} "
                f"floats, found {len(blob) // 4}"
            )
        return list(struct.unpack(f"<{self.dims}f", blob))

    def _row_to_field(self, row: sqlite3.Row) -> Field:
        centroid = row["centroid"]
        return Field(
            id=row["id"],
            parent_id=row["parent_id"],
            level=int(row["level"]),
            name=row["name"],
            description=row["description"],
            description_embedding=self._decode_vector(
                row["desc_embedding"], label="description embedding"
            ),
            centroid=(
                None
                if centroid is None
                else self._decode_vector(centroid, label="centroid")
            ),
            member_count=int(row["member_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _rows_to_fields(self, rows: Sequence[sqlite3.Row]) -> list[Field]:
        return [self._row_to_field(row) for row in rows]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_name(name: str) -> None:
        if not name:
            raise FieldStoreError("field name must not be empty")
        if len(name) > MAX_NAME_LENGTH:
            raise FieldStoreError(
                f"field name must be at most {MAX_NAME_LENGTH} characters"
            )

    @staticmethod
    def _check_description(description: str) -> None:
        if not description:
            raise FieldStoreError("field description must not be empty")
        if len(description) > MAX_DESCRIPTION_LENGTH:
            raise FieldStoreError(
                f"field description must be at most {MAX_DESCRIPTION_LENGTH} characters"
            )

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------
    def create_field(
        self,
        *,
        parent_id: str | None,
        name: str,
        description: str,
        description_embedding: Sequence[float],
        field_id: str | None = None,
    ) -> Field:
        """Create and return a field at ``parent.level + 1`` (or level 1)."""
        display_name = name.strip()
        display_description = description.strip()
        self._check_name(display_name)
        self._check_description(display_description)
        normalized = _normalize_name(display_name)
        if not normalized:
            raise FieldStoreError("field name must contain a letter or digit")

        embedding_blob = self._encode_vector(
            description_embedding, label="description_embedding", normalize=True
        )

        conn = self._connection()
        with self._transaction():
            if parent_id is None:
                level = 1
            else:
                parent_row = conn.execute(
                    "SELECT level FROM fields WHERE id = ?", (parent_id,)
                ).fetchone()
                if parent_row is None:
                    raise FieldStoreError(f"parent field not found: {parent_id}")
                level = int(parent_row["level"]) + 1

            if level > self.max_depth:
                raise FieldStoreError(
                    f"field level {level} exceeds max_depth {self.max_depth}"
                )

            if self.find_sibling(parent_id, display_name) is not None:
                raise DuplicateFieldError(
                    f"a sibling named {display_name!r} already exists"
                )

            if field_id is None:
                digest = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"whack26/field/{parent_id or 'root'}/{normalized}",
                ).hex[:12]
                new_id = f"fld_{digest}"
            else:
                new_id = field_id

            if conn.execute(
                "SELECT 1 FROM fields WHERE id = ?", (new_id,)
            ).fetchone() is not None:
                raise DuplicateFieldError(f"field id already exists: {new_id}")

            now = utc_now_iso()
            try:
                conn.execute(
                    """
                    INSERT INTO fields
                      (id, parent_id, level, name, description, desc_embedding,
                       centroid, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                    """,
                    (
                        new_id,
                        parent_id,
                        level,
                        display_name,
                        display_description,
                        embedding_blob,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateFieldError(
                    f"could not create field {new_id!r}: {exc}"
                ) from exc

        field = self.get_field(new_id)
        if field is None:  # pragma: no cover - defensive
            raise FieldStoreError(f"created field disappeared: {new_id}")
        return field

    def get_field(self, field_id: str) -> Field | None:
        conn = self._connection()
        row = conn.execute(
            _SELECT_FIELDS + " WHERE f.id = ?", (field_id,)
        ).fetchone()
        return None if row is None else self._row_to_field(row)

    def children(self, parent_id: str | None) -> list[Field]:
        conn = self._connection()
        if parent_id is None:
            rows = conn.execute(
                _SELECT_FIELDS
                + " WHERE f.parent_id IS NULL ORDER BY lower(f.name), f.id"
            ).fetchall()
        else:
            rows = conn.execute(
                _SELECT_FIELDS
                + " WHERE f.parent_id = ? ORDER BY lower(f.name), f.id",
                (parent_id,),
            ).fetchall()
        return self._rows_to_fields(rows)

    def roots(self) -> list[Field]:
        return self.children(None)

    def all_fields(self) -> list[Field]:
        conn = self._connection()
        rows = conn.execute(
            _SELECT_FIELDS + " ORDER BY f.level, lower(f.name), f.id"
        ).fetchall()
        return self._rows_to_fields(rows)

    def find_sibling(self, parent_id: str | None, name: str) -> Field | None:
        """Find a sibling whose normalized name matches ``name``."""
        conn = self._connection()
        if parent_id is None:
            rows = conn.execute(
                "SELECT id, name FROM fields WHERE parent_id IS NULL ORDER BY id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name FROM fields WHERE parent_id = ? ORDER BY id",
                (parent_id,),
            ).fetchall()
        key = _normalize_name(name)
        for row in rows:
            if _normalize_name(row["name"]) == key:
                return self.get_field(row["id"])
        return None

    def update_field(
        self,
        field_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        description_embedding: Sequence[float] | None = None,
    ) -> Field:
        """Update supplied attributes and return the refreshed field."""
        conn = self._connection()
        with self._transaction():
            row = conn.execute(
                "SELECT parent_id FROM fields WHERE id = ?", (field_id,)
            ).fetchone()
            if row is None:
                raise FieldStoreError(f"field not found: {field_id}")

            updates: dict[str, object] = {}

            if name is not None:
                display_name = name.strip()
                self._check_name(display_name)
                if not _normalize_name(display_name):
                    raise FieldStoreError(
                        "field name must contain a letter or digit"
                    )
                existing = self.find_sibling(row["parent_id"], display_name)
                if existing is not None and existing.id != field_id:
                    raise DuplicateFieldError(
                        f"a sibling named {display_name!r} already exists"
                    )
                updates["name"] = display_name

            if description is not None:
                display_description = description.strip()
                self._check_description(display_description)
                updates["description"] = display_description

            if description_embedding is not None:
                updates["desc_embedding"] = self._encode_vector(
                    description_embedding,
                    label="description_embedding",
                    normalize=True,
                )

            if updates:
                updates["updated_at"] = utc_now_iso()
                assignments = ", ".join(f"{column} = ?" for column in updates)
                try:
                    conn.execute(
                        f"UPDATE fields SET {assignments} WHERE id = ?",
                        (*updates.values(), field_id),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DuplicateFieldError(
                        f"could not update field {field_id!r}: {exc}"
                    ) from exc

        field = self.get_field(field_id)
        if field is None:  # pragma: no cover - defensive
            raise FieldStoreError(f"updated field disappeared: {field_id}")
        return field

    def delete_field(self, field_id: str, *, cascade: bool = False) -> None:
        """Delete a field, refusing to orphan children unless ``cascade``."""
        conn = self._connection()
        with self._transaction():
            row = conn.execute(
                "SELECT 1 FROM fields WHERE id = ?", (field_id,)
            ).fetchone()
            if row is None:
                raise FieldStoreError(f"field not found: {field_id}")

            has_children = conn.execute(
                "SELECT 1 FROM fields WHERE parent_id = ? LIMIT 1", (field_id,)
            ).fetchone()
            if has_children is not None and not cascade:
                raise FieldStoreError(
                    f"field {field_id!r} has children; pass cascade=True"
                )

            # Foreign keys ON DELETE CASCADE remove descendants and memberships.
            conn.execute("DELETE FROM fields WHERE id = ?", (field_id,))

    def delete_empty_leaf_fields(self) -> list[str]:
        """Delete memberless leaves that have no children.

        Only leaves that are empty when the call starts are removed.  Callers
        that want to prune parents exposed by those deletions repeat the call
        until it returns an empty list.
        """
        conn = self._connection()
        with self._transaction():
            rows = conn.execute(
                """
                SELECT f.id
                FROM fields AS f
                WHERE NOT EXISTS (
                    SELECT 1 FROM field_members AS m
                    WHERE m.field_id = f.id
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM fields AS c WHERE c.parent_id = f.id
                  )
                """
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                conn.executemany(
                    "DELETE FROM fields WHERE id = ?", [(item,) for item in ids]
                )
        return sorted(ids)

    # ------------------------------------------------------------------
    # Datasets and memberships
    # ------------------------------------------------------------------
    def commit_path(
        self,
        *,
        dataset_id: str,
        sha256: str,
        profile_json: str,
        summary: str,
        embedding: Sequence[float],
        path_json: str,
        confidence: float,
        refined_after_research: bool,
        field_ids: Sequence[str],
        cache_key: str | None = None,
    ) -> list[str]:
        """Upsert a dataset and replace its membership rows atomically.

        ``cache_key`` is an opaque caller identity used to decide whether a
        committed path can be reused.  Returns the sorted union of the
        dataset's old and new field ids so the caller can recompute exactly the
        affected centroids.
        """
        if not dataset_id:
            raise FieldStoreError("dataset_id must not be empty")
        ids = list(field_ids)
        if not ids:
            raise FieldStoreError("a dataset path must contain at least one field")
        if len(set(ids)) != len(ids):
            raise FieldStoreError("a dataset path must not repeat field ids")
        if not all(isinstance(field_id, str) for field_id in ids):
            raise FieldStoreError("field_ids must contain only strings")

        # The stored path and the membership rows must describe the same
        # hierarchy, in the same order.  ``dataset_path`` returns fields in
        # ``path_json`` order, so an inconsistent pair would silently display
        # a path that disagrees with the memberships.
        try:
            decoded_path = json.loads(path_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FieldStoreError("path_json must be valid JSON") from exc
        if not isinstance(decoded_path, list) or not all(
            isinstance(item, str) for item in decoded_path
        ):
            raise FieldStoreError(
                "path_json must be a JSON array of field id strings"
            )
        if decoded_path != ids:
            raise FieldStoreError(
                "path_json must match field_ids in the same order"
            )

        embedding_blob = self._encode_vector(
            embedding, label="dataset embedding", normalize=False
        )

        conn = self._connection()
        affected: set[str] = set()
        with self._transaction():
            placeholders = ", ".join("?" for _ in ids)
            known = {
                row["id"]
                for row in conn.execute(
                    f"SELECT id FROM fields WHERE id IN ({placeholders})", ids
                ).fetchall()
            }
            missing = [field_id for field_id in ids if field_id not in known]
            if missing:
                raise FieldStoreError(
                    "unknown field ids in dataset path: " + ", ".join(missing)
                )

            old_ids = [
                row["field_id"]
                for row in conn.execute(
                    "SELECT field_id FROM field_members WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchall()
            ]

            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO datasets
                  (id, sha256, profile_json, summary, embedding, path_json,
                   confidence, refined_after_research, cache_key, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  sha256 = excluded.sha256,
                  profile_json = excluded.profile_json,
                  summary = excluded.summary,
                  embedding = excluded.embedding,
                  path_json = excluded.path_json,
                  confidence = excluded.confidence,
                  refined_after_research = excluded.refined_after_research,
                  cache_key = excluded.cache_key,
                  updated_at = excluded.updated_at
                """,
                (
                    dataset_id,
                    sha256,
                    profile_json,
                    summary,
                    embedding_blob,
                    path_json,
                    float(confidence),
                    int(bool(refined_after_research)),
                    cache_key,
                    now,
                ),
            )

            conn.execute(
                "DELETE FROM field_members WHERE dataset_id = ?", (dataset_id,)
            )
            conn.executemany(
                "INSERT INTO field_members (field_id, dataset_id) VALUES (?, ?)",
                [(field_id, dataset_id) for field_id in ids],
            )

            affected.update(old_ids)
            affected.update(ids)

        return sorted(affected)

    def get_dataset(self, dataset_id: str) -> dict | None:
        conn = self._connection()
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "sha256": row["sha256"],
            "profile_json": row["profile_json"],
            "summary": row["summary"],
            "embedding": self._decode_vector(
                row["embedding"], label="dataset embedding"
            ),
            "path_json": row["path_json"],
            "confidence": float(row["confidence"]),
            "refined_after_research": bool(row["refined_after_research"]),
            "cache_key": row["cache_key"],
            "updated_at": row["updated_at"],
        }

    def dataset_path(self, dataset_id: str) -> list[Field]:
        """Return the dataset's fields in the exact order in ``path_json``."""
        dataset = self.get_dataset(dataset_id)
        if dataset is None:
            return []

        try:
            items = json.loads(dataset["path_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise FieldStoreError(
                f"dataset {dataset_id!r} has invalid path_json"
            ) from exc
        if not isinstance(items, list):
            raise FieldStoreError(
                f"dataset {dataset_id!r} path_json must be a JSON array"
            )

        fields: list[Field] = []
        for item in items:
            if isinstance(item, str):
                field_id = item
            elif isinstance(item, dict):
                field_id = item.get("field_id") or item.get("id")
            else:
                field_id = None
            if not isinstance(field_id, str):
                raise FieldStoreError(
                    f"dataset {dataset_id!r} path_json contains an invalid entry"
                )
            field = self.get_field(field_id)
            if field is None:
                raise FieldStoreError(
                    f"dataset {dataset_id!r} path references missing field "
                    f"{field_id!r}"
                )
            fields.append(field)
        return fields

    # ------------------------------------------------------------------
    # Centroids
    # ------------------------------------------------------------------
    def recompute_centroid(self, field_id: str) -> Field | None:
        """Recompute one centroid from its members, or clear it when empty."""
        conn = self._connection()
        with self._transaction():
            row = conn.execute(
                "SELECT centroid FROM fields WHERE id = ?", (field_id,)
            ).fetchone()
            if row is None:
                return None

            member_rows = conn.execute(
                """
                SELECT d.embedding
                FROM datasets AS d
                JOIN field_members AS m ON m.dataset_id = d.id
                WHERE m.field_id = ?
                ORDER BY d.id
                """,
                (field_id,),
            ).fetchall()

            if member_rows:
                total = [0.0] * self.dims
                for member_row in member_rows:
                    vector = self._decode_vector(
                        member_row["embedding"], label="dataset embedding"
                    )
                    for index, value in enumerate(vector):
                        total[index] += value
                count = len(member_rows)
                mean = [value / count for value in total]
                new_centroid = self._encode_vector(
                    l2_normalize(mean), label="centroid", normalize=False
                )
            else:
                new_centroid = None

            if new_centroid != row["centroid"]:
                conn.execute(
                    "UPDATE fields SET centroid = ?, updated_at = ? WHERE id = ?",
                    (new_centroid, utc_now_iso(), field_id),
                )

        return self.get_field(field_id)

    def recompute_centroids(self, field_ids: Sequence[str]) -> list[Field]:
        """Recompute several centroids, ignoring duplicates and missing fields."""
        seen: set[str] = set()
        results: list[Field] = []
        for field_id in field_ids:
            if field_id in seen:
                continue
            seen.add(field_id)
            field = self.recompute_centroid(field_id)
            if field is not None:
                results.append(field)
        return results

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        conn = self._connection()
        row = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM fields) AS fields,
              (SELECT COUNT(*) FROM fields WHERE parent_id IS NULL) AS roots,
              (SELECT COUNT(*) FROM datasets) AS datasets,
              (SELECT COUNT(*) FROM field_members) AS memberships,
              (SELECT COUNT(*) FROM fields AS f
               WHERE NOT EXISTS (
                   SELECT 1 FROM field_members AS m WHERE m.field_id = f.id
               )) AS orphan_fields
            """
        ).fetchone()
        return {
            "fields": int(row["fields"]),
            "roots": int(row["roots"]),
            "datasets": int(row["datasets"]),
            "memberships": int(row["memberships"]),
            "orphan_fields": int(row["orphan_fields"]),
        }
