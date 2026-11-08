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
from .embeddings import cosine, l2_normalize
from .research.scrape.models import (
    Document,
    DocumentComparison,
    DocumentCritique,
    DocumentLink,
)

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

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('dataset','search','paper','web')),
  source TEXT NOT NULL,
  source_id TEXT,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_embedding BLOB NOT NULL,
  embedding_model TEXT NOT NULL,
  canonical_url TEXT,
  url_hash TEXT UNIQUE,
  content_sha256 TEXT,
  content_sha512 TEXT UNIQUE,
  authors_json TEXT NOT NULL DEFAULT '[]',
  publisher TEXT,
  published_at TEXT,
  license TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  credibility_score REAL NOT NULL DEFAULT 0.0,
  credibility_json TEXT NOT NULL DEFAULT '{}',
  credible INTEGER NOT NULL DEFAULT 0,
  preloaded INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_kind ON documents(kind);
CREATE INDEX IF NOT EXISTS documents_credible ON documents(credible);
CREATE INDEX IF NOT EXISTS documents_preloaded ON documents(preloaded);

CREATE TABLE IF NOT EXISTS datasets (
  document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  profile_json TEXT NOT NULL,
  path_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  refined_after_research INTEGER NOT NULL DEFAULT 0,
  cache_key TEXT
);

CREATE VIEW IF NOT EXISTS searches AS
  SELECT * FROM documents WHERE kind = 'search';
CREATE VIEW IF NOT EXISTS papers AS
  SELECT * FROM documents WHERE kind = 'paper';

CREATE TABLE IF NOT EXISTS document_links (
  from_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  to_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  relation TEXT NOT NULL CHECK (
    relation IN ('references','cites','linked_from','same_as','dataset_paper','search_result')
  ),
  url TEXT,
  anchor_text TEXT,
  discovered_at TEXT NOT NULL,
  PRIMARY KEY (from_id, to_id, relation)
);
CREATE INDEX IF NOT EXISTS document_links_to ON document_links(to_id);

CREATE TABLE IF NOT EXISTS document_fields (
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  field_id TEXT NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  confidence REAL NOT NULL,
  ordinal INTEGER NOT NULL DEFAULT 0,
  assigned_at TEXT NOT NULL,
  PRIMARY KEY (document_id, field_id)
);
CREATE INDEX IF NOT EXISTS document_fields_field ON document_fields(field_id);

CREATE TABLE IF NOT EXISTS document_comparisons (
  left_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  right_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  method_version TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (left_id, right_id, kind, method_version)
);

CREATE TABLE IF NOT EXISTS document_critiques (
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  corpus_version TEXT NOT NULL,
  verdict TEXT NOT NULL,
  issues_json TEXT NOT NULL,
  missing_information_json TEXT NOT NULL DEFAULT '[]',
  summary TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (document_id, corpus_version)
);

CREATE TABLE IF NOT EXISTS document_fetches (
  url_hash TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  status INTEGER NOT NULL,
  etag TEXT,
  last_modified TEXT,
  body TEXT NOT NULL,
  title TEXT,
  content_type TEXT,
  links_json TEXT NOT NULL DEFAULT '[]',
  content_sha512 TEXT,
  fetched_at TEXT NOT NULL,
  ttl_seconds INTEGER NOT NULL
);

CREATE VIEW IF NOT EXISTS field_members AS
  SELECT df.field_id, df.document_id AS dataset_id
  FROM document_fields AS df
  JOIN documents AS d ON d.id = df.document_id
  WHERE d.kind = 'dataset';
"""

_SELECT_FIELDS = """
SELECT f.id, f.parent_id, f.level, f.name, f.description, f.desc_embedding,
       f.centroid, f.created_at, f.updated_at,
       (SELECT COUNT(*) FROM document_fields AS m WHERE m.field_id = f.id)
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
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Create the v2 schema or migrate a v1 database in place.

        v1 used ``datasets`` (id, sha256, profile_json, summary, embedding,
        path_json, confidence, refined_after_research, cache_key, updated_at)
        and ``field_members`` (field_id, dataset_id).  v2 moves the common
        document fields into ``documents``, keeps dataset-specific fields in
        the ``datasets`` extension table, and renames membership to
        ``document_fields``.
        """
        conn = self._connection()
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        is_v1 = (
            version < 2
            and "datasets" in tables
            and "documents" not in tables
            and "sha256" in columns
        )
        if not is_v1:
            conn.executescript(_SCHEMA)
            self._ensure_v2_columns(conn)
            if version < 2:
                conn.execute("PRAGMA user_version = 2")
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations "
                    "(version, applied_at, note) VALUES (2, ?, ?)",
                    (utc_now_iso(), "created v2 schema"),
                )
                conn.commit()
            return

        # v1 migration.  Rename the old tables first so the new ``datasets``
        # extension table and the ``field_members`` compatibility view can be
        # created with their v2 definitions.
        has_field_members = "field_members" in tables
        conn.execute("ALTER TABLE datasets RENAME TO datasets_v1")
        if has_field_members:
            conn.execute("ALTER TABLE field_members RENAME TO field_members_v1")
        # Older v1 stores predate the cache_key column; add it before the
        # backfill so the SELECT below works on every v1 database.
        v1_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets_v1)").fetchall()
        }
        if "cache_key" not in v1_columns:
            conn.execute("ALTER TABLE datasets_v1 ADD COLUMN cache_key TEXT")
        if "refined_after_research" not in v1_columns:
            conn.execute(
                "ALTER TABLE datasets_v1 ADD COLUMN refined_after_research "
                "INTEGER NOT NULL DEFAULT 0"
            )
        conn.commit()
        conn.executescript(_SCHEMA)
        self._ensure_v2_columns(conn)
        now = utc_now_iso()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO documents
                  (id, kind, source, source_id, title, summary, summary_embedding,
                   embedding_model, canonical_url, url_hash, content_sha256,
                   content_sha512, authors_json, publisher, published_at, license,
                   metadata_json, credibility_score, credibility_json, credible,
                   preloaded, status, created_at, updated_at)
                SELECT id, 'dataset', 'user', NULL, id, summary, embedding, '',
                       NULL, NULL, sha256, NULL, '[]', NULL, NULL, NULL, '{}',
                       0.0, '{}', 0, 0, 'classified', updated_at, updated_at
                FROM datasets_v1
                """
            )
            for row in conn.execute(
                "SELECT id, profile_json FROM datasets_v1"
            ).fetchall():
                title = row["id"]
                try:
                    profile = json.loads(row["profile_json"])
                    source_path = profile.get("source_path")
                    if source_path:
                        title = Path(source_path).name or title
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                conn.execute(
                    "UPDATE documents SET title = ? WHERE id = ?",
                    (title, row["id"]),
                )
            conn.execute(
                """
                INSERT INTO datasets
                  (document_id, profile_json, path_json, confidence,
                   refined_after_research, cache_key)
                SELECT id, profile_json, path_json, confidence,
                       refined_after_research, cache_key
                FROM datasets_v1
                """
            )
            if has_field_members:
                conn.execute(
                    """
                    INSERT INTO document_fields
                      (document_id, field_id, confidence, ordinal, assigned_at)
                    SELECT dataset_id, field_id, 1.0, 0, ?
                    FROM field_members_v1
                    """,
                    (now,),
                )
            conn.execute("DROP TABLE IF EXISTS field_members_v1")
            conn.execute("DROP TABLE IF EXISTS datasets_v1")
            conn.execute("PRAGMA user_version = 2")
            conn.execute(
                "INSERT OR REPLACE INTO schema_migrations "
                "(version, applied_at, note) VALUES (2, ?, ?)",
                (now, "migrated v1 datasets and field_members"),
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise FieldStoreError(
                "schema migration left foreign-key violations: "
                + repr([tuple(row) for row in foreign_key_errors[:5]])
            )
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise FieldStoreError(
                f"schema migration integrity check failed: {integrity[0] if integrity else 'no result'}"
            )

    def _ensure_v2_columns(self, conn: sqlite3.Connection) -> None:
        """Add columns introduced after the v2 schema was first created."""
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        if columns and "cache_key" not in columns:
            conn.execute("ALTER TABLE datasets ADD COLUMN cache_key TEXT")
            conn.commit()
        fetch_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(document_fetches)").fetchall()
        }
        fetch_additions = {
            "title": "TEXT",
            "content_type": "TEXT",
            "links_json": "TEXT NOT NULL DEFAULT '[]'",
            "content_sha512": "TEXT",
        }
        changed = False
        for name, definition in fetch_additions.items():
            if name not in fetch_columns:
                conn.execute(
                    f"ALTER TABLE document_fetches ADD COLUMN {name} {definition}"
                )
                changed = True
        if changed:
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
                    SELECT 1 FROM document_fields AS m
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
        title = dataset_id
        try:
            profile = json.loads(profile_json)
            source_path = profile.get("source_path")
            if source_path:
                title = Path(source_path).name or title
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

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
                    "SELECT field_id FROM document_fields WHERE document_id = ?",
                    (dataset_id,),
                ).fetchall()
            ]

            now = utc_now_iso()
            conn.execute(
                """
                INSERT INTO documents
                  (id, kind, source, source_id, title, summary, summary_embedding,
                   embedding_model, canonical_url, url_hash, content_sha256,
                   content_sha512, authors_json, publisher, published_at, license,
                   metadata_json, credibility_score, credibility_json, credible,
                   preloaded, status, created_at, updated_at)
                VALUES (?, 'dataset', 'user', NULL, ?, ?, ?, '', NULL, NULL, ?,
                        NULL, '[]', NULL, NULL, NULL, '{}', 0.0, '{}', 0, 0,
                        'classified', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  kind = excluded.kind,
                  source = excluded.source,
                  title = excluded.title,
                  summary = excluded.summary,
                  summary_embedding = excluded.summary_embedding,
                  content_sha256 = excluded.content_sha256,
                  status = excluded.status,
                  updated_at = excluded.updated_at
                """,
                (
                    dataset_id,
                    title,
                    summary,
                    embedding_blob,
                    sha256,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO datasets
                  (document_id, profile_json, path_json, confidence,
                   refined_after_research, cache_key)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                  profile_json = excluded.profile_json,
                  path_json = excluded.path_json,
                  confidence = excluded.confidence,
                  refined_after_research = excluded.refined_after_research,
                  cache_key = excluded.cache_key
                """,
                (
                    dataset_id,
                    profile_json,
                    path_json,
                    float(confidence),
                    int(bool(refined_after_research)),
                    cache_key,
                ),
            )

            conn.execute(
                "DELETE FROM document_fields WHERE document_id = ?", (dataset_id,)
            )
            conn.executemany(
                "INSERT INTO document_fields "
                "(document_id, field_id, confidence, ordinal, assigned_at) "
                "VALUES (?, ?, 1.0, ?, ?)",
                [
                    (dataset_id, field_id, index, now)
                    for index, field_id in enumerate(ids)
                ],
            )

            affected.update(old_ids)
            affected.update(ids)

        return sorted(affected)

    def get_dataset(self, dataset_id: str) -> dict | None:
        conn = self._connection()
        row = conn.execute(
            """
            SELECT d.id, d.summary, d.summary_embedding, d.content_sha256,
                   d.updated_at, ds.profile_json, ds.path_json, ds.confidence,
                   ds.refined_after_research, ds.cache_key
            FROM documents AS d
            JOIN datasets AS ds ON ds.document_id = d.id
            WHERE d.id = ? AND d.kind = 'dataset'
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "sha256": row["content_sha256"] or "",
            "profile_json": row["profile_json"],
            "summary": row["summary"],
            "embedding": self._decode_vector(
                row["summary_embedding"], label="dataset embedding"
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
    # Documents (datasets, searches, papers, web)
    # ------------------------------------------------------------------
    @staticmethod
    def _json_value(raw: str | None, default: object) -> object:
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return default

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            kind=row["kind"],
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            summary=row["summary"],
            summary_embedding=self._decode_vector(
                row["summary_embedding"], label="summary embedding"
            ),
            embedding_model=row["embedding_model"],
            canonical_url=row["canonical_url"],
            url_hash=row["url_hash"],
            content_sha256=row["content_sha256"],
            content_sha512=row["content_sha512"],
            authors=self._json_value(row["authors_json"], []),
            publisher=row["publisher"],
            published_at=row["published_at"],
            license=row["license"],
            metadata=self._json_value(row["metadata_json"], {}),
            credibility_score=float(row["credibility_score"]),
            credibility=self._json_value(row["credibility_json"], {}),
            credible=bool(row["credible"]),
            preloaded=bool(row["preloaded"]),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_document(self, document_id: str) -> Document | None:
        row = self._connection().execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return None if row is None else self._row_to_document(row)

    def find_document_by_url(self, url_hash_value: str) -> Document | None:
        row = self._connection().execute(
            "SELECT * FROM documents WHERE url_hash = ?", (url_hash_value,)
        ).fetchone()
        return None if row is None else self._row_to_document(row)

    def find_document_by_content(self, content_sha512_value: str) -> Document | None:
        row = self._connection().execute(
            "SELECT * FROM documents WHERE content_sha512 = ?",
            (content_sha512_value,),
        ).fetchone()
        return None if row is None else self._row_to_document(row)

    def upsert_document(self, document: Document) -> Document:
        conn = self._connection()
        now = utc_now_iso()
        created = document.created_at or now
        updated = document.updated_at or now
        vector = document.summary_embedding or [0.0] * self.dims
        embedding_blob = self._encode_vector(
            vector, label="summary embedding", normalize=False
        )
        try:
            with self._transaction():
                conn.execute(
                    """
                    INSERT INTO documents
                      (id, kind, source, source_id, title, summary,
                       summary_embedding, embedding_model, canonical_url, url_hash,
                       content_sha256, content_sha512, authors_json, publisher,
                       published_at, license, metadata_json, credibility_score,
                       credibility_json, credible, preloaded, status, created_at,
                       updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      kind = excluded.kind,
                      source = excluded.source,
                      source_id = excluded.source_id,
                      title = excluded.title,
                      summary = excluded.summary,
                      summary_embedding = excluded.summary_embedding,
                      embedding_model = excluded.embedding_model,
                      canonical_url = excluded.canonical_url,
                      url_hash = excluded.url_hash,
                      content_sha256 = excluded.content_sha256,
                      content_sha512 = excluded.content_sha512,
                      authors_json = excluded.authors_json,
                      publisher = excluded.publisher,
                      published_at = excluded.published_at,
                      license = excluded.license,
                      metadata_json = excluded.metadata_json,
                      credibility_score = excluded.credibility_score,
                      credibility_json = excluded.credibility_json,
                      credible = excluded.credible,
                      preloaded = excluded.preloaded,
                      status = excluded.status,
                      updated_at = excluded.updated_at
                    """,
                    (
                        document.id,
                        document.kind,
                        document.source,
                        document.source_id,
                        document.title,
                        document.summary,
                        embedding_blob,
                        document.embedding_model,
                        document.canonical_url,
                        document.url_hash,
                        document.content_sha256,
                        document.content_sha512,
                        json.dumps(document.authors),
                        document.publisher,
                        document.published_at,
                        document.license,
                        json.dumps(document.metadata),
                        float(document.credibility_score),
                        json.dumps(document.credibility),
                        int(bool(document.credible)),
                        int(bool(document.preloaded)),
                        document.status,
                        created,
                        updated,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise FieldStoreError(
                f"document identity conflict for {document.id!r}: {exc}"
            ) from exc
        result = self.get_document(document.id)
        if result is None:  # pragma: no cover - defensive
            raise FieldStoreError(f"upserted document disappeared: {document.id}")
        return result

    def add_document_link(self, link: DocumentLink) -> None:
        conn = self._connection()
        discovered = link.discovered_at or utc_now_iso()
        with self._transaction():
            conn.execute(
                """
                INSERT INTO document_links
                  (from_id, to_id, relation, url, anchor_text, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(from_id, to_id, relation) DO UPDATE SET
                  url = excluded.url,
                  anchor_text = excluded.anchor_text
                """,
                (
                    link.from_id,
                    link.to_id,
                    link.relation,
                    link.url,
                    link.anchor_text,
                    discovered,
                ),
            )

    def document_links(
        self, document_id: str, *, direction: str = "out"
    ) -> list[DocumentLink]:
        column = "from_id" if direction == "out" else "to_id"
        rows = self._connection().execute(
            f"SELECT * FROM document_links WHERE {column} = ? "
            "ORDER BY discovered_at, relation, from_id, to_id",
            (document_id,),
        ).fetchall()
        return [
            DocumentLink(
                from_id=row["from_id"],
                to_id=row["to_id"],
                relation=row["relation"],
                url=row["url"],
                anchor_text=row["anchor_text"],
                discovered_at=row["discovered_at"],
            )
            for row in rows
        ]

    def assign_document_path(
        self,
        document_id: str,
        field_ids: Sequence[str],
        *,
        confidence: float = 1.0,
    ) -> list[str]:
        """Replace a document's field path and return affected field ids."""
        ids = list(field_ids)
        if not ids:
            raise FieldStoreError("a document path must contain at least one field")
        if len(set(ids)) != len(ids):
            raise FieldStoreError("a document path must not repeat field ids")
        if not all(isinstance(field_id, str) for field_id in ids):
            raise FieldStoreError("field_ids must contain only strings")
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
                    "unknown field ids in document path: " + ", ".join(missing)
                )
            old_ids = [
                row["field_id"]
                for row in conn.execute(
                    "SELECT field_id FROM document_fields WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
            ]
            now = utc_now_iso()
            conn.execute(
                "DELETE FROM document_fields WHERE document_id = ?", (document_id,)
            )
            conn.executemany(
                "INSERT INTO document_fields "
                "(document_id, field_id, confidence, ordinal, assigned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (document_id, field_id, float(confidence), index, now)
                    for index, field_id in enumerate(ids)
                ],
            )
            affected.update(old_ids)
            affected.update(ids)
        return sorted(affected)

    def document_path(self, document_id: str) -> list[Field]:
        rows = self._connection().execute(
            _SELECT_FIELDS
            + " JOIN document_fields AS m ON m.field_id = f.id "
            "WHERE m.document_id = ? ORDER BY m.ordinal, f.id",
            (document_id,),
        ).fetchall()
        return self._rows_to_fields(rows)

    def neighbors(self, document_id: str, limit: int = 10) -> list[Document]:
        """Return the most similar documents by summary embedding."""
        target = self.get_document(document_id)
        if target is None or not target.summary_embedding:
            return []
        rows = self._connection().execute(
            "SELECT * FROM documents WHERE id != ?", (document_id,)
        ).fetchall()
        scored: list[tuple[float, Document]] = []
        for row in rows:
            candidate = self._row_to_document(row)
            if not candidate.summary_embedding:
                continue
            try:
                similarity = cosine(
                    target.summary_embedding, candidate.summary_embedding
                )
            except (ValueError, TypeError):
                continue
            scored.append((similarity, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [candidate for _, candidate in scored[: max(0, limit)]]

    def put_comparison(self, comparison: DocumentComparison) -> None:
        created = comparison.created_at or utc_now_iso()
        with self._transaction():
            self._connection().execute(
                """
                INSERT INTO document_comparisons
                  (left_id, right_id, kind, method_version, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(left_id, right_id, kind, method_version) DO UPDATE SET
                  result_json = excluded.result_json,
                  created_at = excluded.created_at
                """,
                (
                    comparison.left_id,
                    comparison.right_id,
                    comparison.kind,
                    comparison.method_version,
                    comparison.model_dump_json(),
                    created,
                ),
            )

    def get_comparison(
        self,
        left_id: str,
        right_id: str,
        kind: str,
        method_version: str = "1.0.0",
    ) -> DocumentComparison | None:
        row = self._connection().execute(
            "SELECT result_json FROM document_comparisons "
            "WHERE left_id = ? AND right_id = ? AND kind = ? "
            "AND method_version = ?",
            (left_id, right_id, kind, method_version),
        ).fetchone()
        if row is None:
            return None
        return DocumentComparison.model_validate_json(row["result_json"])

    def put_critique(self, critique: DocumentCritique) -> None:
        created = critique.created_at or utc_now_iso()
        with self._transaction():
            self._connection().execute(
                """
                INSERT INTO document_critiques
                  (document_id, corpus_version, verdict, issues_json,
                   missing_information_json, summary, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, corpus_version) DO UPDATE SET
                  verdict = excluded.verdict,
                  issues_json = excluded.issues_json,
                  missing_information_json = excluded.missing_information_json,
                  summary = excluded.summary,
                  model = excluded.model,
                  created_at = excluded.created_at
                """,
                (
                    critique.document_id,
                    critique.corpus_version,
                    critique.verdict,
                    json.dumps(critique.issues),
                    json.dumps(critique.missing_information),
                    critique.summary,
                    critique.model,
                    created,
                ),
            )

    def get_critique(
        self, document_id: str, corpus_version: str
    ) -> DocumentCritique | None:
        row = self._connection().execute(
            "SELECT * FROM document_critiques "
            "WHERE document_id = ? AND corpus_version = ?",
            (document_id, corpus_version),
        ).fetchone()
        if row is None:
            return None
        return DocumentCritique(
            document_id=row["document_id"],
            corpus_version=row["corpus_version"],
            verdict=row["verdict"],
            issues=self._json_value(row["issues_json"], []),
            missing_information=self._json_value(
                row["missing_information_json"], []
            ),
            summary=row["summary"],
            model=row["model"],
            created_at=row["created_at"],
        )

    def record_fetch(
        self,
        *,
        url_hash_value: str,
        url: str,
        status: int,
        body: str,
        etag: str | None = None,
        last_modified: str | None = None,
        ttl_seconds: int = 604800,
        title: str | None = None,
        content_type: str | None = None,
        links: Sequence[str] | None = None,
        content_sha512: str | None = None,
    ) -> None:
        with self._transaction():
            self._connection().execute(
                """
                INSERT INTO document_fetches
                  (url_hash, url, status, etag, last_modified, body, title,
                   content_type, links_json, content_sha512, fetched_at,
                   ttl_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                  url = excluded.url,
                  status = excluded.status,
                  etag = excluded.etag,
                  last_modified = excluded.last_modified,
                  body = excluded.body,
                  title = excluded.title,
                  content_type = excluded.content_type,
                  links_json = excluded.links_json,
                  content_sha512 = excluded.content_sha512,
                  fetched_at = excluded.fetched_at,
                  ttl_seconds = excluded.ttl_seconds
                """,
                (
                    url_hash_value,
                    url,
                    int(status),
                    etag,
                    last_modified,
                    body,
                    title,
                    content_type,
                    json.dumps(list(links or [])),
                    content_sha512,
                    utc_now_iso(),
                    int(ttl_seconds),
                ),
            )

    def get_fetch(self, url_hash_value: str) -> dict | None:
        row = self._connection().execute(
            "SELECT * FROM document_fetches WHERE url_hash = ?",
            (url_hash_value,),
        ).fetchone()
        return None if row is None else dict(row)

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
                SELECT d.summary_embedding AS embedding
                FROM documents AS d
                JOIN document_fields AS m ON m.document_id = d.id
                WHERE m.field_id = ?
                ORDER BY d.id
                """,
                (field_id,),
            ).fetchall()

            if member_rows:
                total = [0.0] * self.dims
                for member_row in member_rows:
                    vector = self._decode_vector(
                        member_row["embedding"], label="document embedding"
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
              (SELECT COUNT(*) FROM documents) AS documents,
              (SELECT COUNT(*) FROM document_fields) AS memberships,
              (SELECT COUNT(*) FROM document_links) AS links,
              (SELECT COUNT(*) FROM document_comparisons) AS comparisons,
              (SELECT COUNT(*) FROM document_critiques) AS critiques,
              (SELECT COUNT(*) FROM fields AS f
               WHERE NOT EXISTS (
                   SELECT 1 FROM document_fields AS m WHERE m.field_id = f.id
               )) AS orphan_fields
            """
        ).fetchone()
        return {
            "fields": int(row["fields"]),
            "roots": int(row["roots"]),
            "datasets": int(row["datasets"]),
            "documents": int(row["documents"]),
            "memberships": int(row["memberships"]),
            "links": int(row["links"]),
            "comparisons": int(row["comparisons"]),
            "critiques": int(row["critiques"]),
            "orphan_fields": int(row["orphan_fields"]),
        }
