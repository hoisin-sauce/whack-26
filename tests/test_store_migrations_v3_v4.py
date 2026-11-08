from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from whack26.store import FieldStore

DIMS = 2


def build_v2_shaped_store(path: Path) -> dict[str, str]:
    """Create a current store, then strip the v3/v4 additions from it.

    The result matches the schema an old v2 store would have: no
    ``documents.leaf_field_id``, no ``datasets.collection_json``, none of the
    v3/v4 tables, and ``PRAGMA user_version = 2``.
    """
    with FieldStore(path, dims=DIMS) as store:
        root = store.create_field(
            parent_id=None,
            name="Housing",
            description="Housing markets.",
            description_embedding=[1.0, 0.0],
        )
        member_leaf = store.create_field(
            parent_id=root.id,
            name="Rents",
            description="Rental markets.",
            description_embedding=[0.0, 1.0],
        )
        path_leaf = store.create_field(
            parent_id=root.id,
            name="Prices",
            description="House prices.",
            description_embedding=[0.5, 0.5],
        )
        store.commit_path(
            dataset_id="ds_members",
            sha256="members-sha256",
            profile_json="{}",
            summary="A dataset with membership rows.",
            embedding=[1.0, 0.0],
            path_json=json.dumps([root.id, member_leaf.id]),
            confidence=1.0,
            refined_after_research=False,
            field_ids=[root.id, member_leaf.id],
        )
        store.commit_path(
            dataset_id="ds_path",
            sha256="path-sha256",
            profile_json="{}",
            summary="A dataset with only path metadata.",
            embedding=[0.0, 1.0],
            path_json=json.dumps([root.id, path_leaf.id]),
            confidence=1.0,
            refined_after_research=False,
            field_ids=[root.id, path_leaf.id],
        )

    with sqlite3.connect(path) as conn:
        # Remove the v3 additions.
        conn.execute("ALTER TABLE documents DROP COLUMN leaf_field_id")
        conn.execute("DROP TABLE field_operations")
        conn.execute("DROP TABLE store_meta")
        # Remove the v4 additions.
        conn.execute("ALTER TABLE datasets DROP COLUMN collection_json")
        conn.execute("DROP TABLE research_models")
        # ds_path keeps only its datasets.path_json row, so migration must
        # recover its leaf from that JSON array rather than from memberships.
        conn.execute(
            "DELETE FROM document_fields WHERE document_id = ?", ("ds_path",)
        )
        conn.execute("DELETE FROM schema_migrations WHERE version IN (3, 4)")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()

    return {
        "root": root.id,
        "member_leaf": member_leaf.id,
        "path_leaf": path_leaf.id,
    }


@pytest.fixture()
def legacy_v2_store(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    path = tmp_path / "legacy-v2.db"
    identifiers = build_v2_shaped_store(path)
    return path, identifiers


def test_v2_store_migrates_to_v4_with_backfills(legacy_v2_store) -> None:
    path, identifiers = legacy_v2_store

    with FieldStore(path, dims=DIMS) as store:
        conn = store._connection()

        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4

        # leaf_field_id comes from the deepest document_fields ordinal ...
        assert store.get_leaf("ds_members") == identifiers["member_leaf"]
        # ... and from the last element of datasets.path_json otherwise.
        assert store.get_leaf("ds_path") == identifiers["path_leaf"]

        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"field_operations", "store_meta", "research_models"} <= tables

        meta = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM store_meta").fetchall()
        }
        assert meta["hierarchy_version"] == "1"

        dataset_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(datasets)").fetchall()
        }
        assert "collection_json" in dataset_columns

        migrated_versions = {
            row["version"]
            for row in conn.execute(
                "SELECT version FROM schema_migrations"
            ).fetchall()
        }
        assert {3, 4} <= migrated_versions


def test_reopening_migrated_v2_store_is_idempotent(legacy_v2_store) -> None:
    path, identifiers = legacy_v2_store

    with FieldStore(path, dims=DIMS) as store:
        assert store.get_leaf("ds_members") == identifiers["member_leaf"]
        assert store.get_leaf("ds_path") == identifiers["path_leaf"]

    with FieldStore(path, dims=DIMS) as reopened:
        conn = reopened._connection()

        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert reopened.get_leaf("ds_members") == identifiers["member_leaf"]
        assert reopened.get_leaf("ds_path") == identifiers["path_leaf"]
        assert reopened.hierarchy_version() == 1

        counts = {
            row["version"]: row["count"]
            for row in conn.execute(
                "SELECT version, COUNT(*) AS count FROM schema_migrations "
                "GROUP BY version"
            ).fetchall()
        }
        assert counts[3] == 1
        assert counts[4] == 1
