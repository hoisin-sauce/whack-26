from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from whack26.research.scrape.models import (
    Document,
    DocumentComparison,
    DocumentCritique,
    DocumentLink,
)
from whack26.store import FieldStore, FieldStoreError

DIMS = 2
TIMESTAMP = "2026-11-08T10:00:00Z"


def make_document(
    identifier: str,
    *,
    title: str = "A paper",
    summary: str = "A summary.",
    embedding: list[float] | None = None,
    url_hash: str | None = None,
    content_sha512: str | None = None,
    kind: str = "paper",
    source: str = "openalex",
) -> Document:
    return Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source=source,
        source_id=identifier,
        title=title,
        summary=summary,
        summary_embedding=embedding if embedding is not None else [1.0, 0.0],
        embedding_model="mock@2",
        url_hash=url_hash,
        content_sha512=content_sha512,
        authors=["A. Author"],
        metadata={"links": ["https://example.test/a"]},
        status="summarized",
    )


def make_field(store: FieldStore, name: str, *, parent_id: str | None = None):
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} description.",
        description_embedding=[1.0, 0.0],
    )


def test_new_store_creates_v2_schema(tmp_path: Path) -> None:
    path = tmp_path / "fields.db"
    with FieldStore(path, dims=DIMS) as store:
        version = store._connection().execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row["name"]
            for row in store._connection()
            .execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
            .fetchall()
        }
        assert version == 4
        assert {
            "documents",
            "datasets",
            "document_links",
            "document_fields",
            "document_comparisons",
            "document_critiques",
            "document_fetches",
            "field_operations",
            "store_meta",
            "research_models",
            "schema_migrations",
            "searches",
            "papers",
            "field_members",
        } <= tables
        assert store.stats()["documents"] == 0


def build_v1_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE fields (
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
            CREATE TABLE datasets (
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
            CREATE TABLE field_members (
              field_id TEXT NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
              dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
              PRIMARY KEY (field_id, dataset_id)
            );
            """
        )
        import struct

        conn.execute(
            "INSERT INTO fields VALUES (?, NULL, 1, ?, ?, ?, NULL, ?, ?)",
            (
                "fld_root",
                "Housing",
                "Housing markets.",
                struct.pack("<2f", 1.0, 0.0),
                TIMESTAMP,
                TIMESTAMP,
            ),
        )
        conn.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ds_old",
                "abc123",
                json.dumps(
                    {
                        "dataset_id": "ds_old",
                        "source_path": "data/old_housing.csv",
                        "sha256": "abc123",
                        "format": "csv",
                        "rows": 10,
                        "columns": [],
                        "description": "Old housing data.",
                        "target": None,
                        "time_column": None,
                        "warnings": [],
                    }
                ),
                "Old housing summary.",
                struct.pack("<2f", 1.0, 0.0),
                json.dumps(["fld_root"]),
                0.8,
                1,
                "cache-key",
                TIMESTAMP,
            ),
        )
        conn.execute(
            "INSERT INTO field_members VALUES (?, ?)", ("fld_root", "ds_old")
        )
        conn.commit()


def test_v1_migration_preserves_datasets_and_memberships(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    build_v1_database(path)

    with FieldStore(path, dims=DIMS) as store:
        version = store._connection().execute("PRAGMA user_version").fetchone()[0]
        assert version == 4
        dataset = store.get_dataset("ds_old")
        assert dataset is not None
        assert dataset["summary"] == "Old housing summary."
        assert dataset["sha256"] == "abc123"
        assert dataset["cache_key"] == "cache-key"
        assert dataset["embedding"] == pytest.approx([1.0, 0.0])
        assert [field.id for field in store.dataset_path("ds_old")] == ["fld_root"]
        assert store.stats()["documents"] == 1
        assert store.stats()["datasets"] == 1
        compatibility = store._connection().execute(
            "SELECT field_id, dataset_id FROM field_members"
        ).fetchall()
        assert [(row["field_id"], row["dataset_id"]) for row in compatibility] == [
            ("fld_root", "ds_old")
        ]

    # Reopening is idempotent.
    with FieldStore(path, dims=DIMS) as reopened:
        assert reopened.get_dataset("ds_old") is not None
        assert reopened.stats()["documents"] == 1


def test_v1_migration_adds_missing_cache_key_column(tmp_path: Path) -> None:
    path = tmp_path / "legacy-no-cache-key.db"
    build_v1_database(path)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE datasets DROP COLUMN cache_key")
        conn.commit()

    with FieldStore(path, dims=DIMS) as store:
        dataset = store.get_dataset("ds_old")
        assert dataset is not None
        assert dataset["cache_key"] is None
        assert dataset["summary"] == "Old housing summary."
        assert store.stats()["documents"] == 1


def test_document_upsert_and_identity_lookups(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        document = make_document(
            "doc_a", url_hash="url-a", content_sha512="content-a"
        )

        stored = store.upsert_document(document)

        assert stored.id == "doc_a"
        assert stored.authors == ["A. Author"]
        assert stored.metadata["links"]
        assert store.get_document("doc_a") == stored
        assert store.find_document_by_url("url-a") == stored
        assert store.find_document_by_content("content-a") == stored

        duplicate = make_document("doc_b", url_hash="url-a")
        with pytest.raises(FieldStoreError, match="identity conflict"):
            store.upsert_document(duplicate)


def test_document_links_path_neighbors_and_comparisons(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        root = make_field(store, "Research")
        child = make_field(store, "Methods", parent_id=root.id)
        first = make_document("doc_a", embedding=[1.0, 0.0])
        second = make_document("doc_b", embedding=[0.9, 0.1])
        store.upsert_document(first)
        store.upsert_document(second)

        store.add_document_link(
            DocumentLink(
                from_id="doc_a",
                to_id="doc_b",
                relation="references",
                url="https://example.test/b",
            )
        )
        links = store.document_links("doc_a")
        assert len(links) == 1
        assert links[0].to_id == "doc_b"
        assert store.document_links("doc_b", direction="in")[0].from_id == "doc_a"

        affected = store.assign_document_path("doc_a", [root.id, child.id])
        assert set(affected) == {root.id, child.id}
        assert [field.id for field in store.document_path("doc_a")] == [
            root.id,
            child.id,
        ]

        neighbors = store.neighbors("doc_a", limit=5)
        assert [item.id for item in neighbors] == ["doc_b"]

        comparison = DocumentComparison(
            left_id="doc_a",
            right_id="doc_b",
            kind="method_overlap",
            similarity=0.9,
            verdict="supports",
            rationale="Both use OLS.",
        )
        store.put_comparison(comparison)
        cached = store.get_comparison("doc_a", "doc_b", "method_overlap")
        assert cached is not None
        assert cached.verdict == "supports"


def test_document_critiques_and_fetches(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        store.upsert_document(make_document("doc_a"))
        critique = DocumentCritique(
            document_id="doc_a",
            corpus_version="1.0.0+abc",
            verdict="refine",
            issues=[{"severity": "major", "finding": "No split protocol."}],
            missing_information=["split protocol"],
            summary="A method review.",
            model="mock",
        )
        store.put_critique(critique)
        loaded = store.get_critique("doc_a", "1.0.0+abc")
        assert loaded is not None
        assert loaded.verdict == "refine"
        assert loaded.missing_information == ["split protocol"]

        store.record_fetch(
            url_hash_value="url-a",
            url="https://example.test/a",
            status=200,
            body='{"ok": true}',
            etag="etag-1",
            ttl_seconds=60,
        )
        fetch = store.get_fetch("url-a")
        assert fetch is not None
        assert fetch["status"] == 200
        assert fetch["etag"] == "etag-1"


def test_leaf_operations_and_hierarchy_version(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        root = make_field(store, "Research")
        child = make_field(store, "Methods", parent_id=root.id)
        store.upsert_document(make_document("doc_a"))

        store.set_leaf("doc_a", child.id)
        assert store.get_leaf("doc_a") == child.id

        before = store.hierarchy_version()
        operation_id = store.log_field_operation(
            operation="create",
            field_id=child.id,
            old_parent_id=root.id,
            new_parent_id=root.id,
            old_level=1,
            new_level=2,
            reason="test",
        )
        after = store.bump_hierarchy_version()

        assert after == before + 1
        assert operation_id.startswith("fop_")
        operations = store.field_operations(field_id=child.id)
        assert operations[0]["operation"] == "create"
        assert operations[0]["reason"] == "test"


def test_research_models_and_collection_metadata(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        store.upsert_document(make_document("doc_a"))
        model_id = store.put_research_model(
            document_id="doc_a",
            family="linear",
            spec_json='{"family": "linear"}',
            mapping_json='{"predictors": ["x"]}',
            evidence="The paper reports an OLS model.",
            applied=True,
        )

        models = store.list_research_models(document_id="doc_a")
        assert models[0]["id"] == model_id
        assert models[0]["family"] == "linear"
        assert models[0]["spec"] == {"family": "linear"}
        assert models[0]["mapping"] == {"predictors": ["x"]}
        assert models[0]["applied"] is True

        root = make_field(store, "Research")
        store.commit_path(
            dataset_id="ds_test",
            sha256="abc123",
            profile_json="{}",
            summary="A test dataset.",
            embedding=[1.0, 0.0],
            path_json=json.dumps([root.id]),
            confidence=1.0,
            refined_after_research=False,
            field_ids=[root.id],
        )
        store.set_collection(
            "ds_test",
            '{"design": "observational", "unit_of_analysis": "block group"}',
        )
        assert store.get_collection("ds_test") == {
            "design": "observational",
            "unit_of_analysis": "block group",
        }
