"""Tests for hierarchy path recomputation and verification."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from whack26.hierarchy.paths import recompute_paths, verify_paths
from whack26.hierarchy.types import HierarchyIntegrityError
from whack26.research.scrape.models import Document
from whack26.store import Field, FieldStore

DIMS = 8
DATASET_ID = "ds_paths"


def vec(*values: float) -> list[float]:
    assert len(values) <= DIMS
    return [*values, *([0.0] * (DIMS - len(values)))]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[FieldStore]:
    with FieldStore(tmp_path / "fields.db", dims=DIMS, max_depth=4) as opened:
        yield opened


def make_field(
    store: FieldStore,
    name: str,
    *,
    parent_id: str | None = None,
    embedding: list[float] | None = None,
) -> Field:
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} description",
        description_embedding=embedding if embedding is not None else vec(1.0),
    )


def make_document(
    store: FieldStore,
    identifier: str,
    *,
    kind: str = "paper",
    embedding: list[float] | None = None,
) -> Document:
    document = Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source="test",
        title=identifier,
        summary=f"{identifier} summary",
        summary_embedding=embedding if embedding is not None else vec(1.0),
        embedding_model="fake@1",
        status="classified",
    )
    store.upsert_document(document)
    return document


def commit_dataset(
    store: FieldStore,
    field_ids: list[str],
    *,
    dataset_id: str = DATASET_ID,
    confidence: float = 0.75,
) -> None:
    store.commit_path(
        dataset_id=dataset_id,
        sha256="a" * 64,
        profile_json='{"rows": 10}',
        summary=f"{dataset_id} summary",
        embedding=vec(1.0),
        path_json=json.dumps(field_ids),
        confidence=confidence,
        refined_after_research=False,
        field_ids=field_ids,
    )


def path_ids(store: FieldStore, document_id: str) -> list[str]:
    return [field.id for field in store.document_path(document_id)]


def membership_rows(store: FieldStore, document_id: str) -> list[tuple[str, int]]:
    rows = store._connection().execute(
        "SELECT field_id, ordinal FROM document_fields "
        "WHERE document_id = ? ORDER BY ordinal",
        (document_id,),
    ).fetchall()
    return [(row["field_id"], row["ordinal"]) for row in rows]


def stored_path_json(store: FieldStore, dataset_id: str = DATASET_ID) -> list[str]:
    dataset = store.get_dataset(dataset_id)
    assert dataset is not None
    return json.loads(dataset["path_json"])


def set_leaf_without_foreign_key(
    store: FieldStore, document_id: str, field_id: str
) -> None:
    conn = store._connection()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            "UPDATE documents SET leaf_field_id = ? WHERE id = ?",
            (field_id, document_id),
        )
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def move_field(
    store: FieldStore,
    field_id: str,
    *,
    parent_id: str,
    level: int,
) -> None:
    with store._transaction():
        store._connection().execute(
            "UPDATE fields SET parent_id = ?, level = ? WHERE id = ?",
            (parent_id, level, field_id),
        )


def test_verify_paths_accepts_a_clean_hierarchy(store: FieldStore) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)

    commit_dataset(store, [root.id, child.id, leaf.id])
    make_document(store, "doc_paper")
    store.assign_document_path("doc_paper", [root.id, child.id], confidence=0.4)
    store.set_leaf("doc_paper", child.id)
    make_document(store, "doc_unclassified")

    assert verify_paths(store) == []

    assert membership_rows(store, DATASET_ID) == [
        (root.id, 0),
        (child.id, 1),
        (leaf.id, 2),
    ]
    assert store.get_leaf(DATASET_ID) == leaf.id
    assert path_ids(store, DATASET_ID) == [root.id, child.id, leaf.id]
    assert stored_path_json(store) == [root.id, child.id, leaf.id]

    # An unclassified document has no path to rebuild or verify.
    assert recompute_paths(store, document_ids=["doc_unclassified"]) == 0
    assert verify_paths(store) == []


def test_recompute_paths_repairs_a_moved_subtree(store: FieldStore) -> None:
    old_root = make_field(store, "Research")
    moved = make_field(store, "Methods", parent_id=old_root.id)
    leaf = make_field(store, "Panel Data", parent_id=moved.id)

    commit_dataset(store, [old_root.id, moved.id, leaf.id])
    make_document(store, "doc_attached_to_moved")
    store.assign_document_path("doc_attached_to_moved", [old_root.id, moved.id])
    store.recompute_centroids([old_root.id, moved.id, leaf.id])
    assert store.get_field(old_root.id).member_count == 2  # type: ignore[union-attr]

    # Simulate a promotion that moves ``moved`` and its subtree under a new
    # branch.  The stored document paths still name the old root.
    new_root = make_field(store, "Statistics")
    intermediate = make_field(store, "Modeling", parent_id=new_root.id)
    move_field(store, moved.id, parent_id=intermediate.id, level=3)
    with store._transaction():
        store._connection().execute(
            "UPDATE fields SET level = 4 WHERE id = ?", (leaf.id,)
        )

    assert path_ids(store, DATASET_ID) == [old_root.id, moved.id, leaf.id]

    rebuilt = recompute_paths(store, field_ids=[moved.id])

    assert rebuilt == 2
    expected_dataset_path = [new_root.id, intermediate.id, moved.id, leaf.id]
    assert path_ids(store, DATASET_ID) == expected_dataset_path
    assert membership_rows(store, DATASET_ID) == [
        (new_root.id, 0),
        (intermediate.id, 1),
        (moved.id, 2),
        (leaf.id, 3),
    ]
    assert stored_path_json(store) == expected_dataset_path
    assert store.get_leaf(DATASET_ID) == leaf.id

    assert path_ids(store, "doc_attached_to_moved") == [
        new_root.id,
        intermediate.id,
        moved.id,
    ]
    assert store.get_leaf("doc_attached_to_moved") == moved.id

    assert verify_paths(store) == []
    old = store.get_field(old_root.id)
    new = store.get_field(new_root.id)
    assert old is not None and old.member_count == 0 and old.centroid is None
    assert new is not None and new.member_count == 2
    assert new.centroid is not None


def test_verify_paths_detects_corruption_and_recompute_repairs_it(
    store: FieldStore,
) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)
    other = make_field(store, "Other", parent_id=root.id)
    commit_dataset(store, [root.id, child.id, leaf.id])

    # Replace the real membership with an unrelated but valid sibling and make
    # path_json agree with the corruption, leaving leaf_field_id at the truth.
    with store._transaction():
        conn = store._connection()
        conn.execute(
            "DELETE FROM document_fields WHERE document_id = ? AND field_id = ?",
            (DATASET_ID, leaf.id),
        )
        conn.execute(
            "DELETE FROM document_fields WHERE document_id = ? AND field_id = ?",
            (DATASET_ID, child.id),
        )
        conn.execute(
            "INSERT INTO document_fields "
            "(document_id, field_id, confidence, ordinal, assigned_at) "
            "VALUES (?, ?, 0.5, 1, '2026-01-01T00:00:00Z')",
            (DATASET_ID, other.id),
        )
        conn.execute(
            "UPDATE datasets SET path_json = ? WHERE document_id = ?",
            (json.dumps([root.id, other.id]), DATASET_ID),
        )

    problems = verify_paths(store)
    assert any("document_fields" in problem for problem in problems)
    assert any("path_json" in problem for problem in problems)

    rebuilt = recompute_paths(store, document_ids=[DATASET_ID])

    assert rebuilt == 1
    assert path_ids(store, DATASET_ID) == [root.id, child.id, leaf.id]
    assert stored_path_json(store) == [root.id, child.id, leaf.id]
    assert membership_rows(store, DATASET_ID) == [
        (root.id, 0),
        (child.id, 1),
        (leaf.id, 2),
    ]
    assert verify_paths(store) == []


def test_recompute_paths_rebuilds_dataset_path_json(store: FieldStore) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    commit_dataset(store, [root.id, child.id])

    with store._transaction():
        store._connection().execute(
            "UPDATE datasets SET path_json = ? WHERE document_id = ?",
            ("[]", DATASET_ID),
        )

    problems = verify_paths(store)
    assert any("path_json" in problem for problem in problems)

    rebuilt = recompute_paths(store, field_ids=[child.id])

    assert rebuilt == 1
    assert membership_rows(store, DATASET_ID) == [(root.id, 0), (child.id, 1)]
    assert stored_path_json(store) == [root.id, child.id]
    assert verify_paths(store) == []


def test_recompute_paths_rejects_a_missing_leaf(store: FieldStore) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    commit_dataset(store, [root.id, child.id])
    set_leaf_without_foreign_key(store, DATASET_ID, "fld_missing")

    problems = verify_paths(store)
    assert any("does not exist" in problem for problem in problems)

    with pytest.raises(HierarchyIntegrityError, match="missing leaf field"):
        recompute_paths(store, document_ids=[DATASET_ID])

    # Validation happens before any write.
    assert path_ids(store, DATASET_ID) == [root.id, child.id]
    assert stored_path_json(store) == [root.id, child.id]


def test_recompute_paths_rejects_a_missing_parent(store: FieldStore) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)
    commit_dataset(store, [root.id, child.id, leaf.id])

    conn = store._connection()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            "UPDATE fields SET parent_id = 'fld_missing' WHERE id = ?",
            (child.id,),
        )
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    problems = verify_paths(store)
    assert any("references missing parent" in problem for problem in problems)

    with pytest.raises(HierarchyIntegrityError, match="missing parent field"):
        recompute_paths(store, field_ids=[leaf.id])

    assert path_ids(store, DATASET_ID) == [root.id, child.id, leaf.id]


def test_recompute_paths_rejects_a_cycle(store: FieldStore) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)
    commit_dataset(store, [root.id, child.id, leaf.id])

    move_field(store, root.id, parent_id=leaf.id, level=1)

    problems = verify_paths(store)
    assert any("cycle" in problem for problem in problems)

    with pytest.raises(HierarchyIntegrityError, match="cycle"):
        recompute_paths(store, document_ids=[DATASET_ID])

    # The failed recompute did not touch the stored rows.
    assert path_ids(store, DATASET_ID) == [root.id, child.id, leaf.id]
    assert stored_path_json(store) == [root.id, child.id, leaf.id]


def test_recompute_paths_derives_a_leaf_from_membership_ordinals(
    store: FieldStore,
) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)
    commit_dataset(store, [root.id, child.id, leaf.id])

    with store._transaction():
        store._connection().execute(
            "UPDATE documents SET leaf_field_id = NULL WHERE id = ?", (DATASET_ID,)
        )

    assert any("no leaf_field_id" in problem for problem in verify_paths(store))

    rebuilt = recompute_paths(store, document_ids=[DATASET_ID])

    assert rebuilt == 1
    assert store.get_leaf(DATASET_ID) == leaf.id
    assert path_ids(store, DATASET_ID) == [root.id, child.id, leaf.id]
    assert verify_paths(store) == []


def test_recompute_paths_derives_a_leaf_from_dataset_path_json(
    store: FieldStore,
) -> None:
    root = make_field(store, "Research")
    child = make_field(store, "Methods", parent_id=root.id)
    leaf = make_field(store, "Panel Data", parent_id=child.id)
    commit_dataset(store, [root.id, child.id, leaf.id])

    with store._transaction():
        conn = store._connection()
        conn.execute(
            "DELETE FROM document_fields WHERE document_id = ?", (DATASET_ID,)
        )
        conn.execute(
            "UPDATE documents SET leaf_field_id = NULL WHERE id = ?", (DATASET_ID,)
        )

    assert any("path_json but no leaf_field_id" in item for item in verify_paths(store))

    rebuilt = recompute_paths(store, document_ids=[DATASET_ID])

    assert rebuilt == 1
    assert store.get_leaf(DATASET_ID) == leaf.id
    assert membership_rows(store, DATASET_ID) == [
        (root.id, 0),
        (child.id, 1),
        (leaf.id, 2),
    ]
    confidence = store._connection().execute(
        "SELECT confidence FROM document_fields "
        "WHERE document_id = ? AND field_id = ?",
        (DATASET_ID, leaf.id),
    ).fetchone()["confidence"]
    assert confidence == pytest.approx(0.75)
    assert verify_paths(store) == []


def test_verify_paths_reports_levels_and_duplicate_sibling_names(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)

    with store._transaction():
        conn = store._connection()
        # Bypass the sibling unique index with a punctuation variant that has
        # the same normalized name after verification normalizes it.
        conn.execute(
            """
            INSERT INTO fields
              (id, parent_id, level, name, description, desc_embedding,
               centroid, created_at, updated_at)
            SELECT ?, parent_id, level, ?, description, desc_embedding,
                   centroid, created_at, updated_at
            FROM fields WHERE id = ?
            """,
            ("fld_duplicate_root", "Root!", root.id),
        )
        conn.execute("UPDATE fields SET level = 3 WHERE id = ?", (child.id,))

    problems = verify_paths(store)

    assert any("duplicate sibling name" in problem for problem in problems)
    assert any("level" in problem for problem in problems)
