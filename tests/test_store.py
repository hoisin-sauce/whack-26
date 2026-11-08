from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from whack26.store import (
    DuplicateFieldError,
    Field,
    FieldStore,
    FieldStoreError,
)

DIMS = 8
DATASET_ID = "ds_123456789abc"


def vec(*values: float) -> list[float]:
    """Pad values with zeros to the store's embedding size."""
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
    description: str | None = None,
    embedding: list[float] | None = None,
    field_id: str | None = None,
) -> Field:
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=description if description is not None else f"{name} description",
        description_embedding=embedding if embedding is not None else vec(1.0),
        field_id=field_id,
    )


def commit(
    store: FieldStore,
    field_ids: list[str],
    *,
    dataset_id: str = DATASET_ID,
    embedding: list[float] | None = None,
    path_json: str | None = None,
    summary: str = "A test dataset",
) -> list[str]:
    return store.commit_path(
        dataset_id=dataset_id,
        sha256="a" * 64,
        profile_json='{"rows": 10}',
        summary=summary,
        embedding=embedding if embedding is not None else vec(1.0),
        path_json=path_json if path_json is not None else json.dumps(field_ids),
        confidence=0.75,
        refined_after_research=False,
        field_ids=field_ids,
    )


def test_create_root_and_child_sets_levels_and_stable_ids(store: FieldStore) -> None:
    root = make_field(store, "Data Science")

    expected_root_id = (
        "fld_"
        + uuid.uuid5(
            uuid.NAMESPACE_URL, "whack26/field/root/data science"
        ).hex[:12]
    )
    assert root.id == expected_root_id
    assert root.parent_id is None
    assert root.level == 1
    assert root.member_count == 0
    assert root.centroid is None
    assert root.created_at == root.updated_at
    assert root.created_at.endswith("Z")

    child = make_field(store, "Neural Networks", parent_id=root.id)
    expected_child_id = (
        "fld_"
        + uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"whack26/field/{root.id}/neural networks",
        ).hex[:12]
    )
    assert child.id == expected_child_id
    assert child.parent_id == root.id
    assert child.level == 2

    grandchild = make_field(store, "Transformers", parent_id=child.id)
    assert grandchild.level == 3
    assert grandchild.parent_id == child.id


def test_duplicate_sibling_names_are_rejected(store: FieldStore) -> None:
    root = make_field(store, "Data Science")

    for variant in ("data science", "DATA   SCIENCE!", "Data-Science"):
        with pytest.raises(DuplicateFieldError):
            make_field(store, variant)

    with pytest.raises(DuplicateFieldError):
        make_field(store, "Another Name", field_id=root.id)

    # The same display name is fine under a different parent.
    other = make_field(store, "Other")
    child = make_field(store, "Data Science", parent_id=other.id)
    assert child.parent_id == other.id

    assert store.find_sibling(None, "data-science") == root
    assert store.find_sibling(other.id, "DATA SCIENCE") == child
    assert store.find_sibling(None, "Missing") is None


def test_max_depth_is_enforced(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "shallow.db", dims=DIMS, max_depth=2) as store:
        root = make_field(store, "Root")
        child = make_field(store, "Child", parent_id=root.id)
        assert child.level == 2
        with pytest.raises(FieldStoreError, match="max_depth"):
            make_field(store, "Grandchild", parent_id=child.id)


def test_invalid_constructor_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FieldStore(tmp_path / "dims.db", dims=0)
    with pytest.raises(ValueError):
        FieldStore(tmp_path / "deep.db", max_depth=5)
    with pytest.raises(ValueError):
        FieldStore(tmp_path / "shallow.db", max_depth=0)


def test_vector_dimensions_are_validated(store: FieldStore) -> None:
    with pytest.raises(FieldStoreError, match="dimension"):
        store.create_field(
            parent_id=None,
            name="Bad vector",
            description="Wrong length.",
            description_embedding=[0.0] * (DIMS - 1),
        )

    field = make_field(store, "Good vector")
    with pytest.raises(FieldStoreError, match="dimension"):
        store.update_field(
            field.id, description_embedding=[0.0] * (DIMS + 1)
        )


def test_description_embedding_is_normalized_and_stored_as_float32(
    store: FieldStore, tmp_path: Path
) -> None:
    field = make_field(store, "Housing", embedding=vec(3.0, 4.0))

    expected = vec(0.6, 0.8)
    assert field.description_embedding == pytest.approx(expected, abs=1e-6)
    norm = math.sqrt(sum(value * value for value in field.description_embedding))
    assert norm == pytest.approx(1.0, abs=1e-6)

    conn = sqlite3.connect(tmp_path / "fields.db")
    try:
        blob = conn.execute(
            "SELECT desc_embedding FROM fields WHERE id = ?", (field.id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert isinstance(blob, bytes)
    assert len(blob) == DIMS * 4


def test_update_field_changes_supplied_attributes(store: FieldStore) -> None:
    field = make_field(
        store, "Old Name", description="Old description", embedding=vec(1.0)
    )

    updated = store.update_field(
        field.id,
        name="New Name",
        description="New description",
        description_embedding=vec(0.0, 3.0, 4.0),
    )

    assert updated.id == field.id
    assert updated.parent_id == field.parent_id
    assert updated.level == field.level
    assert updated.name == "New Name"
    assert updated.description == "New description"
    assert updated.description_embedding == pytest.approx(
        vec(0.0, 0.6, 0.8), abs=1e-6
    )
    assert updated.created_at == field.created_at
    assert store.find_sibling(None, "new-name") == updated
    assert store.find_sibling(None, "old name") is None


def test_update_rejects_duplicate_sibling_and_missing_field(
    store: FieldStore,
) -> None:
    first = make_field(store, "Alpha")
    second = make_field(store, "Beta")

    with pytest.raises(DuplicateFieldError):
        store.update_field(second.id, name="alpha!")

    unchanged = store.get_field(second.id)
    assert unchanged is not None
    assert unchanged.name == "Beta"
    assert first.name == "Alpha"

    # Updating to the field's own current name is allowed.
    assert store.update_field(second.id, name="Beta").name == "Beta"

    with pytest.raises(FieldStoreError, match="not found"):
        store.update_field("fld_missing", name="Gamma")


def test_name_and_description_validation(store: FieldStore) -> None:
    with pytest.raises(FieldStoreError, match="empty"):
        store.create_field(
            parent_id=None,
            name="   ",
            description="Description.",
            description_embedding=vec(1.0),
        )
    with pytest.raises(FieldStoreError, match="80"):
        make_field(store, "x" * 81)
    with pytest.raises(FieldStoreError, match="empty"):
        make_field(store, "Valid", description="   ")
    with pytest.raises(FieldStoreError, match="500"):
        make_field(store, "Valid", description="x" * 501)


def test_get_children_roots_and_all_fields(store: FieldStore) -> None:
    root_a = make_field(store, "Alpha")
    root_b = make_field(store, "Beta")
    child_a = make_field(store, "Child A", parent_id=root_a.id)
    grandchild = make_field(store, "Grandchild", parent_id=child_a.id)

    assert {field.id for field in store.roots()} == {root_a.id, root_b.id}
    assert {field.id for field in store.children(None)} == {
        root_a.id,
        root_b.id,
    }
    assert [field.id for field in store.children(root_a.id)] == [child_a.id]
    assert [field.id for field in store.children(child_a.id)] == [
        grandchild.id
    ]
    assert store.children(grandchild.id) == []

    assert store.get_field(root_a.id) == root_a
    assert store.get_field("fld_missing") is None
    assert {field.id for field in store.all_fields()} == {
        root_a.id,
        root_b.id,
        child_a.id,
        grandchild.id,
    }


def test_commit_path_inserts_dataset_and_preserves_path_order(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)
    grandchild = make_field(store, "Grandchild", parent_id=child.id)

    # Deliberately not level order: dataset_path must follow path_json, and
    # field_ids must describe the same order.
    stored_order = [child.id, grandchild.id, root.id]
    affected = store.commit_path(
        dataset_id=DATASET_ID,
        sha256="a" * 64,
        profile_json='{"rows": 10}',
        summary="A test dataset",
        embedding=vec(3.0, 4.0),
        path_json=json.dumps(stored_order),
        confidence=0.75,
        refined_after_research=True,
        field_ids=stored_order,
    )

    assert affected == sorted([root.id, child.id, grandchild.id])
    dataset = store.get_dataset(DATASET_ID)
    assert dataset is not None
    assert dataset["id"] == DATASET_ID
    assert dataset["sha256"] == "a" * 64
    assert dataset["profile_json"] == '{"rows": 10}'
    assert dataset["summary"] == "A test dataset"
    assert dataset["embedding"] == pytest.approx(vec(3.0, 4.0), abs=1e-6)
    assert dataset["path_json"] == json.dumps(stored_order)
    assert dataset["confidence"] == pytest.approx(0.75)
    assert dataset["refined_after_research"] is True
    assert dataset["updated_at"].endswith("Z")

    assert [field.id for field in store.dataset_path(DATASET_ID)] == stored_order
    assert store.get_dataset("ds_missing") is None
    assert store.dataset_path("ds_missing") == []


def test_commit_path_accepts_matching_path_json_and_field_ids(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)
    order = [child.id, root.id]

    commit(store, order)

    dataset = store.get_dataset(DATASET_ID)
    assert dataset is not None
    assert dataset["path_json"] == json.dumps(order)
    assert [field.id for field in store.dataset_path(DATASET_ID)] == order


def test_commit_path_rejects_mismatched_order(store: FieldStore) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)

    with pytest.raises(FieldStoreError, match="same order"):
        commit(
            store,
            [root.id, child.id],
            path_json=json.dumps([child.id, root.id]),
        )

    assert store.get_dataset(DATASET_ID) is None
    assert store.stats()["datasets"] == 0


def test_commit_path_rejects_mismatched_ids(store: FieldStore) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)

    with pytest.raises(FieldStoreError, match="same order"):
        commit(
            store,
            [root.id],
            path_json=json.dumps([child.id]),
        )
    with pytest.raises(FieldStoreError, match="same order"):
        commit(
            store,
            [root.id, child.id],
            path_json=json.dumps([root.id, "fld_missing"]),
        )

    assert store.get_dataset(DATASET_ID) is None
    assert store.stats()["datasets"] == 0


def test_commit_path_rejects_malformed_path_json(store: FieldStore) -> None:
    root = make_field(store, "Root")

    with pytest.raises(FieldStoreError, match="valid JSON"):
        commit(store, [root.id], path_json="not json")
    with pytest.raises(FieldStoreError, match="JSON array of field id strings"):
        commit(store, [root.id], path_json='{"field_id": "fld_x"}')
    with pytest.raises(FieldStoreError, match="JSON array of field id strings"):
        commit(store, [root.id], path_json=json.dumps([1]))

    assert store.get_dataset(DATASET_ID) is None
    assert store.stats()["datasets"] == 0


def test_dataset_path_raises_when_a_referenced_field_is_missing(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)
    commit(store, [root.id, child.id])

    store.delete_field(child.id)

    with pytest.raises(FieldStoreError, match="missing field"):
        store.dataset_path(DATASET_ID)


def test_commit_path_upserts_and_replaces_memberships(store: FieldStore) -> None:
    root_a = make_field(store, "Alpha")
    child_a = make_field(store, "Child A", parent_id=root_a.id)
    root_b = make_field(store, "Beta")

    first = commit(store, [root_a.id, child_a.id], summary="First version")
    assert first == sorted([root_a.id, child_a.id])
    assert store.stats()["memberships"] == 2

    second = commit(store, [root_b.id], summary="Second version")
    assert second == sorted([root_a.id, child_a.id, root_b.id])
    assert store.stats()["memberships"] == 1

    dataset = store.get_dataset(DATASET_ID)
    assert dataset is not None
    assert dataset["summary"] == "Second version"
    assert [field.id for field in store.dataset_path(DATASET_ID)] == [root_b.id]


def test_move_between_fields_nulls_old_centroid(store: FieldStore) -> None:
    root_a = make_field(store, "Alpha")
    root_b = make_field(store, "Beta")

    affected = commit(store, [root_a.id], embedding=vec(1.0))
    store.recompute_centroids(affected)

    old = store.get_field(root_a.id)
    assert old is not None
    assert old.member_count == 1
    assert old.centroid is not None

    affected = commit(store, [root_b.id], embedding=vec(1.0))
    assert affected == sorted([root_a.id, root_b.id])
    store.recompute_centroids(affected)

    old = store.get_field(root_a.id)
    new = store.get_field(root_b.id)
    assert old is not None and new is not None
    assert old.member_count == 0
    assert old.centroid is None
    assert new.member_count == 1
    assert new.centroid == pytest.approx(vec(1.0), abs=1e-6)
    assert [field.id for field in store.dataset_path(DATASET_ID)] == [root_b.id]


def test_recompute_centroid_averages_and_normalizes_members(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)

    commit(store, [root.id, child.id], dataset_id="ds_one", embedding=vec(1.0, 0.0))
    commit(store, [root.id, child.id], dataset_id="ds_two", embedding=vec(0.0, 1.0))

    fields = store.recompute_centroids(
        [root.id, child.id, root.id, "fld_missing"]
    )
    assert len(fields) == 2
    assert store.recompute_centroid("fld_missing") is None

    expected = vec(math.sqrt(0.5), math.sqrt(0.5))
    for field in fields:
        assert field.member_count == 2
        assert field.centroid == pytest.approx(expected, abs=1e-6)
        norm = math.sqrt(sum(value * value for value in field.centroid or []))
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_delete_empty_leaf_fields_keeps_members_and_parents(
    store: FieldStore,
) -> None:
    empty_root = make_field(store, "Empty Root")
    empty_child = make_field(store, "Empty Child", parent_id=empty_root.id)
    used_root = make_field(store, "Used Root")
    used_child = make_field(store, "Used Child", parent_id=used_root.id)

    commit(store, [used_child.id])

    # Each call removes the current empty leaves; repeating exposes parents
    # that the first pass could not remove because they still had children.
    first_pass = store.delete_empty_leaf_fields()
    assert first_pass == [empty_child.id]

    deleted = list(first_pass)
    while True:
        batch = store.delete_empty_leaf_fields()
        if not batch:
            break
        deleted.extend(batch)

    assert set(deleted) == {empty_child.id, empty_root.id}
    assert len(deleted) == 2
    assert store.get_field(empty_child.id) is None
    assert store.get_field(empty_root.id) is None
    # The used child has a member; its parent has a child, so both survive.
    assert store.get_field(used_child.id) is not None
    assert store.get_field(used_root.id) is not None


def test_cascade_delete_removes_descendants_and_memberships(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)
    grandchild = make_field(store, "Grandchild", parent_id=child.id)
    commit(store, [root.id, child.id, grandchild.id])

    with pytest.raises(FieldStoreError, match="children"):
        store.delete_field(root.id)
    assert store.get_field(root.id) is not None

    store.delete_field(root.id, cascade=True)

    assert store.get_field(root.id) is None
    assert store.get_field(child.id) is None
    assert store.get_field(grandchild.id) is None
    assert store.stats()["fields"] == 0
    assert store.stats()["memberships"] == 0
    # The dataset row remains even when its path fields disappear.
    assert store.get_dataset(DATASET_ID) is not None
    with pytest.raises(FieldStoreError, match="missing field"):
        store.dataset_path(DATASET_ID)

    with pytest.raises(FieldStoreError, match="not found"):
        store.delete_field(root.id)


def test_delete_leaf_without_cascade(store: FieldStore) -> None:
    root = make_field(store, "Root")
    child = make_field(store, "Child", parent_id=root.id)

    store.delete_field(child.id)

    assert store.get_field(child.id) is None
    assert store.get_field(root.id) is not None
    assert store.children(root.id) == []


def test_commit_path_rejects_invalid_paths_and_dimension_mismatch(
    store: FieldStore,
) -> None:
    root = make_field(store, "Root")

    with pytest.raises(FieldStoreError, match="at least one"):
        commit(store, [])
    with pytest.raises(FieldStoreError, match="repeat"):
        commit(store, [root.id, root.id])
    with pytest.raises(FieldStoreError, match="unknown"):
        commit(store, [root.id, "fld_missing"])
    with pytest.raises(FieldStoreError, match="dimension"):
        commit(store, [root.id], embedding=[0.0] * (DIMS - 1))

    assert store.get_dataset(DATASET_ID) is None
    assert store.stats()["datasets"] == 0


def test_stats_counts(store: FieldStore) -> None:
    root_a = make_field(store, "Alpha")
    root_b = make_field(store, "Beta")
    child = make_field(store, "Child", parent_id=root_a.id)

    commit(store, [root_a.id, child.id])

    assert store.stats() == {
        "fields": 3,
        "roots": 2,
        "datasets": 1,
        "documents": 1,
        "memberships": 2,
        "links": 0,
        "comparisons": 0,
        "critiques": 0,
        "orphan_fields": 1,
    }


def test_close_is_idempotent_and_reopen_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "fields.db"
    store = FieldStore(path, dims=DIMS)
    try:
        field = make_field(store, "Persisted")
        store.close()
        store.close()
        with pytest.raises(FieldStoreError, match="closed"):
            store.all_fields()
    finally:
        store.close()

    with FieldStore(path, dims=DIMS) as reopened:
        assert reopened.get_field(field.id) == field


def test_reopen_adds_cache_key_column_to_legacy_store(tmp_path: Path) -> None:
    path = tmp_path / "fields.db"
    with FieldStore(path, dims=DIMS) as store:
        root = make_field(store, "Root")
        commit(store, [root.id])

    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE datasets DROP COLUMN cache_key")

    with FieldStore(path, dims=DIMS) as reopened:
        dataset = reopened.get_dataset(DATASET_ID)
        assert dataset is not None
        assert "cache_key" in dataset
        assert dataset["cache_key"] is None
