from __future__ import annotations

import pytest

from tests.contracts.fakes import make_document

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.hierarchy.operations import branch_field, promote_field
from whack26.hierarchy.place import attach_document
from whack26.hierarchy.types import HierarchyError
from whack26.llm import MockLLMClient
from whack26.store import FieldStore

DIMS = 2


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "llm_provider": "mock",
        "embed_provider": "mock",
        "embed_dims": DIMS,
        "desc_weight": 1.0,
        "centroid_weight": 0.0,
        "hierarchy_tau_1": 0.50,
        "hierarchy_tau_2": 0.50,
        "hierarchy_tau_3": 0.50,
        "hierarchy_tau_4": 0.50,
        "specificity_margin": 0.05,
        "promotion_margin": 0.05,
        "duplicate_cosine": 0.90,
        "max_depth": 4,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def add_field(store: FieldStore, name: str, embedding: list[float], *, parent=None):
    return store.create_field(
        parent_id=parent.id if parent is not None else None,
        name=name,
        description=f"{name} field description.",
        description_embedding=embedding,
    )


def place_document_at(store: FieldStore, identifier: str, field, embedding):
    document = make_document(identifier, embedding=embedding)
    store.upsert_document(document)
    attach_document(document, field, store=store, settings=make_settings())
    return document


def test_branch_creates_child_and_logs_create() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Research", [1.0, 0.0])
        settings = make_settings()
        before = store.hierarchy_version()

        result = branch_field(
            parent=root,
            name="Methods",
            description="Documents about research methods.",
            description_embedding=[0.0, 1.0],
            store=store,
            llm=MockLLMClient(),
            settings=settings,
        )

        assert result.created is True
        assert result.merged_into is None
        child = store.get_field(result.field_id)
        assert child is not None
        assert child.parent_id == root.id and child.level == 2
        assert result.path_ids == [root.id, child.id]
        assert store.hierarchy_version() == before + 1
        operations = store.field_operations(field_id=child.id)
        assert any(item["operation"] == "create" for item in operations)


def test_branch_merges_duplicate_sibling_instead_of_creating() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Research", [1.0, 0.0])
        sibling = add_field(store, "Methods", [1.0, 0.0], parent=root)
        fields_before = store.stats()["fields"]

        result = branch_field(
            parent=root,
            name="Research Methods",
            description="A near-duplicate description.",
            description_embedding=[1.0, 0.0],
            store=store,
            llm=MockLLMClient(),
            settings=make_settings(),
        )

        assert result.created is False
        assert result.field_id == sibling.id
        assert result.merged_into == sibling.id
        assert store.stats()["fields"] == fields_before
        operations = store.field_operations(field_id=sibling.id)
        assert any(item["operation"] == "merge" for item in operations)


def test_branch_rejects_empty_and_overlong_names() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Research", [1.0, 0.0])
        with pytest.raises(HierarchyError, match="must not be empty"):
            branch_field(
                parent=root,
                name="   ",
                description="A description.",
                description_embedding=[0.0, 1.0],
                store=store,
                llm=MockLLMClient(),
                settings=make_settings(),
            )
        with pytest.raises(HierarchyError, match="at most"):
            branch_field(
                parent=root,
                name="x" * 200,
                description="A description.",
                description_embedding=[0.0, 1.0],
                store=store,
                llm=MockLLMClient(),
                settings=make_settings(),
            )


def test_branch_at_max_depth_is_rejected() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        level1 = add_field(store, "One", [1.0, 0.0])
        level2 = add_field(store, "Two", [1.0, 0.0], parent=level1)
        level3 = add_field(store, "Three", [1.0, 0.0], parent=level2)
        level4 = add_field(store, "Four", [1.0, 0.0], parent=level3)

        with pytest.raises(HierarchyError, match="max_depth"):
            branch_field(
                parent=level4,
                name="Five",
                description="A field below the maximum depth.",
                description_embedding=[0.0, 1.0],
                store=store,
                llm=MockLLMClient(),
                settings=make_settings(),
            )


def test_branch_promotes_new_field_above_its_parent() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        llm = MockLLMClient(
            {
                "hierarchy_specificity": {
                    "relation": "new_field_is_parent",
                    "confidence": 0.9,
                    "reason": "statistics is more general than methods",
                }
            }
        )

        result = branch_field(
            parent=parent,
            name="Statistics",
            description="Documents about statistics as a discipline.",
            description_embedding=[1.0, 0.0],
            store=store,
            llm=llm,
            settings=make_settings(),
        )

        assert result.created is True
        assert result.promoted is True
        promoted = store.get_field(result.field_id)
        moved_parent = store.get_field(parent.id)
        assert promoted is not None and moved_parent is not None
        assert promoted.parent_id == grandparent.id and promoted.level == 2
        assert moved_parent.parent_id == promoted.id and moved_parent.level == 3
        assert result.path_ids == [grandparent.id, promoted.id]
        operations = store.field_operations(field_id=promoted.id)
        assert any(item["operation"] == "promote" for item in operations)


def test_branch_keeps_child_when_promotion_exceeds_max_depth() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        child = add_field(store, "Trials", [1.0, 0.0], parent=parent)
        add_field(store, "Sub-trials", [1.0, 0.0], parent=child)
        llm = MockLLMClient(
            {
                "hierarchy_specificity": {
                    "relation": "new_field_is_parent",
                    "confidence": 0.9,
                    "reason": "more general",
                }
            }
        )

        result = branch_field(
            parent=parent,
            name="Statistics",
            description="Documents about statistics as a discipline.",
            description_embedding=[0.0, 1.0],
            store=store,
            llm=llm,
            settings=make_settings(),
        )

        assert result.created is True
        assert result.promoted is False
        created = store.get_field(result.field_id)
        unchanged_parent = store.get_field(parent.id)
        assert created is not None and unchanged_parent is not None
        assert created.parent_id == parent.id and created.level == 3
        assert unchanged_parent.parent_id == grandparent.id
        assert unchanged_parent.level == 2
        assert "promotion_rejected_depth" in result.reason
        operations = store.field_operations(field_id=created.id)
        assert any(
            item["operation"] == "promote"
            and "promotion_rejected_depth" in item["reason"]
            for item in operations
        )


def test_promote_leaf_swaps_levels_and_logs_audit_row() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        new_field = add_field(store, "Statistics", [1.0, 0.0], parent=parent)
        before = store.hierarchy_version()

        result = promote_field(
            new_field,
            parent,
            store=store,
            llm=MockLLMClient(),
            settings=make_settings(),
        )

        assert result.rejected is False
        assert result.levels_changed == 2
        promoted = store.get_field(new_field.id)
        moved = store.get_field(parent.id)
        assert promoted is not None and moved is not None
        assert promoted.parent_id == grandparent.id and promoted.level == 2
        assert moved.parent_id == promoted.id and moved.level == 3
        assert store.hierarchy_version() == before + 1
        operations = store.field_operations(field_id=new_field.id)
        promote_rows = [item for item in operations if item["operation"] == "promote"]
        assert promote_rows and promote_rows[0]["new_parent_id"] == grandparent.id
        assert promote_rows[0]["old_level"] == 3
        assert promote_rows[0]["new_level"] == 2


def test_promote_subtree_increments_every_descendant_level() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        child = add_field(store, "Trials", [1.0, 0.0], parent=parent)
        new_field = add_field(store, "Statistics", [1.0, 0.0], parent=parent)

        result = promote_field(
            new_field,
            parent,
            store=store,
            llm=MockLLMClient(),
            settings=make_settings(),
        )

        assert result.rejected is False
        assert result.levels_changed == 3
        promoted = store.get_field(new_field.id)
        moved = store.get_field(parent.id)
        grandchild = store.get_field(child.id)
        assert promoted is not None and moved is not None and grandchild is not None
        assert promoted.parent_id == grandparent.id and promoted.level == 2
        assert moved.parent_id == promoted.id and moved.level == 3
        assert grandchild.parent_id == moved.id and grandchild.level == 4


def test_promotion_rejects_a_subtree_that_would_exceed_max_depth() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        child = add_field(store, "Trials", [1.0, 0.0], parent=parent)
        add_field(store, "Sub-trials", [1.0, 0.0], parent=child)
        new_field = add_field(store, "Statistics", [1.0, 0.0], parent=parent)

        result = promote_field(
            new_field,
            parent,
            store=store,
            llm=MockLLMClient(),
            settings=make_settings(),
        )

        assert result.rejected is True
        assert "promotion_rejected_depth" in result.reason
        unchanged = store.get_field(parent.id)
        unchanged_new = store.get_field(new_field.id)
        assert unchanged is not None and unchanged_new is not None
        assert unchanged.parent_id == grandparent.id and unchanged.level == 2
        assert unchanged_new.parent_id == parent.id and unchanged_new.level == 3
        operations = store.field_operations(field_id=new_field.id)
        assert any(
            item["operation"] == "promote"
            and "promotion_rejected_depth" in item["reason"]
            for item in operations
        )


def test_promotion_reassigns_closer_documents_and_keeps_the_rest() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        grandparent = add_field(store, "Science", [1.0, 0.0])
        parent = add_field(store, "Methods", [0.0, 1.0], parent=grandparent)
        new_field = add_field(store, "Statistics", [1.0, 0.0], parent=parent)

        closer = place_document_at(store, "doc_closer", parent, [1.0, 0.0])
        kept = place_document_at(store, "doc_kept", parent, [0.0, 1.0])

        result = promote_field(
            new_field,
            parent,
            store=store,
            llm=MockLLMClient(),
            settings=make_settings(),
        )

        assert result.rejected is False
        assert result.affected_documents == 2
        assert result.reassigned_documents == 1
        assert store.get_leaf(closer.id) == new_field.id
        assert store.get_leaf(kept.id) == parent.id
        assert [field.id for field in store.document_path(closer.id)] == [
            grandparent.id,
            new_field.id,
        ]
        assert [field.id for field in store.document_path(kept.id)] == [
            grandparent.id,
            new_field.id,
            parent.id,
        ]

        operations = store.field_operations(limit=100)
        assert any(
            item["operation"] == "reassign_leaf"
            and item["document_id"] == closer.id
            for item in operations
        )
        assert any(
            item["operation"] == "recompute_path"
            and item["document_id"] == kept.id
            for item in operations
        )
        assert any(item["operation"] == "promote" for item in operations)


def test_promotion_requires_a_direct_child() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Science", [1.0, 0.0])
        unrelated = add_field(store, "Statistics", [1.0, 0.0])

        with pytest.raises(HierarchyError, match="not a child"):
            promote_field(
                unrelated,
                root,
                store=store,
                llm=MockLLMClient(),
                settings=make_settings(),
            )
