from __future__ import annotations

import json
import math

import pytest

from tests.contracts.fakes import make_document

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.hierarchy.place import place_document
from whack26.llm import MockLLMClient
from whack26.research.scrape.models import Document
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


def add_field(
    store: FieldStore,
    name: str,
    embedding: list[float],
    *,
    parent=None,
):
    return store.create_field(
        parent_id=parent.id if parent is not None else None,
        name=name,
        description=f"{name} field description.",
        description_embedding=embedding,
    )


def store_document(store: FieldStore, identifier: str, embedding: list[float]):
    document = make_document(identifier, embedding=embedding)
    store.upsert_document(document)
    return document


def test_attach_at_root_when_root_directly_covers_document() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Statistics", [1.0, 0.0])
        document = store_document(store, "doc_root", [1.0, 0.0])
        llm = MockLLMClient()

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert result.leaf_field_id == root.id
        assert result.path_ids == [root.id]
        assert result.created_fields == []
        assert store.get_leaf(document.id) == root.id
        assert [field.id for field in store.document_path(document.id)] == [root.id]
        assert llm.calls == []


def test_descend_when_child_is_direct_and_more_specific() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Statistics", [0.8, 0.6])
        child = add_field(store, "Regression", [1.0, 0.0], parent=root)
        document = store_document(store, "doc_regression", [1.0, 0.0])

        result = place_document(
            document,
            store=store,
            llm=MockLLMClient(),
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert result.leaf_field_id == child.id
        assert result.path_ids == [root.id, child.id]
        assert [
            field.id for field in store.document_path(document.id)
        ] == [root.id, child.id]


def test_branch_when_no_child_passes_and_current_is_only_weakly_related() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(
            store, "Urban Economics", [0.52, math.sqrt(1.0 - 0.52**2)]
        )
        document = store_document(store, "doc_rent", [1.0, 0.0])
        llm = MockLLMClient(
            {
                "hierarchy_field": {
                    "name": "Housing Prices",
                    "description": "Documents about housing prices and rents.",
                    "reason": "narrower concept",
                }
            }
        )

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(duplicate_cosine=1.0),
        )

        assert result.leaf_field_id != root.id
        assert len(result.created_fields) == 1
        assert result.created_fields[0] == result.leaf_field_id
        assert result.path_ids[0] == root.id
        assert result.path_ids[-1] == result.leaf_field_id
        created = store.get_field(result.leaf_field_id)
        assert created is not None
        assert created.parent_id == root.id
        assert store.get_leaf(document.id) == result.leaf_field_id


def test_create_root_when_no_existing_root_is_directly_related() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        document = store_document(store, "doc_new_domain", [1.0, 0.0])
        llm = MockLLMClient(
            {
                "hierarchy_field": {
                    "name": "Urban Studies",
                    "description": "Documents about cities and urban systems.",
                    "reason": "no root fits",
                }
            }
        )

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert result.created_fields == [result.leaf_field_id]
        assert result.path_ids == [result.leaf_field_id]
        assert store.stats()["roots"] == 1
        root = store.get_field(result.leaf_field_id)
        assert root is not None and root.parent_id is None and root.level == 1
        assert store.get_leaf(document.id) == root.id


def test_max_depth_attaches_instead_of_branching() -> None:
    with FieldStore(":memory:", dims=DIMS, max_depth=3) as store:
        root = add_field(store, "Research", [1.0, 0.0])
        middle = add_field(store, "Methods", [1.0, 0.0], parent=root)
        leaf = add_field(store, "Statistics", [1.0, 0.0], parent=middle)
        document = store_document(store, "doc_deep", [1.0, 0.0])
        llm = MockLLMClient(
            {
                "hierarchy_specificity": {
                    "relation": "needs_child",
                    "confidence": 0.8,
                    "reason": "narrower",
                }
            }
        )
        settings = make_settings(max_depth=3)

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=settings,
        )

        assert result.leaf_field_id == leaf.id
        assert result.path_ids == [root.id, middle.id, leaf.id]
        assert result.created_fields == []
        assert store.stats()["fields"] == 3
        assert llm.calls.count("hierarchy_specificity") == 2


def test_deterministic_margin_avoids_the_judge() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Statistics", [0.8, 0.6])
        add_field(store, "Regression", [1.0, 0.0], parent=root)
        document = store_document(store, "doc_margin", [1.0, 0.0])
        llm = MockLLMClient()

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert store.get_field(result.leaf_field_id).name == "Regression"
        assert llm.calls == []


def test_ambiguous_margin_calls_the_judge_and_attaches_at_current() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(
            store, "Statistics", [0.98, math.sqrt(1.0 - 0.98**2)]
        )
        add_field(store, "Regression", [1.0, 0.0], parent=root)
        document = store_document(store, "doc_ambiguous", [1.0, 0.0])
        llm = MockLLMClient(
            {
                "hierarchy_specificity": {
                    "relation": "attach_here",
                    "confidence": 0.7,
                    "reason": "root already covers the document",
                }
            }
        )

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert result.leaf_field_id == root.id
        assert llm.calls.count("hierarchy_specificity") == 1


def test_embedding_is_computed_when_the_summary_embedding_is_missing() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        document = make_document("doc_no_embedding")
        document = document.model_copy(update={"summary_embedding": []})
        store.upsert_document(document)
        embeddings = MockEmbeddingClient(DIMS)
        llm = MockLLMClient(
            {
                "hierarchy_field": {
                    "name": "General Research",
                    "description": "General research documents.",
                    "reason": "fallback root",
                }
            }
        )

        result = place_document(
            document,
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=make_settings(),
        )

        assert embeddings.calls
        assert embeddings.calls[0] == [document.summary]
        assert store.get_leaf(document.id) == result.leaf_field_id


def test_dataset_path_json_and_ordinals_stay_consistent_after_attach() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        root = add_field(store, "Statistics", [0.8, 0.6])
        child = add_field(store, "Regression", [1.0, 0.0], parent=root)
        store.commit_path(
            dataset_id="ds_housing",
            sha256="abc123",
            profile_json="{}",
            summary="A tabular statistics dataset.",
            embedding=[1.0, 0.0],
            path_json=json.dumps([root.id]),
            confidence=1.0,
            refined_after_research=False,
            field_ids=[root.id],
        )
        document = store.get_document("ds_housing")
        assert document is not None

        result = place_document(
            document,
            store=store,
            llm=MockLLMClient(),
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert result.path_ids == [root.id, child.id]
        dataset = store.get_dataset("ds_housing")
        assert dataset is not None
        assert json.loads(dataset["path_json"]) == [root.id, child.id]
        rows = store._connection().execute(
            "SELECT field_id, ordinal FROM document_fields "
            "WHERE document_id = ? ORDER BY ordinal",
            ("ds_housing",),
        ).fetchall()
        assert [(row["field_id"], row["ordinal"]) for row in rows] == [
            (root.id, 0),
            (child.id, 1),
        ]


def test_missing_document_embedding_and_client_is_an_error() -> None:
    with FieldStore(":memory:", dims=DIMS) as store:
        add_field(store, "Statistics", [1.0, 0.0])
        document = make_document("doc_broken")
        document = document.model_copy(update={"summary_embedding": []})
        store.upsert_document(document)

        with pytest.raises(Exception, match="no embedding"):
            place_document(
                document,
                store=store,
                llm=MockLLMClient(),
                embeddings=None,
                settings=make_settings(),
            )
