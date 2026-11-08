"""End-to-end hierarchy placement through the scrape classifier adapter."""

from __future__ import annotations

from typing import Any

from tests.contracts.fakes import make_document

from whack26.config import Settings
from whack26.hierarchy.place import attach_document
from whack26.llm import MockLLMClient
from whack26.research.scrape.classify import classify_document_hierarchically
from whack26.store import FieldStore

DIMS = 2


class FixedEmbeddings:
    """Return a deterministic vector so promotion and reassignment are stable."""

    def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def make_settings() -> Settings:
    return Settings(
        llm_provider="mock",
        embed_provider="mock",
        embed_dims=DIMS,
        desc_weight=1.0,
        centroid_weight=0.0,
        hierarchy_tau_1=0.50,
        hierarchy_tau_2=0.87,
        hierarchy_tau_3=0.87,
        hierarchy_tau_4=0.87,
        specificity_margin=0.05,
        promotion_margin=0.05,
        max_depth=4,
    )


def add_field(
    store: FieldStore,
    name: str,
    embedding: list[float],
    *,
    parent_id: str | None = None,
):
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} field description.",
        description_embedding=embedding,
    )


def test_hierarchy_placement_promotes_and_reassigns() -> None:
    settings = make_settings()
    with FieldStore(":memory:", dims=DIMS, max_depth=4) as store:
        root = add_field(store, "Science", [0.8, 0.6])
        methods = add_field(store, "Methods", [0.9, 0.43589], parent_id=root.id)

        existing = make_document("doc_existing", embedding=[1.0, 0.0])
        store.upsert_document(existing)
        attach_document(existing, methods, store=store, settings=settings)

        new_document = make_document("doc_new", embedding=[1.0, 0.0])
        store.upsert_document(new_document)

        llm = MockLLMClient(
            fixtures={
                "hierarchy_field": {
                    "name": "Statistics",
                    "description": "Statistical methods and inference.",
                    "reason": "the document is about statistics",
                },
                "hierarchy_specificity": {
                    "relation": "new_field_is_parent",
                    "confidence": 0.9,
                    "reason": "statistics is more general than methods",
                },
            }
        )

        path = classify_document_hierarchically(
            new_document,
            store=store,
            llm=llm,
            embeddings=FixedEmbeddings(),
            settings=settings,
        )

        statistics = store.find_sibling(root.id, "Statistics")
        assert statistics is not None
        assert path == [root.id, statistics.id]
        assert statistics.level == 2
        moved_methods = store.get_field(methods.id)
        assert moved_methods is not None
        assert moved_methods.parent_id == statistics.id
        assert moved_methods.level == 3

        # The existing document was closer to the promoted field, so it moved.
        existing_path = store.document_path(existing.id)
        assert existing_path[-1].id == statistics.id
        assert store.get_leaf(existing.id) == statistics.id

        operations = store.field_operations(field_id=statistics.id)
        assert any(item["operation"] == "promote" for item in operations)
        assert any(item["operation"] == "reassign_leaf" for item in operations)
        assert store.hierarchy_version() >= 1


def test_hierarchy_placement_attaches_when_already_covered() -> None:
    settings = make_settings()
    with FieldStore(":memory:", dims=DIMS, max_depth=4) as store:
        root = add_field(store, "Housing", [1.0, 0.0])
        document = make_document("doc_housing", embedding=[1.0, 0.0])
        store.upsert_document(document)

        path = classify_document_hierarchically(
            document,
            store=store,
            llm=MockLLMClient(),
            embeddings=FixedEmbeddings(),
            settings=settings,
        )

        assert path == [root.id]
        operations = store.field_operations()
        assert any(
            item["operation"] == "reassign_leaf"
            and item["document_id"] == document.id
            for item in operations
        )
