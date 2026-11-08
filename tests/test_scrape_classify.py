from __future__ import annotations

from typing import Any

import pytest

from whack26.config import Settings
from whack26.embeddings import MockEmbeddingClient
from whack26.llm import MockLLMClient
from whack26.research.scrape.classify import (
    DOCUMENT_CHOICE_SCHEMA,
    DOCUMENT_SYSTEM_PROMPT,
    classify_document,
)
from whack26.research.scrape.models import Document
from whack26.store import Field, FieldStore

DIMS = 2


def make_settings(*, max_depth: int = 4) -> Settings:
    return Settings(embed_dims=DIMS, max_depth=max_depth)


def make_store(*, max_depth: int = 4) -> FieldStore:
    return FieldStore(":memory:", dims=DIMS, max_depth=max_depth)


def make_field(
    store: FieldStore,
    name: str,
    *,
    parent_id: str | None = None,
    embedding: tuple[float, ...] = (1.0, 0.0),
) -> Field:
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} description for documents.",
        description_embedding=list(embedding),
    )


def make_document(
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    summary: str = "A study of housing prices and residential demand.",
    title: str = "Housing prices over time",
) -> Document:
    return Document(
        id="doc_paper_1",
        kind="paper",
        source="openalex",
        source_id="W123",
        title=title,
        summary=summary,
        summary_embedding=list(embedding),
        embedding_model="mock@2",
        status="summarized",
    )


def make_llm(payload: dict[str, Any]) -> MockLLMClient:
    return MockLLMClient(fixtures={"document_classification": payload})


def test_document_choice_schema_and_prompt_are_document_specific() -> None:
    properties = DOCUMENT_CHOICE_SCHEMA["properties"]
    assert properties["action"]["enum"] == ["assign", "create", "stop"]
    assert properties["field_id"]["maxLength"] == 64
    assert set(DOCUMENT_CHOICE_SCHEMA["required"]) == {"action", "reason"}
    assert DOCUMENT_CHOICE_SCHEMA["propertyOrdering"][0] == "action"
    assert "research document" in DOCUMENT_SYSTEM_PROMPT
    assert "not a dataset" in DOCUMENT_SYSTEM_PROMPT
    assert "Never create a new root" in DOCUMENT_SYSTEM_PROMPT


def test_classify_document_assigns_an_existing_field() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        document = make_document()
        store.upsert_document(document)
        llm = make_llm(
            {"action": "assign", "field_id": root.id, "reason": "housing fits"}
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == [root.id]
        assert llm.calls and llm.calls[0] == "document_classification"
    finally:
        store.close()


def test_classify_document_assigns_an_existing_child() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        child = make_field(store, "Housing Markets", parent_id=root.id)
        document = make_document()
        store.upsert_document(document)
        llm = make_llm(
            {"action": "assign", "field_id": child.id, "reason": "the child fits"}
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == [root.id, child.id]
        assert store.document_path(document.id)[-1].name == "Housing Markets"
    finally:
        store.close()


def test_classify_document_creates_a_child_when_no_sibling_fits() -> None:
    store = make_store(max_depth=2)
    try:
        root = make_field(store, "Housing")
        document = make_document()
        store.upsert_document(document)
        llm = make_llm(
            {
                "action": "create",
                "name": "Housing Prices",
                "description": "Prices and demand in residential housing markets.",
                "reason": "no existing sibling is specific enough",
            }
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(max_depth=2),
        )

        assert len(path) == 2
        assert path[0] == root.id
        child = store.get_field(path[1])
        assert child is not None
        assert child.parent_id == root.id
        assert child.level == 2
        assert child.name == "Housing Prices"
        # Creating a child must not create another root.
        assert [item.id for item in store.roots()] == [root.id]
    finally:
        store.close()


def test_classify_document_stops_at_the_current_leaf() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        document = make_document()
        store.upsert_document(document)
        llm = make_llm({"action": "stop", "reason": "the current field fits"})

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == [root.id]
        assert [item.id for item in store.document_path(document.id)] == [root.id]
    finally:
        store.close()


def test_classify_document_falls_back_to_the_top_candidate() -> None:
    store = make_store()
    try:
        top = make_field(store, "Housing", embedding=(1.0, 0.0))
        make_field(store, "Biology", embedding=(0.0, 1.0))
        document = make_document(embedding=(1.0, 0.0))
        store.upsert_document(document)
        llm = make_llm(
            {
                "action": "assign",
                "field_id": "fld_does_not_exist",
                "reason": "invalid candidate",
            }
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == [top.id]
        # The invalid response is retried once per level, then the fallback
        # chooses the top-scoring root deterministically.
        assert len(llm.calls) >= 2
    finally:
        store.close()


def test_classify_document_falls_back_on_malformed_schema() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        document = make_document()
        store.upsert_document(document)
        llm = make_llm({"action": "invent", "reason": "not an allowed action"})

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == [root.id]
        assert store.stats()["fields"] == 1
    finally:
        store.close()


def test_classify_document_never_creates_a_new_root() -> None:
    store = make_store()
    try:
        document = make_document()
        store.upsert_document(document)
        llm = make_llm(
            {
                "action": "create",
                "name": "Housing",
                "description": "Housing markets and residential property.",
                "reason": "should not create a root",
            }
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        assert path == []
        assert store.roots() == []
        assert store.stats()["fields"] == 0
        assert llm.calls == []
    finally:
        store.close()


def test_classify_document_persists_the_path_in_document_fields() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        child = make_field(store, "Housing Markets", parent_id=root.id)
        document = make_document()
        store.upsert_document(document)
        llm = make_llm(
            {"action": "assign", "field_id": child.id, "reason": "the child fits"}
        )

        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=MockEmbeddingClient(DIMS),
            settings=make_settings(),
        )

        rows = store._connection().execute(
            "SELECT field_id, ordinal FROM document_fields "
            "WHERE document_id = ? ORDER BY ordinal",
            (document.id,),
        ).fetchall()
        assert path == [root.id, child.id]
        assert [(row["field_id"], row["ordinal"]) for row in rows] == [
            (root.id, 0),
            (child.id, 1),
        ]
    finally:
        store.close()


def test_classify_document_uses_the_stored_summary_embedding() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        document = make_document(embedding=(1.0, 0.0))
        store.upsert_document(document)
        embeddings = MockEmbeddingClient(DIMS)
        llm = make_llm(
            {"action": "assign", "field_id": root.id, "reason": "the root fits"}
        )

        classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=make_settings(),
        )

        assert embeddings.calls == []
    finally:
        store.close()


def test_classify_document_does_not_mutate_its_input() -> None:
    store = make_store()
    try:
        root = make_field(store, "Housing")
        # Deliberately missing the embedding so the classifier must embed the
        # summary; the function must not write that vector back to the document.
        document = make_document(embedding=())
        store.upsert_document(document)
        before = document.model_dump()
        embeddings = MockEmbeddingClient(DIMS)
        llm = make_llm(
            {"action": "assign", "field_id": root.id, "reason": "the root fits"}
        )

        classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=make_settings(),
        )

        assert document.model_dump() == before
        assert document.summary_embedding == []
        assert embeddings.calls == [[document.summary]]
    finally:
        store.close()
