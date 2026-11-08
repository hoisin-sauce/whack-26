"""Tests for the document catalog query API."""

from __future__ import annotations

from pathlib import Path

import pytest

from whack26.research.catalog import (
    CatalogError,
    CatalogQuery,
    catalog_research_records,
    search_documents,
)
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

DIMS = 2


def make_document(
    identifier: str,
    *,
    kind: str = "paper",
    source: str = "openalex",
    title: str | None = None,
    summary: str = "A summary.",
    embedding: list[float] | None = None,
    credible: bool = False,
    credibility_score: float = 0.0,
    preloaded: bool = False,
    published_at: str | None = None,
    metadata: dict | None = None,
) -> Document:
    return Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source=source,
        title=title or identifier,
        summary=summary,
        summary_embedding=embedding if embedding is not None else [1.0, 0.0],
        embedding_model="mock@2",
        credible=credible,
        credibility_score=credibility_score,
        preloaded=preloaded,
        published_at=published_at,
        metadata=metadata or {},
        status="classified",
    )


def test_catalog_research_records_returns_related_credible_documents(
    store: FieldStore,
) -> None:
    dataset = make_document(
        "ds_target",
        kind="dataset",
        source="user",
        title="Target dataset",
        summary="The user's dataset.",
        embedding=[1.0, 0.0],
        credible=True,
    )
    related = make_document(
        "pap_related",
        title="Related paper",
        summary="A related paper.",
        embedding=[1.0, 0.0],
        credible=True,
        published_at="2021",
        metadata={"doi": "10.1234/example"},
    )
    unrelated = make_document(
        "pap_unrelated",
        title="Unrelated paper",
        summary="An unrelated paper.",
        embedding=[0.0, 1.0],
        credible=True,
    )
    low_credibility = make_document(
        "pap_low",
        title="Low credibility",
        summary="A low-credibility paper.",
        embedding=[1.0, 0.0],
        credible=False,
    )
    for document in (dataset, related, unrelated, low_credibility):
        store.upsert_document(document)

    records = catalog_research_records(
        store, dataset_id=dataset.id, limit=10
    )

    assert [record.id for record in records] == ["pap_related"]
    record = records[0]
    assert record.source == "catalog"
    assert record.query == "catalog"
    assert record.doi == "10.1234/example"
    assert record.year == 2021
    assert record.relevance == pytest.approx(1.0)


def make_field(store: FieldStore, name: str, *, parent_id: str | None = None):
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} description.",
        description_embedding=[1.0, 0.0],
    )


@pytest.fixture
def store(tmp_path: Path):
    with FieldStore(tmp_path / "catalog.db", dims=DIMS) as field_store:
        yield field_store


@pytest.fixture
def catalog(store: FieldStore) -> dict[str, object]:
    root = make_field(store, "Housing")
    child = make_field(store, "Prices", parent_id=root.id)
    grandchild = make_field(store, "Rents", parent_id=child.id)
    health = make_field(store, "Health")

    documents = [
        make_document(
            "pap_a",
            title="Housing prices",
            summary="A study of housing prices.",
            embedding=[1.0, 0.0],
            credible=True,
            credibility_score=0.9,
            published_at="2020-05-01",
        ),
        make_document(
            "pap_b",
            source="arxiv",
            title="Housing rents",
            summary="Rent determinants.",
            embedding=[0.8, 0.6],
            credibility_score=0.6,
            published_at="2022",
        ),
        make_document(
            "ds_c",
            kind="dataset",
            source="huggingface",
            title="Housing dataset",
            summary="A housing dataset.",
            embedding=[0.0, 1.0],
            credible=True,
            credibility_score=0.4,
            preloaded=True,
            metadata={"year": 2021},
        ),
        make_document(
            "web_d",
            kind="web",
            source="websearch",
            title="Health outcomes",
            summary="An article about health.",
            embedding=[0.6, 0.8],
            credibility_score=0.9,
            published_at="unknown",
        ),
    ]
    for document in documents:
        store.upsert_document(document)

    store.assign_document_path("pap_a", [root.id, child.id])
    store.assign_document_path("pap_b", [root.id, child.id, grandchild.id])
    store.assign_document_path("ds_c", [child.id])
    store.assign_document_path("web_d", [health.id])

    return {
        "root": root,
        "child": child,
        "grandchild": grandchild,
        "health": health,
    }


def ids(documents: list[Document]) -> list[str]:
    return [document.id for document in documents]


def test_kind_and_source_filters(store: FieldStore, catalog: dict) -> None:
    assert ids(search_documents(store, CatalogQuery(kinds=["paper"]))) == [
        "pap_a",
        "pap_b",
    ]
    assert ids(search_documents(store, CatalogQuery(sources=["arxiv"]))) == [
        "pap_b"
    ]
    assert ids(
        search_documents(
            store, CatalogQuery(kinds=["PAPER"], sources=["openalex"])
        )
    ) == ["pap_a"]
    assert ids(
        search_documents(store, CatalogQuery(kinds=["dataset", "web"]))
    ) == ["web_d", "ds_c"]
    # Duplicate and blank values collapse.
    assert ids(
        search_documents(store, CatalogQuery(sources=["arxiv", "arxiv", ""]))
    ) == ["pap_b"]


def test_field_subtree_filter(store: FieldStore, catalog: dict) -> None:
    root = catalog["root"]
    child = catalog["child"]
    grandchild = catalog["grandchild"]
    health = catalog["health"]

    assert ids(
        search_documents(
            store, CatalogQuery(field_id=root.id, include_descendants=True)
        )
    ) == ["pap_a", "pap_b", "ds_c"]
    assert ids(
        search_documents(
            store, CatalogQuery(field_id=root.id, include_descendants=False)
        )
    ) == ["pap_a", "pap_b"]
    assert ids(
        search_documents(
            store, CatalogQuery(field_id=child.id, include_descendants=True)
        )
    ) == ["pap_a", "pap_b", "ds_c"]
    assert ids(
        search_documents(
            store, CatalogQuery(field_id=child.id, include_descendants=False)
        )
    ) == ["pap_a", "pap_b", "ds_c"]
    assert ids(
        search_documents(
            store, CatalogQuery(field_id=grandchild.id, include_descendants=True)
        )
    ) == ["pap_b"]
    assert ids(
        search_documents(
            store, CatalogQuery(field_id=health.id, include_descendants=True)
        )
    ) == ["web_d"]
    assert search_documents(store, CatalogQuery(field_id="fld_missing")) == []


def test_year_filter_uses_published_at_and_metadata(
    store: FieldStore, catalog: dict
) -> None:
    assert ids(search_documents(store, CatalogQuery(year_min=2021))) == [
        "pap_b",
        "ds_c",
    ]
    assert ids(search_documents(store, CatalogQuery(year_max=2020))) == ["pap_a"]
    assert ids(
        search_documents(
            store, CatalogQuery(year_min=2021, year_max=2022)
        )
    ) == ["pap_b", "ds_c"]
    # The invalid "unknown" date and the missing date are excluded.
    assert "web_d" not in ids(
        search_documents(store, CatalogQuery(year_min=1900))
    )


def test_credibility_and_preloaded_filters(
    store: FieldStore, catalog: dict
) -> None:
    assert ids(search_documents(store, CatalogQuery(credible_only=True))) == [
        "pap_a",
        "ds_c",
    ]
    assert ids(search_documents(store, CatalogQuery(preloaded=True))) == ["ds_c"]
    assert ids(search_documents(store, CatalogQuery(preloaded=False))) == [
        "pap_a",
        "web_d",
        "pap_b",
    ]
    assert ids(
        search_documents(
            store, CatalogQuery(credible_only=True, preloaded=True)
        )
    ) == ["ds_c"]


def test_text_filter_is_case_insensitive(
    store: FieldStore, catalog: dict
) -> None:
    assert ids(search_documents(store, CatalogQuery(text="rents"))) == ["pap_b"]
    assert ids(search_documents(store, CatalogQuery(text="HOUSING"))) == [
        "pap_a",
        "pap_b",
        "ds_c",
    ]
    assert ids(search_documents(store, CatalogQuery(text="rent determinants"))) == [
        "pap_b"
    ]
    assert search_documents(store, CatalogQuery(text="not present")) == []


def test_similarity_ordering_by_document(store: FieldStore, catalog: dict) -> None:
    results = search_documents(store, CatalogQuery(similar_to="pap_a"))

    # pap_a is the reference and is excluded; the rest order by cosine:
    # pap_b 0.8, web_d 0.6, ds_c 0.0.
    assert ids(results) == ["pap_b", "web_d", "ds_c"]


def test_similarity_ordering_by_text(store: FieldStore, catalog: dict) -> None:
    class FixedEmbeddings:
        def embed(self, texts: list[str], *, model: str | None = None):
            return [[1.0, 0.0] for _ in texts]

    results = search_documents(
        store,
        CatalogQuery(similar_to="housing prices"),
        embeddings=FixedEmbeddings(),
    )

    # The reference is text, so pap_a remains in the result with similarity 1.
    assert ids(results) == ["pap_a", "pap_b", "web_d", "ds_c"]

    with pytest.raises(CatalogError, match="no embedding client"):
        search_documents(store, CatalogQuery(similar_to="not a document"))


def test_empty_results(store: FieldStore, catalog: dict) -> None:
    assert search_documents(store, CatalogQuery(kinds=["search"])) == []
    assert search_documents(store, CatalogQuery(text="zzzz")) == []
    assert search_documents(store, CatalogQuery(limit=0)) == []


def test_pagination_and_deterministic_order(
    store: FieldStore, catalog: dict
) -> None:
    # credibility desc, then year desc (unknown last), then id asc:
    # pap_a (0.9, 2020), web_d (0.9, unknown), pap_b (0.6, 2022), ds_c (0.4, 2021)
    expected = ["pap_a", "web_d", "pap_b", "ds_c"]
    assert ids(search_documents(store, CatalogQuery())) == expected
    assert ids(search_documents(store, CatalogQuery())) == expected

    assert ids(search_documents(store, CatalogQuery(limit=2))) == expected[:2]
    assert ids(
        search_documents(store, CatalogQuery(limit=2, offset=2))
    ) == expected[2:]
    assert ids(search_documents(store, CatalogQuery(limit=2, offset=5))) == []
    assert ids(search_documents(store, CatalogQuery(offset=1, limit=1))) == [
        expected[1]
    ]


def test_id_breaks_full_ties(store: FieldStore) -> None:
    for identifier in ("doc_z", "doc_y", "doc_x"):
        store.upsert_document(
            make_document(
                identifier,
                credibility_score=0.5,
                published_at="2020",
            )
        )

    assert ids(search_documents(store, CatalogQuery())) == [
        "doc_x",
        "doc_y",
        "doc_z",
    ]
