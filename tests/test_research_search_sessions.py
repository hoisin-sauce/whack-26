"""Tests for search-session documents and the search graph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whack26.research.scrape.models import Document
from whack26.research.search_sessions import (
    SearchSession,
    SearchSessionError,
    build_search_document,
    link_dataset_document,
    link_search_follow_up,
    record_search_session,
    search_document_id,
)
from whack26.store import FieldStore

DIMS = 2


def make_document(
    identifier: str,
    *,
    kind: str = "paper",
    source: str = "openalex",
    title: str | None = None,
    summary: str = "A summary.",
    canonical_url: str | None = None,
    embedding: list[float] | None = None,
    credible: bool = False,
    credibility_score: float = 0.0,
) -> Document:
    return Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source=source,
        title=title or identifier,
        summary=summary,
        summary_embedding=embedding if embedding is not None else [1.0, 0.0],
        embedding_model="mock@2",
        canonical_url=canonical_url,
        credible=credible,
        credibility_score=credibility_score,
        status="classified",
    )


@pytest.fixture
def store(tmp_path: Path):
    with FieldStore(tmp_path / "search.db", dims=DIMS) as field_store:
        yield field_store


def test_one_search_document_per_query(store: FieldStore) -> None:
    store.upsert_document(make_document("pap_a"))
    store.upsert_document(make_document("pap_b"))

    first = record_search_session(
        store,
        SearchSession(
            query="housing prices",
            source="openalex",
            session_id="run-1",
            parameters={"limit": 10},
            result_ids=["pap_a", "pap_b"],
        ),
    )
    second = record_search_session(
        store,
        SearchSession(
            query="rent determinants",
            source="arxiv",
            session_id="run-1",
            parameters={"limit": 5},
            result_ids=["pap_b"],
        ),
    )

    assert first.id != second.id
    assert first.id.startswith("srch_")
    assert second.id.startswith("srch_")
    assert store.stats()["documents"] == 4
    assert store.get_document(first.id).kind == "search"  # type: ignore[union-attr]

    metadata = first.metadata
    assert metadata["query"] == "housing prices"
    assert metadata["source"] == "openalex"
    assert metadata["parameters"] == {"limit": 10}
    assert metadata["result_count"] == 2
    assert metadata["session_id"] == "run-1"
    assert metadata["timestamp"]
    assert first.title == "housing prices"
    assert "housing prices" in first.summary
    assert "openalex" in first.summary
    assert "results=2" in first.summary
    assert 'parameters={"limit":10}' in first.summary


def test_search_result_and_follow_up_links(store: FieldStore) -> None:
    store.upsert_document(
        make_document(
            "pap_a", title="A paper", canonical_url="https://example.test/a"
        )
    )
    store.upsert_document(make_document("pap_b", title="B paper"))
    store.upsert_document(make_document("pap_c", title="C paper"))

    first = record_search_session(
        store,
        query="housing",
        source="openalex",
        session_id="run-1",
        result_ids=["pap_a", "pap_b"],
    )
    follow_up = record_search_session(
        store,
        query="housing rents",
        source="arxiv",
        session_id="run-1",
        result_ids=["pap_c"],
        follow_up_of=first.id,
    )

    outgoing = store.document_links(first.id)
    by_target = {(link.to_id, link.relation) for link in outgoing}
    assert ("pap_a", "search_result") in by_target
    assert ("pap_b", "search_result") in by_target
    assert (follow_up.id, "search_result") in by_target

    result_link = next(link for link in outgoing if link.to_id == "pap_a")
    assert result_link.url == "https://example.test/a"
    assert result_link.anchor_text == "A paper"

    assert [link.to_id for link in store.document_links(follow_up.id)] == ["pap_c"]
    incoming = store.document_links(follow_up.id, direction="in")
    assert [(link.from_id, link.relation) for link in incoming] == [
        (first.id, "search_result")
    ]


def test_dataset_links_to_search_and_paper(store: FieldStore) -> None:
    store.upsert_document(make_document("ds_home", kind="dataset", source="user"))
    store.upsert_document(make_document("pap_a", title="A paper"))
    search = record_search_session(
        store,
        query="housing",
        source="openalex",
        session_id="run-1",
        result_ids=["pap_a"],
        dataset_ids=["ds_home"],
    )
    link_dataset_document(store, "ds_home", "pap_a")

    outgoing = {
        (link.to_id, link.relation)
        for link in store.document_links("ds_home")
    }
    assert ("pap_a", "dataset_paper") in outgoing
    assert (search.id, "search_result") in outgoing

    # The explicit helper also validates the endpoint kinds.
    with pytest.raises(SearchSessionError, match="not a dataset"):
        link_dataset_document(store, "pap_a", search.id)


def test_follow_up_and_result_validation(store: FieldStore) -> None:
    search = record_search_session(
        store,
        query="housing",
        source="openalex",
        session_id="run-1",
    )
    other = record_search_session(
        store,
        query="health",
        source="openalex",
        session_id="run-1",
    )

    link_search_follow_up(store, search.id, other.id, relation="linked_from")
    assert store.document_links(search.id)[0].relation == "linked_from"

    with pytest.raises(SearchSessionError, match="prior search not found"):
        link_search_follow_up(store, "pap_missing", other.id)

    with pytest.raises(SearchSessionError, match="result document not found"):
        record_search_session(
            store,
            query="new query",
            source="openalex",
            session_id="run-1",
            result_ids=["pap_missing"],
        )


def test_rerun_is_idempotent(store: FieldStore) -> None:
    store.upsert_document(make_document("pap_a"))
    store.upsert_document(make_document("pap_b"))
    session = SearchSession(
        query="housing prices",
        source="openalex",
        session_id="run-1",
        parameters={"limit": 10},
        result_ids=["pap_a", "pap_b"],
    )

    first = record_search_session(store, session)
    documents_after_first = store.stats()["documents"]
    links_after_first = store.stats()["links"]
    timestamp = first.metadata["timestamp"]

    second = record_search_session(store, session)

    assert second.id == first.id
    assert store.stats()["documents"] == documents_after_first == 3
    assert store.stats()["links"] == links_after_first == 2
    assert second.metadata["timestamp"] == timestamp
    assert {link.to_id for link in store.document_links(first.id)} == {
        "pap_a",
        "pap_b",
    }


def test_dedup_of_queries_results_and_sessions(store: FieldStore) -> None:
    store.upsert_document(make_document("pap_a"))

    first = record_search_session(
        store,
        query="  Housing   Prices ",
        source="OpenAlex",
        session_id="run-1",
        parameters={"limit": 10},
        result_ids=["pap_a", "pap_a"],
    )
    same = record_search_session(
        store,
        query="housing prices",
        source="openalex",
        session_id="run-1",
        parameters={"limit": 10},
        result_ids=["pap_a"],
    )
    different_session = record_search_session(
        store,
        query="housing prices",
        source="openalex",
        session_id="run-2",
        parameters={"limit": 10},
        result_ids=["pap_a"],
    )
    different_parameters = record_search_session(
        store,
        query="housing prices",
        source="openalex",
        session_id="run-1",
        parameters={"limit": 5},
        result_ids=["pap_a"],
    )

    assert same.id == first.id
    assert different_session.id != first.id
    assert different_parameters.id != first.id
    assert store.stats()["documents"] == 4  # one result + three search docs
    assert len(store.document_links(first.id)) == 1

    # Ids are derived only from stable, redacted identity fields.
    assert search_document_id(
        SearchSession(
            query=" housing  prices ",
            source=" OPENALEX ",
            session_id="run-1",
            parameters={"limit": 10},
        )
    ) == first.id


def test_search_documents_contain_no_secrets(store: FieldStore) -> None:
    session = SearchSession(
        query="housing token=abc123 api_key=sk-live",
        source="openalex",
        session_id="run-1",
        parameters={
            "api_key": "sk-xyz",
            "Authorization": "Bearer top-secret",
            "limit": 10,
            "nested": {"password": "hunter2"},
        },
    )

    document = record_search_session(store, session)
    rendered = json.dumps(
        {
            "title": document.title,
            "summary": document.summary,
            "metadata": document.metadata,
        }
    )

    for secret in (
        "abc123",
        "sk-live",
        "sk-xyz",
        "top-secret",
        "hunter2",
    ):
        assert secret not in rendered
    assert "[redacted]" in rendered
    assert document.metadata["parameters"]["limit"] == 10


def test_build_search_document_defaults_and_status() -> None:
    session = SearchSession(
        query="housing",
        source="openalex",
        session_id="run-1",
        status="compared",
        result_ids=["pap_a", "pap_a", "pap_b"],
        timestamp="2026-11-08T10:00:00Z",
    )

    document = build_search_document(session)

    assert document.id == search_document_id(session)
    assert document.status == "compared"
    assert document.metadata["result_count"] == 2
    assert document.metadata["timestamp"] == "2026-11-08T10:00:00Z"
    assert document.summary_embedding == []


def test_minimal_store_using_put_document() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.documents: dict[str, Document] = {}
            self.links: list = []

        def put_document(self, document: Document) -> Document:
            self.documents[document.id] = document
            return document

        def add_document_link(self, link) -> None:
            self.links.append(link)

    fake = FakeStore()
    document = record_search_session(
        fake,
        query="housing",
        source="openalex",
        session_id="run-1",
        result_ids=["pap_a"],
    )

    assert fake.documents[document.id] is document
    assert [(link.from_id, link.to_id, link.relation) for link in fake.links] == [
        (document.id, "pap_a", "search_result")
    ]

