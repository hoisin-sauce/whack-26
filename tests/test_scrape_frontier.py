"""Tests for the semantic-scraping frontier."""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from whack26.config import Settings
from whack26.research.scrape.frontier import expand_frontier
from whack26.research.scrape.identity import canonicalize_url, url_hash
from whack26.research.scrape.links import LinkCandidate
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

DIMS = 4
TARGET = [1.0, 0.0, 0.0, 0.0]


def vec(x: float, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> list[float]:
    return [x, y, z, w]


def at_similarity(value: float) -> list[float]:
    return vec(value, math.sqrt(max(0.0, 1.0 - value * value)))


@pytest.fixture
def store() -> Iterator[FieldStore]:
    with FieldStore(":memory:", dims=DIMS) as opened:
        yield opened


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "scrape_max_depth": 2,
        "scrape_max_resources": 10,
        "scrape_max_links_per_resource": 25,
        "scrape_max_fetches": 10,
        "scrape_summary_threshold": 0.62,
        "scrape_link_threshold": 0.68,
    }
    values.update(overrides)
    return Settings(**values)


def make_document(
    key: str,
    url: str,
    *,
    metadata: dict | None = None,
    content: str | None = None,
    embedding: list[float] | None = None,
) -> Document:
    canonical = canonicalize_url(url)
    return Document(
        id=f"pap_{key}",
        kind="paper",
        source="test",
        title=key,
        summary="",
        summary_embedding=list(embedding or []),
        canonical_url=canonical,
        url_hash=url_hash(canonical),
        content_sha512=content,
        metadata=dict(metadata or {}),
    )


class FakeProcess:
    def __init__(self, embeddings: dict[str, list[float]], *, mutate: bool = False):
        self.embeddings = embeddings
        self.calls: list[str] = []
        self.mutate = mutate

    def __call__(self, document: Document) -> Document:
        self.calls.append(document.canonical_url or document.id)
        if self.mutate:
            document.metadata["mutated"] = True
            document.title = "mutated"
        vector = self.embeddings.get(document.canonical_url or "", at_similarity(0.0))
        return document.model_copy(
            update={
                "summary": f"summary:{document.title}",
                "summary_embedding": list(vector),
                "status": "summarized",
            }
        )


class FakeFetch:
    def __init__(self, graph: dict[str, Document | None], *, mutate: bool = False):
        self.graph = graph
        self.calls: list[str] = []
        self.mutate = mutate

    def __call__(self, candidate: LinkCandidate) -> Document | None:
        self.calls.append(candidate.url)
        if self.mutate:
            candidate.title = "mutated"
        document = self.graph.get(candidate.url)
        return document.model_copy(deep=True) if document is not None else None


def seed_with_links(
    key: str,
    url: str,
    *,
    links: list[dict],
    embedding: list[float],
) -> Document:
    return make_document(
        key,
        url,
        metadata={"links": links},
        embedding=embedding,
    )


def test_high_similarity_follows_all_links_with_fetch(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    b_url = "https://example.com/b"
    seed = seed_with_links(
        "seed",
        seed_url,
        links=[
            {"url": a_url, "title": "A", "relation": "cites"},
            {"url": b_url, "title": "B"},
        ],
        embedding=vec(1.0),
    )
    process = FakeProcess(
        {
            seed_url: vec(1.0),
            a_url: at_similarity(0.9),
            b_url: at_similarity(0.4),
        }
    )
    fetch = FakeFetch({})

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert fetch.calls == [a_url, b_url]
    assert {document.canonical_url for document in result.documents} == {
        seed_url,
        a_url,
        b_url,
    }
    assert result.stats["documents"] == 3
    assert result.stats["links"] == 2
    assert result.stats["fetches"] == 2

    rows = store.document_links("pap_seed")
    assert len(rows) == 2
    assert {row.to_id for row in rows} == {
        document.id for document in result.documents if document.id != "pap_seed"
    }
    assert {row.relation for row in rows} == {"cites", "references"}


def test_low_similarity_fetches_and_gates_each_link(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    b_url = "https://example.com/b"
    a2_url = "https://example.com/a2"
    b2_url = "https://example.com/b2"

    seed = seed_with_links(
        "seed",
        seed_url,
        links=[{"url": a_url}, {"url": b_url}],
        embedding=at_similarity(0.0),
    )
    graph = {
        a_url: make_document(
            "a", a_url, metadata={"links": [{"url": a2_url}]}
        ),
        b_url: make_document(
            "b", b_url, metadata={"links": [{"url": b2_url}]}
        ),
    }
    process = FakeProcess(
        {
            seed_url: at_similarity(0.0),
            a_url: at_similarity(0.9),
            b_url: at_similarity(0.5),
            a2_url: vec(1.0),
            b2_url: vec(1.0),
        }
    )
    fetch = FakeFetch(graph)

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert fetch.calls == [a_url, b_url, a2_url]
    assert {document.canonical_url for document in result.documents} == {
        seed_url,
        a_url,
        b_url,
    }
    assert store.find_document_by_url(url_hash(b2_url)) is None
    assert result.stats["skipped_threshold"] == 1
    assert any(
        warning == f"scrape_skipped_threshold:{b_url}"
        for warning in result.warnings
    )


def test_duplicate_url_is_not_processed_twice(store: FieldStore) -> None:
    s1_url = "https://example.com/s1"
    s2_url = "https://example.com/s2"
    a_url = "https://example.com/a"
    s1 = seed_with_links(
        "s1", s1_url, links=[{"url": a_url}], embedding=vec(1.0)
    )
    s2 = seed_with_links(
        "s2", s2_url, links=[{"url": a_url}], embedding=vec(1.0)
    )
    process = FakeProcess(
        {s1_url: vec(1.0), s2_url: vec(1.0), a_url: vec(1.0)}
    )
    fetch = FakeFetch({a_url: make_document("a", a_url)})

    result = expand_frontier(
        [s1, s2],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert fetch.calls == [a_url]
    assert len(result.documents) == 3
    assert process.calls.count(a_url) == 1
    assert any(
        warning.startswith("scrape_duplicate_url:") for warning in result.warnings
    )
    assert store.find_document_by_url(url_hash(a_url)) is not None
    assert len(store.document_links("pap_s1")) == 1
    assert len(store.document_links("pap_s2")) == 1


def test_duplicate_content_at_different_url_is_merged(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    x_url = "https://example.com/x"
    y_url = "https://example.com/y"
    shared_content = "a" * 128

    seed = seed_with_links(
        "seed",
        seed_url,
        links=[{"url": x_url}, {"url": y_url}],
        embedding=at_similarity(0.0),
    )
    graph = {
        x_url: make_document("x", x_url, content=shared_content),
        y_url: make_document("y", y_url, content=shared_content),
    }
    process = FakeProcess(
        {seed_url: at_similarity(0.0), x_url: vec(1.0), y_url: vec(1.0)}
    )
    fetch = FakeFetch(graph)

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert len(result.documents) == 2
    assert process.calls.count(x_url) == 1
    assert process.calls.count(y_url) == 0
    assert store.find_document_by_url(url_hash(y_url)) is None
    assert any(
        warning.startswith("scrape_duplicate_content:")
        for warning in result.warnings
    )
    relations = {
        row.relation
        for row in store.document_links("pap_seed")
    }
    assert relations == {"references", "same_as"}


def test_depth_limit_stops_expansion(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    a2_url = "https://example.com/a2"
    seed = seed_with_links(
        "seed", seed_url, links=[{"url": a_url}], embedding=vec(1.0)
    )
    graph = {
        a_url: make_document("a", a_url, metadata={"links": [{"url": a2_url}]})
    }
    process = FakeProcess(
        {seed_url: vec(1.0), a_url: vec(1.0), a2_url: vec(1.0)}
    )
    fetch = FakeFetch(graph)

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(scrape_max_depth=1),
    )

    assert {document.canonical_url for document in result.documents} == {
        seed_url,
        a_url,
    }
    assert store.find_document_by_url(url_hash(a2_url)) is None
    assert result.stats["max_depth_reached"] == 1


def test_document_budget_stops_the_run(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    link_urls = [f"https://example.com/{index}" for index in range(4)]
    seed = seed_with_links(
        "seed",
        seed_url,
        links=[
            {"url": url, "title": f"Link {index}"}
            for index, url in enumerate(link_urls)
        ],
        embedding=vec(1.0),
    )
    embeddings = {seed_url: vec(1.0)}
    embeddings.update({url: vec(1.0) for url in link_urls})
    process = FakeProcess(embeddings)

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=FakeFetch({}),
        settings=make_settings(scrape_max_resources=2),
    )

    assert result.stats["processed"] == 2
    assert result.stats["documents"] == 2
    assert "scrape_budget_exhausted" in result.warnings


def test_fetch_budget_stops_the_run(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    b_url = "https://example.com/b"
    seed = seed_with_links(
        "seed",
        seed_url,
        links=[{"url": a_url}, {"url": b_url}],
        embedding=at_similarity(0.0),
    )
    graph = {
        a_url: make_document("a", a_url),
        b_url: make_document("b", b_url),
    }
    process = FakeProcess(
        {seed_url: at_similarity(0.0), a_url: vec(1.0), b_url: vec(1.0)}
    )
    fetch = FakeFetch(graph)

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(scrape_max_fetches=1),
    )

    assert fetch.calls == [a_url]
    assert result.stats["documents"] == 2
    assert "scrape_fetch_budget_exhausted" in result.warnings


def test_fetch_failure_records_a_warning_and_continues(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    b_url = "https://example.com/b"
    seed = seed_with_links(
        "seed",
        seed_url,
        links=[{"url": a_url}, {"url": b_url}],
        embedding=at_similarity(0.0),
    )
    process = FakeProcess(
        {seed_url: at_similarity(0.0), b_url: vec(1.0)}
    )
    fetch = FakeFetch({b_url: make_document("b", b_url)})

    result = expand_frontier(
        [seed],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert fetch.calls == [a_url, b_url]
    assert result.stats["documents"] == 2
    assert any(
        warning == f"scrape_fetch_failed:{a_url}"
        for warning in result.warnings
    )


def test_inputs_are_not_mutated(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    a_url = "https://example.com/a"
    seed = seed_with_links(
        "seed",
        seed_url,
        links=[{"url": a_url, "title": "A"}],
        embedding=at_similarity(0.0),
    )
    seeds = [seed]
    seed_before = seed.model_dump()
    graph_document = make_document("a", a_url)
    graph_before = graph_document.model_dump()
    process = FakeProcess(
        {seed_url: at_similarity(0.0), a_url: vec(1.0)}, mutate=True
    )
    fetch = FakeFetch({a_url: graph_document}, mutate=True)

    expand_frontier(
        seeds,
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=fetch,
        settings=make_settings(),
    )

    assert seeds == [seed]
    assert seed.model_dump() == seed_before
    assert graph_document.model_dump() == graph_before


def test_existing_store_document_is_not_summarized_again(store: FieldStore) -> None:
    seed_url = "https://example.com/seed"
    existing = make_document(
        "existing", seed_url, embedding=vec(1.0)
    ).model_copy(update={"summary": "already summarized", "status": "summarized"})
    store.upsert_document(existing)
    process = FakeProcess({seed_url: vec(1.0)})

    result = expand_frontier(
        [existing],
        target_embedding=TARGET,
        store=store,
        process=process,
        fetch=FakeFetch({}),
        settings=make_settings(),
    )

    assert process.calls == []
    assert result.stats["documents"] == 1
    assert result.stats["duplicates"] == 1
