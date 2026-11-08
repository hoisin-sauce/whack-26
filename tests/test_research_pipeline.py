from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import httpx
import pytest

from whack26.config import Settings
from whack26.embeddings import EmbeddingError
from whack26.llm import LLMError
from whack26.profile import DatasetProfile, TargetChoice
from whack26.research.arxiv import arXivAdapter
from whack26.research.base import ResearchRecord
from whack26.research.cache import ResearchCache
from whack26.research.pipeline import (
    ResearchResult,
    generate_queries,
    refine_after_research,
    run_research,
)

SUMMARY = "Housing prices by block group."

QUERY_PAYLOAD = {
    "queries": [
        {"source": "openalex", "query": "housing price determinants", "rationale": "papers"},
        {"source": "arxiv", "query": "housing price prediction", "rationale": "papers"},
        {"source": "huggingface", "query": "housing dataset", "rationale": "dataset"},
        {"source": "zenodo", "query": "housing survey dataset", "rationale": "dataset"},
        {"source": "wikipedia", "query": "housing market", "rationale": "background"},
    ]
}

OPENALEX_PAYLOAD = {
    "meta": {"cost_usd": 0.001},
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1234/housing",
            "title": "Income and housing prices",
            "publication_year": 2020,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
            "primary_location": {
                "landing_page_url": "https://example.test/W1",
                "license": "cc-by",
            },
            "open_access": {"oa_status": "gold"},
            "cited_by_count": 10,
            "abstract_inverted_index": {
                "Income": [0],
                "predicts": [1],
                "housing": [2],
                "prices": [3],
            },
            "topics": [],
            "keywords": [],
            "is_retracted": False,
        }
    ],
}

ARXIV_PAYLOAD = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>Housing price prediction with gradients</title>
    <summary>We model housing prices and income.</summary>
    <author><name>Grace Hopper</name></author>
    <arxiv:doi>10.1234/arxiv-housing</arxiv:doi>
    <link rel="alternate" href="https://arxiv.org/abs/2401.12345v1" />
  </entry>
</feed>
"""

HF_PAYLOAD = [
    {
        "id": "user/housing-dataset",
        "author": "user",
        "description": "Housing prices and income records.",
        "createdAt": "2023-05-01T00:00:00Z",
        "tags": ["license:mit"],
        "private": False,
        "gated": False,
    }
]

ZENODO_PAYLOAD = {
    "hits": {
        "hits": [
            {
                "id": 123,
                "doi": "10.5281/zenodo.123",
                "metadata": {
                    "title": "Housing survey",
                    "description": "A survey of housing and income.",
                    "publication_date": "2022-03-01",
                    "access_right": "open",
                    "license": {"id": "cc-by-4.0"},
                    "creators": [{"name": "Alan Turing"}],
                },
                "links": {"self_html": "https://zenodo.org/records/123"},
            }
        ]
    }
}

WIKIPEDIA_SEARCH_PAYLOAD = {
    "query": {"search": [{"title": "Housing market", "pageid": 42}]}
}

WIKIPEDIA_SUMMARY_PAYLOAD = {
    "pageid": 42,
    "title": "Housing market",
    "extract": "A housing market matches housing supply and demand.",
    "timestamp": "2021-01-01T00:00:00Z",
    "content_urls": {
        "desktop": {"page": "https://en.wikipedia.org/wiki/Housing_market"}
    },
}


def make_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_aaaaaaaaaaaa",
        source_path="data/housing.csv",
        sha256="a" * 64,
        format="csv",
        rows=100,
        columns=[],
        description="Housing prices by block group.",
        target=TargetChoice(
            name="median_house_value", source="user", confidence=1.0
        ),
        time_column=None,
        warnings=[],
    )


class FakeEmbeddings:
    def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        return [
            [1.0, 0.0] if "housing" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


class FailingEmbeddings:
    def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        raise EmbeddingError("embedding provider unavailable")


class ScriptedLLM:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, purpose: str, **kwargs: Any) -> Any:
        self.calls.append({"purpose": purpose, **kwargs})
        if isinstance(self.payload, Exception):
            raise self.payload
        return copy.deepcopy(self.payload)


def make_handler(
    calls: list[str],
    *,
    fail_host: str | None = None,
    empty_long_hf: bool = False,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(str(request.url))
        if fail_host is not None and host == fail_host:
            return httpx.Response(404, text="not found")
        if host == "api.openalex.org":
            return httpx.Response(200, json=OPENALEX_PAYLOAD)
        if host == "export.arxiv.org":
            return httpx.Response(200, text=ARXIV_PAYLOAD)
        if host == "huggingface.co":
            search = request.url.params.get("search", "")
            if empty_long_hf and len(search.split()) > 2:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=HF_PAYLOAD)
        if host == "zenodo.org":
            return httpx.Response(200, json=ZENODO_PAYLOAD)
        if host == "en.wikipedia.org":
            if request.url.path == "/w/api.php":
                return httpx.Response(200, json=WIKIPEDIA_SEARCH_PAYLOAD)
            if request.url.path.startswith("/api/rest_v1/page/summary/"):
                return httpx.Response(200, json=WIKIPEDIA_SUMMARY_PAYLOAD)
        raise AssertionError(f"unexpected request: {request.url}")

    return httpx.MockTransport(handler)


def make_client(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport, follow_redirects=True)


def run(
    llm: ScriptedLLM,
    cache_path: Path,
    client: httpx.Client,
    *,
    offline: bool = False,
    embeddings: Any = None,
) -> ResearchResult:
    return run_research(
        profile=make_profile(),
        summary=SUMMARY,
        path_names=["Housing", "Housing Markets"],
        llm=llm,
        embeddings=embeddings if embeddings is not None else FakeEmbeddings(),
        settings=Settings(
            embed_dims=2,
            openalex_api_key="test-openalex-key",
            hf_token="test-hf-token",
            zenodo_token="test-zenodo-token",
            offline=offline,
        ),
        client=client,
        cache_path=cache_path,
        offline=offline,
    )


def test_generate_queries_accepts_constrained_payload() -> None:
    llm = ScriptedLLM(QUERY_PAYLOAD)
    queries, warnings = generate_queries(
        summary=SUMMARY,
        path_names=["Housing", "Housing Markets"],
        profile=make_profile(),
        llm=llm,
    )
    assert [query.source for query in queries] == [
        "openalex",
        "arxiv",
        "huggingface",
        "zenodo",
        "wikipedia",
    ]
    assert warnings == []
    assert llm.calls[0]["purpose"] == "queries"


def test_generate_queries_dedupes_and_drops_invalid_items() -> None:
    payload = {
        "queries": [
            {"source": "openalex", "query": "housing price determinants", "rationale": "a"},
            {"source": "openalex", "query": "  Housing   Price Determinants ", "rationale": "dup"},
            {"source": "arxiv", "query": "housing AND income", "rationale": "boolean"},
            {"source": "arxiv", "query": "housing price prediction", "rationale": "papers"},
            {"source": "huggingface", "query": "housing dataset", "rationale": "dataset"},
            {"source": "zenodo", "query": "one two three four five six seven eight nine ten eleven twelve thirteen", "rationale": "too long"},
            {"source": "zenodo", "query": "housing survey dataset", "rationale": "dataset"},
            {"source": "wikipedia", "query": "housing market", "rationale": "background"},
            {"source": "wikipedia", "query": "housing", "rationale": "second wiki"},
            {"source": "unknown", "query": "housing", "rationale": "bad source"},
        ]
    }
    queries, warnings = generate_queries(
        summary=SUMMARY,
        path_names=["Housing"],
        profile=make_profile(),
        llm=ScriptedLLM(payload),
    )
    assert [query.query for query in queries] == [
        "housing price determinants",
        "housing price prediction",
        "housing dataset",
        "housing survey dataset",
        "housing market",
    ]
    assert warnings == []


def test_generate_queries_falls_back_on_llm_failure() -> None:
    queries, warnings = generate_queries(
        summary=SUMMARY,
        path_names=["Housing", "Housing Markets"],
        profile=make_profile(),
        llm=ScriptedLLM(LLMError("quota", purpose="queries")),
    )
    assert len(queries) == 5
    assert queries[0].source == "openalex"
    assert any("query_generation_failed" in warning for warning in warnings)


def test_generate_queries_falls_back_when_constraints_unmet() -> None:
    payload = {
        "queries": [
            {"source": "openalex", "query": "housing prices", "rationale": "one"},
            {"source": "huggingface", "query": "housing dataset", "rationale": "two"},
        ]
    }
    queries, warnings = generate_queries(
        summary=SUMMARY,
        path_names=["Housing", "Housing Markets"],
        profile=make_profile(),
        llm=ScriptedLLM(payload),
    )
    assert len(queries) == 5
    assert any("query_generation_invalid" in warning for warning in warnings)


def test_generate_queries_falls_back_when_response_shape_is_invalid() -> None:
    queries, warnings = generate_queries(
        summary=SUMMARY,
        path_names=["Housing"],
        profile=make_profile(),
        llm=ScriptedLLM({"queries": "not a list"}),
    )
    assert len(queries) == 5
    assert warnings == []


def test_generate_queries_reuses_cached_queries(tmp_path: Path) -> None:
    cache = ResearchCache(tmp_path / "cache.db")
    llm = ScriptedLLM(QUERY_PAYLOAD)
    try:
        first, warnings = generate_queries(
            summary=SUMMARY,
            path_names=["Housing", "Housing Markets"],
            profile=make_profile(),
            llm=llm,
            cache=cache,
        )
        assert warnings == []
        assert len(llm.calls) == 1

        second_llm = ScriptedLLM(LLMError("should not be called", purpose="queries"))
        second, warnings = generate_queries(
            summary=SUMMARY,
            path_names=["Housing", "Housing Markets"],
            profile=make_profile(),
            llm=second_llm,
            cache=cache,
        )
    finally:
        cache.close()

    assert second == first
    assert warnings == []
    assert second_llm.calls == []


def test_generate_queries_invalidates_cache_when_field_path_changes(
    tmp_path: Path,
) -> None:
    cache = ResearchCache(tmp_path / "cache.db")
    try:
        first, warnings = generate_queries(
            summary=SUMMARY,
            path_names=["Housing", "Housing Markets"],
            profile=make_profile(),
            llm=ScriptedLLM(QUERY_PAYLOAD),
            cache=cache,
        )
        assert warnings == []

        changed_payload = copy.deepcopy(QUERY_PAYLOAD)
        changed_payload["queries"][0]["query"] = "property price determinants"
        second_llm = ScriptedLLM(changed_payload)
        second, warnings = generate_queries(
            summary=SUMMARY,
            path_names=["Property", "Property Statistics"],
            profile=make_profile(),
            llm=second_llm,
            cache=cache,
        )
    finally:
        cache.close()

    assert warnings == []
    assert len(second_llm.calls) == 1
    assert second[0].query == "property price determinants"
    assert second != first


def test_run_research_fetches_ranks_and_caches_all_sources(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    calls: list[str] = []
    client = make_client(make_handler(calls))
    try:
        result = run(ScriptedLLM(QUERY_PAYLOAD), tmp_path / "cache.db", client)
    finally:
        client.close()

    ids = {record.id for record in result.records}
    assert ids == {
        "openalex:W1",
        "arxiv:2401.12345v1",
        "huggingface:user/housing-dataset",
        "zenodo:123",
        "wikipedia:42",
    }
    assert set(result.sources_used) == {
        "openalex",
        "arxiv",
        "huggingface",
        "zenodo",
        "wikipedia",
    }
    assert result.warnings == []
    assert result.limited_mode is False
    assert result.offline is False
    assert "housing" in result.keywords
    assert result.cache["writes"] == 6
    assert len(calls) == 6

    by_source = {record.source: record for record in result.records}
    assert by_source["openalex"].doi == "10.1234/housing"
    assert by_source["openalex"].authors == ["Ada Lovelace"]
    assert by_source["arxiv"].year == 2024
    assert by_source["huggingface"].license == "mit"
    assert by_source["zenodo"].authors == ["Alan Turing"]
    assert by_source["wikipedia"].license == "CC BY-SA 4.0"
    assert all(record.relevance > 0 for record in result.records)


def test_run_research_replays_cache_without_http(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    cache_path = tmp_path / "cache.db"
    first_calls: list[str] = []
    client = make_client(make_handler(first_calls))
    try:
        first = run(ScriptedLLM(QUERY_PAYLOAD), cache_path, client)
    finally:
        client.close()

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"cache miss caused HTTP: {request.url}")

    second_llm = ScriptedLLM(LLMError("query cache miss", purpose="queries"))
    client = httpx.Client(transport=httpx.MockTransport(fail))
    try:
        second = run(second_llm, cache_path, client)
    finally:
        client.close()

    assert [record.id for record in second.records] == [
        record.id for record in first.records
    ]
    assert second_llm.calls == []
    assert second.cache["writes"] == 0
    assert second.cache["hits"] == 6


def test_run_research_offline_replays_stale_cache(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    cache_path = tmp_path / "cache.db"
    client = make_client(make_handler([]))
    try:
        run(ScriptedLLM(QUERY_PAYLOAD), cache_path, client)
    finally:
        client.close()

    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"offline mode caused HTTP: {request.url}")

    offline_llm = ScriptedLLM(LLMError("query cache miss", purpose="queries"))
    client = httpx.Client(transport=httpx.MockTransport(fail))
    try:
        offline = run(offline_llm, cache_path, client, offline=True)
    finally:
        client.close()

    assert offline.offline is True
    assert offline_llm.calls == []
    assert len(offline.records) == 5
    assert offline.warnings == []
    assert offline.cache["hits"] == 6


def test_run_research_surfaces_embedding_failure_warning(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    client = make_client(make_handler([]))
    try:
        result = run(
            ScriptedLLM(QUERY_PAYLOAD),
            tmp_path / "cache.db",
            client,
            embeddings=FailingEmbeddings(),
        )
    finally:
        client.close()

    assert len(result.records) == 5
    assert any(
        warning.startswith("research_ranking_embeddings_failed")
        for warning in result.warnings
    )


def test_run_research_continues_after_one_source_fails(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    calls: list[str] = []
    client = make_client(make_handler(calls, fail_host="api.openalex.org"))
    try:
        result = run(
            ScriptedLLM(QUERY_PAYLOAD), tmp_path / "cache.db", client
        )
    finally:
        client.close()

    assert "openalex" not in result.sources_used
    assert set(result.sources_used) == {
        "arxiv",
        "huggingface",
        "zenodo",
        "wikipedia",
    }
    assert len(result.records) == 4
    assert any(
        warning.startswith("source_failed: openalex")
        for warning in result.warnings
    )


def test_run_research_retries_long_huggingface_query(tmp_path: Path) -> None:
    arXivAdapter.reset_rate_limit()
    payload = copy.deepcopy(QUERY_PAYLOAD)
    payload["queries"][2]["query"] = "palmer penguins dataset"
    calls: list[str] = []
    client = make_client(make_handler(calls, empty_long_hf=True))
    try:
        result = run(
            ScriptedLLM(payload), tmp_path / "cache.db", client
        )
    finally:
        client.close()

    assert "huggingface" in result.sources_used
    assert any(
        warning.startswith("source_retry: huggingface")
        for warning in result.warnings
    )
    assert (
        sum(1 for call in calls if "huggingface.co" in call) == 2
    )
    assert any(
        record.id == "huggingface:user/housing-dataset"
        for record in result.records
    )


class StubClassifier:
    def __init__(self, should: bool, refined: Any) -> None:
        self.should = should
        self.refined = refined
        self.should_calls = 0
        self.refine_calls: list[dict[str, Any]] = []

    def should_refine(
        self, path: Any, research_titles: Any = ()
    ) -> bool:
        self.should_calls += 1
        self.titles = list(research_titles)
        return self.should

    def refine(
        self,
        profile: DatasetProfile,
        frame: Any,
        path: Any,
        *,
        research_titles: Any = (),
        keywords: Any = (),
        force: bool = False,
    ) -> Any:
        self.refine_calls.append(
            {
                "path": path,
                "research_titles": list(research_titles),
                "keywords": list(keywords),
                "force": force,
            }
        )
        return self.refined


def make_record(identifier: str, title: str) -> ResearchRecord:
    return ResearchRecord(
        id=identifier,
        source="openalex",
        source_id=identifier,
        title=title,
        url="https://example.test",
        retrieved_at="2026-01-01T00:00:00Z",
        query="housing",
    )


def test_refine_after_research_skips_without_records() -> None:
    classifier = StubClassifier(True, "refined")
    path = "original"
    result = refine_after_research(
        classifier=classifier,
        profile=make_profile(),
        frame=None,
        path=path,
        result=ResearchResult(records=[]),
    )
    assert result == "original"
    assert classifier.should_calls == 0


def test_refine_after_research_returns_unchanged_when_trigger_is_false() -> None:
    classifier = StubClassifier(False, "refined")
    result = refine_after_research(
        classifier=classifier,
        profile=make_profile(),
        frame=None,
        path="original",
        result=ResearchResult(
            records=[make_record("openalex:W1", "Housing prices")],
            keywords=["housing"],
        ),
    )
    assert result == "original"
    assert classifier.should_calls == 1
    assert classifier.refine_calls == []


def test_refine_after_research_calls_classifier_once() -> None:
    records = [
        make_record(f"openalex:W{index}", f"Housing paper {index}")
        for index in range(1, 8)
    ]
    classifier = StubClassifier(True, "refined")
    result = refine_after_research(
        classifier=classifier,
        profile=make_profile(),
        frame=None,
        path="original",
        result=ResearchResult(records=records, keywords=["housing", "income"]),
    )
    assert result == "refined"
    assert classifier.should_calls == 1
    assert len(classifier.refine_calls) == 1
    call = classifier.refine_calls[0]
    assert call["path"] == "original"
    assert call["research_titles"] == [
        f"Housing paper {index}" for index in range(1, 6)
    ]
    assert call["keywords"] == ["housing", "income"]
    assert call["force"] is True
