"""Integration check for the research-time metrics enrichment hook."""

from __future__ import annotations

from pathlib import Path

import httpx

from whack26.config import Settings
from whack26.profile import DatasetProfile, TargetChoice
from whack26.research import pipeline
from whack26.research.base import ResearchRecord


class _FakeAdapter:
    name = "openalex"
    ttl_seconds = 60

    def search(self, query: str, limit: int, client: httpx.Client):
        return [
            ResearchRecord(
                id="openalex:W1",
                source="openalex",
                source_id="W1",
                title="Housing prices and income",
                url="https://openalex.org/W1",
                retrieved_at="2026-01-01T00:00:00Z",
                query=query,
                cited_by_count=7,
                author_ids=["openalex:A1"],
                metrics_source="openalex",
            )
        ]


class _FakeEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def _profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_metrics_integration",
        source_path="data/housing.csv",
        sha256="a" * 64,
        format="csv",
        rows=10,
        columns=[],
        description=None,
        target=TargetChoice(name="price", source="user", confidence=1.0),
        time_column=None,
        warnings=[],
    )


def test_run_research_invokes_enrichment_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        pipeline,
        "generate_queries",
        lambda **kwargs: (
            [pipeline.ResearchQuery(source="openalex", query="housing prices")],
            [],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_adapters",
        lambda *args, **kwargs: {"openalex": _FakeAdapter()},
    )

    seen: list[int] = []

    def fake_enrich(records, **kwargs):
        seen.append(len(records))
        return records

    monkeypatch.setattr(pipeline, "enrich_records", fake_enrich)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = pipeline.run_research(
            profile=_profile(),
            summary="Housing prices.",
            path_names=["Housing"],
            llm=object(),
            embeddings=_FakeEmbeddings(),
            settings=Settings(
                openalex_api_key="test-key",
                research_metrics_enabled=True,
                embed_dims=1,
            ),
            client=client,
            cache_path=tmp_path / "research.db",
            offline=False,
        )

    assert seen == [1]
    assert [record.id for record in result.records] == ["openalex:W1"]
