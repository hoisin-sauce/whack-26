"""Tests for the field-scoped pitfall stage."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from whack26.config import Settings
from whack26.pitfalls import (
    FieldPitfall,
    FieldPitfallReport,
    applicability,
    build_field_queries,
    build_signals,
    extract_field_pitfalls,
    run_field_pitfalls,
)
from whack26.profile import ColumnProfile, DatasetProfile, TargetChoice
from whack26.research.base import ResearchRecord


def _profile(
    *,
    target: str = "value",
    description: str = "1990 census block-group data.",
    time_column: str | None = None,
    columns: list[ColumnProfile] | None = None,
) -> DatasetProfile:
    if columns is None:
        columns = [
            ColumnProfile(
                name="value",
                dtype="float64",
                role="target",
                missing_pct=0.0,
                cardinality=20,
                sample_values=["1", "2"],
                numeric={"mean": 10.0, "max": 100.0},
                top_values=None,
                datetime=False,
            ),
            ColumnProfile(
                name="latitude",
                dtype="float64",
                role="feature",
                missing_pct=0.0,
                cardinality=20,
                sample_values=["37.1"],
                numeric={"mean": 37.1, "max": 38.0},
                top_values=None,
                datetime=False,
            ),
        ]
    return DatasetProfile(
        dataset_id="ds_test",
        source_path="data/test.csv",
        sha256="a" * 64,
        format="csv",
        rows=100,
        columns=columns,
        description=description,
        target=TargetChoice(name=target, source="user", confidence=1.0),
        time_column=time_column,
        warnings=[],
    )


def _record(
    identifier: str,
    title: str,
    abstract: str,
    *,
    source: str = "openalex",
    relevance: float = 0.8,
    year: int = 2024,
) -> ResearchRecord:
    return ResearchRecord(
        id=identifier,
        source=source,  # type: ignore[arg-type]
        source_id=identifier.split(":", 1)[-1],
        title=title,
        abstract=abstract,
        url=f"https://example.org/{identifier}",
        retrieved_at="2026-01-01T00:00:00Z",
        query="field limitations",
        relevance=relevance,
        year=year,
    )


def _fields() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id="fld_root",
            name="Real Estate and Housing",
            description="Housing markets and property values.",
            level=1,
        ),
        SimpleNamespace(
            id="fld_leaf",
            name="Housing Valuation",
            description="Models of house prices and valuations.",
            level=2,
        ),
    ]


def test_build_field_queries_is_leaf_first_and_short() -> None:
    queries = build_field_queries(
        _fields(), per_field=2, sources=("openalex", "arxiv")
    )
    assert [query.field_id for query in queries] == [
        "fld_leaf",
        "fld_leaf",
        "fld_root",
        "fld_root",
    ]
    assert all(len(query.query.split()) <= 12 for query in queries)
    assert queries[0].source == "openalex"
    assert queries[1].source == "arxiv"


def test_extract_keeps_only_pitfall_records_and_marks_spatial() -> None:
    profile = _profile()
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 100.0, 100.0],
            "latitude": [37.1, 37.2, 37.3, 37.4],
        }
    )
    records = [
        _record(
            "openalex:W1",
            "House price prediction",
            "Random cross-validation ignores spatial autocorrelation and "
            "nearby properties, producing optimistic estimates.",
        ),
        _record(
            "openalex:W2",
            "A descriptive note",
            "The paper reports summary statistics for the sample.",
        ),
    ]
    pitfalls = extract_field_pitfalls(
        _fields()[-1], records, profile, frame=frame, max_items=3, scope="leaf"
    )
    assert len(pitfalls) == 1
    pitfall = pitfalls[0]
    assert pitfall.category == "spatial_dependence"
    assert pitfall.applies_to_dataset == "yes"
    assert pitfall.scope == "leaf"
    assert pitfall.evidence_id == "openalex:W1"
    assert pitfall.id.startswith("F:fld_leaf:")
    assert "spatial" in pitfall.check.lower()


def test_extract_marks_censoring_and_missingness_from_profile() -> None:
    columns = [
        ColumnProfile(
            name="value",
            dtype="float64",
            role="target",
            missing_pct=0.0,
            cardinality=20,
            sample_values=["1"],
            numeric={"mean": 10.0, "max": 100.0},
            top_values=None,
            datetime=False,
        ),
        ColumnProfile(
            name="income",
            dtype="float64",
            role="feature",
            missing_pct=3.0,
            cardinality=20,
            sample_values=["1"],
            numeric={"mean": 5.0, "max": 10.0},
            top_values=None,
            datetime=False,
        ),
    ]
    profile = _profile(columns=columns)
    frame = pd.DataFrame(
        {
            "value": [100.0] * 10 + [1.0] * 90,
            "income": [1.0] * 97 + [None] * 3,
        }
    )
    signals = build_signals(profile, frame=frame)
    assert signals.boundary_columns
    applies, reason = applicability("censoring", signals)
    assert applies == "yes"
    assert "value" in reason
    applies, _ = applicability("missing_data", signals)
    assert applies == "yes"

    records = [
        _record(
            "openalex:W3",
            "Top-coded house values",
            "The target is top-coded at a ceiling, so OLS estimates are biased.",
        ),
        _record(
            "openalex:W4",
            "Missing data in surveys",
            "Complete-case analysis with missing data can bias estimates.",
        ),
    ]
    pitfalls = extract_field_pitfalls(_fields()[-1], records, profile, frame=frame)
    categories = {pitfall.category for pitfall in pitfalls}
    assert {"censoring", "missing_data"} <= categories
    for pitfall in pitfalls:
        assert pitfall.applies_to_dataset == "yes"
        assert pitfall.applicability_reason


def test_run_field_pitfalls_mock_uses_existing_records(tmp_path) -> None:
    settings = Settings(
        llm_provider="mock",
        embed_provider="mock",
        field_pitfalls_enabled=True,
        cache_dir=tmp_path,
    )
    records = [
        _record(
            "openalex:W5",
            "Housing data limitations",
            "Selection bias and survivorship bias affect listings and repeat sales.",
        )
    ]
    report = run_field_pitfalls(
        profile=_profile(),
        summary="summary",
        field_path=_fields(),
        settings=settings,
        existing_records=records,
        frame=pd.DataFrame({"value": [1.0, 2.0], "latitude": [37.1, 37.2]}),
        cache_path=tmp_path / "research.db",
    )
    assert isinstance(report, FieldPitfallReport)
    assert report.field_path == ["Real Estate and Housing", "Housing Valuation"]
    assert report.pitfalls
    assert all(pitfall.id.startswith("F:") for pitfall in report.pitfalls)
    assert any("mock_records_only" in warning for warning in report.warnings)


def test_run_field_pitfalls_searches_without_network(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        llm_provider="deepseek",
        embed_provider="gemini",
        field_pitfalls_enabled=True,
        field_pitfalls_max_queries_per_field=1,
        field_pitfalls_max_per_field=2,
        cache_dir=tmp_path,
    )
    seen_queries: list[str] = []

    class FakeAdapter:
        warnings: list[str] = []

        def search(self, query: str, limit: int, client) -> list[ResearchRecord]:
            seen_queries.append(query)
            return [
                _record(
                    "openalex:W6",
                    "Spatial dependence in housing markets",
                    "Nearby properties share unobserved amenities, so random "
                    "cross-validation overstates generalization.",
                )
            ]

    monkeypatch.setattr(
        "whack26.pitfalls.pipeline.build_adapters",
        lambda *args, **kwargs: {"openalex": FakeAdapter()},
    )
    report = run_field_pitfalls(
        profile=_profile(),
        summary="summary",
        field_path=_fields(),
        settings=settings,
        frame=pd.DataFrame({"value": [1.0, 2.0], "latitude": [37.1, 37.2]}),
        cache_path=tmp_path / "research.db",
    )
    assert seen_queries
    assert all("limitations bias" in query for query in seen_queries)
    assert report.pitfalls
    assert report.degraded is False
    assert report.cache


def test_run_field_pitfalls_drops_pitfalls_the_profile_rules_out(
    tmp_path, monkeypatch
) -> None:
    settings = Settings(
        llm_provider="deepseek",
        embed_provider="gemini",
        field_pitfalls_enabled=True,
        field_pitfalls_max_queries_per_field=1,
        cache_dir=tmp_path,
    )

    class FakeAdapter:
        warnings: list[str] = []

        def search(self, query: str, limit: int, client) -> list[ResearchRecord]:
            return [
                _record(
                    "openalex:W7",
                    "Time-series regressions",
                    "The authors use time-series regressions of panel data.",
                )
            ]

    monkeypatch.setattr(
        "whack26.pitfalls.pipeline.build_adapters",
        lambda *args, **kwargs: {"openalex": FakeAdapter()},
    )
    report = run_field_pitfalls(
        profile=_profile(),
        summary="summary",
        field_path=_fields()[:1],
        settings=settings,
        frame=pd.DataFrame({"value": [1.0, 2.0]}),
        cache_path=tmp_path / "research.db",
    )
    assert report.pitfalls == []
    assert any(
        warning.startswith("field_pitfalls_no_applicable_pitfalls")
        for warning in report.warnings
    )


def test_report_by_field_and_citations() -> None:
    pitfall = FieldPitfall(
        id="F:fld_leaf:1",
        field_id="fld_leaf",
        field_name="Housing Valuation",
        title="Source",
        pitfall="Pitfall text",
        evidence_id="openalex:W1",
    )
    report = FieldPitfallReport(pitfalls=[pitfall])
    assert report.citations() == ["F:fld_leaf:1"]
    assert report.by_field()["fld_leaf"] == [pitfall]


def test_empty_field_path_returns_degraded_report() -> None:
    settings = Settings(
        llm_provider="mock",
        embed_provider="mock",
        field_pitfalls_enabled=True,
    )
    report = run_field_pitfalls(
        profile=_profile(),
        summary="summary",
        field_path=[],
        settings=settings,
    )
    assert report.degraded is True
    assert report.pitfalls == []
    assert "field_pitfalls_no_fields" in report.warnings
