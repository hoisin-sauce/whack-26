"""Tests for model-search JSON and Markdown reports."""

from __future__ import annotations

import json

from whack26.model_search.report import (
    render_json,
    render_markdown,
    report_payload,
    write_json,
    write_markdown,
    write_report,
)
from whack26.model_search.select import select_models
from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)


def make_result() -> ModelSearchResult:
    candidate = ModelCandidate(
        spec=ModelSpec(
            family="linear",
            features=["x1", "x2"],
            interactions=[("x1", "x2")],
            target_transform="log1p",
        ),
        folds=[
            FoldResult(
                fold=0,
                train_rows=70,
                test_rows=30,
                metrics={"r2": 0.80},
                baseline_metrics={"r2": 0.10},
            ),
            FoldResult(
                fold=1,
                train_rows=70,
                test_rows=30,
                metrics={"r2": 0.84},
                baseline_metrics={"r2": 0.12},
            ),
            FoldResult(
                fold=2,
                train_rows=70,
                test_rows=30,
                metrics={"r2": 0.82},
                baseline_metrics={"r2": 0.11},
            ),
        ],
        metrics={"r2": 0.82},
        metric_std={"r2": 0.028},
        diagnostics=[
            Diagnostic(name="breusch_pagan", status="warn", value=0.01),
            Diagnostic(name="durbin_watson", status="ok", value=2.01),
        ],
    )
    weaker = ModelCandidate(
        spec=ModelSpec(family="linear", features=["x1"]),
        folds=[
            FoldResult(
                fold=0,
                train_rows=70,
                test_rows=30,
                metrics={"r2": 0.60},
                baseline_metrics={"r2": 0.10},
            )
        ],
        metrics={"r2": 0.60},
        metric_std={"r2": 0.0},
    )
    return select_models(
        [candidate, weaker],
        target_kind="continuous",
        n_rows=1000,
        config=None,
    )


def test_json_payload_contains_selection_and_reports() -> None:
    result = make_result()

    payload = report_payload(result)

    assert payload["target_kind"] == "continuous"
    assert payload["primary_metric"] == "r2"
    assert payload["selection_confidence"] == "high"
    assert payload["selected"]["spec"]["family"] == "linear"
    assert payload["candidates"][0]["baseline_delta"] is not None
    assert payload["candidates"][0]["baseline_delta"] > 0.5
    assert payload["candidates"][0]["diagnostics"][0]["name"] == "breusch_pagan"

    text = render_json(result)
    decoded = json.loads(text)
    assert decoded == json.loads(json.dumps(payload))


def test_markdown_renders_table_selected_model_and_diagnostics() -> None:
    result = make_result()

    markdown = render_markdown(result)

    assert "# Model search" in markdown
    assert "Target kind: **continuous**" in markdown
    assert "Primary metric: **r2**" in markdown
    assert "Selection confidence: **high**" in markdown
    assert "## Selected model" in markdown
    assert "| Rank | Family | Features |" in markdown
    assert "x1×x2" in markdown
    assert "breusch_pagan" in markdown
    assert "**yes**" in markdown
    assert "## Warnings" in markdown


def test_renders_empty_result_without_candidates() -> None:
    result = ModelSearchResult(target_kind="continuous", metric="r2")

    markdown = render_markdown(result)
    payload = report_payload(result)

    assert "No candidate beat the baseline" in markdown
    assert "_No eligible candidates._" in markdown
    assert payload["selection_confidence"] == "low"
    assert payload["candidates"] == []


def test_write_json_markdown_and_report(tmp_path) -> None:
    result = make_result()

    json_path = write_json(result, tmp_path / "nested" / "model_search.json")
    markdown_path = write_markdown(result, tmp_path / "nested" / "model_search.md")
    report_json, report_markdown = write_report(result, tmp_path / "bundle")

    assert json_path.exists()
    assert markdown_path.exists()
    assert report_json.exists() and report_markdown.exists()
    assert report_json.name == "model_search.json"
    assert report_markdown.name == "model_search.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["selected"]
    assert "# Model search" in markdown_path.read_text(encoding="utf-8")
