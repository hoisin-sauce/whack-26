from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from whack26.artifact import validate_artifact
from whack26.cli import main
from whack26.config import Settings
from whack26.research.base import ResearchRecord
from whack26.research.pipeline import ResearchResult
from whack26.run import run_pipeline


def make_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(5.0, 1.0, 40)
    frame = pd.DataFrame(
        {
            "x": x,
            "group": ["a", "b"] * 20,
            "y": 2.0 * x + rng.normal(0.0, 0.2, 40),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def make_settings(tmp_path: Path, *, offline: bool = True) -> Settings:
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        offline=offline,
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_run_pipeline_mock_offline_writes_valid_artifact_and_preview(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"
    messages: list[str] = []

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
        log=messages.append,
    )

    for relative in (
        "artifact.json",
        "preview.md",
        "dataset.csv",
        "factors.json",
        "research.json",
        "logs/run.log",
    ):
        assert (run_dir / relative).is_file(), relative

    artifact, errors = validate_artifact(outcome.artifact_path)
    assert errors == []
    assert artifact is not None
    assert len(artifact.hypotheses) == 3
    assert {item.method_family for item in artifact.hypotheses} == {
        "linear_model",
        "group_comparison",
        "interaction_or_derived",
    }
    assert all(item.execution.status == "ok" for item in artifact.hypotheses)
    assert all(item.critic is not None for item in artifact.hypotheses)
    assert artifact.classification.path
    assert any("run_done" in message for message in messages)

    preview = outcome.preview_path.read_text(encoding="utf-8")
    assert "## Hypotheses" in preview
    assert "## Critic summary" in preview
    for hypothesis in artifact.hypotheses:
        assert hypothesis.title in preview


def test_generated_modules_reproduce_saved_predictions(tmp_path: Path) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
    )
    frame = pd.read_csv(run_dir / "dataset.csv")

    for hypothesis in outcome.artifact.hypotheses:
        module = load_module(run_dir / hypothesis.code_path)
        assert module.applies(frame) is True
        result = module.run(frame.copy(), seed=0)
        predictions = result["predictions"]
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        saved = pd.read_csv(run_dir / hypothesis.execution.predictions_path)
        np.testing.assert_allclose(
            np.asarray(predictions, dtype=float),
            saved["prediction"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        )
        assert saved["row_index"].tolist() == frame.index.tolist()


def test_cli_run_validate_and_preview(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    monkeypatch.setenv("WHACK26_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WHACK26_RUNS_DIR", str(tmp_path / "runs"))
    run_dir = tmp_path / "cli-run"

    code = main(
        [
            "run",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--mock",
            "--offline",
            "--out",
            str(run_dir),
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "artifact:" in output
    artifact_path = run_dir / "artifact.json"
    assert artifact_path.is_file()

    code = main(["validate-artifact", str(artifact_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "OK" in captured.out
    assert "hypotheses: 3" in captured.out

    (run_dir / "preview.md").unlink()
    code = main(["preview", str(artifact_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (run_dir / "preview.md").is_file()


def test_cli_validate_reports_broken_artifact(tmp_path: Path, capsys) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{not json", encoding="utf-8")

    code = main(["validate-artifact", str(artifact_path)])

    captured = capsys.readouterr()
    assert code == 1
    assert "not valid JSON" in captured.err


def test_cli_schema_writes_json_schema(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "artifact.schema.json"

    code = main(["schema", "--out", str(destination)])

    assert code == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["type"] == "object"
    assert "HypothesisSection" in payload["$defs"]
    assert "wrote" in capsys.readouterr().out


def test_run_pipeline_with_scrape_mock_writes_scrape_json(tmp_path: Path) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"
    settings = make_settings(tmp_path).with_overrides(
        scrape_enabled=True,
        scrape_max_resources=10,
        scrape_max_depth=1,
    )

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        scrape=True,
        settings=settings,
        out_dir=run_dir,
    )

    assert outcome.scrape is not None
    assert (run_dir / "scrape.json").is_file()
    assert (run_dir / "scrape.md").is_file()
    artifact, errors = validate_artifact(outcome.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.research.semantic_scraping is not None
    assert artifact.research.semantic_scraping.enabled is True
    assert "## Semantic scraping" in outcome.preview_path.read_text(
        encoding="utf-8"
    )


def test_cli_run_with_scrape_mock(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    monkeypatch.setenv("WHACK26_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WHACK26_RUNS_DIR", str(tmp_path / "runs"))
    run_dir = tmp_path / "cli-scrape-run"

    code = main(
        [
            "run",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--mock",
            "--offline",
            "--scrape",
            "--fetch",
            "--out",
            str(run_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "scrape:" in captured.out
    assert (run_dir / "scrape.json").is_file()


def test_cli_scrape_command_mock(tmp_path: Path, monkeypatch, capsys) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    monkeypatch.setenv("WHACK26_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WHACK26_RUNS_DIR", str(tmp_path / "runs"))
    run_dir = tmp_path / "cli-scrape-command"

    code = main(
        [
            "scrape",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--mock",
            "--offline",
            "--max-resources",
            "5",
            "--max-depth",
            "1",
            "--fetch",
            "--out",
            str(run_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "scrape:" in captured.out
    assert (run_dir / "scrape.json").is_file()
    assert (run_dir / "scrape.md").is_file()


def test_cli_preload_statistics_zero_limit(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "preload-statistics",
            "--mock",
            "--offline",
            "--limit",
            "0",
            "--db",
            str(tmp_path / "fields.db"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "preload: 0 document(s)" in captured.out
    assert "preload_limit_zero" in captured.out


def test_run_pipeline_with_scrape_and_seed_writes_documents(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    record = ResearchRecord(
        id="openalex:W1",
        source="openalex",
        source_id="W1",
        title="A housing study",
        abstract="A study of housing prices and income.",
        authors=["A. Author"],
        year=2024,
        doi=None,
        url="https://example.test/W1",
        license="CC0",
        retrieved_at="2026-11-08T10:00:00Z",
        query="housing prices",
        relevance=0.9,
    )
    monkeypatch.setattr(
        "whack26.run.run_research",
        lambda **kwargs: ResearchResult(
            records=[record],
            queries=[],
            warnings=[],
            keywords=["housing"],
            sources_used=["openalex"],
            cache={"hits": 0, "misses": 0},
        ),
    )
    run_dir = tmp_path / "run"
    settings = make_settings(tmp_path).with_overrides(
        scrape_enabled=True,
        scrape_max_resources=5,
        scrape_max_depth=1,
    )

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        scrape=True,
        settings=settings,
        out_dir=run_dir,
    )

    assert outcome.scrape is not None
    assert len(outcome.scrape.documents) >= 1
    artifact, errors = validate_artifact(outcome.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.research.semantic_scraping is not None
    assert len(artifact.research.semantic_scraping.documents) >= 1
    assert (run_dir / "scrape.json").is_file()
