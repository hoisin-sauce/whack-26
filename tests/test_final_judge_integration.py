"""End-to-end final-judge handoff tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from artifact_fixtures import make_artifact

from whack26.artifact import validate_artifact, write_artifact
from whack26.cli import main
from whack26.config import Settings
from whack26.final_judge.report import render_json
from whack26.final_judge.types import FinalJudgeReport
from whack26.run import run_pipeline


def make_dataset(path: Path) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(0.0, 1.0, 80)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(0.0, 1.0, 80),
            "y": 2.0 * x + rng.normal(0.0, 0.1, 80),
        }
    )
    frame.to_csv(path, index=False)


def make_settings(tmp_path: Path) -> Settings:
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        offline=True,
    )


def test_run_writes_judge_artifact_preview_and_ui_fragment(tmp_path: Path) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        mock=True,
        offline=True,
        model_search=True,
        final_judge=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
    )

    for name in (
        "final_judge.json",
        "final_judge.md",
        "final_judge.html",
        "final_judge.fragment.html",
    ):
        assert (run_dir / name).is_file(), name

    artifact, errors = validate_artifact(outcome.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.final_judge is not None
    assert artifact.final_judge.decision == "insufficient_evidence"
    assert artifact.final_judge.fragment_path == "final_judge.fragment.html"
    assert outcome.final_judge is not None
    assert outcome.final_judge.decision == "insufficient_evidence"

    preview = outcome.preview_path.read_text(encoding="utf-8")
    assert "## Final model judgement" in preview
    assert "Insufficient evidence" in preview

    fragment = (run_dir / "final_judge.fragment.html").read_text(encoding="utf-8")
    assert "Final model judgement" in fragment
    assert "<style" not in fragment
    assert "<script" not in fragment

    payload = json.loads((run_dir / "final_judge.json").read_text(encoding="utf-8"))
    assert payload["decision"] == "insufficient_evidence"


def test_judge_cli_reuses_and_rewrites_the_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"
    run_pipeline(
        input_path=dataset,
        target="y",
        mock=True,
        offline=True,
        model_search=True,
        final_judge=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
    )
    capsys.readouterr()

    code = main(["judge", str(run_dir), "--mock", "--offline"])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "final judge:" in captured.out
    assert "reused final_judge.json" in captured.out
    assert (run_dir / "final_judge.fragment.html").is_file()


def prepare_fixture_run(tmp_path: Path) -> Path:
    artifact, run_dir, _ = make_artifact(tmp_path)
    write_artifact(run_dir / "artifact.json", artifact)
    return run_dir


def test_judge_cli_honours_format_on_the_reuse_path(
    tmp_path: Path, capsys
) -> None:
    run_dir = prepare_fixture_run(tmp_path)
    report = FinalJudgeReport(
        generated_at="2026-11-08T00:00:00Z",
        decision="accepted",
        n_critics_evaluated=9,
        min_required=5,
    )
    (run_dir / "final_judge.json").write_text(render_json(report), encoding="utf-8")
    out_dir = tmp_path / "out"

    code = main(
        ["judge", str(run_dir), "--format", "markdown", "--out", str(out_dir)]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (out_dir / "final_judge.md").is_file()
    assert not (out_dir / "final_judge.json").exists()
    assert not (out_dir / "final_judge.html").exists()
    assert "markdown:" in captured.out
    assert "json:" not in captured.out


def test_judge_cli_fresh_run_and_refresh(tmp_path: Path, capsys) -> None:
    run_dir = prepare_fixture_run(tmp_path)
    out_dir = tmp_path / "out"

    code = main(
        [
            "judge",
            str(run_dir),
            "--mock",
            "--offline",
            "--format",
            "json",
            "--out",
            str(out_dir),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (out_dir / "final_judge.json").is_file()
    assert not (out_dir / "final_judge.md").exists()
    payload = json.loads((out_dir / "final_judge.json").read_text(encoding="utf-8"))
    assert payload["decision"] in {
        "accepted",
        "accepted_with_concerns",
        "insufficient_evidence",
    }

    capsys.readouterr()
    code = main(
        [
            "judge",
            str(run_dir),
            "--mock",
            "--offline",
            "--refresh",
            "--format",
            "html",
            "--out",
            str(out_dir),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "reused" not in captured.out
    assert (out_dir / "final_judge.html").is_file()
    assert (out_dir / "final_judge.fragment.html").is_file()


def test_judge_cli_reports_an_invalid_artifact(tmp_path: Path, capsys) -> None:
    code = main(["judge", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    assert code == 1
    assert "artifact is not valid" in captured.err


def test_judge_cli_rejects_unreadable_existing_report(
    tmp_path: Path, capsys
) -> None:
    run_dir = prepare_fixture_run(tmp_path)
    (run_dir / "final_judge.json").write_text("{not json", encoding="utf-8")

    code = main(["judge", str(run_dir), "--mock", "--offline"])

    captured = capsys.readouterr()
    assert code == 1
    assert "existing final_judge.json is unreadable" in captured.err
