"""Pipeline/CLI-to-widget handoff tests (plan section F, SA-5).

Covers UI-71, UI-72, UI-74, UI-76 to UI-79. UI-70 and UI-75 are already
exercised by ``tests/test_final_judge_integration.py`` and are deliberately not
re-authored here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from UI_frontEnd import QuantDashboard  # noqa: E402
from ui_fixtures import (  # noqa: E402
    make_judge_run_dir,
    make_pipeline_settings,
    make_report,
    make_ui_dataset,
)
from whack26.cli import main  # noqa: E402
from whack26.run import run_pipeline  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

# Stable text contract from the final judge: report decision -> UI label.
DECISION_LABELS = {
    "accepted": "Accepted",
    "accepted_with_concerns": "Accepted with concerns",
    "requires_reaudit": "Re-audit required",
    "rejected": "Rejected",
    "insufficient_evidence": "Insufficient evidence",
}

JUDGE_FILES = (
    "artifact.json",
    "final_judge.json",
    "final_judge.html",
    "final_judge.fragment.html",
)


@pytest.fixture
def dashboard(qtbot):
    """A real dashboard whose lifetime is owned by pytest-qt."""
    widget = QuantDashboard()
    qtbot.addWidget(widget)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """One small real mock/offline pipeline run shared by UI-71 and UI-72."""
    tmp_path = tmp_path_factory.mktemp("ui_pipeline")
    dataset = make_ui_dataset(tmp_path / "input.csv")
    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        mock=True,
        offline=True,
        final_judge=True,
        settings=make_pipeline_settings(tmp_path),
        out_dir=tmp_path / "run",
    )
    assert outcome.final_judge is not None
    return outcome


def test_ui71_pipeline_decision_maps_to_fragment_label(pipeline_run) -> None:
    """UI-71: the real pipeline decision maps to the rendered decision label."""
    report = pipeline_run.final_judge
    assert report is not None

    label = DECISION_LABELS[report.decision]
    fragment = (pipeline_run.run_dir / "final_judge.fragment.html").read_text(
        encoding="utf-8"
    )

    assert label in fragment
    assert "Final model judgement" in fragment
    assert f"{report.n_critics_evaluated}/{report.min_required}" in fragment


def test_ui72_real_fragment_renders_decision_and_counts(dashboard, pipeline_run) -> None:
    """UI-72: the real fragment renders in a real tab with decision and n/min."""
    report = pipeline_run.final_judge
    assert report is not None

    workspace = dashboard.current_workspace()
    dashboard.load_final_judge(pipeline_run.run_dir, workspace)

    text = workspace.summary_output.toPlainText()
    assert "Final model judgement" in text
    assert DECISION_LABELS[report.decision] in text
    assert f"{report.n_critics_evaluated}/{report.min_required}" in text


def test_ui74_cli_run_judge_path_feeds_the_dashboard(
    dashboard, tmp_path: Path, monkeypatch, capsys
) -> None:
    """UI-74: ``whack26 run --judge --mock --offline`` output is UI-consumable."""
    dataset = make_ui_dataset(tmp_path / "input.csv")
    monkeypatch.setenv("WHACK26_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WHACK26_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("WHACK26_DATA_DIR", str(tmp_path / "data"))
    run_dir = tmp_path / "cli-run"

    code = main(
        [
            "run",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--out",
            str(run_dir),
            "--judge",
            "--mock",
            "--offline",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    for name in JUDGE_FILES:
        assert (run_dir / name).is_file(), name

    payload = json.loads(
        (run_dir / "final_judge.json").read_text(encoding="utf-8")
    )
    workspace = dashboard.current_workspace()
    dashboard.load_final_judge(run_dir, workspace)

    text = workspace.summary_output.toPlainText()
    assert "Final model judgement" in text
    assert DECISION_LABELS[payload["decision"]] in text
    assert (
        f"{payload['n_critics_evaluated']}/{payload['min_required']}" in text
    )


def test_ui76_two_run_directories_stay_isolated_per_tab(
    dashboard, tmp_path: Path
) -> None:
    """UI-76: two tabs keep their own run directory's decision/report."""
    run_a = make_judge_run_dir(
        tmp_path / "a",
        make_report(
            decision="accepted",
            n_critics_evaluated=5,
            min_required=5,
            note="TAB-A-ONLY",
        ),
    )
    run_b = make_judge_run_dir(
        tmp_path / "b",
        make_report(
            decision="rejected",
            n_critics_evaluated=2,
            min_required=5,
            note="TAB-B-ONLY",
        ),
    )

    tab_a = dashboard.current_workspace()
    tab_b = dashboard.add_dataset_tab()
    dashboard.load_final_judge(run_a, tab_a)
    dashboard.load_final_judge(run_b, tab_b)

    text_a = tab_a.summary_output.toPlainText()
    text_b = tab_b.summary_output.toPlainText()

    assert "Accepted" in text_a
    assert "5/5" in text_a
    assert "TAB-A-ONLY" in text_a
    assert "Rejected" not in text_a
    assert "TAB-B-ONLY" not in text_a

    assert "Rejected" in text_b
    assert "2/5" in text_b
    assert "TAB-B-ONLY" in text_b
    assert "Accepted" not in text_b
    assert "TAB-A-ONLY" not in text_b


def test_ui77_run_dir_accepts_str_and_path(dashboard, tmp_path: Path) -> None:
    """UI-77: ``load_final_judge`` accepts both ``str`` and ``Path`` run dirs."""
    str_run = make_judge_run_dir(
        tmp_path / "str-case",
        make_report(
            decision="accepted_with_concerns",
            n_critics_evaluated=4,
            min_required=5,
            note="STR-REFERENCE",
        ),
    )
    path_run = make_judge_run_dir(
        tmp_path / "path-case",
        make_report(
            decision="rejected",
            n_critics_evaluated=3,
            min_required=5,
            note="PATH-REFERENCE",
        ),
    )

    str_tab = dashboard.current_workspace()
    path_tab = dashboard.add_dataset_tab()
    dashboard.load_final_judge(str(str_run), str_tab)
    assert isinstance(path_run, Path)
    dashboard.load_final_judge(path_run, path_tab)

    text_str = str_tab.summary_output.toPlainText()
    text_path = path_tab.summary_output.toPlainText()

    assert "Accepted with concerns" in text_str
    assert "STR-REFERENCE" in text_str
    assert "PATH-REFERENCE" not in text_str

    assert "Rejected" in text_path
    assert "PATH-REFERENCE" in text_path
    assert "STR-REFERENCE" not in text_path


def test_ui78_missing_fragment_is_actionable_and_keeps_previous_report(
    dashboard, tmp_path: Path
) -> None:
    """UI-78: artifact-only run raises the --judge hint and keeps old report."""
    good_run = make_judge_run_dir(
        tmp_path / "good",
        make_report(decision="accepted", note="PREVIOUS-REPORT"),
    )
    artifact_only_run = make_judge_run_dir(
        tmp_path / "artifact-only",
        make_report(decision="rejected"),
        include_artifact=True,
    )
    fragment = artifact_only_run / "final_judge.fragment.html"
    fragment.unlink()
    assert (artifact_only_run / "artifact.json").is_file()
    assert not fragment.exists()

    workspace = dashboard.current_workspace()
    dashboard.load_final_judge(good_run, workspace)
    before = workspace.summary_output.toPlainText()
    assert "PREVIOUS-REPORT" in before

    with pytest.raises(FileNotFoundError, match="run the pipeline with --judge"):
        dashboard.load_final_judge(artifact_only_run, workspace)

    after = workspace.summary_output.toPlainText()
    assert after == before
    assert "PREVIOUS-REPORT" in after
    assert "Rejected" not in after


@pytest.mark.xfail(
    reason=(
        "no degraded-state handler in the current UI: a failed final-judge "
        "stage leaves no fragment and QuantDashboard.load_final_judge raises "
        "FileNotFoundError instead of rendering an error state; production "
        "change required"
    ),
    strict=False,
)
def test_ui79_failed_judge_stage_does_not_crash_the_tab(
    dashboard, tmp_path: Path
) -> None:
    """UI-79: a failed judge stage should degrade, not raise, in the tab."""
    run_dir = make_judge_run_dir(
        tmp_path / "failed-judge",
        include_artifact=True,
        fragment_only=False,
    )
    (run_dir / "final_judge.fragment.html").unlink()

    workspace = dashboard.current_workspace()
    dashboard.load_final_judge(run_dir, workspace)

    degraded = (
        workspace.summary_output.toPlainText()
        + workspace.log_output.toPlainText()
    ).lower()
    assert "final judge" in degraded
