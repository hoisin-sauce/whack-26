"""Headless smoke tests for the optional desktop dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import UI_frontEnd  # noqa: E402


pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def write_fragment(run_dir: Path) -> Path:
    fragment = run_dir / "final_judge.fragment.html"
    fragment.write_text(
        "<h1>Final model judgement</h1><p>Accepted with concerns.</p>",
        encoding="utf-8",
    )
    return fragment


def test_workspace_loads_the_final_judge_fragment(app, tmp_path: Path) -> None:
    fragment = write_fragment(tmp_path)
    workspace = UI_frontEnd.DatasetWorkspace()

    workspace.load_final_judge_report(tmp_path)

    text = workspace.summary_output.toPlainText()
    assert "Final model judgement" in text
    assert "Accepted with concerns" in text
    assert fragment.is_file()


def test_workspace_rejects_a_missing_fragment(app, tmp_path: Path) -> None:
    workspace = UI_frontEnd.DatasetWorkspace()

    with pytest.raises(FileNotFoundError, match="run the pipeline with --judge"):
        workspace.load_final_judge_report(tmp_path)


def test_dashboard_loads_the_final_judge_into_the_current_tab(
    app, tmp_path: Path
) -> None:
    write_fragment(tmp_path)
    dashboard = UI_frontEnd.QuantDashboard()
    try:
        workspace = dashboard.add_dataset_tab()
        dashboard.load_final_judge(tmp_path)

        assert "Final model judgement" in workspace.summary_output.toPlainText()
    finally:
        dashboard.close()
