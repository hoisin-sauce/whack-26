"""Automatic pipeline submission from the dashboard.

These tests exercise the GUI-side wiring only. The real pipeline runner is
covered by tests/test_ui_pipeline.py and tests/test_run_flow.py.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

import UI_frontEnd  # noqa: E402
from ui_fixtures import make_ui_dataset, write_fragment  # noqa: E402

pytestmark = pytest.mark.ui


def _wait_for(qtbot, predicate, timeout: int = 4000) -> None:
    qtbot.waitUntil(predicate, timeout=timeout)


def test_auto_run_starts_on_drop_and_loads_the_report(qtbot, tmp_path: Path) -> None:
    dataset = make_ui_dataset(tmp_path / "auto.csv")
    run_dir = tmp_path / "run"
    write_fragment(
        run_dir,
        "<h1>Final model judgement</h1><p>Auto pipeline report</p>",
    )
    release = threading.Event()
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        assert release.wait(5.0)
        return SimpleNamespace(run_dir=run_dir)

    workspace = UI_frontEnd.DatasetWorkspace(
        auto_run_pipeline=True, pipeline_runner=fake_runner
    )
    qtbot.addWidget(workspace)

    workspace.handle_dropped_file(str(dataset))

    assert "Report loading" in workspace.summary_output.toPlainText()
    assert not workspace.run_btn.isEnabled()
    _wait_for(qtbot, lambda: bool(calls))
    assert calls[0]["final_judge"] is True
    assert calls[0]["model_search"] is True
    assert calls[0]["validate"] is True
    assert calls[0]["target"] is None

    release.set()
    _wait_for(
        qtbot,
        lambda: "Auto pipeline report" in workspace.summary_output.toPlainText(),
    )
    assert workspace.run_btn.isEnabled()
    assert workspace.run_btn.text() == UI_frontEnd.PIPELINE_BUTTON_IDLE
    assert str(run_dir) in workspace.log_output.toPlainText()


def test_run_button_starts_pipeline_when_auto_run_is_disabled(
    qtbot, tmp_path: Path
) -> None:
    dataset = make_ui_dataset(tmp_path / "manual.csv")
    run_dir = tmp_path / "run"
    write_fragment(run_dir, "<h1>Final model judgement</h1><p>Manual report</p>")
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_dir=run_dir)

    workspace = UI_frontEnd.DatasetWorkspace(pipeline_runner=fake_runner)
    qtbot.addWidget(workspace)
    workspace.handle_dropped_file(str(dataset))
    assert not calls
    assert workspace.summary_output.toPlainText().strip() == ""

    workspace.run_btn.click()
    _wait_for(
        qtbot,
        lambda: "Manual report" in workspace.summary_output.toPlainText(),
    )
    assert calls
    assert workspace.run_btn.isEnabled()


def test_pipeline_failure_shows_an_error_and_reenables_the_button(
    qtbot, tmp_path: Path
) -> None:
    dataset = make_ui_dataset(tmp_path / "failure.csv")

    def fake_runner(**kwargs):
        raise RuntimeError("provider unavailable")

    workspace = UI_frontEnd.DatasetWorkspace(
        auto_run_pipeline=True, pipeline_runner=fake_runner
    )
    qtbot.addWidget(workspace)
    workspace.handle_dropped_file(str(dataset))

    _wait_for(
        qtbot,
        lambda: "Report unavailable" in workspace.summary_output.toPlainText(),
    )
    text = workspace.summary_output.toPlainText()
    assert "provider unavailable" in text
    assert "Pipeline failed" in workspace.log_output.toPlainText()
    assert workspace.run_btn.isEnabled()


def test_pipeline_without_a_fragment_shows_an_error(qtbot, tmp_path: Path) -> None:
    dataset = make_ui_dataset(tmp_path / "empty.csv")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fake_runner(**kwargs):
        return SimpleNamespace(run_dir=run_dir)

    workspace = UI_frontEnd.DatasetWorkspace(
        auto_run_pipeline=True, pipeline_runner=fake_runner
    )
    qtbot.addWidget(workspace)
    workspace.handle_dropped_file(str(dataset))

    _wait_for(
        qtbot,
        lambda: "Report unavailable" in workspace.summary_output.toPlainText(),
    )
    assert "could not be loaded" in workspace.summary_output.toPlainText()
    assert workspace.run_btn.isEnabled()


def test_dashboard_propagates_auto_run_to_new_tabs(qtbot, tmp_path: Path) -> None:
    dataset = make_ui_dataset(tmp_path / "dashboard.csv")
    run_dir = tmp_path / "run"
    write_fragment(run_dir, "<h1>Final model judgement</h1><p>DASHBOARD-REPORT</p>")
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_dir=run_dir)

    dashboard = UI_frontEnd.QuantDashboard(
        auto_run_pipeline=True, pipeline_runner=fake_runner
    )
    qtbot.addWidget(dashboard)
    workspace = dashboard.current_workspace()

    workspace.handle_dropped_file(str(dataset))

    _wait_for(
        qtbot,
        lambda: "DASHBOARD-REPORT" in workspace.summary_output.toPlainText(),
    )
    assert calls


def test_target_selector_can_override_auto_detect(
    qtbot, tmp_path: Path
) -> None:
    dataset = make_ui_dataset(tmp_path / "target.csv")
    run_dir = tmp_path / "run"
    write_fragment(run_dir, "<h1>Final model judgement</h1><p>TARGET-REPORT</p>")
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_dir=run_dir)

    workspace = UI_frontEnd.DatasetWorkspace(pipeline_runner=fake_runner)
    qtbot.addWidget(workspace)
    workspace.handle_dropped_file(str(dataset))

    labels = [
        workspace.target_combo.itemText(index)
        for index in range(workspace.target_combo.count())
    ]
    assert labels[0] == "Auto-detect"
    assert "y" in labels

    workspace.target_combo.setCurrentIndex(labels.index("y"))
    workspace.run_btn.click()
    _wait_for(
        qtbot,
        lambda: "TARGET-REPORT" in workspace.summary_output.toPlainText(),
    )
    assert calls[-1]["target"] == "y"


def test_auto_run_is_isolated_between_tabs(qtbot, tmp_path: Path) -> None:
    first_dataset = make_ui_dataset(tmp_path / "first.csv")
    second_dataset = make_ui_dataset(tmp_path / "second.csv", seed=1)
    first_run = tmp_path / "first-run"
    second_run = tmp_path / "second-run"
    write_fragment(first_run, "<h1>Final model judgement</h1><p>FIRST-REPORT</p>")
    write_fragment(second_run, "<h1>Final model judgement</h1><p>SECOND-REPORT</p>")

    def make_runner(run_dir: Path):
        def runner(**kwargs):
            return SimpleNamespace(run_dir=run_dir)

        return runner

    first = UI_frontEnd.DatasetWorkspace(
        auto_run_pipeline=True, pipeline_runner=make_runner(first_run)
    )
    second = UI_frontEnd.DatasetWorkspace(
        auto_run_pipeline=True, pipeline_runner=make_runner(second_run)
    )
    qtbot.addWidget(first)
    qtbot.addWidget(second)

    first.handle_dropped_file(str(first_dataset))
    second.handle_dropped_file(str(second_dataset))

    _wait_for(
        qtbot,
        lambda: "FIRST-REPORT" in first.summary_output.toPlainText()
        and "SECOND-REPORT" in second.summary_output.toPlainText(),
    )
    assert "SECOND-REPORT" not in first.summary_output.toPlainText()
    assert "FIRST-REPORT" not in second.summary_output.toPlainText()
