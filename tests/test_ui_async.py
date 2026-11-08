"""Responsiveness, progress, and cancellation checks (UI-90 to UI-95).

``DatasetWorkspace.start_pipeline`` now runs the full pipeline in a background
``_PipelineTask`` with streamed logs, busy state, and failure handling.  These
tests target the legacy ``DatasetWorkspace.run_validation`` path, which is still
synchronous on the GUI thread and has no cancel control.  They document the
intended behavior for that path and are xfailed until it delegates to the
pipeline worker or gains its own async seam.  They deliberately fail on an
assertion rather than skipping so the gap stays visible in the test report.
"""

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QPushButton  # noqa: E402

import UI_frontEnd  # noqa: E402
from ui_fixtures import make_ui_dataset  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

ASYNC_XFAIL_REASON = (
    "UI-90..UI-95 xfail: the legacy DatasetWorkspace.run_validation path is "
    "still synchronous on the GUI thread and has no cancel control. "
    "start_pipeline uses a background _PipelineTask with streamed logs, busy "
    "state, and failure handling; re-enable these once run_validation delegates "
    "to that seam."
)


def _placeholder_runner(**_kwargs):
    """Keep the legacy async placeholders away from live providers."""
    time.sleep(0.25)
    raise RuntimeError("async placeholder")


@pytest.fixture
def dashboard(qtbot):
    widget = UI_frontEnd.QuantDashboard(pipeline_runner=_placeholder_runner)
    qtbot.addWidget(widget)
    return widget


def _new_dashboard(qtbot):
    """Build a dashboard for tests that need more than one tab."""
    widget = UI_frontEnd.QuantDashboard(pipeline_runner=_placeholder_runner)
    qtbot.addWidget(widget)
    return widget


def _prepare_workspace(workspace, tmp_path, name: str) -> None:
    dataset = make_ui_dataset(tmp_path / f"{name}.csv")
    workspace.handle_dropped_file(str(dataset))
    if workspace.loaded_dataframe is None:
        raise RuntimeError("fixture dataset did not load into the workspace")
    # The shared fixture includes a text feature.  Keep this async slice on
    # numeric data so it does not depend on the separate dataset-contract fix.
    frame = workspace.loaded_dataframe
    text_columns = [
        column
        for column in frame.columns[:-1]
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if text_columns:
        workspace.loaded_dataframe = frame.drop(columns=text_columns)
    workspace.strategy_name_input.setText(name)
    workspace.data_desc_input.setText(name)
    workspace.periods_input.setText("252")
    workspace.min_sharpe_input.setText("0.95")
    workspace.max_decay_input.setText("0.25")


@pytest.mark.xfail(strict=False, reason=ASYNC_XFAIL_REASON, raises=AssertionError)
def test_ui_90_event_loop_ticks_while_a_slow_run_is_active(
    qtbot, monkeypatch, tmp_path
) -> None:
    dashboard = _new_dashboard(qtbot)
    workspace = dashboard.current_workspace()
    _prepare_workspace(workspace, tmp_path, "tab-one")

    ticks: list[int] = []
    observed: dict[str, int] = {}

    def slow_engine(**_kwargs):
        before = len(ticks)
        time.sleep(0.2)
        observed["during"] = len(ticks) - before
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", slow_engine)

    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    try:
        QTimer.singleShot(0, workspace.run_btn.click)
        qtbot.wait(500)
    finally:
        timer.stop()

    assert observed.get("during", 0) >= 3, (
        "no event-loop timers fired while the run was active"
    )


@pytest.mark.xfail(strict=False, reason=ASYNC_XFAIL_REASON, raises=AssertionError)
def test_ui_91_log_updates_arrive_before_completion(
    qtbot, monkeypatch, tmp_path
) -> None:
    dashboard = _new_dashboard(qtbot)
    workspace = dashboard.current_workspace()
    _prepare_workspace(workspace, tmp_path, "tab-one")

    state = {"running": False}
    updates_during_run: list[bool] = []

    def slow_engine(**_kwargs):
        state["running"] = True
        try:
            time.sleep(0.2)
        finally:
            state["running"] = False
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", slow_engine)
    workspace.log_output.textChanged.connect(
        lambda: updates_during_run.append(state["running"])
    )

    QTimer.singleShot(0, workspace.run_btn.click)
    qtbot.wait(500)

    assert any(updates_during_run), (
        "no log update was delivered while the run was active"
    )


def test_ui_92_closing_a_tab_mid_run_is_safe(qtbot, monkeypatch, tmp_path) -> None:
    dashboard = _new_dashboard(qtbot)
    workspace = dashboard.current_workspace()
    _prepare_workspace(workspace, tmp_path, "tab-one")

    state = {"finished": False}

    def slow_engine(**_kwargs):
        time.sleep(0.25)
        state["finished"] = True
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", slow_engine)

    def close_mid_run() -> None:
        state["closed_while_running"] = not state["finished"]
        dashboard.close_tab(dashboard.tabs.indexOf(workspace))

    QTimer.singleShot(0, workspace.run_btn.click)
    QTimer.singleShot(50, close_mid_run)
    qtbot.wait(600)

    assert state.get("closed_while_running") is True, (
        "the tab could not be closed until the synchronous run finished"
    )
    assert dashboard.tabs.count() == 1


@pytest.mark.xfail(strict=False, reason=ASYNC_XFAIL_REASON, raises=AssertionError)
def test_ui_93_two_concurrent_runs_keep_tabs_isolated(
    qtbot, monkeypatch, tmp_path
) -> None:
    dashboard = _new_dashboard(qtbot)
    first = dashboard.current_workspace()
    second = dashboard.add_dataset_tab()
    _prepare_workspace(first, tmp_path, "tab-one")
    _prepare_workspace(second, tmp_path, "tab-two")

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_engine(**kwargs):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.2)
        with lock:
            state["active"] -= 1
        return {"passed": True, "tag": kwargs.get("data_description")}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", slow_engine)

    QTimer.singleShot(0, first.run_btn.click)
    QTimer.singleShot(30, second.run_btn.click)
    qtbot.wait(800)

    assert state["max_active"] == 2, "the two runs never overlapped"

    first_text = first.summary_output.toPlainText()
    second_text = second.summary_output.toPlainText()
    assert "tab-one" in first_text
    assert "tab-one" not in second_text
    assert "tab-two" in second_text
    assert "tab-two" not in first_text


@pytest.mark.xfail(strict=False, reason=ASYNC_XFAIL_REASON, raises=AssertionError)
def test_ui_94_pipeline_exception_re_enables_the_run_button(
    qtbot, monkeypatch, tmp_path
) -> None:
    dashboard = _new_dashboard(qtbot)
    workspace = dashboard.current_workspace()
    _prepare_workspace(workspace, tmp_path, "tab-one")

    state: dict[str, bool | None] = {"disabled_during_run": None}

    def failing_engine(**_kwargs):
        state["disabled_during_run"] = not workspace.run_btn.isEnabled()
        time.sleep(0.15)
        raise RuntimeError("UI-94 boom")

    monkeypatch.setattr(UI_frontEnd, "validation_engine", failing_engine)

    QTimer.singleShot(0, workspace.run_btn.click)
    qtbot.wait(500)

    assert state["disabled_during_run"] is True, (
        "the run button was never disabled while the run was active"
    )
    assert workspace.run_btn.isEnabled(), "the run button stayed disabled"
    assert "UI-94 boom" in workspace.log_output.toPlainText()


@pytest.mark.xfail(strict=False, reason=ASYNC_XFAIL_REASON, raises=AssertionError)
def test_ui_95_cancel_button_stops_a_slow_run(
    qtbot, monkeypatch, tmp_path
) -> None:
    dashboard = _new_dashboard(qtbot)
    workspace = dashboard.current_workspace()
    _prepare_workspace(workspace, tmp_path, "tab-one")

    state = {"cancel_buttons": 0, "cancelled": False}

    def slow_engine(**_kwargs):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if state["cancelled"]:
                return {"passed": False, "cancelled": True}
            time.sleep(0.01)
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", slow_engine)

    def request_cancel() -> None:
        buttons = [
            button
            for button in workspace.findChildren(QPushButton)
            if "cancel" in button.text().lower()
        ]
        state["cancel_buttons"] = len(buttons)
        if buttons:
            state["cancelled"] = True
            buttons[0].click()

    QTimer.singleShot(0, workspace.run_btn.click)
    QTimer.singleShot(50, request_cancel)
    qtbot.wait(1200)

    assert state["cancel_buttons"] > 0, "the dashboard has no cancel control"
    assert state["cancelled"] is True
    assert "cancel" in workspace.log_output.toPlainText().lower()
