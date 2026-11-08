"""Legacy validation-path tests for the optional dashboard.

Covers UI-30 to UI-35 from docs/spec/ui-integration-test-plan.md.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

import UI_frontEnd  # noqa: E402

pytestmark = pytest.mark.ui


@pytest.fixture
def workspace(qtbot):
    widget = UI_frontEnd.DatasetWorkspace()
    qtbot.addWidget(widget)
    return widget


def _frame(rows: int = 80, seed: int = 0) -> pd.DataFrame:
    """Numeric feature frame with a continuous target in the last column."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2.0 * x - z + rng.normal(scale=0.2, size=rows),
        }
    )


def _set_restrictions(
    workspace,
    *,
    periods: str = "252",
    min_sharpe: str = "0.95",
    max_decay: str = "0.25",
) -> None:
    workspace.periods_input.setText(periods)
    workspace.min_sharpe_input.setText(min_sharpe)
    workspace.max_decay_input.setText(max_decay)


def _stub_trials(monkeypatch, sharpes=(0.4, 0.8)) -> None:
    """Make the trial list deterministic without changing the UI contract."""

    def fake_trial_sharpes(features, target, periods_per_year, model_predict=None):
        return list(sharpes), np.zeros_like(np.asarray(target, dtype=float))

    monkeypatch.setattr(UI_frontEnd, "trial_sharpes", fake_trial_sharpes)


def _engine_result(passed: bool) -> dict:
    return {
        "passed": passed,
        "dsr_score": 0.97,
        "p_value": 0.01,
        "oos_score": 0.82,
        "decay_rate": 0.11,
        "tolerance": 0.25,
        "tolerance_clamped": False,
        "parameter_sensitivity": "PASS",
        "layers": {"layer_1": "pass"},
        "diagnostics": {"note": "ok"},
        "message": "Synthetic engine result.",
    }


# --------------------------------------------------------------------- UI-30
@pytest.mark.parametrize("passed", [True, False])
def test_ui_30_valid_run_writes_status_and_every_engine_key(
    workspace, monkeypatch, passed: bool
) -> None:
    _stub_trials(monkeypatch)
    result = _engine_result(passed)
    monkeypatch.setattr(UI_frontEnd, "validation_engine", lambda **kwargs: result)
    workspace.loaded_dataframe = _frame(rows=80)
    _set_restrictions(workspace)

    workspace.run_validation()

    expected_status = "PASSED" if passed else "REJECTED"
    summary = workspace.summary_output.toPlainText()
    assert f"Status: {expected_status}" in summary
    for key in result:
        if key == "passed":
            continue
        assert f"{key.replace('_', ' ').title()}:" in summary
    log = workspace.log_output.toPlainText()
    assert f"Validation cycle completed: {expected_status} (2 real trials)." in log


# --------------------------------------------------------------------- UI-31
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("periods_input", ""),
        ("min_sharpe_input", "not-a-number"),
        ("max_decay_input", ""),
    ],
)
def test_ui_31_missing_or_invalid_restriction_fields_skip_the_engine(
    workspace, monkeypatch, field: str, bad_value: str
) -> None:
    calls: list = []

    def fake_engine(**kwargs):
        calls.append(kwargs)
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", fake_engine)
    workspace.loaded_dataframe = _frame(rows=80)
    _set_restrictions(workspace)
    getattr(workspace, field).setText(bad_value)

    workspace.run_validation()

    assert calls == []
    log = workspace.log_output.toPlainText()
    assert "Missing or Invalid Inputs" in log
    assert "valid numbers" in log
    assert workspace.summary_output.toPlainText().strip() == ""
    assert workspace.isEnabled()


# --------------------------------------------------------------------- UI-32
def test_ui_32_fewer_than_two_real_trial_signals_is_rejected(
    workspace, monkeypatch
) -> None:
    def one_trial(features, target, periods_per_year, model_predict=None):
        return [0.5], np.zeros_like(np.asarray(target, dtype=float))

    monkeypatch.setattr(UI_frontEnd, "trial_sharpes", one_trial)
    calls: list = []

    def fake_engine(**kwargs):
        calls.append(kwargs)
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", fake_engine)
    workspace.loaded_dataframe = _frame(rows=80)
    _set_restrictions(workspace)

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "At least two real trial signals" in summary
    assert "DSR" in summary
    assert "Validation rejected" in workspace.log_output.toPlainText()


# --------------------------------------------------------------------- UI-33
@pytest.mark.parametrize("max_decay", ["0.05", "0.90"])
def test_ui_33_out_of_range_max_decay_reports_the_clamp(
    workspace, monkeypatch, max_decay: str
) -> None:
    _stub_trials(monkeypatch)
    captured: dict = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"passed": True, "tolerance_clamped": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", fake_engine)
    workspace.loaded_dataframe = _frame(rows=80)
    _set_restrictions(workspace, max_decay=max_decay)

    workspace.run_validation()

    assert captured["base_params"]["max_decay_tolerance"] == pytest.approx(
        float(max_decay)
    )
    summary = workspace.summary_output.toPlainText()
    assert "Status: PASSED" in summary
    assert (
        "Warning: max_decay_tolerance was clamped to the supported "
        "0.15-0.50 range." in summary
    )


# --------------------------------------------------------------------- UI-34
def test_ui_34_engine_exception_is_reported_and_window_survives(
    qtbot, monkeypatch
) -> None:
    _stub_trials(monkeypatch)

    def exploding_engine(**kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(UI_frontEnd, "validation_engine", exploding_engine)
    dashboard = UI_frontEnd.QuantDashboard()
    qtbot.addWidget(dashboard)
    workspace = dashboard.current_workspace()
    workspace.loaded_dataframe = _frame(rows=80)
    _set_restrictions(workspace)

    workspace.run_validation()

    log = workspace.log_output.toPlainText()
    summary = workspace.summary_output.toPlainText()
    assert "Engine Execution Error: engine exploded" in log
    assert "Error executing engine: engine exploded" in summary
    assert workspace.isEnabled()

    second = dashboard.add_dataset_tab()
    assert second is not workspace
    assert dashboard.tabs.count() == 2


# --------------------------------------------------------------------- UI-35
def test_ui_35_engine_receives_params_features_targets_and_label_space(
    workspace, monkeypatch
) -> None:
    frame = _frame(rows=60)
    workspace.loaded_dataframe = frame
    _set_restrictions(workspace)
    workspace.data_desc_input.setText("Synthetic continuous returns")

    builder_calls: list = []

    def fake_builder(params, X, y):
        builder_calls.append((params, np.array(X), np.array(y)))
        training_mean = float(np.mean(y))
        return lambda new_X: np.full(np.asarray(new_X).shape[0], training_mean)

    winning_returns = np.linspace(-0.01, 0.01, len(frame))

    def fake_trial_sharpes(features, target, periods_per_year, model_predict=None):
        return [0.4, 0.7], winning_returns

    captured: dict = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "build_linear_model", fake_builder)
    monkeypatch.setattr(UI_frontEnd, "trial_sharpes", fake_trial_sharpes)
    monkeypatch.setattr(UI_frontEnd, "validation_engine", fake_engine)

    workspace.run_validation()

    expected_features = frame[["x", "z"]].to_numpy(dtype=float)
    expected_targets = frame["y"].to_numpy(dtype=float)
    split_index = int(len(frame) * 0.70)

    assert len(builder_calls) == 1
    params, X_train, y_train = builder_calls[0]
    assert params == {
        "ridge": 1e-6,
        "min_dsr_limit": 0.95,
        "max_decay_tolerance": 0.25,
    }
    np.testing.assert_allclose(X_train, expected_features[:split_index])
    np.testing.assert_allclose(y_train, expected_targets[:split_index])

    assert captured["base_params"] == params
    assert captured["base_model_builder"] is fake_builder
    np.testing.assert_allclose(captured["features"], expected_features)
    np.testing.assert_allclose(captured["targets"], expected_targets)
    np.testing.assert_allclose(
        captured["actuals"], expected_targets[split_index:]
    )
    assert captured["predictions"].shape == captured["actuals"].shape
    assert np.all(np.isfinite(captured["predictions"]))
    np.testing.assert_allclose(
        captured["predictions"],
        np.full(
            len(frame) - split_index,
            float(np.mean(expected_targets[:split_index])),
        ),
    )
    np.testing.assert_allclose(captured["winning_returns"], winning_returns)
    assert captured["all_trial_sharpes"] == [0.4, 0.7]
    assert captured["periods_per_year"] == 252.0
    assert captured["metric_func"] is UI_frontEnd.r2_score
    assert captured["higher_is_better"] is True
    assert captured["data_description"] == "Synthetic continuous returns"
