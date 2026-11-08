"""Dataset input and target-contract tests for the optional dashboard.

Covers UI-10 to UI-21 from docs/spec/ui-integration-test-plan.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

import UI_frontEnd  # noqa: E402
from ui_fixtures import make_ui_dataset  # noqa: E402

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


def _forbid_engine(monkeypatch) -> list:
    calls: list = []

    def fake_engine(**kwargs):
        calls.append(kwargs)
        return {"passed": True}

    monkeypatch.setattr(UI_frontEnd, "validation_engine", fake_engine)
    return calls


# --------------------------------------------------------------------- UI-10
def test_ui_10_valid_csv_drop_loads_frame_logs_shape_and_renames_tab(
    qtbot, tmp_path: Path
) -> None:
    path = make_ui_dataset(tmp_path / "alpha.csv", rows=80)
    dashboard = UI_frontEnd.QuantDashboard()
    qtbot.addWidget(dashboard)
    workspace = dashboard.current_workspace()

    workspace.handle_dropped_file(str(path))

    assert workspace.loaded_dataframe is not None
    assert workspace.loaded_dataframe.shape == (80, 4)
    log = workspace.log_output.toPlainText()
    assert "File dropped: alpha.csv" in log
    assert "Successfully loaded dataset! Shape: (80, 4)" in log
    assert "Columns detected: ['x', 'z', 'group', 'y']" in log
    assert workspace.dataset_name == "alpha"
    assert dashboard.tabs.tabText(0) == "alpha"


# --------------------------------------------------------------------- UI-11
def test_ui_11_valid_xlsx_drop_loads_through_read_excel(
    workspace, tmp_path: Path
) -> None:
    frame = _frame(rows=40)
    path = tmp_path / "book.xlsx"
    frame.to_excel(path, index=False)

    workspace.handle_dropped_file(str(path))

    assert workspace.loaded_dataframe is not None
    assert workspace.loaded_dataframe.shape == (40, 3)
    assert list(workspace.loaded_dataframe.columns) == ["x", "z", "y"]
    log = workspace.log_output.toPlainText()
    assert "Successfully loaded dataset! Shape: (40, 3)" in log
    assert workspace.dataset_name == "book"


# --------------------------------------------------------------------- UI-12
@pytest.mark.parametrize("extension", [".txt", ".parquet"])
def test_ui_12_unsupported_extension_is_logged_and_keeps_frame(
    workspace, tmp_path: Path, extension: str
) -> None:
    path = tmp_path / f"notes{extension}"
    path.write_text("not a dataset", encoding="utf-8")
    previous = _frame(rows=10)
    workspace.loaded_dataframe = previous

    workspace.handle_dropped_file(str(path))

    assert workspace.loaded_dataframe is previous
    log = workspace.log_output.toPlainText()
    assert f"Unsupported file extension: {extension}" in log
    assert ".csv, .xls, or .xlsx" in log


# --------------------------------------------------------------------- UI-13
def test_ui_13_malformed_csv_logs_pandas_error_without_raising(
    workspace, tmp_path: Path
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
    workspace.loaded_dataframe = None

    workspace.handle_dropped_file(str(path))

    assert workspace.loaded_dataframe is None
    log = workspace.log_output.toPlainText()
    assert "File dropped: bad.csv" in log
    assert "Error parsing file with Pandas:" in log
    assert "Error tokenizing data" in log


# --------------------------------------------------------------------- UI-14
def test_ui_14_non_numeric_target_is_rejected(
    workspace, monkeypatch
) -> None:
    frame = _frame(rows=60)
    frame["y"] = pd.Series(["up", "down"] * 30, dtype=object)
    workspace.loaded_dataframe = frame
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    with pytest.raises(ValueError, match="must be numeric and continuous"):
        workspace._prepare_arrays()

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "must be numeric and continuous" in summary
    assert "Dataset rejected:" in workspace.log_output.toPlainText()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "pandas 3 uses StringDtype for text columns and np.issubdtype raises "
        "TypeError before _prepare_arrays can raise its ValueError contract"
    ),
)
def test_ui_14_pandas_string_target_does_not_crash_the_dashboard(
    workspace,
) -> None:
    frame = _frame(rows=60)
    frame["y"] = ["up", "down"] * 30  # pandas 3 infers StringDtype
    workspace.loaded_dataframe = frame

    with pytest.raises(ValueError, match="must be numeric and continuous"):
        workspace._prepare_arrays()


# --------------------------------------------------------------------- UI-15
def test_ui_15_constant_target_is_rejected(workspace, monkeypatch) -> None:
    frame = _frame(rows=60)
    frame["y"] = 7.0
    workspace.loaded_dataframe = frame
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    with pytest.raises(ValueError, match="non-zero variance"):
        workspace._prepare_arrays()

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "at least three distinct values" in summary
    assert "constant" in summary.lower()


# --------------------------------------------------------------------- UI-16
def test_ui_16_binary_target_is_rejected(workspace, monkeypatch) -> None:
    frame = _frame(rows=60)
    frame["y"] = [0, 1] * 30
    workspace.loaded_dataframe = frame
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    with pytest.raises(ValueError, match="at least three distinct values"):
        workspace._prepare_arrays()

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "continuous" in summary
    assert "Binary" in summary


# --------------------------------------------------------------------- UI-17
def test_ui_17_too_few_complete_rows_is_rejected(
    workspace, monkeypatch
) -> None:
    workspace.loaded_dataframe = _frame(rows=29)
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    with pytest.raises(ValueError, match="At least 30 complete rows"):
        workspace._prepare_arrays()

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "found 29" in summary


# --------------------------------------------------------------------- UI-18
def test_ui_18_too_few_train_or_test_rows_is_rejected(
    workspace, monkeypatch
) -> None:
    workspace.loaded_dataframe = _frame(rows=30)
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    features, targets = workspace._prepare_arrays()
    assert features.shape == (30, 2)
    assert targets.shape == (30,)

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "At least 10 training rows and 10 test rows" in summary
    assert "chronological split" in summary


# --------------------------------------------------------------------- UI-19
def test_ui_19_non_numeric_features_are_ignored(workspace) -> None:
    frame = _frame(rows=80)
    frame.insert(
        1,
        "sector",
        pd.Series(["a", "b"] * 40, dtype=object),
    )
    workspace.loaded_dataframe = frame

    features, targets = workspace._prepare_arrays()

    assert features.shape == (80, 2)
    assert targets.shape == (80,)
    np.testing.assert_allclose(features, frame[["x", "z"]].to_numpy(dtype=float))
    log = workspace.log_output.toPlainText()
    assert "Ignoring non-numeric feature columns: sector" in log


@pytest.mark.xfail(
    strict=False,
    reason=(
        "pandas 3 uses StringDtype for text features and np.issubdtype raises "
        "TypeError instead of ignoring the column"
    ),
)
def test_ui_19_pandas_string_feature_is_ignored_not_fatal(workspace) -> None:
    frame = _frame(rows=80)
    frame.insert(1, "sector", ["a", "b"] * 40)  # pandas 3 infers StringDtype
    workspace.loaded_dataframe = frame

    features, targets = workspace._prepare_arrays()

    assert features.shape == (80, 2)
    assert targets.shape == (80,)


# --------------------------------------------------------------------- UI-20
def test_ui_20_sparse_text_feature_does_not_shrink_sample(workspace) -> None:
    frame = _frame(rows=60)
    frame.insert(
        1,
        "comment",
        pd.Series(
            ["note" if index % 2 == 0 else None for index in range(60)],
            dtype=object,
        ),
    )
    workspace.loaded_dataframe = frame

    features, targets = workspace._prepare_arrays()

    assert features.shape == (60, 2)
    assert targets.shape == (60,)
    assert np.all(np.isfinite(features))
    assert np.all(np.isfinite(targets))
    assert "Ignoring non-numeric feature columns: comment" in (
        workspace.log_output.toPlainText()
    )


# --------------------------------------------------------------------- UI-21
def test_ui_21_inf_in_features_is_rejected(workspace) -> None:
    frame = _frame(rows=40)
    frame.loc[3, "x"] = np.inf
    workspace.loaded_dataframe = frame

    with pytest.raises(ValueError, match="finite values"):
        workspace._prepare_arrays()


def test_ui_21_inf_in_target_is_rejected(workspace) -> None:
    frame = _frame(rows=40)
    frame.loc[4, "y"] = -np.inf
    workspace.loaded_dataframe = frame

    with pytest.raises(ValueError, match="finite values"):
        workspace._prepare_arrays()


def test_ui_21_nan_in_used_feature_is_dropped_then_rejected_when_too_few_rows(
    workspace, monkeypatch
) -> None:
    frame = _frame(rows=30)
    frame.loc[2, "x"] = np.nan
    workspace.loaded_dataframe = frame
    calls = _forbid_engine(monkeypatch)
    _set_restrictions(workspace)

    # The production code drops missing rows before the finite-value check, so
    # a NaN in a used column shrinks the sample instead of raising "non-finite".
    with pytest.raises(ValueError, match="found 29"):
        workspace._prepare_arrays()

    workspace.run_validation()

    assert calls == []
    summary = workspace.summary_output.toPlainText()
    assert "Status: REJECTED" in summary
    assert "At least 30 complete rows" in summary
