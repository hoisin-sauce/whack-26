"""L2 widget tests for dashboard bootstrap and tab lifecycle.

Covers UI-01 to UI-07 from ``docs/spec/ui-integration-test-plan.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt  # noqa: E402

import UI_frontEnd  # noqa: E402
from ui_fixtures import make_ui_dataset  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pytestqt.qtbot import QtBot


pytestmark = pytest.mark.ui


@pytest.fixture
def dashboard(qtbot: QtBot) -> UI_frontEnd.QuantDashboard:
    widget = UI_frontEnd.QuantDashboard()
    qtbot.addWidget(widget)
    return widget


def write_dataset(path: Path) -> Path:
    """Write a deterministic UI dataset, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return make_ui_dataset(path)


def test_ui_01_dashboard_bootstrap_has_one_named_tab(
    dashboard: UI_frontEnd.QuantDashboard,
) -> None:
    """UI-01: a fresh dashboard has exactly one tab and a matching title."""
    assert dashboard.tabs.count() == 1

    tab_name = dashboard.tabs.tabText(0)
    assert tab_name
    assert tab_name in dashboard.windowTitle()
    assert isinstance(dashboard.current_workspace(), UI_frontEnd.DatasetWorkspace)


def test_ui_02_second_tab_is_independent_and_uniquely_named(
    dashboard: UI_frontEnd.QuantDashboard,
) -> None:
    """UI-02: adding a tab creates an isolated workspace with a unique name."""
    first = dashboard.current_workspace()

    second = dashboard.add_dataset_tab()

    assert dashboard.tabs.count() == 2
    assert dashboard.workspaces() == [first, second]
    assert first is not second

    names = [dashboard.tabs.tabText(i) for i in range(2)]
    assert names == ["Dataset 1", "Dataset 2"]
    assert dashboard.current_workspace() is second

    first.strategy_name_input.setText("alpha")
    assert second.strategy_name_input.text() == ""

    second.strategy_name_input.setText("beta")
    assert first.strategy_name_input.text() == "alpha"


def test_ui_03_same_stem_datasets_get_a_numeric_suffix(
    dashboard: UI_frontEnd.QuantDashboard, tmp_path: Path
) -> None:
    """UI-03: a second dataset with the same stem becomes ``name (2)``."""
    first_path = write_dataset(tmp_path / "left" / "returns.csv")
    second_path = write_dataset(tmp_path / "right" / "returns.csv")

    first = dashboard.current_workspace()
    first.handle_dropped_file(str(first_path))
    assert dashboard.tabs.tabText(0) == "returns"

    second = dashboard.add_dataset_tab()
    second.handle_dropped_file(str(second_path))

    assert dashboard.tabs.tabText(0) == "returns"
    assert dashboard.tabs.tabText(1) == "returns (2)"
    assert first.loaded_dataframe is not second.loaded_dataframe


def test_ui_04_rename_then_reset_restores_auto_naming(
    dashboard: UI_frontEnd.QuantDashboard, tmp_path: Path
) -> None:
    """UI-04: an explicit name wins until it is reset to dataset-based naming."""
    path = write_dataset(tmp_path / "prices.csv")
    workspace = dashboard.current_workspace()
    workspace.handle_dropped_file(str(path))
    assert dashboard.tabs.tabText(0) == "prices"

    dashboard.set_tab_name("  Risk   Model  ", workspace)
    assert dashboard.tabs.tabText(0) == "Risk Model"

    dashboard.set_tab_name(None, workspace)
    assert dashboard.tabs.tabText(0) == "prices"

    dashboard.set_tab_name("Pinned", tab_ref=0)
    assert dashboard.tabs.tabText(0) == "Pinned"

    dashboard.set_tab_name("", tab_ref=0)
    assert dashboard.tabs.tabText(0) == "prices"


def test_ui_05_closing_a_tab_renumbers_remaining_names(
    dashboard: UI_frontEnd.QuantDashboard, tmp_path: Path
) -> None:
    """UI-05: closing a tab recomputes the remaining auto-generated names."""
    first_path = write_dataset(tmp_path / "left" / "returns.csv")
    second_path = write_dataset(tmp_path / "right" / "returns.csv")

    first = dashboard.current_workspace()
    first.handle_dropped_file(str(first_path))
    second = dashboard.add_dataset_tab()
    second.handle_dropped_file(str(second_path))
    assert [dashboard.tabs.tabText(i) for i in range(2)] == [
        "returns",
        "returns (2)",
    ]

    dashboard.close_tab(0)

    assert dashboard.tabs.count() == 1
    assert dashboard.current_workspace() is second
    assert dashboard.tabs.tabText(0) == "returns"


def test_ui_05_closing_the_last_tab_creates_a_fresh_one(
    dashboard: UI_frontEnd.QuantDashboard,
) -> None:
    """UI-05: closing the final tab replaces it instead of leaving an empty shell."""
    stale = dashboard.current_workspace()

    dashboard.close_tab(0)

    assert dashboard.tabs.count() == 1
    fresh = dashboard.current_workspace()
    assert fresh is not stale
    assert dashboard.tabs.tabText(0).startswith("Dataset ")
    assert fresh.loaded_dataframe is None
    assert dashboard.tabs.tabText(0) in dashboard.windowTitle()


def test_ui_06_ctrl_t_and_ctrl_w_match_the_buttons(
    qtbot: QtBot, dashboard: UI_frontEnd.QuantDashboard
) -> None:
    """UI-06: the keyboard shortcuts add and close tabs like the controls."""
    with qtbot.waitExposed(dashboard):
        dashboard.show()
    with qtbot.waitActive(dashboard):
        dashboard.activateWindow()
    baseline = dashboard.tabs.count()

    qtbot.keyClick(
        dashboard, Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier
    )
    assert dashboard.tabs.count() == baseline + 1

    dashboard.add_tab_button.click()
    assert dashboard.tabs.count() == baseline + 2

    qtbot.keyClick(
        dashboard, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier
    )
    assert dashboard.tabs.count() == baseline + 1

    qtbot.keyClick(
        dashboard, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier
    )
    assert dashboard.tabs.count() == baseline


def test_ui_07_tab_tooltip_follows_description_then_title(
    dashboard: UI_frontEnd.QuantDashboard, tmp_path: Path
) -> None:
    """UI-07: the tooltip mirrors the description and falls back to the title."""
    path = write_dataset(tmp_path / "prices.csv")
    workspace = dashboard.current_workspace()
    workspace.handle_dropped_file(str(path))
    assert dashboard.tabs.tabText(0) == "prices"

    workspace.data_desc_input.setText("Daily prices with macro features")
    assert dashboard.tabs.tabToolTip(0) == "Daily prices with macro features"

    workspace.data_desc_input.setText("   ")
    assert dashboard.tabs.tabToolTip(0) == dashboard.tabs.tabText(0) == "prices"

    dashboard.set_tab_name("Pinned", workspace)
    workspace.data_desc_input.setText("A custom description")
    assert dashboard.tabs.tabToolTip(0) == "A custom description"

    workspace.data_desc_input.clear()
    assert dashboard.tabs.tabToolTip(0) == "Pinned"
