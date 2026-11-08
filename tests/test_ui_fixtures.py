"""L0 checks for the optional-dependency boundary."""

from __future__ import annotations

from ui_fixtures import ui_skip_reason


def test_ui_skip_reason_when_pyqt_is_missing() -> None:
    assert ui_skip_reason(lambda name: None) == (
        "optional UI stack not installed: PyQt6"
    )


def test_ui_skip_reason_when_pyqt_is_present() -> None:
    assert ui_skip_reason(lambda name: object()) is None
