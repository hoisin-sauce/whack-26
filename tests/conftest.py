"""Shared pytest setup.

Keep the repository root and the tests directory importable regardless of the
working directory the suite is launched from. ``pyproject.toml`` also sets
``pythonpath = ["."]``; this fallback keeps direct invocations working on older
pytest versions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui_fixtures import ui_skip_reason  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip ``ui`` tests when the optional PyQt6 stack is not installed."""
    reason = ui_skip_reason()
    if reason is None:
        return
    for item in items:
        if "ui" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(autouse=True)
def _restore_optional_model_fitters():
    """Keep backend registration from leaking between tests.

    ``run_model_search`` registers optional tree/SINDy fitters on import of the
    pipeline. That is process-global state, so a test that selects the sklearn
    or full backend would otherwise make the "backend unavailable" tests fail
    only when they run later in the same process.
    """
    from whack26.model_search import fit as model_fit

    saved = dict(model_fit._OPTIONAL_FITTERS)
    try:
        yield
    finally:
        model_fit._OPTIONAL_FITTERS.clear()
        model_fit._OPTIONAL_FITTERS.update(saved)
