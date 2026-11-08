"""Shared fixtures for the UI integration test plan.

Keep this module free of PyQt imports so the core suite can import it on a
machine without the optional UI stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

from whack26.config import Settings
from whack26.final_judge.report import write_reports
from whack26.final_judge.types import FinalJudgeReport


def ui_skip_reason(
    find_spec: Callable[[str], Any] | None = None,
) -> str | None:
    """Return a skip reason when PyQt6 is unavailable, else ``None``.

    The hook deliberately checks only PyQt6. pytest-qt is optional for the
    existing QApplication-based smoke tests; modules that use ``qtbot`` add
    their own ``pytest.importorskip("pytestqt")``.
    """
    probe = find_spec or importlib.util.find_spec
    if probe("PyQt6") is None:
        return "optional UI stack not installed: PyQt6"
    return None


def make_ui_dataset(
    path: Path, rows: int = 80, seed: int = 0
) -> Path:
    """Write a small dataset whose last column is a continuous target."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": ["a", "b"] * (rows // 2),
            "y": 2.0 * x - z + rng.normal(scale=0.2, size=rows),
        }
    )
    frame.to_csv(path, index=False)
    return path


def make_pipeline_settings(tmp_path: Path) -> Settings:
    """Settings that keep a mock pipeline offline and inside the temp tree."""
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        offline=True,
    )


def make_report(
    decision: str = "insufficient_evidence",
    **overrides: Any,
) -> FinalJudgeReport:
    """Build a final-judge report without invoking an LLM."""
    values: dict[str, Any] = {
        "generated_at": "2026-11-08T10:00:00Z",
        "decision": decision,
        "n_critics_evaluated": 0,
        "min_required": 5,
        "unevaluated_critics": ["overfitting_skeptic"],
        "note": "The audit did not complete; this is NOT a pass.",
    }
    values.update(overrides)
    return FinalJudgeReport(**values)


def write_fragment(
    run_dir: Path,
    html: str | None = None,
) -> Path:
    """Write a UI fragment with a sentinel when no custom HTML is given."""
    run_dir.mkdir(parents=True, exist_ok=True)
    fragment = run_dir / "final_judge.fragment.html"
    fragment.write_text(
        html
        or (
            "<h1>Final model judgement</h1>"
            "<p>Insufficient evidence. This is NOT a pass.</p>"
        ),
        encoding="utf-8",
    )
    return fragment


def make_judge_run_dir(
    tmp_path: Path,
    report: FinalJudgeReport | None = None,
    *,
    include_artifact: bool = False,
    fragment_only: bool = True,
) -> Path:
    """Create a run directory containing the final-judge fragment."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = report or make_report()
    if fragment_only:
        write_reports(run_dir, report, formats=("fragment",))
    else:
        write_reports(run_dir, report)
    if include_artifact:
        (run_dir / "artifact.json").write_text(
            '{"schema_version": "1.5.0"}', encoding="utf-8"
        )
    return run_dir


__all__ = [
    "make_judge_run_dir",
    "make_pipeline_settings",
    "make_report",
    "make_ui_dataset",
    "ui_skip_reason",
    "write_fragment",
]
