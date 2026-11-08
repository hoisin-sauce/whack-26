"""Integration tests for optional model-search backends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from whack26.config import Settings
from whack26.io import read_dataset
from whack26.model_search import sindy, trees
from whack26.model_search.pipeline import run_model_search
from whack26.profile import build_profile


def make_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, rows)
    z = rng.normal(0.0, 1.0, rows)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2.0 * x - 0.5 * z + rng.normal(0.0, 0.1, rows),
        }
    )


def make_profile(tmp_path: Path, frame: pd.DataFrame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    loaded = read_dataset(path, target="y")
    return build_profile(loaded, target="y"), loaded.frame


def test_sklearn_backend_adds_tree_candidates(tmp_path: Path) -> None:
    frame = make_frame()
    profile, loaded = make_profile(tmp_path, frame)
    settings = Settings(
        model_search_backend="sklearn",
        model_search_folds=3,
        model_search_test_fraction=0.30,
        model_search_top_n=10,
    )

    result = run_model_search(profile, loaded, settings=settings)

    families = {candidate.spec.family for candidate in result.candidates}
    if trees.sklearn_available():
        assert {"tree", "forest", "gradient_boosting"} <= families
        assert result.selected is not None
    else:
        assert not ({"tree", "forest", "gradient_boosting"} & families)


def test_full_backend_adds_sindy_for_time_series(tmp_path: Path) -> None:
    if not sindy.pysindy_available():
        return
    rows = 80
    t = np.arange(rows)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "x": np.sin(t / 5.0),
            "y": np.exp(-t / 40.0) + 0.01 * np.sin(t / 3.0),
        }
    )
    profile, loaded = make_profile(tmp_path, frame)
    settings = Settings(
        model_search_backend="full",
        model_search_folds=3,
        model_search_test_fraction=0.30,
        model_search_top_n=10,
    )

    result = run_model_search(profile, loaded, settings=settings)

    assert result.target_kind == "time_series"
    assert "sindy" in {candidate.spec.family for candidate in result.candidates}


def test_core_backend_stays_numpy_scipy_only(tmp_path: Path) -> None:
    frame = make_frame()
    profile, loaded = make_profile(tmp_path, frame)
    settings = Settings(model_search_backend="core")

    result = run_model_search(profile, loaded, settings=settings)

    families = {candidate.spec.family for candidate in result.candidates}
    assert not ({"tree", "forest", "gradient_boosting", "sindy"} & families)
