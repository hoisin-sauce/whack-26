"""Producer-side coverage for the evidence consumed by the final judge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from whack26.config import Settings
from whack26.io import read_dataset
from whack26.model_search.pipeline import run_model_search
from whack26.profile import build_profile


def make_profile_and_frame(tmp_path: Path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=120)
    z = rng.normal(size=120)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": ["a", "b", "c"] * 40,
            "y": 2.0 * x + 0.4 * z + rng.normal(scale=0.2, size=120),
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    loaded = read_dataset(path, target="y")
    profile = build_profile(
        loaded,
        target="y",
        description="A synthetic regression dataset.",
    )
    return profile, loaded.frame


def test_run_model_search_populates_final_judge_evidence(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    settings = Settings(
        model_search_folds=3,
        model_search_seed=0,
        model_search_max_candidates=12,
        model_search_top_n=3,
    )

    result = run_model_search(profile, frame, settings=settings)

    assert result.dataset_info is not None
    assert result.dataset_info["target"]["name"] == "y"
    assert result.screen is not None
    assert result.screen["features"]
    assert result.selected is not None
    assert result.selected.folds

    for fold in result.selected.folds:
        assert fold.train_metrics
        assert "r2" in fold.train_metrics
        assert fold.metrics
        assert "r2" in fold.metrics
        assert fold.train_target_min is not None
        assert fold.train_target_max is not None
        assert fold.prediction_min is not None
        assert fold.prediction_max is not None
        assert np.isfinite(fold.train_target_min)
        assert np.isfinite(fold.prediction_max)

    # The serialized copies survive a JSON round trip, which is what the
    # artifact and the judge CLI rely on.
    restored = result.__class__.model_validate_json(result.model_dump_json())
    assert restored.dataset_info == result.dataset_info
    assert restored.screen == result.screen
    assert restored.selected is not None
    assert restored.selected.folds[0].train_metrics == result.selected.folds[0].train_metrics


def test_run_model_search_skips_without_a_target_or_enough_rows(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    profile = profile.model_copy(update={"target": None})

    missing_target = run_model_search(profile, frame)
    too_small = run_model_search(
        build_profile(
            read_dataset(
                _write_small_dataset(tmp_path), target="y"
            ),
            target="y",
        ),
        pd.read_csv(_write_small_dataset(tmp_path)),
    )

    assert missing_target.selected is None
    assert missing_target.warnings == ["model_search_skipped:no_target"]
    assert too_small.selected is None
    assert too_small.warnings[0].startswith("model_search_skipped:too_few_rows")


def _write_small_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "small.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}).to_csv(
        path, index=False
    )
    return path
