"""Tests for model-search feature screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whack26.model_search.screen import (
    benjamini_hochberg,
    compute_vif,
    screen_features,
)


def _by_name(screen) -> dict[str, object]:
    return {feature.feature: feature for feature in screen.features}


def test_pearson_and_spearman_known_signal() -> None:
    rng = np.random.default_rng(0)
    rows = 400
    x = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=rows),
            "y": 3.0 * x + 0.05 * rng.normal(size=rows),
        }
    )
    screen = screen_features(frame, "y", max_features=1)
    features = _by_name(screen)
    assert screen.selected == ["x"]
    assert features["x"].pearson is not None
    assert features["x"].pearson > 0.99
    assert features["x"].spearman is not None
    assert features["x"].spearman > 0.99
    assert features["x"].adjusted_p is not None
    assert features["x"].adjusted_p < 0.01
    assert features["x"].significant is True
    assert abs(features["noise"].pearson) < 0.2
    assert features["noise"].significant is False


def test_spearman_detects_monotonic_non_linear_relationship() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(0.1, 5.0, 300)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": x**3 + 0.1 * rng.normal(size=x.size),
        }
    )
    screen = screen_features(frame, "y", max_features=1)
    feature = _by_name(screen)["x"]
    assert feature.spearman is not None
    assert feature.spearman > 0.99
    assert feature.p_value is not None
    assert feature.p_value < 0.01


def test_mutual_information_ranks_a_real_non_linear_signal() -> None:
    rng = np.random.default_rng(2)
    rows = 600
    x = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=rows),
            "y": (x + 0.1 * rng.normal(size=rows) > 0.0).astype(int),
        }
    )
    screen = screen_features(frame, "y", max_features=1)
    features = _by_name(screen)
    assert screen.target_kind == "binary"
    assert screen.selected == ["x"]
    assert features["x"].mutual_info is not None
    assert features["noise"].mutual_info is not None
    assert features["x"].mutual_info > 0.05
    assert features["x"].mutual_info > features["noise"].mutual_info


def test_anova_finds_categorical_group_means() -> None:
    rng = np.random.default_rng(3)
    rows = 300
    categories = np.repeat(["a", "b", "c"], rows // 3)
    effects = {"a": 0.0, "b": 5.0, "c": 10.0}
    frame = pd.DataFrame(
        {
            "cat": categories,
            "y": [effects[value] for value in categories]
            + rng.normal(scale=0.2, size=rows),
        }
    )
    screen = screen_features(frame, "y", max_features=1)
    feature = _by_name(screen)["cat"]
    assert screen.selected == ["cat"]
    assert feature.test == "anova"
    assert feature.p_value is not None
    assert feature.p_value < 1e-6
    assert feature.effect_name == "eta_squared"
    assert feature.effect_size is not None
    assert feature.effect_size > 0.5


def test_chi_square_finds_categorical_association() -> None:
    rng = np.random.default_rng(4)
    rows = 500
    category = rng.choice(["a", "b"], size=rows)
    y = np.asarray(category == "a", dtype=int)
    flips = rng.random(size=rows) < 0.05
    y = np.where(flips, 1 - y, y)
    frame = pd.DataFrame({"cat": category, "y": y})
    screen = screen_features(frame, "y", max_features=1)
    feature = _by_name(screen)["cat"]
    assert screen.selected == ["cat"]
    assert feature.test == "chi_square"
    assert feature.p_value is not None
    assert feature.p_value < 1e-6
    assert feature.effect_name == "cramers_v"
    assert feature.effect_size is not None
    assert feature.effect_size > 0.5


def test_vif_flags_collinearity() -> None:
    rng = np.random.default_rng(5)
    rows = 200
    x1 = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": 2.0 * x1 + 0.001 * rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )
    vif = compute_vif(frame, ["x1", "x2", "x3"])
    assert vif["x1"] is not None and vif["x1"] > 10.0
    assert vif["x2"] is not None and vif["x2"] > 10.0
    assert vif["x3"] is not None and vif["x3"] < 5.0

    screen = screen_features(frame, "y")
    features = _by_name(screen)
    assert features["x1"].vif is not None and features["x1"].vif > 10.0
    assert features["x3"].vif is not None and features["x3"].vif < 5.0


def test_vif_single_feature_is_one() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    assert compute_vif(frame, ["x"]) == {"x": 1.0}


def test_benjamini_hochberg_known_values() -> None:
    adjusted = benjamini_hochberg([0.001, 0.01, 0.02, 0.5])
    assert adjusted == pytest.approx([0.004, 0.02, 0.0266666667, 0.5])


def test_benjamini_hochberg_preserves_none_and_order() -> None:
    adjusted = benjamini_hochberg([0.04, None, 0.01])
    assert adjusted[1] is None
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[2] == pytest.approx(0.02)


def test_benjamini_hochberg_is_monotone_in_sorted_order() -> None:
    pvalues = [0.5, 0.001, 0.2, 0.03, 0.9]
    adjusted = benjamini_hochberg(pvalues)
    ordered = sorted(
        (value, pvalue) for pvalue, value in zip(pvalues, adjusted)
    )
    values = [value for value, _ in ordered]
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_benjamini_hochberg_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], alpha=0.0)


def test_no_target_or_leakage_feature_is_screened() -> None:
    rng = np.random.default_rng(6)
    rows = 300
    y = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "id": [f"row-{index}" for index in range(rows)],
            "x": rng.normal(size=rows),
            "copy_of_y": y.copy(),
            "future_y": y * 3.0,
            "y": y,
        }
    )
    screen = screen_features(frame, "y", max_features=10)
    names = {feature.feature for feature in screen.features}
    assert names == {"x"}
    assert "y" not in screen.selected
    assert "copy_of_y" not in screen.selected
    assert "future_y" not in screen.selected
    assert screen.selected == ["x"]


def test_missing_values_are_handled_without_imputation() -> None:
    rng = np.random.default_rng(7)
    rows = 200
    x = rng.normal(size=rows)
    y = 2.0 * x + rng.normal(size=rows)
    x[rng.random(size=rows) < 0.1] = np.nan
    y[rng.random(size=rows) < 0.1] = np.nan
    frame = pd.DataFrame({"x": x, "y": y})
    before = frame.copy(deep=True)
    screen = screen_features(frame, "y")
    assert frame.equals(before)
    feature = _by_name(screen)["x"]
    assert feature.pearson is not None and feature.pearson > 0.8
    assert 0.0 < feature.missing_pct < 100.0


def test_datetime_feature_is_screened_when_time_aware() -> None:
    rows = 200
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "x": np.arange(rows, dtype=float),
            "y": np.arange(rows, dtype=float) * 2.0 + 1.0,
        }
    )
    screen = screen_features(
        frame, "y", time_column="when", time_aware=True
    )
    feature = _by_name(screen)["when"]
    assert feature.kind == "datetime"
    assert feature.test == "pearson"
    assert feature.pearson is not None
    assert feature.pearson > 0.99


def test_screen_reports_optional_backend_and_warnings() -> None:
    frame = pd.DataFrame({"x": [1.0, 0.0, 1.0, 0.0], "y": [1.0, 2.0, 3.0, 4.0]})
    screen = screen_features(frame, "y")
    assert screen.method in {"histogram", "sklearn"}
    assert screen.target == "y"
    assert screen.n_rows == 4
    assert screen.features


def test_screen_with_no_usable_features_warns() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    screen = screen_features(frame, "y")
    assert screen.features == []
    assert screen.selected == []
    assert "no_usable_features" in screen.warnings
