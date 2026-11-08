"""Tests for target/feature typing in automated model search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whack26.model_search.data import (
    build_dataset_info,
    classify_feature,
    describe_target,
    infer_target_kind,
    usable_feature_frame,
)


@pytest.mark.parametrize(
    ("name", "series", "time_column", "expected"),
    [
        ("y", pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]), None, "continuous"),
        ("outcome", pd.Series([0, 1, 0, 1, 1]), None, "binary"),
        ("label", pd.Series(["yes", "no", "yes", "no", "no"]), None, "binary"),
        (
            "grade",
            pd.Series(["a", "b", "c", "a", "b", "c"]),
            None,
            "multiclass",
        ),
        ("y", pd.Series([0, 3, 1, 2, 3, 0, 3]), None, "count"),
        (
            "visits",
            pd.Series([0, 3, 1, 2, 3, 0, 3]),
            None,
            "count",
        ),
        ("y", pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]), "when", "time_series"),
    ],
)
def test_infer_target_kind(
    name: str,
    series: pd.Series,
    time_column: str | None,
    expected: str,
) -> None:
    assert infer_target_kind(series, name, time_column=time_column) == expected


def test_small_integer_codes_are_multiclass_not_count() -> None:
    series = pd.Series([0, 1, 2, 3, 4] * 20)
    assert infer_target_kind(series, "class_code") == "multiclass"


def test_count_like_name_beats_small_integer_code_rule() -> None:
    series = pd.Series([0, 1, 2, 3, 4] * 20)
    assert infer_target_kind(series, "event_count") == "count"


def test_describe_target_numeric_summary() -> None:
    frame = pd.DataFrame({"y": [1.0, 2.0, np.nan, 4.0]})
    info = describe_target(frame, "y")
    assert info.kind == "continuous"
    assert info.count == 3
    assert info.missing == 1
    assert info.missing_pct == 25.0
    assert info.stats is not None
    assert info.stats["mean"] == pytest.approx(7.0 / 3.0)
    assert info.stats["min"] == 1.0
    assert info.stats["max"] == 4.0


def test_describe_target_categorical_classes() -> None:
    frame = pd.DataFrame({"outcome": ["yes", "no", "yes", "no"]})
    info = describe_target(frame, "outcome")
    assert info.kind == "binary"
    assert set(info.classes) == {"yes", "no"}
    assert info.positive_class == "yes"
    assert info.top_values


def test_feature_typing_and_usable_columns() -> None:
    rng = np.random.default_rng(0)
    rows = 60
    frame = pd.DataFrame(
        {
            "amount": rng.normal(size=rows),
            "region": rng.choice(["north", "south"], size=rows),
            "when": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "row_id": [f"row-{index}" for index in range(rows)],
            "future_value": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )
    info = build_dataset_info(frame, "y", time_column="when")
    kinds = {feature.name: feature.kind for feature in info.features}
    assert kinds["amount"] == "numeric"
    assert kinds["region"] == "categorical"
    assert kinds["when"] == "datetime"
    assert kinds["row_id"] == "id"
    assert kinds["future_value"] == "leakage_risk"
    assert info.usable_features == ["amount", "region"]
    assert info.excluded["when"] == "time"
    assert info.excluded["row_id"] == "id"
    assert info.excluded["future_value"] == "leakage_name"
    assert info.excluded["y"] == "target"
    assert info.time_column == "when"
    assert info.time_aware is False


def test_time_column_is_usable_only_when_time_aware() -> None:
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=10, freq="D"),
            "x": np.arange(10, dtype=float),
            "y": np.arange(10, dtype=float),
        }
    )
    without = build_dataset_info(frame, "y", time_column="when")
    assert "when" not in without.usable_features
    with_time = build_dataset_info(
        frame, "y", time_column="when", time_aware=True
    )
    assert "when" in with_time.usable_features


def test_target_copy_is_leakage_risk() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    frame = pd.DataFrame({"x": rng.normal(size=50), "mystery": y * 2.0 + 1.0, "y": y})
    info = build_dataset_info(frame, "y")
    mystery = next(feature for feature in info.features if feature.name == "mystery")
    assert mystery.kind == "leakage_risk"
    assert mystery.reason == "target_copy"
    assert "mystery" not in info.usable_features


def test_missingness_is_recorded() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, np.nan],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    info = build_dataset_info(frame, "y")
    assert info.missingness["x"] == 50.0
    feature = next(feature for feature in info.features if feature.name == "x")
    assert feature.missing_pct == 50.0


def test_constant_and_all_missing_columns_are_unusable() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "empty": [np.nan, np.nan, np.nan, np.nan],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    info = build_dataset_info(frame, "y")
    assert "constant" not in info.usable_features
    assert "empty" not in info.usable_features
    assert info.excluded["constant"] == "constant"
    assert info.excluded["empty"] == "all_missing"


def test_group_column_is_excluded_and_marks_group_aware() -> None:
    frame = pd.DataFrame(
        {
            "site": [1, 1, 2, 2, 3, 3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    info = build_dataset_info(frame, "y", group_column="site")
    assert info.group_aware is True
    assert "site" not in info.usable_features
    assert info.excluded["site"] == "group_column"


def test_usable_feature_frame_is_a_copy() -> None:
    frame = pd.DataFrame({"x": [1.0, 0.0, 1.0], "y": [1.0, 2.0, 3.0]})
    info = build_dataset_info(frame, "y")
    features = usable_feature_frame(frame, info)
    assert list(features.columns) == ["x"]
    features.loc[:, "x"] = 0.0
    assert frame["x"].tolist() == [1.0, 0.0, 1.0]


def test_classify_feature_rejects_target_name() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    info = classify_feature(
        "y",
        series,
        rows=3,
        target_name="y",
        target_series=series,
    )
    assert info.usable is False
    assert info.reason == "target"
