from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from whack26.validation.adapter import (
    AdapterSkip,
    align_predictions,
    build_feature_matrix,
    infer_periods_per_year,
)


def make_frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "x": x,
            "group": (["a", "b", "c"] * rows)[:rows],
            "y": 2.0 * x + rng.normal(0.0, 0.1, size=rows),
        }
    )


def test_numeric_and_categorical_round_trip() -> None:
    frame = make_frame()
    matrix = build_feature_matrix(
        frame,
        target="y",
        required_columns=["x", "group"],
        max_categories=5,
    )

    assert matrix.rows == 60
    assert matrix.feature_columns == ["x", "group"]
    assert matrix.X.shape == (60, 4)  # x + three one-hot group columns
    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert list(reconstructed.columns) == ["x", "group", "y"]
    np.testing.assert_allclose(reconstructed["x"], frame["x"])
    assert reconstructed["group"].tolist() == frame["group"].tolist()
    np.testing.assert_allclose(reconstructed["y"], frame["y"])


def test_prediction_frame_excludes_target() -> None:
    frame = make_frame()
    matrix = build_feature_matrix(
        frame,
        target="y",
        required_columns=["x"],
    )

    prediction_frame = matrix.frame_from_matrix(matrix.X)

    assert list(prediction_frame.columns) == ["x"]
    assert "y" not in prediction_frame.columns


def test_missing_rows_are_dropped_and_reported() -> None:
    frame = make_frame()
    frame.loc[0, "x"] = np.nan
    frame.loc[1, "y"] = np.nan

    matrix = build_feature_matrix(
        frame,
        target="y",
        required_columns=["x"],
    )

    assert matrix.rows == 58
    assert matrix.dropped_rows == 2
    assert 0 not in matrix.retained_index
    assert 1 not in matrix.retained_index


def test_binary_target_is_encoded_and_constant_is_skipped() -> None:
    frame = make_frame()
    frame["binary"] = (frame["y"] > 0).astype(float)
    frame["constant"] = 1.0

    matrix = build_feature_matrix(frame, target="binary", required_columns=["x"])
    assert matrix.target_kind == "binary"
    assert matrix.target_labels == {0: 0.0, 1: 1.0}
    assert set(matrix.y.tolist()) == {0.0, 1.0}

    with pytest.raises(AdapterSkip) as constant:
        build_feature_matrix(frame, target="constant", required_columns=["x"])
    assert constant.value.reason == "target_contract"


def test_small_integer_category_target_is_skipped() -> None:
    frame = make_frame()
    frame["label"] = np.arange(len(frame)) % 3

    with pytest.raises(AdapterSkip) as error:
        build_feature_matrix(frame, target="label", required_columns=["x"])

    assert error.value.reason == "target_contract"


def test_too_few_complete_rows_is_skipped() -> None:
    frame = make_frame(rows=30)
    frame.loc[:19, "x"] = np.nan

    with pytest.raises(AdapterSkip) as error:
        build_feature_matrix(frame, target="y", required_columns=["x"])

    assert error.value.reason == "insufficient_rows"


def test_short_target_is_skipped_by_contract() -> None:
    frame = make_frame(rows=10)

    with pytest.raises(AdapterSkip) as error:
        build_feature_matrix(frame, target="y", required_columns=["x"])

    assert error.value.reason == "target_contract"


def test_periods_inference_from_daily_time_column() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "y": np.linspace(0.0, 1.0, 30) + np.sin(np.arange(30)),
        }
    )

    periods, source = infer_periods_per_year(frame, "date")

    assert periods == 252.0
    assert source == "time_column"


def test_periods_default_without_time_column() -> None:
    frame = make_frame()
    assert infer_periods_per_year(frame, None) == (1.0, "default")


def test_align_predictions_by_row_index() -> None:
    predictions = pd.DataFrame(
        {"row_index": [2, 0, 1], "prediction": [20.0, 0.0, 10.0]}
    )

    aligned = align_predictions(predictions, [0, 1, 2])

    np.testing.assert_allclose(aligned, [0.0, 10.0, 20.0])


def test_period_override_wins() -> None:
    frame = make_frame()
    matrix = build_feature_matrix(
        frame,
        target="y",
        required_columns=["x"],
        period_override=12.0,
    )
    assert matrix.periods_per_year == 12.0
    assert matrix.periods_per_year_source == "config"
    assert math.isfinite(matrix.periods_per_year)
