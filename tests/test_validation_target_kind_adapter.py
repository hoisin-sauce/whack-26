"""Target-kind aware adapter coverage.

Synthetic numpy/pandas data only. These tests exercise the public adapter API:
``infer_target_kind``, ``encode_target``, the ``FeatureMatrix`` target fields,
and ``build_feature_matrix(target_kind=...)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whack26.validation.adapter import (
    AdapterSkip,
    build_feature_matrix,
    encode_target,
    infer_target_kind,
)


def _frame(rows: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2.0 * x + 0.25 * z + rng.normal(0.0, 0.1, size=rows),
        }
    )


def _build(
    frame: pd.DataFrame,
    target: str,
    kind: str | None = None,
):
    kwargs = {} if kind is None else {"target_kind": kind}
    return build_feature_matrix(
        frame,
        target=target,
        required_columns=["x", "z"],
        **kwargs,
    )


def _reason(frame: pd.DataFrame, target: str, kind: str | None) -> str:
    with pytest.raises(AdapterSkip) as error:
        _build(frame, target, kind)
    return error.value.reason


# ---------------------------------------------------------------------------
# infer_target_kind
# ---------------------------------------------------------------------------


def test_infer_bool_is_binary() -> None:
    series = pd.Series([True, False] * 30, name="won")
    assert infer_target_kind(series) == "binary"


def test_infer_two_labels_is_binary() -> None:
    assert infer_target_kind(pd.Series(["down", "up"] * 30, name="outcome")) == (
        "binary"
    )
    assert infer_target_kind(pd.Series([0.0, 1.0] * 30, name="win")) == "binary"


def test_infer_nonnumeric_three_to_ten_levels_is_multiclass() -> None:
    three = pd.Series(["alpha", "beta", "gamma"] * 20, name="class")
    ten = pd.Series([str(value) for value in range(10)] * 6, name="label")
    assert infer_target_kind(three) == "multiclass"
    assert infer_target_kind(ten) == "multiclass"


def test_infer_nonnumeric_high_cardinality_is_continuous() -> None:
    series = pd.Series([f"item-{value}" for value in range(60)], name="label")
    assert infer_target_kind(series) == "continuous"


def test_infer_integral_with_many_levels_is_count() -> None:
    series = pd.Series(np.arange(60) % 12, name="value")
    assert infer_target_kind(series) == "count"


def test_infer_integral_small_levels_with_count_name_is_count() -> None:
    assert infer_target_kind(pd.Series(np.arange(60) % 4, name="visits")) == "count"
    assert infer_target_kind(pd.Series(np.arange(60) % 4, name="event_count")) == (
        "count"
    )


def test_infer_small_integer_codes_without_count_name_is_continuous() -> None:
    series = pd.Series(np.arange(60) % 4, name="score")
    assert infer_target_kind(series) == "continuous"


def test_infer_float_measure_is_continuous() -> None:
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(size=60), name="return")
    assert infer_target_kind(series) == "continuous"


def test_infer_empty_or_all_missing_is_continuous() -> None:
    assert infer_target_kind(pd.Series([], dtype=float, name="y")) == "continuous"
    assert infer_target_kind(pd.Series([np.nan] * 30, name="y")) == "continuous"


# ---------------------------------------------------------------------------
# encode_target
# ---------------------------------------------------------------------------


def test_encode_binary_sorted_round_trip() -> None:
    series = pd.Series(["b", "a", "b", "a"], name="outcome")

    codes, labels = encode_target(series, "binary")

    assert labels == {0: "a", 1: "b"}
    np.testing.assert_array_equal(codes, [1.0, 0.0, 1.0, 0.0])


def test_encode_multiclass_sorted_round_trip() -> None:
    series = pd.Series(["beta", "alpha", "gamma", "alpha"], name="class")

    codes, labels = encode_target(series, "multiclass")

    assert labels == {0: "alpha", 1: "beta", 2: "gamma"}
    np.testing.assert_array_equal(codes, [1.0, 0.0, 2.0, 0.0])


def test_encode_count_preserves_values() -> None:
    codes, labels = encode_target(pd.Series([2, 0, 4, 1], name="events"), "count")

    assert labels == {}
    np.testing.assert_array_equal(codes, [2.0, 0.0, 4.0, 1.0])


def test_encode_continuous_preserves_values() -> None:
    series = pd.Series([0.5, -1.25, 3.0], name="y")

    codes, labels = encode_target(series, "continuous")

    assert labels == {}
    np.testing.assert_allclose(codes, [0.5, -1.25, 3.0])


@pytest.mark.parametrize("kind", ["continuous", "binary", "multiclass", "count"])
def test_encode_constant_target_is_skipped(kind: str) -> None:
    with pytest.raises(AdapterSkip) as error:
        encode_target(pd.Series([1, 1, 1, 1], name="target"), kind)

    assert error.value.reason == "target_contract"


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, -1],
        [0, 1, 1.5],
        [0, 1, np.inf],
        [0, 1, np.nan],
    ],
)
def test_encode_invalid_count_is_rejected(values: list[float]) -> None:
    with pytest.raises(AdapterSkip) as error:
        encode_target(pd.Series(values, name="events"), "count")

    assert error.value.reason == "target_contract"


def test_encode_unknown_kind_is_rejected() -> None:
    with pytest.raises(AdapterSkip) as error:
        encode_target(pd.Series([0, 1], name="target"), "ordinal")

    assert error.value.reason == "target_contract"


def test_encode_legacy_categorical_kind_falls_back_to_inference() -> None:
    series = pd.Series(["b", "a", "b", "a"], name="outcome")

    codes, labels = encode_target(series, "categorical")

    assert labels == {0: "a", 1: "b"}
    np.testing.assert_array_equal(codes, [1.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# build_feature_matrix: binary
# ---------------------------------------------------------------------------


def test_binary_string_labels_round_trip() -> None:
    frame = _frame()
    frame["outcome"] = np.where(frame["x"] > 0.0, "up", "down")

    matrix = _build(frame, "outcome", "binary")

    assert matrix.target_kind == "binary"
    assert matrix.target_labels == {0: "down", 1: "up"}
    np.testing.assert_array_equal(
        matrix.y, (frame["outcome"] == "up").to_numpy(dtype=float)
    )

    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert list(reconstructed.columns) == ["x", "z", "outcome"]
    assert reconstructed["outcome"].tolist() == frame["outcome"].tolist()

    predictions_only = matrix.frame_from_matrix(matrix.X)
    assert list(predictions_only.columns) == ["x", "z"]


def test_bool_labels_round_trip() -> None:
    frame = _frame()
    frame["flag"] = frame["x"] > 0.0

    matrix = _build(frame, "flag", "binary")

    assert matrix.target_kind == "binary"
    assert matrix.target_labels == {0: False, 1: True}
    np.testing.assert_array_equal(matrix.y, frame["flag"].to_numpy(dtype=float))

    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["flag"].tolist() == frame["flag"].tolist()


def test_zero_one_integer_labels_round_trip() -> None:
    frame = _frame()
    frame["win"] = (frame["x"] > 0.0).astype(int)

    matrix = _build(frame, "win", "binary")

    assert matrix.target_labels == {0: 0, 1: 1}
    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["win"].tolist() == frame["win"].tolist()


def test_binary_inferred_without_explicit_kind() -> None:
    frame = _frame()
    frame["win"] = (frame["x"] > 0.0).astype(int)

    matrix = _build(frame, "win")

    assert matrix.target_kind == "binary"
    assert matrix.target_labels == {0: 0, 1: 1}


def test_binary_with_three_classes_is_rejected() -> None:
    frame = _frame()
    frame["class"] = np.arange(len(frame)) % 3

    assert _reason(frame, "class", "binary") == "target_contract"


# ---------------------------------------------------------------------------
# build_feature_matrix: multiclass
# ---------------------------------------------------------------------------


def test_multiclass_string_labels_round_trip() -> None:
    frame = _frame()
    frame["kind"] = np.array(["alpha", "beta", "gamma"])[
        np.arange(len(frame)) % 3
    ]

    matrix = _build(frame, "kind", "multiclass")

    assert matrix.target_kind == "multiclass"
    assert matrix.target_labels == {0: "alpha", 1: "beta", 2: "gamma"}
    np.testing.assert_array_equal(matrix.y, np.arange(len(frame)) % 3)

    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["kind"].tolist() == frame["kind"].tolist()


def test_multiclass_inferred_without_explicit_kind() -> None:
    frame = _frame()
    frame["kind"] = np.array(["alpha", "beta", "gamma"])[
        np.arange(len(frame)) % 3
    ]

    matrix = _build(frame, "kind")

    assert matrix.target_kind == "multiclass"
    assert matrix.target_labels == {0: "alpha", 1: "beta", 2: "gamma"}


def test_explicit_multiclass_overrides_continuous_inference() -> None:
    frame = _frame()
    frame["code"] = np.arange(len(frame)) % 3

    assert infer_target_kind(frame["code"]) == "continuous"

    matrix = _build(frame, "code", "multiclass")

    assert matrix.target_kind == "multiclass"
    assert matrix.target_labels == {0: 0, 1: 1, 2: 2}
    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["code"].tolist() == frame["code"].tolist()


def test_multiclass_labels_survive_dropped_rows() -> None:
    frame = _frame()
    frame["kind"] = np.array(["alpha", "beta", "gamma"])[
        np.arange(len(frame)) % 3
    ]
    frame.loc[0, "x"] = np.nan

    matrix = _build(frame, "kind", "multiclass")

    assert matrix.dropped_rows == 1
    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["kind"].tolist() == frame["kind"].iloc[1:].tolist()


def test_frame_from_matrix_rejects_fractional_class_codes() -> None:
    frame = _frame()
    frame["win"] = (frame["x"] > 0.0).astype(int)
    matrix = _build(frame, "win", "binary")

    with pytest.raises(ValueError):
        matrix.frame_from_matrix(matrix.X, matrix.y + 0.25)


def test_frame_from_matrix_rejects_unknown_class_codes() -> None:
    frame = _frame()
    frame["win"] = (frame["x"] > 0.0).astype(int)
    matrix = _build(frame, "win", "binary")

    with pytest.raises(ValueError):
        matrix.frame_from_matrix(matrix.X, matrix.y + 2.0)


# ---------------------------------------------------------------------------
# build_feature_matrix: count
# ---------------------------------------------------------------------------


def test_count_integers_round_trip() -> None:
    rng = np.random.default_rng(21)
    frame = _frame()
    frame["events"] = rng.integers(0, 25, size=len(frame))

    matrix = _build(frame, "events", "count")

    assert matrix.target_kind == "count"
    assert matrix.target_labels == {}
    np.testing.assert_allclose(
        matrix.y, frame["events"].to_numpy(dtype=float)
    )

    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    assert reconstructed["events"].tolist() == frame["events"].tolist()


def test_count_inferred_from_many_levels() -> None:
    frame = _frame()
    frame["value"] = np.arange(len(frame)) % 12

    matrix = _build(frame, "value")

    assert matrix.target_kind == "count"
    assert matrix.target_labels == {}


def test_count_inferred_from_count_like_name() -> None:
    frame = _frame()
    frame["visits"] = np.arange(len(frame)) % 4

    matrix = _build(frame, "visits")

    assert matrix.target_kind == "count"
    assert matrix.target_labels == {}


def test_negative_count_is_rejected() -> None:
    frame = _frame()
    frame["delta"] = np.where(np.arange(len(frame)) % 2 == 0, -1, 2).astype(int)

    assert _reason(frame, "delta", "count") == "target_contract"


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_non_finite_count_is_rejected(bad: float) -> None:
    frame = _frame()
    frame["events"] = (np.arange(len(frame)) % 12).astype(float)
    frame.loc[3, "events"] = bad

    assert _reason(frame, "events", "count") == "target_contract"


def test_non_integer_count_is_rejected() -> None:
    frame = _frame()
    frame["events"] = (np.arange(len(frame)) % 12).astype(float) + 0.5

    assert _reason(frame, "events", "count") == "target_contract"


# ---------------------------------------------------------------------------
# continuous contract remains unchanged
# ---------------------------------------------------------------------------


def test_continuous_path_is_unchanged() -> None:
    frame = _frame()

    matrix = _build(frame, "y")

    assert matrix.target_kind == "continuous"
    assert matrix.target_labels == {}
    np.testing.assert_allclose(matrix.y, frame["y"].to_numpy(dtype=float))

    reconstructed = matrix.frame_from_matrix(matrix.X, matrix.y)
    np.testing.assert_allclose(reconstructed["y"], frame["y"])


def test_small_integer_category_still_skipped_by_default() -> None:
    frame = _frame()
    frame["label"] = np.arange(len(frame)) % 3

    assert _reason(frame, "label", None) == "target_contract"


def test_non_finite_continuous_target_is_rejected() -> None:
    frame = _frame()
    frame.loc[0, "y"] = np.inf

    assert _reason(frame, "y", "continuous") == "target_contract"


def test_time_series_kind_uses_continuous_contract() -> None:
    frame = _frame()

    matrix = _build(frame, "y", "time_series")

    assert matrix.target_kind == "continuous"
    assert matrix.target_labels == {}


# ---------------------------------------------------------------------------
# contract rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [None, "continuous", "binary", "multiclass", "count"])
def test_constant_target_is_skipped(kind: str | None) -> None:
    frame = _frame()
    frame["constant"] = 1.0

    assert _reason(frame, "constant", kind) == "target_contract"


def test_constant_string_target_is_skipped() -> None:
    frame = _frame()
    frame["constant"] = "same"

    assert _reason(frame, "constant", None) == "target_contract"
    assert _reason(frame, "constant", "binary") == "target_contract"


@pytest.mark.parametrize("kind", ["ordinal", "unknown"])
def test_unknown_target_kind_is_rejected(kind: str) -> None:
    frame = _frame()

    assert _reason(frame, "y", kind) == "target_contract"


def test_legacy_numeric_kind_falls_back_to_inference() -> None:
    frame = _frame()

    matrix = _build(frame, "y", "numeric")

    assert matrix.target_kind == "continuous"
    assert matrix.target_labels == {}


def test_legacy_categorical_kind_infers_binary() -> None:
    frame = _frame()
    frame["outcome"] = np.where(frame["x"] > 0.0, "up", "down")

    matrix = _build(frame, "outcome", "categorical")

    assert matrix.target_kind == "binary"
    assert matrix.target_labels == {0: "down", 1: "up"}


def test_legacy_categorical_kind_infers_multiclass() -> None:
    frame = _frame()
    frame["kind"] = np.array(["alpha", "beta", "gamma"])[
        np.arange(len(frame)) % 3
    ]

    matrix = _build(frame, "kind", "categorical")

    assert matrix.target_kind == "multiclass"
    assert matrix.target_labels == {0: "alpha", 1: "beta", 2: "gamma"}


def test_short_classification_target_is_skipped() -> None:
    frame = _frame(rows=10)
    frame["outcome"] = ["down", "up"] * 5

    assert _reason(frame, "outcome", "binary") == "target_contract"
