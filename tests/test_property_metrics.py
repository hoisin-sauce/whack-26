"""Property-based tests for validation metrics and model-search splits.

Kept intentionally small and fast: every property runs at most 30 examples
with Hypothesis' wall-clock deadline disabled.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from stats_engine import brier_score, log_loss, r2_score, roc_auc_score
from whack26.model_search.splits import build_splits

PROPERTY_SETTINGS = settings(max_examples=30, deadline=None)


def _finite_floats(low: float, high: float) -> st.SearchStrategy[float]:
    return st.floats(
        min_value=low,
        max_value=high,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@st.composite
def _spread_array(
    draw: st.DrawFn, size: int | None = None
) -> np.ndarray:
    """Finite 1D array whose values are not all (numerically) equal."""
    if size is None:
        size = draw(st.integers(min_value=2, max_value=25))
    values = draw(
        st.lists(
            _finite_floats(-50.0, 50.0),
            min_size=size,
            max_size=size,
        )
    )
    array = np.asarray(values, dtype=float)
    if float(np.ptp(array)) < 1e-3:
        array = array.copy()
        array[0] += 1.0
    return array


@st.composite
def _binary_probabilities_and_labels(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities in [0, 1] paired with labels containing both classes."""
    size = draw(st.integers(min_value=2, max_value=40))
    probabilities = draw(
        npst.arrays(
            dtype=np.float64,
            shape=size,
            elements=_finite_floats(0.0, 1.0),
        )
    )
    extras = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=size - 2,
            max_size=size - 2,
        )
    )
    labels = np.asarray([0, 1, *extras], dtype=int)
    return probabilities, labels


@st.composite
def _binary_scores_and_labels(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Arbitrary finite scores paired with labels containing both classes."""
    size = draw(st.integers(min_value=2, max_value=40))
    scores = draw(
        npst.arrays(
            dtype=np.float64,
            shape=size,
            elements=_finite_floats(-1e6, 1e6),
        )
    )
    extras = draw(
        st.lists(
            st.integers(min_value=0, max_value=1),
            min_size=size - 2,
            max_size=size - 2,
        )
    )
    labels = np.asarray([0, 1, *extras], dtype=int)
    return scores, labels


def _continuous_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )


@PROPERTY_SETTINGS
@given(actuals=_spread_array())
def test_r2_score_perfect_predictions_equal_one(actuals: np.ndarray) -> None:
    assert r2_score(actuals, actuals) == 1.0


@PROPERTY_SETTINGS
@given(
    data=st.data(),
    shift=_finite_floats(-10.0, 10.0),
)
def test_r2_score_invariant_to_common_shift(
    data: st.DataObject, shift: float
) -> None:
    actuals = data.draw(_spread_array())
    predictions = data.draw(_spread_array(size=actuals.size))

    baseline = r2_score(predictions, actuals)
    shifted = r2_score(predictions + shift, actuals + shift)

    assert math.isclose(baseline, shifted, rel_tol=1e-9, abs_tol=1e-9)


@PROPERTY_SETTINGS
@given(_binary_probabilities_and_labels())
def test_brier_score_stays_in_unit_interval(
    payload: tuple[np.ndarray, np.ndarray],
) -> None:
    probabilities, labels = payload
    score = brier_score(probabilities, labels)

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0


@PROPERTY_SETTINGS
@given(_binary_probabilities_and_labels())
def test_log_loss_finite_and_non_negative(
    payload: tuple[np.ndarray, np.ndarray],
) -> None:
    probabilities, labels = payload
    score = log_loss(probabilities, labels)

    assert math.isfinite(score)
    assert score >= 0.0


@PROPERTY_SETTINGS
@given(_binary_scores_and_labels())
def test_roc_auc_score_stays_in_unit_interval(
    payload: tuple[np.ndarray, np.ndarray],
) -> None:
    scores, labels = payload
    auc = roc_auc_score(scores, labels)

    assert math.isfinite(auc)
    assert 0.0 <= auc <= 1.0


@PROPERTY_SETTINGS
@given(
    rows=st.integers(min_value=3, max_value=120),
    seed=st.integers(min_value=0, max_value=50_000),
    n_splits=st.integers(min_value=2, max_value=5),
    test_fraction=_finite_floats(0.10, 0.90),
)
def test_build_splits_partitions_rows_and_is_deterministic(
    rows: int, seed: int, n_splits: int, test_fraction: float
) -> None:
    frame = _continuous_frame(rows, seed)
    kwargs = {
        "target_kind": "continuous",
        "n_splits": n_splits,
        "test_fraction": test_fraction,
        "seed": seed,
    }
    first = build_splits(frame, "y", **kwargs)
    second = build_splits(frame, "y", **kwargs)

    all_rows = set(range(rows))
    assert len(first.splits) == n_splits
    for split, repeated in zip(first.splits, second.splits):
        train = set(split.train_index)
        test = set(split.test_index)

        assert train.isdisjoint(test)
        assert train | test == all_rows
        assert split.train_rows == len(split.train_index)
        assert split.test_rows == len(split.test_index)
        assert (split.train_index, split.test_index) == (
            repeated.train_index,
            repeated.test_index,
        )


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=50_000))
def test_build_splits_different_seeds_can_change_partitions(seed: int) -> None:
    frame = _continuous_frame(80, seed)
    first = build_splits(frame, "y", target_kind="continuous", seed=seed)
    other = build_splits(frame, "y", target_kind="continuous", seed=seed + 1_000)

    first_partitions = [(s.train_index, s.test_index) for s in first.splits]
    other_partitions = [(s.train_index, s.test_index) for s in other.splits]

    assert first_partitions != other_partitions


@PROPERTY_SETTINGS
@given(
    rows=st.integers(min_value=10, max_value=180),
    seed=st.integers(min_value=0, max_value=50_000),
)
def test_model_search_fold_rows_sum_to_frame_length(
    rows: int, seed: int
) -> None:
    frame = _continuous_frame(rows, seed)
    plan = build_splits(
        frame,
        "y",
        target_kind="continuous",
        n_splits=3,
        test_fraction=0.30,
        seed=seed,
    )

    # Rows stay below final_holdout_min_rows (200), so every fold sees the
    # whole frame exactly once.
    assert plan.final_holdout is None
    for split in plan.splits:
        assert split.train_rows == len(split.train_index)
        assert split.test_rows == len(split.test_index)
        assert split.train_rows + split.test_rows == len(frame)
