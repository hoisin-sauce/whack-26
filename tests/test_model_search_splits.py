"""Tests for repeated, stratified, group-aware, and chronological splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from whack26.model_search.splits import build_splits


def _continuous_frame(rows: int, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "noise": rng.normal(size=rows),
            "y": rng.normal(size=rows),
        }
    )


def test_three_repeated_holdouts_are_disjoint() -> None:
    frame = _continuous_frame(120)
    plan = build_splits(frame, "y")
    assert plan.kind == "random"
    assert len(plan.splits) == 3
    assert plan.seeds == [0, 1, 2]
    assert len(set(plan.seeds)) == 3
    assert plan.final_holdout is None
    assert plan.low_confidence is True

    all_rows = set(range(len(frame)))
    test_sets = []
    for split in plan.splits:
        train = set(split.train_index)
        test = set(split.test_index)
        assert train and test
        assert train.isdisjoint(test)
        assert train | test == all_rows
        assert split.train_rows == len(train)
        assert split.test_rows == len(test)
        test_sets.append(frozenset(test))
    assert len(set(test_sets)) == 3


def test_final_holdout_is_reserved_before_repeated_splits() -> None:
    frame = _continuous_frame(300)
    plan = build_splits(frame, "y")
    assert plan.final_holdout is not None
    assert plan.low_confidence is False

    holdout = plan.final_holdout
    holdout_rows = set(holdout.test_index)
    pool_rows = set(holdout.train_index)
    assert len(holdout_rows) == 60  # 20% of 300
    assert holdout_rows.isdisjoint(pool_rows)
    assert holdout_rows | pool_rows == set(range(len(frame)))

    for split in plan.splits:
        used = set(split.train_index) | set(split.test_index)
        assert used == pool_rows
        assert used.isdisjoint(holdout_rows)


def test_small_n_has_no_final_holdout_and_marks_low_confidence() -> None:
    frame = _continuous_frame(100)
    plan = build_splits(frame, "y", final_holdout_min_rows=200)
    assert plan.final_holdout is None
    assert plan.low_confidence is True
    assert any("final_holdout_skipped_small_n" in warning for warning in plan.warnings)


def test_stratified_splits_keep_class_balance() -> None:
    rng = np.random.default_rng(2)
    rows = 300
    y = np.concatenate([np.zeros(240, dtype=int), np.ones(60, dtype=int)])
    rng.shuffle(y)
    frame = pd.DataFrame({"x": rng.normal(size=rows), "y": y})
    plan = build_splits(frame, "y")
    assert plan.kind == "stratified"
    assert plan.target_kind == "binary"

    overall_rate = float(frame["y"].mean())
    assert plan.final_holdout is not None
    assert set(frame["y"].iloc[plan.final_holdout.test_index]) == {0, 1}
    for split in plan.splits:
        test_values = frame["y"].iloc[split.test_index]
        train_values = frame["y"].iloc[split.train_index]
        assert set(test_values) == {0, 1}
        assert set(train_values) == {0, 1}
        assert float(test_values.mean()) == pytest.approx(overall_rate, abs=0.08)


def test_group_splits_never_share_a_group() -> None:
    rng = np.random.default_rng(3)
    rows = 240
    groups = np.repeat(np.arange(24), 10)
    frame = pd.DataFrame(
        {
            "group": groups,
            "x": rng.normal(size=rows),
            "y": rng.integers(0, 2, size=rows),
        }
    )
    plan = build_splits(frame, "y", group_column="group", seed=7)
    assert plan.kind == "group"
    assert plan.seeds == [7, 8, 9]
    assert plan.final_holdout is not None

    holdout_groups = set(frame["group"].iloc[plan.final_holdout.test_index])
    pool_groups = set(frame["group"].iloc[plan.final_holdout.train_index])
    assert holdout_groups.isdisjoint(pool_groups)
    for split in plan.splits:
        train_groups = set(frame["group"].iloc[split.train_index])
        test_groups = set(frame["group"].iloc[split.test_index])
        assert train_groups.isdisjoint(test_groups)
        assert test_groups <= pool_groups


def test_chronological_splits_are_ordered_and_expanding() -> None:
    rows = 180
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2023-01-01", periods=rows, freq="D"),
            "x": rng.normal(size=rows),
            "y": np.arange(rows, dtype=float) + rng.normal(size=rows),
        }
    )
    plan = build_splits(frame, "y", time_column="when", time_aware=True)
    assert plan.kind == "chronological"
    assert plan.target_kind == "time_series"
    assert plan.final_holdout is None
    assert len(plan.splits) == 3

    previous_train: set[int] | None = None
    test_sets: list[frozenset[int]] = []
    for split in plan.splits:
        train_times = frame["when"].iloc[split.train_index]
        test_times = frame["when"].iloc[split.test_index]
        assert train_times.max() < test_times.min()
        assert split.train_rows + split.test_rows <= rows
        if previous_train is not None:
            assert previous_train <= set(split.train_index)
        previous_train = set(split.train_index)
        test_sets.append(frozenset(split.test_index))
    assert len(set(test_sets)) == 3


def test_chronological_final_holdout_is_the_latest_period() -> None:
    rows = 300
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "x": rng.normal(size=rows),
            "y": np.arange(rows, dtype=float),
        }
    )
    frame = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    plan = build_splits(frame, "y", time_column="when")
    assert plan.final_holdout is not None
    holdout_times = frame["when"].iloc[plan.final_holdout.test_index]
    pool_times = frame["when"].iloc[plan.final_holdout.train_index]
    assert holdout_times.min() > pool_times.max()
    for split in plan.splits:
        assert (
            frame["when"].iloc[split.train_index].max()
            < frame["when"].iloc[split.test_index].min()
        )


def test_missing_target_rows_are_excluded_without_imputation() -> None:
    rows = 120
    rng = np.random.default_rng(6)
    y = rng.normal(size=rows)
    y[:10] = np.nan
    frame = pd.DataFrame({"x": rng.normal(size=rows), "y": y})
    before = frame.copy(deep=True)
    plan = build_splits(frame, "y")
    assert frame.equals(before)  # no imputation or mutation
    used = set()
    for split in plan.splits:
        used.update(split.train_index)
        used.update(split.test_index)
    assert used == set(np.flatnonzero(~np.isnan(y)).tolist())
    assert any("rows_excluded" in warning for warning in plan.warnings)


def test_duplicate_seeds_are_rejected() -> None:
    frame = _continuous_frame(120)
    with pytest.raises(ValueError, match="distinct"):
        build_splits(frame, "y", seeds=[0, 0, 1])


def test_wrong_seed_count_is_rejected() -> None:
    frame = _continuous_frame(120)
    with pytest.raises(ValueError, match="expected 3 seeds"):
        build_splits(frame, "y", seeds=[0, 1])


def test_time_series_requires_a_time_column() -> None:
    frame = _continuous_frame(120)
    with pytest.raises(ValueError, match="time column"):
        build_splits(frame, "y", target_kind="time_series")


def test_group_column_with_one_group_falls_back_safely() -> None:
    frame = _continuous_frame(120)
    frame["group"] = "only"
    plan = build_splits(frame, "y", group_column="group")
    assert plan.kind == "group"
    assert any("group_split_fallback" in warning for warning in plan.warnings)
    for split in plan.splits:
        assert set(split.train_index).isdisjoint(split.test_index)
