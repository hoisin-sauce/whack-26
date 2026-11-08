"""Splitting strategies for automated regression model search.

Default evaluation uses three repeated holdouts with distinct seeds. The
splitter only returns positional row indices; it never copies, imputes, or
transforms the frame, so every preprocessing step remains the fitter's
responsibility and must be fitted inside each training fold.

Strategies:

* random 70/30 holdouts for continuous and count targets;
* stratified holdouts for binary and multiclass targets;
* group-aware holdouts when a group/entity column is supplied, so no group
  appears in both train and test;
* chronological expanding-window holdouts for time-series or time-aware
  models, where every training row precedes every test row.

For ``n >= 200`` a final holdout is reserved *before* the repeated splits, so
it can never appear in a training or evaluation fold. Small datasets report
``low_confidence=True`` instead.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .data import infer_target_kind
from .types import TargetKind

SplitKind = Literal["random", "stratified", "group", "chronological"]

_MIN_SPLIT_ROWS = 2


class DataSplit(BaseModel):
    """One train/test partition, expressed as positional row indices."""

    fold: int
    seed: int
    kind: SplitKind
    train_index: list[int] = Field(default_factory=list)
    test_index: list[int] = Field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0


class SplitPlan(BaseModel):
    """Repeated holdouts plus the optional one-shot final holdout."""

    target: str
    target_kind: TargetKind
    kind: SplitKind
    seeds: list[int] = Field(default_factory=list)
    splits: list[DataSplit] = Field(default_factory=list)
    final_holdout: DataSplit | None = None
    low_confidence: bool = False
    warnings: list[str] = Field(default_factory=list)


def _as_positions(indices: np.ndarray) -> list[int]:
    return [int(value) for value in indices]


def _clamp_test_size(size: int, test_fraction: float) -> int:
    if size < _MIN_SPLIT_ROWS:
        return 0
    test = int(round(size * test_fraction))
    return min(max(test, 1), size - 1)


def _random_partition(
    indices: np.ndarray,
    *,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    test_size = _clamp_test_size(len(shuffled), test_fraction)
    test = np.sort(shuffled[:test_size])
    train = np.sort(shuffled[test_size:])
    return train, test


def _stratified_partition(
    indices: np.ndarray,
    labels: pd.Series,
    *,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    codes, _ = pd.factorize(labels.iloc[indices], sort=False)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for code in range(int(codes.max()) + 1 if len(codes) else 0):
        members = indices[codes == code]
        if members.size == 0:
            continue
        shuffled = np.array(members, copy=True)
        rng.shuffle(shuffled)
        test_size = _clamp_test_size(members.size, test_fraction)
        test_parts.append(shuffled[:test_size])
        train_parts.append(shuffled[test_size:])
    train = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    return train, test


def _group_keys(values: Sequence[object]) -> np.ndarray:
    """Build hashable group tokens, treating each missing value as its own group."""
    keys: list[str] = []
    for position, value in enumerate(values):
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            keys.append(f"__missing__:{position}")
        else:
            # ``repr`` keeps 1 and "1" in separate groups and is stable for
            # mixed-type object columns.
            keys.append(f"{type(value).__name__}:{value!r}")
    return np.asarray(keys, dtype=object)


def _group_partition(
    indices: np.ndarray,
    group_keys: np.ndarray,
    *,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split whole groups, returning ``(train, test, grouped)``."""
    keys = group_keys[indices]
    unique = list(pd.unique(keys))
    if len(unique) < 2:
        train, test = _random_partition(
            indices, test_fraction=test_fraction, rng=rng
        )
        return train, test, False

    order = sorted(range(len(unique)), key=lambda position: str(unique[position]))
    unique = [unique[position] for position in order]
    rng.shuffle(unique)

    sizes = Counter(keys)
    target_rows = max(1, int(round(len(indices) * test_fraction)))
    test_groups: list[object] = []
    test_rows = 0
    # Always leave at least one group for training.
    for key in unique[:-1]:
        test_groups.append(key)
        test_rows += sizes[key]
        if test_rows >= target_rows:
            break
    test_set = set(test_groups)
    test_mask = np.asarray([key in test_set for key in keys], dtype=bool)
    train = np.sort(indices[~test_mask])
    test = np.sort(indices[test_mask])
    if train.size == 0 or test.size == 0:
        train, test = _random_partition(
            indices, test_fraction=test_fraction, rng=rng
        )
        return train, test, False
    return train, test, True


def _time_values(frame: pd.DataFrame, time_column: str) -> pd.Series:
    series = frame[time_column]
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce", utc=True)


def _chronological_partitions(
    indices: np.ndarray,
    times: pd.Series,
    *,
    n_splits: int,
    test_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window chronological splits over time-ordered rows.

    The last fold is the classic 70/30 cut. Earlier folds move the test window
    back and shrink the training window so every fold evaluates a distinct,
    later period while keeping ``train < test`` in time.
    """
    ordered = np.asarray(
        sorted(indices, key=lambda position: (times.iloc[position], position)),
        dtype=int,
    )
    total = len(ordered)
    if total < _MIN_SPLIT_ROWS:
        return []
    test_size = max(1, int(round(total * test_fraction)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if n_splits <= 1:
        test_start = max(1, total - test_size)
        folds.append((ordered[:test_start], ordered[test_start:]))
        return folds

    # Test windows end at 70%, 85%, ..., 100% of the ordered pool, and each
    # window is `test_fraction` long. Fold 0 therefore still has 70% of the
    # pool available as future test data, while its training set is shorter.
    step = test_fraction / float(n_splits - 1)
    for fold in range(n_splits):
        end_fraction = 1.0 - float(n_splits - 1 - fold) * step
        start_fraction = end_fraction - test_fraction
        start = int(round(total * start_fraction))
        end = int(round(total * end_fraction))
        start = min(max(start, 1), total - 1)
        end = min(max(end, start + 1), total)
        train = ordered[:start]
        test = ordered[start:end]
        if train.size and test.size:
            folds.append((train, test))
    return folds


def _final_holdout_indices(
    eligible: np.ndarray,
    *,
    fraction: float,
    target_kind: TargetKind,
    labels: pd.Series,
    group_keys: np.ndarray | None,
    chronological: bool,
    times: pd.Series | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the final holdout from ``eligible`` before any fold is built."""
    rng = np.random.default_rng(seed)
    size = int(round(len(eligible) * fraction))
    size = min(max(size, 1), max(len(eligible) - 1, 1))

    if chronological and times is not None:
        ordered = np.asarray(
            sorted(
                eligible,
                key=lambda position: (times.iloc[position], position),
            ),
            dtype=int,
        )
        holdout = np.sort(ordered[-size:])
        pool = np.sort(ordered[:-size])
        return pool, holdout

    if group_keys is not None:
        pool, holdout, grouped = _group_partition(
            eligible,
            group_keys,
            test_fraction=fraction,
            rng=rng,
        )
        if grouped:
            return pool, holdout

    if target_kind in ("binary", "multiclass"):
        pool, holdout = _stratified_partition(
            eligible,
            labels,
            test_fraction=fraction,
            rng=rng,
        )
        if holdout.size and pool.size:
            return pool, holdout

    _, holdout = _random_partition(
        eligible, test_fraction=fraction, rng=rng
    )
    holdout_set = set(int(value) for value in holdout)
    pool = np.asarray(
        [int(value) for value in eligible if int(value) not in holdout_set],
        dtype=int,
    )
    return pool, np.sort(holdout)


def build_splits(
    frame: pd.DataFrame,
    target: str,
    *,
    target_kind: TargetKind | None = None,
    n_splits: int = 3,
    test_fraction: float = 0.30,
    seed: int = 0,
    seeds: Sequence[int] | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    time_aware: bool = False,
    final_holdout_fraction: float = 0.20,
    final_holdout_min_rows: int = 200,
) -> SplitPlan:
    """Build repeated holdouts plus the optional untouched final holdout.

    ``frame`` is never modified. Returned indices are positional row numbers
    into the original frame. Rows with a missing target (and, for chronological
    splits, a missing time) are excluded and recorded in ``warnings``.
    """
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} is not in the dataset")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0.0 <= final_holdout_fraction < 1.0:
        raise ValueError("final_holdout_fraction must be in [0, 1)")

    labels = frame[target]
    resolved_kind = target_kind or infer_target_kind(
        labels,
        target,
        time_column=time_column,
    )
    classification = resolved_kind in ("binary", "multiclass")
    chronological = resolved_kind == "time_series" or time_aware

    warnings: list[str] = []
    if chronological:
        if time_column is None or time_column not in frame.columns:
            raise ValueError(
                "chronological splits require a time column present in the frame"
            )
        times = _time_values(frame, time_column)
    else:
        times = None

    valid = labels.notna().to_numpy(copy=True)
    if times is not None:
        valid &= times.notna().to_numpy()
    eligible = np.flatnonzero(valid)
    dropped = int(len(frame) - len(eligible))
    if dropped:
        warnings.append(f"rows_excluded_missing_target_or_time:{dropped}")
    if len(eligible) < _MIN_SPLIT_ROWS:
        raise ValueError(
            "at least two rows with a non-missing target are required for splits"
        )

    if seeds is None:
        resolved_seeds = [int(seed) + index for index in range(n_splits)]
    else:
        resolved_seeds = [int(value) for value in seeds]
        if len(resolved_seeds) != n_splits:
            raise ValueError(
                f"expected {n_splits} seeds, received {len(resolved_seeds)}"
            )
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("repeated holdout seeds must be distinct")

    group_keys: np.ndarray | None = None
    if group_column is not None:
        if group_column not in frame.columns:
            warnings.append(f"group_column_missing:{group_column}")
        else:
            group_keys = _group_keys(frame[group_column].tolist())

    use_final_holdout = (
        final_holdout_fraction > 0.0
        and len(eligible) >= final_holdout_min_rows
    )
    final_holdout: DataSplit | None = None
    pool = eligible
    if use_final_holdout:
        pool, holdout = _final_holdout_indices(
            eligible,
            fraction=final_holdout_fraction,
            target_kind=resolved_kind,
            labels=labels,
            group_keys=group_keys,
            chronological=chronological,
            times=times,
            seed=seed,
        )
        final_holdout = DataSplit(
            fold=-1,
            seed=int(seed),
            kind=(
                "chronological"
                if chronological
                else "group"
                if group_keys is not None
                else "stratified"
                if classification
                else "random"
            ),
            train_index=_as_positions(pool),
            test_index=_as_positions(holdout),
            train_rows=int(len(pool)),
            test_rows=int(len(holdout)),
        )
    elif final_holdout_fraction > 0.0:
        warnings.append(
            "final_holdout_skipped_small_n:"
            f"n={len(eligible)}<{final_holdout_min_rows}"
        )

    if len(pool) < _MIN_SPLIT_ROWS:
        raise ValueError(
            "not enough rows remain after reserving the final holdout"
        )

    splits: list[DataSplit] = []
    if chronological:
        partitions = _chronological_partitions(
            pool,
            times,
            n_splits=n_splits,
            test_fraction=test_fraction,
        )
        kind: SplitKind = "chronological"
        for fold, (train, test) in enumerate(partitions):
            splits.append(
                DataSplit(
                    fold=fold,
                    seed=resolved_seeds[fold],
                    kind=kind,
                    train_index=_as_positions(train),
                    test_index=_as_positions(test),
                    train_rows=int(len(train)),
                    test_rows=int(len(test)),
                )
            )
    else:
        grouped = group_keys is not None
        kind = "group" if grouped else "stratified" if classification else "random"
        for fold, fold_seed in enumerate(resolved_seeds):
            rng = np.random.default_rng(fold_seed)
            if grouped:
                train, test, grouped_ok = _group_partition(
                    pool,
                    group_keys,
                    test_fraction=test_fraction,
                    rng=rng,
                )
                if not grouped_ok:
                    warnings.append(
                        f"group_split_fallback:{fold}:fewer_than_two_groups"
                    )
            elif classification:
                train, test = _stratified_partition(
                    pool,
                    labels,
                    test_fraction=test_fraction,
                    rng=rng,
                )
            else:
                train, test = _random_partition(
                    pool, test_fraction=test_fraction, rng=rng
                )
            splits.append(
                DataSplit(
                    fold=fold,
                    seed=fold_seed,
                    kind=kind,
                    train_index=_as_positions(train),
                    test_index=_as_positions(test),
                    train_rows=int(len(train)),
                    test_rows=int(len(test)),
                )
            )

    if len(splits) < n_splits:
        warnings.append(
            f"fewer_splits_than_requested:{len(splits)}<{n_splits}"
        )

    return SplitPlan(
        target=target,
        target_kind=resolved_kind,
        kind=kind,
        seeds=resolved_seeds,
        splits=splits,
        final_holdout=final_holdout,
        low_confidence=final_holdout is None,
        warnings=warnings,
    )
