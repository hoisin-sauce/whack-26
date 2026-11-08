"""Target and feature typing for automated regression model search.

The functions here only inspect data. They never transform a frame, fit an
encoder, or impute a value, so feature typing can be used before splitting
without leaking test information into training fits.

Target kinds follow the model-search specification:

* ``binary`` -- exactly two non-null values;
* ``multiclass`` -- 3-20 categorical values or small integer codes;
* ``count`` -- non-negative integers, a count-like name, or a mean/variance
  ratio near one;
* ``time_series`` -- an ordered numeric target with a detected time column;
* ``continuous`` -- everything else.

Feature kinds are ``numeric``, ``categorical``, ``datetime``, ``id`` and
``leakage_risk``. Identifier, target-derived, and time columns are excluded
from the usable feature set unless the caller declares the model time-aware.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..profile import (
    _LEAKAGE_NAME_RE,
    _is_datetime_series,
    _is_numeric_series,
    _looks_like_id,
    _looks_like_time,
)
from .types import FeatureKind, TargetKind

_MAX_UNIQUE_VALUES = 20
_MAX_TOP_VALUES = 20
_COUNT_DISPERSION = (0.8, 1.25)
_TARGET_COPY_CORRELATION = 0.999999

_COUNT_NAME_TOKENS = frozenset(
    {
        "count",
        "counts",
        "num",
        "number",
        "n",
        "freq",
        "frequency",
        "occurrence",
        "occurrences",
        "event",
        "events",
        "visit",
        "visits",
        "trip",
        "trips",
        "quantity",
        "qty",
    }
)
_CATEGORY_NAME_TOKENS = frozenset(
    {
        "class",
        "classes",
        "category",
        "categories",
        "cat",
        "label",
        "labels",
        "group",
        "groups",
        "status",
        "code",
        "codes",
        "type",
        "types",
        "kind",
        "kinds",
        "grade",
        "grades",
        "level",
        "levels",
    }
)


class TargetInfo(BaseModel):
    """Typed summary of the prediction target."""

    name: str
    kind: TargetKind
    dtype: str
    n_rows: int
    count: int
    missing: int
    missing_pct: float
    cardinality: int
    unique_values: list[str] = Field(default_factory=list)
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    stats: dict[str, float | int | None] | None = None
    classes: list[str] = Field(default_factory=list)
    positive_class: str | None = None
    warnings: list[str] = Field(default_factory=list)


class FeatureInfo(BaseModel):
    """Typed summary of one candidate feature column."""

    name: str
    kind: FeatureKind
    dtype: str
    cardinality: int
    missing_pct: float
    usable: bool
    reason: str = ""
    numeric_summary: dict[str, float | int | None] | None = None
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    min_value: str | None = None
    max_value: str | None = None


class DatasetInfo(BaseModel):
    """Dataset typing result consumed by screening, splits, and candidates."""

    n_rows: int
    target: TargetInfo
    features: list[FeatureInfo] = Field(default_factory=list)
    usable_features: list[str] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    missingness: dict[str, float] = Field(default_factory=dict)
    time_column: str | None = None
    group_column: str | None = None
    time_aware: bool = False
    group_aware: bool = False
    warnings: list[str] = Field(default_factory=list)


def _name_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _name_has_token(name: str, tokens: frozenset[str]) -> bool:
    return bool(set(_name_tokens(name)) & tokens)


def _clean(value: Any, *, limit: int = 80) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def _missing_pct(series: pd.Series) -> float:
    rows = max(len(series), 1)
    return round(float(series.isna().sum()) / rows * 100.0, 2)


def _is_integral_non_negative(series: pd.Series) -> bool:
    if not _is_numeric_series(series):
        return False
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(np.equal(np.mod(values, 1.0), 0.0))
    )


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "zeros": 0,
            "negatives": 0,
            "mean_variance_ratio": None,
        }
    variance = float(np.var(values))
    mean = float(np.mean(values))
    return {
        "count": int(values.size),
        "mean": mean,
        "std": float(np.std(values)),
        "variance": variance,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "zeros": int(np.sum(values == 0.0)),
        "negatives": int(np.sum(values < 0.0)),
        "mean_variance_ratio": (
            mean / variance if variance > 0.0 else None
        ),
    }


def _top_values(series: pd.Series) -> list[tuple[str, int]]:
    counts = series.value_counts(dropna=True)
    return [
        (_clean(label), int(count))
        for label, count in counts.head(_MAX_TOP_VALUES).items()
    ]


def _unique_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for value in series.dropna():
        text = _clean(value)
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
        if len(values) >= _MAX_UNIQUE_VALUES:
            break
    return values


def _is_integer_dtype(series: pd.Series) -> bool:
    return bool(pd.api.types.is_integer_dtype(series.dtype))


def _looks_like_small_codes(values: np.ndarray, *, maximum_levels: int = 20) -> bool:
    """Contiguous, roughly balanced small integer codes.

    Counts are commonly imbalanced and may have gaps; categorical codes are
    usually a complete 0..k/1..k range with comparable group sizes.
    """
    if values.size == 0:
        return False
    minimum = int(np.min(values))
    maximum = int(np.max(values))
    levels = maximum - minimum + 1
    if levels < 3 or levels > maximum_levels:
        return False
    counts = np.bincount(
        values.astype(int) - minimum, minlength=levels
    ).astype(float)
    present = counts[counts > 0]
    if present.size != levels:
        return False
    return float(present.max() / present.min()) <= 2.5


def _positive_class(series: pd.Series) -> str | None:
    values = series.dropna()
    if values.nunique() != 2:
        return None
    labels = list(pd.unique(values))
    for preferred in (1, True, "1", "yes", "true"):
        if preferred in labels:
            return _clean(preferred)
    return _clean(labels[-1])


def infer_target_kind(
    series: pd.Series,
    name: str,
    *,
    time_column: str | None = None,
) -> TargetKind:
    """Infer one of the five target kinds from a target column.

    ``time_column`` only promotes a numeric target to ``time_series``; a
    two-class or categorical target keeps its classification kind.
    """
    values = series.dropna()
    if values.empty:
        return "continuous"

    unique = int(values.nunique())
    numeric = _is_numeric_series(series)
    is_bool = bool(pd.api.types.is_bool_dtype(series.dtype))

    if is_bool or unique == 2:
        return "binary"

    if time_column is not None and numeric:
        return "time_series"

    if not numeric:
        if 3 <= unique <= _MAX_UNIQUE_VALUES:
            return "multiclass"
        return "continuous"

    integral = _is_integral_non_negative(series)
    count_like_name = _name_has_token(name, _COUNT_NAME_TOKENS)
    category_like_name = _name_has_token(name, _CATEGORY_NAME_TOKENS)

    if integral:
        if count_like_name:
            return "count"
        raw = values.to_numpy(dtype=float)
        if _is_integer_dtype(series):
            if 3 <= unique <= _MAX_UNIQUE_VALUES and _looks_like_small_codes(raw):
                return "multiclass"
            # Integer measures with gaps or imbalanced levels are counts.
            return "count"
        # Float columns holding whole numbers are usually continuous measures
        # that lose precision to a NaN; only a categorical name, a near-Poisson
        # mean/variance ratio, or many levels promotes them to a non-continuous
        # kind.
        if 3 <= unique <= _MAX_UNIQUE_VALUES and category_like_name:
            return "multiclass"
        if unique <= _MAX_UNIQUE_VALUES:
            ratio = _numeric_summary(series)["mean_variance_ratio"]
            if (
                ratio is not None
                and _COUNT_DISPERSION[0] <= float(ratio) <= _COUNT_DISPERSION[1]
            ):
                return "count"
            return "continuous"
        return "count"

    if 3 <= unique <= _MAX_UNIQUE_VALUES and category_like_name:
        return "multiclass"
    return "continuous"


def describe_target(
    frame: pd.DataFrame,
    target: str,
    *,
    time_column: str | None = None,
) -> TargetInfo:
    """Return the typed target summary, raising when the target is absent."""
    if target not in frame.columns:
        raise ValueError(
            f"target column {target!r} is not in the dataset; "
            f"available columns: {', '.join(str(name) for name in frame.columns)}"
        )
    series = frame[target]
    missing = int(series.isna().sum())
    warnings: list[str] = []
    kind = infer_target_kind(series, target, time_column=time_column)
    cardinality = int(series.nunique(dropna=True))
    if cardinality == 0:
        warnings.append("target_all_missing")
    elif kind == "continuous" and not _is_numeric_series(series):
        warnings.append("target_high_cardinality_categorical")
    is_classification = kind in ("binary", "multiclass")
    return TargetInfo(
        name=target,
        kind=kind,
        dtype=str(series.dtype),
        n_rows=int(len(frame)),
        count=int(series.notna().sum()),
        missing=missing,
        missing_pct=_missing_pct(series),
        cardinality=cardinality,
        unique_values=_unique_values(series) if is_classification else [],
        top_values=_top_values(series) if cardinality <= 50 else [],
        stats=_numeric_summary(series) if _is_numeric_series(series) else None,
        classes=_unique_values(series) if is_classification else [],
        positive_class=_positive_class(series) if kind == "binary" else None,
        warnings=warnings,
    )


def _is_target_copy(
    series: pd.Series,
    target_series: pd.Series,
) -> bool:
    """Detect a numeric feature that is an affine copy of a numeric target."""
    if not (_is_numeric_series(series) and _is_numeric_series(target_series)):
        return False
    pair = pd.DataFrame(
        {
            "feature": pd.to_numeric(series, errors="coerce"),
            "target": pd.to_numeric(target_series, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3:
        return False
    feature = pair["feature"].to_numpy(dtype=float)
    target = pair["target"].to_numpy(dtype=float)
    if float(np.std(feature)) == 0.0 or float(np.std(target)) == 0.0:
        return False
    correlation = float(np.corrcoef(feature, target)[0, 1])
    return bool(
        math.isfinite(correlation)
        and abs(correlation) >= _TARGET_COPY_CORRELATION
    )


def classify_feature(
    name: str,
    series: pd.Series,
    *,
    rows: int,
    target_name: str | None = None,
    target_series: pd.Series | None = None,
    group_column: str | None = None,
) -> FeatureInfo:
    """Type one feature column and decide whether it is usable.

    Leakage is checked before time and identifier heuristics: a column named
    ``future_outcome`` must not become a usable ``datetime`` feature just
    because the model is time-aware.
    """
    cardinality = int(series.nunique(dropna=True))
    reason = ""

    if target_name is not None and name == target_name:
        kind: FeatureKind = "id"
        reason = "target"
    elif _LEAKAGE_NAME_RE.search(name.lower()) is not None:
        kind = "leakage_risk"
        reason = "leakage_name"
    elif group_column is not None and name == group_column:
        kind = "id"
        reason = "group_column"
    elif target_series is not None and _is_target_copy(series, target_series):
        kind = "leakage_risk"
        reason = "target_copy"
    elif _looks_like_time(name, series):
        kind = "datetime"
    elif _looks_like_id(name, series, cardinality, rows):
        kind = "id"
    elif _is_numeric_series(series):
        kind = "numeric"
    elif _is_datetime_series(series):
        kind = "datetime"
    else:
        kind = "categorical"

    if cardinality == 0:
        reason = reason or "all_missing"
    elif cardinality < 2:
        reason = reason or "constant"

    numeric_summary = (
        _numeric_summary(series) if kind == "numeric" else None
    )
    min_value: str | None = None
    max_value: str | None = None
    if _is_datetime_series(series):
        parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        if not parsed.empty:
            min_value = str(parsed.min())
            max_value = str(parsed.max())

    return FeatureInfo(
        name=name,
        kind=kind,
        dtype=str(series.dtype),
        cardinality=cardinality,
        missing_pct=_missing_pct(series),
        usable=False,
        reason=reason,
        numeric_summary=numeric_summary,
        top_values=(
            _top_values(series)
            if kind in ("categorical", "id", "leakage_risk")
            and cardinality <= 50
            else []
        ),
        min_value=min_value,
        max_value=max_value,
    )


def _feature_is_usable(
    feature: FeatureInfo,
    *,
    time_aware: bool,
) -> tuple[bool, str]:
    if feature.reason in ("target", "leakage_name", "target_copy", "group_column"):
        return False, feature.reason
    if feature.kind in ("id", "leakage_risk"):
        return False, feature.reason or feature.kind
    if feature.kind == "datetime" and not time_aware:
        return False, "time"
    if feature.cardinality < 2:
        return False, feature.reason or "constant"
    if feature.missing_pct >= 100.0:
        return False, "all_missing"
    return True, ""


def build_dataset_info(
    frame: pd.DataFrame,
    target: str,
    *,
    time_column: str | None = None,
    group_column: str | None = None,
    time_aware: bool | None = None,
) -> DatasetInfo:
    """Type the target and every feature, and list the usable columns.

    ``time_aware`` defaults to whether a time column was supplied. Time
    columns are excluded from features when the model is not time-aware, as
    required by the specification.
    """
    if target not in frame.columns:
        raise ValueError(
            f"target column {target!r} is not in the dataset; "
            f"available columns: {', '.join(str(name) for name in frame.columns)}"
        )

    warnings: list[str] = []
    if time_column is not None and time_column not in frame.columns:
        warnings.append(f"time_column_missing:{time_column}")
        time_column = None
    if group_column is not None and group_column not in frame.columns:
        warnings.append(f"group_column_missing:{group_column}")
        group_column = None

    aware = False if time_aware is None else bool(time_aware)
    if aware and time_column is None:
        warnings.append("time_aware_without_time_column")

    target_info = describe_target(frame, target, time_column=time_column)
    warnings.extend(target_info.warnings)

    target_series = frame[target]
    rows = len(frame)
    features: list[FeatureInfo] = []
    usable: list[str] = []
    excluded: dict[str, str] = {target: "target"}

    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        info = classify_feature(
            name,
            frame[name],
            rows=rows,
            target_name=target,
            target_series=target_series,
            group_column=group_column,
        )
        is_usable, reason = _feature_is_usable(info, time_aware=aware)
        info.usable = is_usable
        info.reason = reason
        features.append(info)
        if is_usable:
            usable.append(name)
        else:
            excluded[name] = reason

    missingness = {
        str(column): _missing_pct(frame[column]) for column in frame.columns
    }
    if group_column is not None:
        excluded.setdefault(group_column, "group_column")
    return DatasetInfo(
        n_rows=rows,
        target=target_info,
        features=features,
        usable_features=usable,
        excluded=excluded,
        missingness=missingness,
        time_column=time_column,
        group_column=group_column,
        time_aware=aware,
        group_aware=group_column is not None,
        warnings=warnings,
    )


def usable_feature_frame(
    frame: pd.DataFrame, info: DatasetInfo
) -> pd.DataFrame:
    """Return a copy containing only the usable feature columns."""
    columns = [name for name in info.usable_features if name in frame.columns]
    return frame.loc[:, columns].copy()


def detect_time_column(frame: pd.DataFrame) -> str | None:
    """Return the first datetime-typed column, or ``None``."""
    for column in frame.columns:
        if _is_datetime_series(frame[column]):
            return str(column)
    return None
