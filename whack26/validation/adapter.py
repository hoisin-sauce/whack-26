"""Feature-matrix construction and row alignment for model validation.

The four-layer engine expects a numeric feature matrix, a numeric target, and
the ability to reconstruct a DataFrame for the generated model. This module
owns that translation. It is intentionally conservative: if the target or the
features cannot be represented without guessing, validation is skipped rather
than forced through an incompatible engine contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

_MISSING_CATEGORY = "__other__"

TargetKind = Literal["continuous", "binary", "multiclass", "count"]

_CONTINUOUS_KINDS = frozenset(
    {"continuous", "time_series", "regression", "regressor"}
)
# Legacy screen kinds: ``numeric`` and ``categorical`` predate the typed target
# vocabulary, so they fall back to data-driven inference.
_INFERRED_KINDS = frozenset({"numeric", "categorical"})
_CLASSIFICATION_KINDS = frozenset({"binary", "multiclass"})
_KNOWN_KINDS = frozenset({"continuous", *_CLASSIFICATION_KINDS, "count"})

# Tokens that mark an integer target as an event count rather than a
# continuous measure. Mirrors the model-search target typing vocabulary.
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


class AdapterSkip(Exception):
    """A validation precondition failed; the report should be ``skipped``."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.message = message or reason


@dataclass
class ColumnSpec:
    name: str
    kind: str
    dtype: str
    start: int = 0
    stop: int = 0
    categories: list[Any] = field(default_factory=list)
    origin: pd.Timestamp | None = None


@dataclass
class FeatureMatrix:
    X: np.ndarray
    y: np.ndarray
    specs: list[ColumnSpec]
    feature_columns: list[str]
    retained_index: list[Any]
    dropped_rows: int
    target_name: str
    target_dtype: str
    periods_per_year: float
    periods_per_year_source: str
    target_kind: str = "continuous"
    target_labels: dict[int, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.X.shape[0])

    def frame_from_matrix(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Reconstruct the model-facing frame from a numeric matrix.

        When ``y`` is supplied the target column is included (for ``_fit``).
        When it is omitted the frame contains feature columns only (for
        ``_predict``), which prevents a model from reading the target at
        prediction time.
        """
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.X.shape[1]:
            raise ValueError(
                "feature matrix has the wrong shape: "
                f"{values.shape} != (n, {self.X.shape[1]})"
            )
        if values.shape[0] == 0:
            raise ValueError("feature matrix is empty")

        columns: dict[str, pd.Series] = {}
        for spec in self.specs:
            block = values[:, spec.start : spec.stop]
            if spec.kind == "numeric":
                series = pd.Series(block[:, 0], name=spec.name)
                series = _safe_cast(series, spec.dtype)
            elif spec.kind == "bool":
                series = pd.Series(block[:, 0] > 0.5, name=spec.name)
            elif spec.kind == "datetime":
                origin = spec.origin or pd.Timestamp(0, tz="UTC")
                series = pd.Series(
                    origin + pd.to_timedelta(block[:, 0], unit="D"),
                    name=spec.name,
                )
            elif spec.kind == "categorical":
                indices = np.argmax(block, axis=1)
                labels = [spec.categories[int(index)] for index in indices]
                series = pd.Series(labels, name=spec.name)
                series = _safe_cast(series, spec.dtype)
            else:  # pragma: no cover - guarded by construction
                raise ValueError(f"unknown column kind {spec.kind!r}")
            columns[spec.name] = series.reset_index(drop=True)

        frame = pd.DataFrame(columns)
        if y is not None:
            target = np.asarray(y, dtype=float).ravel()
            if target.shape[0] != values.shape[0]:
                raise ValueError("target length does not match the feature matrix")
            if self.target_kind in _CLASSIFICATION_KINDS and self.target_labels:
                try:
                    labels = [
                        self.target_labels[code]
                        for code in _target_codes(target)
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown target code {exc.args[0]!r}"
                    ) from exc
                target_series = pd.Series(labels, name=self.target_name)
            else:
                target_series = pd.Series(target, name=self.target_name)
            frame[self.target_name] = _safe_cast(
                target_series, self.target_dtype
            ).reset_index(drop=True)
        return frame


def _target_codes(target: np.ndarray) -> list[int]:
    """Convert an encoded class target back to integer codes."""
    values = np.asarray(target, dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError("classification target codes contain non-finite values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded):
        raise ValueError("classification target codes must be integers")
    return [int(code) for code in rounded]


def _safe_cast(series: pd.Series, dtype: str) -> pd.Series:
    try:
        if dtype == "category":
            return series.astype("category")
        if dtype.startswith(("Int", "Float", "boolean")):
            return series.astype(dtype)
        return series.astype(dtype)
    except (TypeError, ValueError):
        return series


def _is_datetime(series: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(series.dtype))


def _is_bool(series: pd.Series) -> bool:
    return bool(pd.api.types.is_bool_dtype(series.dtype))


def _is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not _is_bool(series)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _name_has_token(name: Any, tokens: frozenset[str]) -> bool:
    if name is None:
        return False
    parts = [part for part in re.split(r"[^a-z0-9]+", str(name).lower()) if part]
    return bool(set(parts) & tokens)


def _is_integral_non_negative(series: pd.Series) -> bool:
    if not _is_numeric(series):
        return False
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return False
    return bool(
        np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(np.mod(values, 1.0) == 0.0)
    )


def infer_target_kind(series: pd.Series) -> TargetKind:
    """Infer the engine target kind from a target column.

    The rules are deliberately simple and order-sensitive:

    * a boolean column or a column with exactly two non-null values is
      ``binary``;
    * a non-numeric column with 3-10 non-null values is ``multiclass``;
    * a non-negative integral column with 3+ values and either 11+ levels or
      a count-like name is ``count``;
    * everything else is ``continuous``.
    """
    values = series.dropna()
    if values.empty:
        return "continuous"
    if _is_bool(series):
        return "binary"
    distinct = int(values.nunique())
    if distinct == 2:
        return "binary"
    if not _is_numeric(series):
        if 3 <= distinct <= 10:
            return "multiclass"
        return "continuous"
    if distinct >= 3 and _is_integral_non_negative(series):
        if distinct >= 11 or _name_has_token(series.name, _COUNT_NAME_TOKENS):
            return "count"
    return "continuous"


def _normalise_target_kind(target_kind: str | None) -> str | None:
    if target_kind is None:
        return None
    kind = str(target_kind).strip().lower()
    if not kind or kind in _INFERRED_KINDS:
        return None
    if kind in _CONTINUOUS_KINDS:
        return "continuous"
    return kind


def _plain_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sorted_labels(values: Any) -> list[Any]:
    labels = [_plain_label(value) for value in pd.unique(values).tolist()]
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=str)


def _encode_continuous(series: pd.Series) -> tuple[np.ndarray, dict[int, Any]]:
    if _is_bool(series) or not _is_numeric(series):
        raise AdapterSkip(
            "target_contract",
            f"target is not a continuous numeric series (dtype {series.dtype})",
        )
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.all(np.isfinite(values)):
        raise AdapterSkip(
            "target_contract", "the continuous target contains non-finite values"
        )
    if np.unique(values).size < 2:
        raise AdapterSkip("target_contract", "the target is constant")
    return values.astype(float), {}


def _encode_classes(
    series: pd.Series, kind: str
) -> tuple[np.ndarray, dict[int, Any]]:
    if _is_datetime(series) or pd.api.types.is_complex_dtype(series.dtype):
        raise AdapterSkip(
            "target_contract",
            f"target dtype {series.dtype} cannot be encoded as classes",
        )
    if series.isna().any():
        raise AdapterSkip(
            "target_contract", "the class target contains missing values"
        )
    labels = _sorted_labels(series)
    if len(labels) < 2:
        raise AdapterSkip("target_contract", "the target is constant")
    if kind == "binary" and len(labels) != 2:
        raise AdapterSkip(
            "target_contract",
            f"a binary target requires exactly 2 classes, found {len(labels)}",
        )
    lookup = {label: index for index, label in enumerate(labels)}
    codes = np.empty(len(series), dtype=float)
    for row, value in enumerate(series.tolist()):
        label = _plain_label(value)
        index = lookup.get(label)
        if index is None:
            raise AdapterSkip(
                "target_contract", f"unencodable target value {value!r}"
            )
        codes[row] = float(index)
    return codes, {index: label for index, label in enumerate(labels)}


def _encode_count(series: pd.Series) -> tuple[np.ndarray, dict[int, Any]]:
    if _is_bool(series) or not _is_numeric(series):
        raise AdapterSkip(
            "target_contract",
            f"a count target must be numeric (dtype {series.dtype})",
        )
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.all(np.isfinite(values)):
        raise AdapterSkip(
            "target_contract", "the count target contains non-finite values"
        )
    if np.any(values < 0.0):
        raise AdapterSkip(
            "target_contract", "the count target contains negative values"
        )
    if not np.all(np.mod(values, 1.0) == 0.0):
        raise AdapterSkip(
            "target_contract", "the count target contains non-integer values"
        )
    if np.unique(values).size < 2:
        raise AdapterSkip("target_contract", "the target is constant")
    return values.astype(float), {}


def encode_target(
    series: pd.Series, kind: str
) -> tuple[np.ndarray, dict[int, Any]]:
    """Encode a target column for the engine, returning codes and labels.

    Binary and multiclass targets are encoded with stable, sorted class labels;
    the returned mapping lets :meth:`FeatureMatrix.frame_from_matrix` restore
    the original values. Counts keep their numeric values. Continuous targets
    stay numeric. Constant targets and invalid count targets raise
    :class:`AdapterSkip` with reason ``target_contract``.
    """
    resolved = _normalise_target_kind(kind)
    if resolved is None:
        resolved = infer_target_kind(series)
    if resolved == "continuous":
        return _encode_continuous(series)
    if resolved in _CLASSIFICATION_KINDS:
        return _encode_classes(series, resolved)
    if resolved == "count":
        return _encode_count(series)
    raise AdapterSkip(
        "target_contract", f"unsupported target kind {kind!r}"
    )


def _continuous_target_contract(series: pd.Series, target: str) -> pd.Series:
    if _is_bool(series) or not _is_numeric(series):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} is not a continuous numeric series "
            f"(dtype {series.dtype})",
        )
    values = pd.to_numeric(series, errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    valid = values[finite]
    if len(valid) < 20:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has fewer than 20 finite rows",
        )
    distinct = int(valid.nunique(dropna=True))
    if distinct < 3:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has fewer than 3 distinct values",
        )
    if float(valid.std(ddof=1)) == 0.0:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has zero variance",
        )
    # A small integer-coded category set is not a return-like series even when
    # it passes the engine's numeric checks. Counts with many levels are left
    # to the engine; this guard only catches obvious class labels.
    if pd.api.types.is_integer_dtype(series.dtype) and distinct <= 10:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} looks like an integer-coded category "
            f"(only {distinct} distinct values)",
        )
    return values


def _class_target_contract(
    series: pd.Series, target: str, kind: str
) -> None:
    if _is_datetime(series) or pd.api.types.is_complex_dtype(series.dtype):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has dtype {series.dtype}, which cannot be "
            "encoded as class labels",
        )
    values = series.dropna()
    if len(values) < 20:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has fewer than 20 usable rows",
        )
    distinct = int(values.nunique())
    if distinct < 2:
        raise AdapterSkip("target_contract", f"target {target!r} is constant")
    if kind == "binary" and distinct != 2:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} is not binary: found {distinct} classes",
        )


def _count_target_contract(series: pd.Series, target: str) -> pd.Series:
    if _is_bool(series) or not _is_numeric(series):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} is not a numeric count series "
            f"(dtype {series.dtype})",
        )
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.all(np.isfinite(values)):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} contains non-finite count values",
        )
    if np.any(values < 0.0):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} contains negative count values",
        )
    if not np.all(np.mod(values, 1.0) == 0.0):
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} contains non-integer count values",
        )
    if values.size < 20:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has fewer than 20 usable rows",
        )
    if np.unique(values).size < 2:
        raise AdapterSkip("target_contract", f"target {target!r} is constant")
    return series


def _target_contract(
    frame: pd.DataFrame, target: str, target_kind: str | None
) -> tuple[pd.Series, str]:
    if target not in frame.columns:
        raise AdapterSkip(
            "target_contract",
            f"target column {target!r} is not in the dataset",
        )
    series = frame[target]
    kind = _normalise_target_kind(target_kind)
    if kind is None:
        kind = infer_target_kind(series)
    if kind not in _KNOWN_KINDS:
        raise AdapterSkip(
            "target_contract",
            f"target {target!r} has unsupported target kind {target_kind!r}",
        )
    if kind == "continuous":
        return _continuous_target_contract(series, target), kind
    if kind in _CLASSIFICATION_KINDS:
        _class_target_contract(series, target, kind)
        return series, kind
    return _count_target_contract(series, target), kind


def _select_features(
    frame: pd.DataFrame,
    target: str,
    required: Sequence[str],
    optional: Sequence[str],
    excluded: set[str],
    max_features: int,
) -> list[str]:
    declared = list(dict.fromkeys([*required, *optional]))
    if declared:
        missing_required = [name for name in required if name not in frame.columns]
        if missing_required:
            raise AdapterSkip(
                "missing_required_columns",
                "required columns are missing: " + ", ".join(missing_required),
            )
        selected = [
            name
            for name in declared
            if name in frame.columns and name != target
        ]
        excluded_hits = [name for name in selected if name in excluded]
        if excluded_hits:
            raise AdapterSkip(
                "excluded_feature_column",
                "feature columns have an id or leakage role: "
                + ", ".join(excluded_hits),
            )
    else:
        selected = [
            name
            for name in frame.columns
            if name != target
            and name not in excluded
            and _is_numeric(frame[name])
            and int(frame[name].nunique(dropna=True)) > 1
        ]
    selected = list(dict.fromkeys(selected))[:max_features]
    if not selected:
        raise AdapterSkip("no_features", "no usable feature columns were found")
    return selected


def _categorical_values(series: pd.Series, max_categories: int) -> tuple[list[Any], list[str]]:
    values = series.dropna()
    categories = sorted(values.unique().tolist(), key=lambda value: str(value))
    if len(categories) <= max_categories:
        return categories, [str(value) for value in categories]
    counts = values.value_counts()
    keep = [
        label
        for label, _ in sorted(
            counts.items(), key=lambda item: (-int(item[1]), str(item[0]))
        )[: max_categories - 1]
    ]
    categories = sorted(keep, key=lambda value: str(value)) + [_MISSING_CATEGORY]
    return categories, [str(value) for value in categories]


def _build_columns(
    working: pd.DataFrame,
    feature_columns: Sequence[str],
    max_categories: int,
) -> tuple[list[ColumnSpec], list[np.ndarray]]:
    specs: list[ColumnSpec] = []
    blocks: list[np.ndarray] = []
    cursor = 0
    for name in feature_columns:
        series = working[name]
        if _is_datetime(series):
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            origin = parsed.min()
            if pd.isna(origin):
                raise AdapterSkip("feature_contract", f"column {name!r} has no dates")
            days = (parsed - origin).dt.total_seconds() / 86400.0
            block = days.to_numpy(dtype=float).reshape(-1, 1)
            spec = ColumnSpec(
                name=name,
                kind="datetime",
                dtype=str(series.dtype),
                start=cursor,
                stop=cursor + 1,
                origin=origin,
            )
        elif _is_bool(series):
            block = series.to_numpy(dtype=float).reshape(-1, 1)
            spec = ColumnSpec(
                name=name,
                kind="bool",
                dtype=str(series.dtype),
                start=cursor,
                stop=cursor + 1,
            )
        elif _is_numeric(series):
            block = series.to_numpy(dtype=float).reshape(-1, 1)
            spec = ColumnSpec(
                name=name,
                kind="numeric",
                dtype=str(series.dtype),
                start=cursor,
                stop=cursor + 1,
            )
        else:
            categories, labels = _categorical_values(series, max_categories)
            block = np.zeros((len(working), len(categories)), dtype=float)
            lookup = {str(value): index for index, value in enumerate(categories)}
            for row, value in enumerate(series.tolist()):
                index = lookup.get(str(value))
                if index is None:
                    if _MISSING_CATEGORY in categories:
                        index = len(categories) - 1
                    else:
                        index = 0
                block[row, index] = 1.0
            spec = ColumnSpec(
                name=name,
                kind="categorical",
                dtype=str(series.dtype),
                start=cursor,
                stop=cursor + len(categories),
                categories=categories,
            )
        specs.append(spec)
        blocks.append(block)
        cursor = spec.stop
    return specs, blocks


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    target: str,
    required_columns: Sequence[str] = (),
    optional_columns: Sequence[str] = (),
    time_column: str | None = None,
    excluded_columns: Sequence[str] = (),
    period_override: float | None = None,
    max_features: int = 10,
    max_categories: int = 20,
    target_kind: str | None = None,
) -> FeatureMatrix:
    """Build the numeric matrix used by ``stats_engine``.

    Raises :class:`AdapterSkip` when the data cannot satisfy the engine's
    contract. The caller turns that into a ``skipped`` report.
    """
    if frame.empty:
        raise AdapterSkip("target_contract", "the dataset is empty")
    target_values, resolved_kind = _target_contract(frame, target, target_kind)
    feature_columns = _select_features(
        frame,
        target,
        required_columns,
        optional_columns,
        set(excluded_columns),
        max_features,
    )

    working = frame.copy()
    mask = target_values.notna()
    for name in feature_columns:
        series = working[name]
        if _is_datetime(series):
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            mask &= parsed.notna()
        elif _is_numeric(series) or _is_bool(series):
            numeric = pd.to_numeric(series, errors="coerce")
            mask &= np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        else:
            mask &= series.notna()

    working = working.loc[mask].copy()
    dropped_rows = int(len(frame) - len(working))
    if len(working) < 20:
        raise AdapterSkip(
            "insufficient_rows",
            f"only {len(working)} complete rows remain after dropping missing "
            "features and target; the engine needs at least 20",
        )

    if time_column and time_column in working.columns:
        parsed_time = pd.to_datetime(working[time_column], errors="coerce")
        if parsed_time.notna().sum() >= 2:
            working = working.assign(__whack26_time=parsed_time).sort_values(
                "__whack26_time", kind="mergesort"
            )
            working = working.drop(columns=["__whack26_time"])

    periods, period_source = infer_periods_per_year(
        working, time_column, period_override
    )
    specs, blocks = _build_columns(working, feature_columns, max_categories)
    X = np.column_stack(blocks) if blocks else np.empty((len(working), 0))
    if X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise AdapterSkip(
            "feature_contract",
            "the feature matrix is empty or contains non-finite values",
        )
    y, target_labels = encode_target(working[target], resolved_kind)

    return FeatureMatrix(
        X=X,
        y=y,
        specs=specs,
        feature_columns=list(feature_columns),
        retained_index=working.index.tolist(),
        dropped_rows=dropped_rows,
        target_name=target,
        target_dtype=str(working[target].dtype),
        periods_per_year=periods,
        periods_per_year_source=period_source,
        target_kind=resolved_kind,
        target_labels=target_labels,
    )


def infer_periods_per_year(
    frame: pd.DataFrame,
    time_column: str | None,
    override: float | None = None,
) -> tuple[float, str]:
    """Infer an annualisation factor, or return the configured value."""
    if override is not None:
        return float(override), "config"
    if not time_column or time_column not in frame.columns:
        return 1.0, "default"
    series = frame[time_column]
    try:
        if _is_numeric(series) and not _is_datetime(series):
            values = pd.to_numeric(series, errors="coerce").dropna().sort_values()
        else:
            values = pd.to_datetime(series, errors="coerce", utc=True).dropna().sort_values()
    except (TypeError, ValueError):
        return 1.0, "default"
    if len(values) < 3:
        return 1.0, "default"
    if _is_numeric(series) and not _is_datetime(series):
        return 1.0, "time_column"
    deltas = values.diff().dropna().dt.total_seconds() / 86400.0
    if deltas.empty:
        return 1.0, "default"
    median = float(np.median(deltas.to_numpy(dtype=float)))
    if median <= 2.0:
        return 252.0, "time_column"
    if median <= 10.0:
        return 52.0, "time_column"
    if median <= 45.0:
        return 12.0, "time_column"
    if median <= 120.0:
        return 4.0, "time_column"
    return 1.0, "time_column"


def align_predictions(
    predictions: pd.DataFrame,
    retained_index: Sequence[Any],
) -> np.ndarray:
    """Align a saved ``row_index,prediction`` CSV to retained rows."""
    if "row_index" not in predictions.columns or "prediction" not in predictions.columns:
        raise ValueError("predictions must have row_index and prediction columns")
    lookup: dict[Any, float] = {}
    for row_index, prediction in zip(
        predictions["row_index"].tolist(), predictions["prediction"].tolist()
    ):
        try:
            value = float(prediction)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            lookup[row_index] = value
    aligned: list[float] = []
    missing: list[Any] = []
    for row_index in retained_index:
        value = lookup.get(row_index)
        if value is None:
            missing.append(row_index)
            aligned.append(float("nan"))
        else:
            aligned.append(value)
    if missing:
        raise ValueError(
            f"predictions are missing {len(missing)} retained row(s)"
        )
    result = np.asarray(aligned, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("aligned predictions contain non-finite values")
    return result
