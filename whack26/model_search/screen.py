"""Feature screening for automated regression model search.

Screening is diagnostic. It ranks features with Pearson/Spearman correlation,
mutual information, ANOVA or chi-square tests, and variance-inflation factors,
then applies a Benjamini-Hochberg correction across the feature tests. It never
claims causality and never edits the frame.

The optional scikit-learn backend is used for mutual information when it is
installed. The core install (numpy/pandas/scipy) has a deterministic
histogram/quantile mutual-information fallback, so screening always works.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from .data import DatasetInfo, build_dataset_info
from .types import FeatureKind, TargetKind

try:  # pragma: no cover - exercised on both backends in the suite
    from sklearn.feature_selection import (
        mutual_info_classif as _sk_mutual_info_classif,
        mutual_info_regression as _sk_mutual_info_regression,
    )

    _HAS_SKLEARN = True
except Exception:  # noqa: BLE001 - sklearn is an optional dependency
    _sk_mutual_info_classif = None
    _sk_mutual_info_regression = None
    _HAS_SKLEARN = False

_MIN_ROWS = 3
_MI_BINS = 10
_MAX_VIF = float("inf")


class FeatureScreen(BaseModel):
    """Screening result for one feature."""

    feature: str
    kind: FeatureKind
    missing_pct: float
    cardinality: int
    test: Literal["pearson", "spearman", "anova", "chi_square", "none"] = "none"
    statistic: float | None = None
    p_value: float | None = None
    adjusted_p: float | None = None
    significant: bool = False
    effect_name: str | None = None
    effect_size: float | None = None
    pearson: float | None = None
    spearman: float | None = None
    mutual_info: float | None = None
    vif: float | None = None
    rank: int | None = None
    selected: bool = False
    notes: list[str] = Field(default_factory=list)


class ScreenResult(BaseModel):
    """Typed screening output for one target."""

    target: str
    target_kind: TargetKind
    n_rows: int
    alpha: float = 0.05
    method: str = "histogram"
    features: list[FeatureScreen] = Field(default_factory=list)
    selected: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: object, digits: int = 6) -> float | None:
    number = _finite(value)
    return None if number is None else round(number, digits)


def _is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _numeric_values(series: pd.Series, kind: FeatureKind) -> pd.Series | None:
    if kind == "numeric" or _is_numeric(series):
        return pd.to_numeric(series, errors="coerce")
    if kind == "datetime":
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        try:
            raw = parsed.astype("int64").astype(float)
        except (TypeError, ValueError, OverflowError):
            raw = parsed.map(
                lambda value: value.value if pd.notna(value) else np.nan
            ).astype(float)
        raw = raw.where(parsed.notna(), np.nan)
        return pd.Series(raw, index=series.index, dtype=float)
    return None


def _correlation_pair(
    x: pd.Series, y: pd.Series
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return Pearson r/p and Spearman rho/p on complete pairs."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < _MIN_ROWS:
        return None, None, None, None
    values_x = frame["x"].to_numpy(dtype=float)
    values_y = frame["y"].to_numpy(dtype=float)
    if float(np.std(values_x)) == 0.0 or float(np.std(values_y)) == 0.0:
        return None, None, None, None
    try:
        pearson = stats.pearsonr(values_x, values_y)
        pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
    except (ValueError, FloatingPointError):
        pearson_r, pearson_p = None, None
    try:
        spearman = stats.spearmanr(values_x, values_y)
        spearman_r, spearman_p = float(spearman.statistic), float(spearman.pvalue)
    except (ValueError, FloatingPointError):
        spearman_r, spearman_p = None, None
    return (
        _round(pearson_r),
        _round(pearson_p),
        _round(spearman_r),
        _round(spearman_p),
    )


def _anova(
    groups: Sequence[pd.Series],
) -> tuple[float | None, float | None, float | None]:
    """One-way ANOVA, returning ``(F, p, eta_squared)``."""
    arrays = [
        group.dropna().to_numpy(dtype=float)
        for group in groups
        if group.dropna().size > 0
    ]
    if len(arrays) < 2:
        return None, None, None
    combined = np.concatenate(arrays)
    if combined.size < _MIN_ROWS:
        return None, None, None
    grand_mean = float(np.mean(combined))
    ss_between = float(
        sum(len(array) * (float(np.mean(array)) - grand_mean) ** 2 for array in arrays)
    )
    ss_within = float(
        sum(float(np.sum((array - float(np.mean(array))) ** 2)) for array in arrays)
    )
    ss_total = ss_between + ss_within
    if ss_total <= 0.0:
        return None, None, None
    eta_squared = ss_between / ss_total
    try:
        result = stats.f_oneway(*arrays)
        statistic = _finite(result.statistic)
        p_value = _finite(result.pvalue)
    except (ValueError, FloatingPointError):
        statistic, p_value = None, None
    return _round(statistic), _round(p_value), _round(eta_squared)


def _chi_square(
    feature: pd.Series, labels: pd.Series
) -> tuple[float | None, float | None, float | None]:
    """Chi-square independence test, returning ``(chi2, p, Cramer's V)``."""
    table = pd.crosstab(feature, labels, dropna=True)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None
    observed = table.to_numpy(dtype=float)
    if float(observed.sum()) < _MIN_ROWS:
        return None, None, None
    try:
        result = stats.chi2_contingency(observed, correction=False)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except (ValueError, FloatingPointError):
        return None, None, None
    n = float(observed.sum())
    denominator = n * (min(observed.shape) - 1)
    cramers_v = math.sqrt(statistic / denominator) if denominator > 0 else None
    return _round(statistic), _round(p_value), _round(cramers_v)


def _rank_bins(values: np.ndarray, *, bins: int = _MI_BINS) -> np.ndarray:
    """Quantile-bin continuous values without requiring pandas categories."""
    size = int(values.size)
    if size == 0:
        return np.array([], dtype=int)
    unique = np.unique(values)
    count = int(min(bins, unique.size))
    if count <= 1:
        return np.zeros(size, dtype=int)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(size, dtype=int)
    ranks[order] = np.arange(size, dtype=int)
    return np.minimum((ranks * count) // size, count - 1)


def _mutual_info_codes(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    x = np.asarray(x_codes, dtype=int)
    y = np.asarray(y_codes, dtype=int)
    if x.size == 0 or x.size != y.size:
        return 0.0
    x = x - int(x.min())
    y = y - int(y.min())
    nx = int(x.max()) + 1
    ny = int(y.max()) + 1
    table = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    total = table.sum()
    if total <= 0:
        return 0.0
    px = table.sum(axis=1) / total
    py = table.sum(axis=0) / total
    joint = table / total
    nonzero = joint > 0
    px_grid = np.broadcast_to(px[:, None], joint.shape)
    py_grid = np.broadcast_to(py[None, :], joint.shape)
    mi = float(
        np.sum(
            joint[nonzero]
            * (
                np.log(joint[nonzero])
                - np.log(px_grid[nonzero])
                - np.log(py_grid[nonzero])
            )
        )
    )
    return max(0.0, mi)


def _encode_categories(values: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(values, sort=False)
    return codes.astype(int)


def _mutual_information(
    x: pd.Series,
    y: pd.Series,
    *,
    classification: bool,
    categorical_feature: bool,
    seed: int,
    sample_size: int,
) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < _MIN_ROWS:
        return None
    if sample_size > 0 and len(frame) > sample_size:
        rng = np.random.default_rng(seed)
        positions = rng.choice(len(frame), size=sample_size, replace=False)
        frame = frame.iloc[np.sort(positions)]

    x_values = frame["x"]
    y_values = frame["y"]

    if _HAS_SKLEARN and (
        _sk_mutual_info_classif is not None
        and _sk_mutual_info_regression is not None
    ):
        try:
            if categorical_feature:
                encoded = _encode_categories(x_values).reshape(-1, 1)
                discrete = [True]
            else:
                encoded = x_values.to_numpy(dtype=float).reshape(-1, 1)
                discrete = [False]
            if classification:
                y_codes = _encode_categories(y_values)
                value = _sk_mutual_info_classif(
                    encoded,
                    y_codes,
                    discrete_features=discrete,
                    random_state=seed,
                )[0]
            else:
                value = _sk_mutual_info_regression(
                    encoded,
                    y_values.to_numpy(dtype=float),
                    discrete_features=discrete,
                    random_state=seed,
                )[0]
            return _round(value)
        except Exception:  # noqa: BLE001 - fall back to the core estimator
            pass

    if classification:
        y_codes = _encode_categories(y_values)
        x_codes = (
            _encode_categories(x_values)
            if categorical_feature
            else _rank_bins(x_values.to_numpy(dtype=float))
        )
    else:
        y_codes = _rank_bins(y_values.to_numpy(dtype=float))
        x_codes = (
            _encode_categories(x_values)
            if categorical_feature
            else _rank_bins(x_values.to_numpy(dtype=float))
        )
    return _round(_mutual_info_codes(x_codes, y_codes))


def benjamini_hochberg(
    pvalues: Sequence[float | None], *, alpha: float = 0.05
) -> list[float | None]:
    """Return Benjamini-Hochberg adjusted p-values in the input order.

    ``None`` entries are ignored and preserved as ``None``. ``alpha`` is used
    only to document the caller's threshold; significance is
    ``adjusted <= alpha``.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    adjusted: list[float | None] = [None] * len(pvalues)
    indexed = [
        (float(value), position)
        for position, value in enumerate(pvalues)
        if value is not None and math.isfinite(float(value))
    ]
    m = len(indexed)
    if m == 0:
        return adjusted
    indexed.sort(key=lambda item: item[0])
    previous = 1.0
    for rank in range(m, 0, -1):
        p_value, position = indexed[rank - 1]
        value = min(previous, p_value * m / rank)
        value = min(1.0, max(0.0, value))
        adjusted[position] = value
        previous = value
    return adjusted


def compute_vif(
    frame: pd.DataFrame, features: Sequence[str]
) -> dict[str, float | None]:
    """Variance inflation factors against the other supplied numeric features.

    Uses OLS via ``numpy.linalg.lstsq`` rather than statsmodels. A single
    numeric feature has VIF 1.0; an exactly collinear feature reports
    ``inf``.
    """
    numeric: dict[str, pd.Series] = {}
    for name in features:
        if name not in frame.columns:
            continue
        series = frame[name]
        if not _is_numeric(series):
            continue
        values = pd.to_numeric(series, errors="coerce")
        if int(values.notna().sum()) < _MIN_ROWS or float(values.std()) == 0.0:
            continue
        numeric[name] = values

    result: dict[str, float | None] = {name: None for name in features}
    if not numeric:
        return result
    if len(numeric) == 1:
        only = next(iter(numeric))
        result[only] = 1.0
        return result

    for name, values in numeric.items():
        others = [column for column in numeric if column != name]
        data = pd.DataFrame({name: values, **{col: numeric[col] for col in others}})
        complete = data.dropna()
        if len(complete) < _MIN_ROWS:
            result[name] = None
            continue
        response = complete[name].to_numpy(dtype=float)
        predictors = complete[others].to_numpy(dtype=float)
        design = np.column_stack(
            [np.ones(len(complete), dtype=float), predictors]
        )
        try:
            coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        except np.linalg.LinAlgError:
            result[name] = None
            continue
        residual = response - design @ coefficients
        total = float(np.sum((response - float(np.mean(response))) ** 2))
        if total <= 0.0:
            result[name] = None
            continue
        r_squared = 1.0 - float(np.sum(residual**2)) / total
        denominator = 1.0 - r_squared
        if denominator <= 1e-12:
            result[name] = _MAX_VIF
        else:
            result[name] = _round(1.0 / denominator)
    return result


def _test_feature(
    name: str,
    kind: FeatureKind,
    series: pd.Series,
    labels: pd.Series,
    target: pd.Series,
    *,
    classification: bool,
    binary: bool,
    seed: int,
    mi_sample_size: int,
) -> FeatureScreen:
    numeric = _numeric_values(series, kind) if kind in ("numeric", "datetime") else None
    screen = FeatureScreen(
        feature=name,
        kind=kind,
        missing_pct=round(float(series.isna().sum()) / max(len(series), 1) * 100.0, 2),
        cardinality=int(series.nunique(dropna=True)),
    )
    if kind == "datetime":
        screen.notes.append("datetime_as_ordinal")

    if classification:
        if numeric is not None:
            class_codes = [
                code
                for code in pd.unique(labels.dropna())
                if float(code) >= 0.0
            ]
            groups = [numeric[labels == code] for code in class_codes]
            statistic, p_value, effect = _anova(groups)
            screen.test = "anova"
            screen.effect_name = "eta_squared"
            screen.effect_size = effect
            screen.statistic = statistic
            screen.p_value = p_value
            pearson_r, pearson_p, spearman_r, _ = _correlation_pair(
                numeric, labels
            )
            screen.pearson = pearson_r
            screen.spearman = spearman_r
            if binary and pearson_p is not None:
                # Point-biserial correlation is the two-group t-test in
                # another form; prefer it for the primary p-value so the
                # sign is visible in the result.
                screen.test = "pearson"
                screen.statistic = pearson_r
                screen.p_value = pearson_p
                screen.effect_name = "pearson_r"
                screen.effect_size = pearson_r
            screen.mutual_info = _mutual_information(
                numeric,
                target,
                classification=True,
                categorical_feature=False,
                seed=seed,
                sample_size=mi_sample_size,
            )
        else:
            statistic, p_value, effect = _chi_square(series, labels)
            screen.test = "chi_square"
            screen.statistic = statistic
            screen.p_value = p_value
            screen.effect_name = "cramers_v"
            screen.effect_size = effect
            screen.mutual_info = _mutual_information(
                series,
                target,
                classification=True,
                categorical_feature=True,
                seed=seed,
                sample_size=mi_sample_size,
            )
    else:
        if numeric is not None:
            pearson_r, pearson_p, spearman_r, _ = _correlation_pair(
                numeric, target
            )
            screen.test = "pearson"
            screen.statistic = pearson_r
            screen.p_value = pearson_p
            screen.pearson = pearson_r
            screen.spearman = spearman_r
            screen.effect_name = "pearson_r"
            screen.effect_size = pearson_r
            screen.mutual_info = _mutual_information(
                numeric,
                target,
                classification=False,
                categorical_feature=False,
                seed=seed,
                sample_size=mi_sample_size,
            )
        else:
            groups = [
                target[series == value]
                for value in pd.unique(series.dropna())
            ]
            statistic, p_value, effect = _anova(groups)
            screen.test = "anova"
            screen.statistic = statistic
            screen.p_value = p_value
            screen.effect_name = "eta_squared"
            screen.effect_size = effect
            screen.mutual_info = _mutual_information(
                series,
                target,
                classification=False,
                categorical_feature=True,
                seed=seed,
                sample_size=mi_sample_size,
            )
    return screen


def screen_features(
    frame: pd.DataFrame,
    target: str,
    *,
    info: DatasetInfo | None = None,
    target_kind: TargetKind | None = None,
    time_column: str | None = None,
    group_column: str | None = None,
    time_aware: bool | None = None,
    max_features: int = 20,
    max_screened: int = 200,
    alpha: float = 0.05,
    mi_sample_size: int = 5000,
    seed: int = 0,
) -> ScreenResult:
    """Screen every usable feature and return typed, ranked results."""
    if max_features < 1:
        raise ValueError("max_features must be positive")
    if max_screened < 1:
        raise ValueError("max_screened must be positive")

    if info is None:
        info = build_dataset_info(
            frame,
            target,
            time_column=time_column,
            group_column=group_column,
            time_aware=time_aware,
        )
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} is not in the dataset")

    resolved_kind = target_kind or info.target.kind
    classification = resolved_kind in ("binary", "multiclass")
    binary = resolved_kind == "binary"
    labels = frame[target]
    if classification:
        codes, _ = pd.factorize(labels, sort=False)
        encoded = pd.Series(codes, index=labels.index, dtype=float).where(
            labels.notna(), np.nan
        )
    else:
        encoded = labels

    feature_info = {feature.name: feature for feature in info.features}
    names = [
        name
        for name in info.usable_features
        if name in frame.columns and name != target
    ][:max_screened]
    warnings: list[str] = list(info.warnings)
    if not names:
        warnings.append("no_usable_features")

    results: list[FeatureScreen] = []
    numeric_for_vif: list[str] = []
    for name in names:
        feature = feature_info.get(name)
        kind: FeatureKind = feature.kind if feature is not None else "numeric"
        if kind == "id" or kind == "leakage_risk":
            # Defensive: a caller-supplied DatasetInfo must not override the
            # target-leakage exclusion.
            continue
        screen = _test_feature(
            name,
            kind,
            frame[name],
            encoded,
            labels,
            classification=classification,
            binary=binary,
            seed=seed,
            mi_sample_size=mi_sample_size,
        )
        results.append(screen)
        if kind == "numeric":
            numeric_for_vif.append(name)

    vif = compute_vif(frame, numeric_for_vif)
    for result in results:
        if result.feature in vif:
            result.vif = vif[result.feature]

    adjusted = benjamini_hochberg(
        [result.p_value for result in results], alpha=alpha
    )
    for result, adjusted_p in zip(results, adjusted):
        result.adjusted_p = _round(adjusted_p)
        result.significant = bool(
            result.adjusted_p is not None and result.adjusted_p <= alpha
        )

    ranked = sorted(
        results,
        key=lambda result: (
            0 if result.significant else 1,
            result.adjusted_p if result.adjusted_p is not None else 1.0,
            -(result.mutual_info if result.mutual_info is not None else -1.0),
            result.feature,
        ),
    )
    for position, result in enumerate(ranked, start=1):
        result.rank = position
    selected = [result.feature for result in ranked[:max_features]]
    selected_set = set(selected)
    for result in results:
        result.selected = result.feature in selected_set

    return ScreenResult(
        target=target,
        target_kind=resolved_kind,
        n_rows=int(len(frame)),
        alpha=alpha,
        method="sklearn" if _HAS_SKLEARN else "histogram",
        features=results,
        selected=selected,
        warnings=warnings,
    )
