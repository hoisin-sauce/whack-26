"""Cheap descriptive screen computed once per run.

The screen is context for hypothesis generation and for the method critic. It
never performs a hypothesis test and never decides whether a hypothesis is
accepted: it only renders distributions, missingness, correlations, group
statistics, and a simple time trend.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .profile import ColumnProfile, DatasetProfile

_MAX_GROUP_CATEGORIES = 20
_MAX_CORRELATION_COLUMNS = 20
_MAX_TOP_CORRELATIONS = 15
_MIN_CORRELATION_ROWS = 10


class TargetDistribution(BaseModel):
    kind: Literal["numeric", "categorical", "none"] = "none"
    name: str | None = None
    count: int = 0
    missing: int = 0
    missing_pct: float = 0.0
    stats: dict[str, float | int | None] | None = None
    top_values: list[tuple[str, int]] | None = None


class CorrelationMatrix(BaseModel):
    columns: list[str] = Field(default_factory=list)
    values: list[list[float | None]] = Field(default_factory=list)


class CorrelationPair(BaseModel):
    a: str
    b: str
    pearson: float | None = None
    spearman: float | None = None


class GroupStat(BaseModel):
    label: str
    n: int
    value: float | None = None


class GroupSummary(BaseModel):
    column: str
    target: str
    kind: Literal["mean", "rate"]
    groups: list[GroupStat] = Field(default_factory=list)


class TimePoint(BaseModel):
    label: str
    n: int
    mean: float | None = None


class TimeTrend(BaseModel):
    column: str
    target: str
    period: Literal["month", "ordered"]
    first: str
    last: str
    points: list[TimePoint] = Field(default_factory=list)


class ScreenSummary(BaseModel):
    """Bounded, JSON-serializable description of the dataset."""

    target: TargetDistribution = Field(default_factory=TargetDistribution)
    missingness: dict[str, float] = Field(default_factory=dict)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    pearson: CorrelationMatrix = Field(default_factory=CorrelationMatrix)
    spearman: CorrelationMatrix = Field(default_factory=CorrelationMatrix)
    top_correlations: list[CorrelationPair] = Field(default_factory=list)
    group_means: list[GroupSummary] = Field(default_factory=list)
    time_trend: TimeTrend | None = None
    warnings: list[str] = Field(default_factory=list)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any) -> float | None:
    number = _finite(value)
    return None if number is None else round(number, 4)


def _is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _is_binary(series: pd.Series) -> bool:
    return int(series.nunique(dropna=True)) == 2


def _target_summary(
    frame: pd.DataFrame, target: str | None
) -> tuple[TargetDistribution, list[str]]:
    warnings: list[str] = []
    if target is None or target not in frame.columns:
        return TargetDistribution(), warnings
    series = frame[target]
    missing = int(series.isna().sum())
    missing_pct = round(missing / max(len(frame), 1) * 100.0, 2)
    if _is_numeric(series):
        unique = set(series.dropna().unique().tolist())
        if len(unique) == 2 and unique <= {0, 1, 0.0, 1.0}:
            # A numeric 0/1 column is a binary outcome, not a continuous one.
            counts = series.value_counts(dropna=True)
            return (
                TargetDistribution(
                    kind="categorical",
                    name=target,
                    count=int(counts.sum()),
                    missing=missing,
                    missing_pct=missing_pct,
                    top_values=[
                        (str(label), int(count)) for label, count in counts.items()
                    ],
                ),
                warnings,
            )
        values = series.dropna()
        if values.empty:
            return (
                TargetDistribution(
                    kind="numeric",
                    name=target,
                    count=0,
                    missing=missing,
                    missing_pct=missing_pct,
                    stats=None,
                ),
                warnings,
            )
        stats: dict[str, float | int | None] = {
            "count": int(values.size),
            "mean": _round(values.mean()),
            "std": _round(values.std()),
            "min": _round(values.min()),
            "p25": _round(values.quantile(0.25)),
            "median": _round(values.median()),
            "p75": _round(values.quantile(0.75)),
            "max": _round(values.max()),
            "zeros": int((values == 0).sum()),
            "negatives": int((values < 0).sum()),
        }
        return (
            TargetDistribution(
                kind="numeric",
                name=target,
                count=int(values.size),
                missing=missing,
                missing_pct=missing_pct,
                stats=stats,
            ),
            warnings,
        )
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return (
            TargetDistribution(
                kind="categorical",
                name=target,
                count=0,
                missing=missing,
                missing_pct=missing_pct,
            ),
            warnings,
        )
    top = [(str(label), int(count)) for label, count in counts.head(20).items()]
    return (
        TargetDistribution(
            kind="categorical",
            name=target,
            count=int(counts.sum()),
            missing=missing,
            missing_pct=missing_pct,
            top_values=top,
        ),
        warnings,
    )


def _missingness(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in frame.columns:
        pct = round(float(frame[name].isna().sum()) / max(len(frame), 1) * 100.0, 2)
        if pct > 0.0:
            result[str(name)] = pct
    return result


def _numeric_columns(
    profile: DatasetProfile, frame: pd.DataFrame
) -> list[str]:
    selected: list[str] = []
    for column in profile.columns:
        if column.numeric is None:
            continue
        if column.role in ("id", "leakage_risk"):
            continue
        if column.name not in frame.columns:
            continue
        if int(frame[column.name].nunique(dropna=True)) < 2:
            continue
        selected.append(column.name)
        if len(selected) >= _MAX_CORRELATION_COLUMNS:
            break
    return selected


def _correlation(
    frame: pd.DataFrame, columns: Sequence[str], method: str
) -> CorrelationMatrix:
    if len(columns) < 2:
        return CorrelationMatrix(columns=list(columns), values=[])
    subset = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    matrix = subset.corr(method=method, min_periods=_MIN_CORRELATION_ROWS)
    values: list[list[float | None]] = []
    for row in matrix.to_numpy(dtype=object):
        values.append([_round(value) for value in row])
    return CorrelationMatrix(columns=list(columns), values=values)


def _top_correlations(
    pearson: CorrelationMatrix, spearman: CorrelationMatrix
) -> list[CorrelationPair]:
    pairs: list[CorrelationPair] = []
    columns = pearson.columns
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            p = pearson.values[i][j] if i < len(pearson.values) else None
            s = (
                spearman.values[i][j]
                if i < len(spearman.values) and j < len(spearman.values[i])
                else None
            )
            if p is None and s is None:
                continue
            pairs.append(CorrelationPair(a=a, b=columns[j], pearson=p, spearman=s))
    pairs.sort(
        key=lambda pair: (
            -abs(pair.pearson if pair.pearson is not None else pair.spearman or 0.0)
        )
    )
    return pairs[:_MAX_TOP_CORRELATIONS]


def _group_summaries(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    target: str | None,
    target_info: TargetDistribution,
) -> tuple[list[GroupSummary], list[str]]:
    warnings: list[str] = []
    if target is None or target not in frame.columns:
        return [], warnings
    binary_target = _is_binary(frame[target])
    if target_info.kind == "categorical" and not binary_target:
        warnings.append(
            "screen_group_means_skipped: categorical target has more than two classes"
        )
        return [], warnings
    numeric_target = target_info.kind == "numeric" and not binary_target
    rate_positive: Any = None
    if binary_target:
        counts = frame[target].value_counts(dropna=True)
        if len(counts) == 2:
            labels = counts.index.tolist()
            if 1 in labels or True in labels:
                rate_positive = 1 if 1 in labels else True
            elif "1" in labels:
                rate_positive = "1"
            else:
                rate_positive = counts.index[-1]

    selected: list[GroupSummary] = []
    for column in profile.columns:
        if column.name == target or column.name not in frame.columns:
            continue
        if column.numeric is not None or column.datetime:
            continue
        if column.role not in ("feature", "unknown"):
            continue
        if column.cardinality < 2 or column.cardinality > _MAX_GROUP_CATEGORIES:
            continue
        series = frame[column.name]
        if series.nunique(dropna=True) > _MAX_GROUP_CATEGORIES:
            continue
        groups: list[GroupStat] = []
        working = pd.DataFrame({"group": series, "target": frame[target]})
        for label, chunk in working.groupby("group", dropna=True, sort=False):
            if numeric_target:
                value = _finite(chunk["target"].mean())
                kind = "mean"
            else:
                positives = int((chunk["target"] == rate_positive).sum())
                denominator = int(chunk["target"].notna().sum())
                value = (
                    round(positives / denominator, 4) if denominator else None
                )
                kind = "rate"
            groups.append(
                GroupStat(label=str(label), n=int(len(chunk)), value=value)
            )
        if not groups:
            continue
        groups.sort(key=lambda item: (-item.n, item.label))
        selected.append(
            GroupSummary(
                column=column.name,
                target=target,
                kind="mean" if numeric_target else "rate",
                groups=groups[:_MAX_GROUP_CATEGORIES],
            )
        )
        if len(selected) >= 8:
            break
    return selected, warnings


def _ordered_chunks(
    labels: pd.Series, values: pd.Series, *, chunks: int = 12
) -> list[TimePoint]:
    frame = pd.DataFrame({"label": labels, "value": values}).dropna(subset=["label"])
    if frame.empty:
        return []
    frame = frame.sort_values("label", kind="mergesort")
    bucket_size = max(1, math.ceil(len(frame) / chunks))
    points: list[TimePoint] = []
    for start in range(0, len(frame), bucket_size):
        chunk = frame.iloc[start : start + bucket_size]
        first_label = str(chunk["label"].iloc[0])
        points.append(
            TimePoint(
                label=first_label,
                n=int(len(chunk)),
                mean=_finite(chunk["value"].mean()),
            )
        )
    return points


def _time_trend(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    target_info: TargetDistribution,
) -> tuple[TimeTrend | None, list[str]]:
    warnings: list[str] = []
    column = profile.time_column
    if column is None or column not in frame.columns:
        return None, warnings
    if target_info.kind != "numeric":
        warnings.append("screen_time_trend_skipped: target is not numeric")
        return None, warnings
    target = target_info.name
    if target is None or target not in frame.columns:
        return None, warnings
    series = frame[column]
    # A numeric year column must be treated as an ordered value, not as
    # nanoseconds since the epoch (pandas' default for integers).
    parsed = (
        pd.Series(pd.NaT, index=series.index)
        if pd.api.types.is_numeric_dtype(series.dtype)
        else pd.to_datetime(series, errors="coerce", utc=True)
    )
    valid_ratio = float(parsed.notna().sum()) / max(len(frame), 1)
    if valid_ratio >= 0.5:
        working = pd.DataFrame({"time": parsed, "value": frame[target]}).dropna(
            subset=["time"]
        )
        if working.empty:
            return None, warnings
        monthly = (
            working.set_index("time")["value"]
            .resample("MS")
            .agg(["count", "mean"])
            .dropna(subset=["count"])
        )
        points = [
            TimePoint(
                label=index.strftime("%Y-%m"),
                n=int(row["count"]),
                mean=_round(row["mean"]),
            )
            for index, row in monthly.iterrows()
        ]
        ordered = working.sort_values("time")
        return (
            TimeTrend(
                column=column,
                target=target,
                period="month",
                first=str(ordered["time"].iloc[0]),
                last=str(ordered["time"].iloc[-1]),
                points=points[-36:],
            ),
            warnings,
        )

    labels = series.astype("string")
    values = pd.to_numeric(frame[target], errors="coerce")
    points = _ordered_chunks(labels, values)
    if not points:
        return None, warnings
    ordered = (
        pd.DataFrame({"label": labels, "value": values})
        .dropna(subset=["label"])
        .sort_values("label", kind="mergesort")
    )
    return (
        TimeTrend(
            column=column,
            target=target,
            period="ordered",
            first=str(ordered["label"].iloc[0]),
            last=str(ordered["label"].iloc[-1]),
            points=points[-24:],
        ),
        warnings,
    )


def _categorical_columns(
    profile: DatasetProfile, frame: pd.DataFrame, target: str | None
) -> list[str]:
    selected: list[str] = []
    for column in profile.columns:
        if column.name == target or column.name not in frame.columns:
            continue
        if column.numeric is not None or column.datetime:
            continue
        if column.role not in ("feature", "unknown"):
            continue
        if 2 <= column.cardinality <= _MAX_GROUP_CATEGORIES:
            selected.append(column.name)
    return selected


def build_screen(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    max_group_summaries: int = 8,
) -> ScreenSummary:
    """Compute the descriptive screen. Never raises on odd columns."""
    target = profile.target.name if profile.target is not None else None
    warnings: list[str] = []

    try:
        target_info, target_warnings = _target_summary(frame, target)
        warnings.extend(target_warnings)
    except Exception as exc:  # noqa: BLE001 - context must not block a run
        target_info = TargetDistribution()
        warnings.append(
            f"screen_target_failed: {type(exc).__name__}: {exc}"
        )

    try:
        missingness = _missingness(frame)
    except Exception as exc:  # noqa: BLE001
        missingness = {}
        warnings.append(
            f"screen_missingness_failed: {type(exc).__name__}: {exc}"
        )

    numeric = _numeric_columns(profile, frame)
    try:
        pearson = _correlation(frame, numeric, "pearson")
        spearman = _correlation(frame, numeric, "spearman")
        top = _top_correlations(pearson, spearman)
    except Exception as exc:  # noqa: BLE001
        pearson = CorrelationMatrix(columns=numeric)
        spearman = CorrelationMatrix(columns=numeric)
        top = []
        warnings.append(
            f"screen_correlations_failed: {type(exc).__name__}: {exc}"
        )

    try:
        groups, group_warnings = _group_summaries(
            profile, frame, target, target_info
        )
        warnings.extend(group_warnings)
        groups = groups[:max_group_summaries]
    except Exception as exc:  # noqa: BLE001
        groups = []
        warnings.append(f"screen_groups_failed: {type(exc).__name__}: {exc}")

    try:
        trend, trend_warnings = _time_trend(profile, frame, target_info)
        warnings.extend(trend_warnings)
    except Exception as exc:  # noqa: BLE001
        trend = None
        warnings.append(f"screen_trend_failed: {type(exc).__name__}: {exc}")

    return ScreenSummary(
        target=target_info,
        missingness=missingness,
        numeric_columns=numeric,
        categorical_columns=_categorical_columns(profile, frame, target),
        pearson=pearson,
        spearman=spearman,
        top_correlations=top,
        group_means=groups,
        time_trend=trend,
        warnings=warnings,
    )


def _format_number(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return "n/a"
    return f"{number:.4g}"


def format_screen(screen: ScreenSummary, *, max_chars: int = 3000) -> str:
    """Render the screen as compact prompt text, bounded by ``max_chars``."""
    lines: list[str] = []
    target = screen.target
    if target.kind == "none":
        lines.append("Target distribution: none")
    elif target.kind == "numeric" and target.stats:
        stats = target.stats
        lines.append(
            "Target distribution: numeric | "
            f"n={target.count}, missing={target.missing} ({target.missing_pct}%) | "
            f"mean={_format_number(stats.get('mean'))}, "
            f"std={_format_number(stats.get('std'))}, "
            f"min={_format_number(stats.get('min'))}, "
            f"median={_format_number(stats.get('median'))}, "
            f"p75={_format_number(stats.get('p75'))}, "
            f"max={_format_number(stats.get('max'))}, "
            f"zeros={stats.get('zeros')}, negatives={stats.get('negatives')}"
        )
    else:
        top = ", ".join(
            f"{label} ({count})" for label, count in (target.top_values or [])[:10]
        )
        lines.append(
            f"Target distribution: categorical | n={target.count}, "
            f"missing={target.missing} ({target.missing_pct}%) | top: {top or 'none'}"
        )

    if screen.missingness:
        missing = ", ".join(
            f"{name}={pct}%" for name, pct in screen.missingness.items()
        )
        lines.append(f"Missingness (>0): {missing}")
    else:
        lines.append("Missingness: none")

    if screen.top_correlations:
        pairs = []
        for pair in screen.top_correlations[:10]:
            pairs.append(
                f"{pair.a}~{pair.b}: pearson={_format_number(pair.pearson)}, "
                f"spearman={_format_number(pair.spearman)}"
            )
        lines.append("Top correlations (descriptive only): " + "; ".join(pairs))

    for group in screen.group_means[:4]:
        shown = ", ".join(
            f"{item.label}: n={item.n}, {group.kind}={_format_number(item.value)}"
            for item in group.groups[:8]
        )
        lines.append(f"Group {group.kind} of {group.target} by {group.column}: {shown}")

    if screen.time_trend is not None:
        trend = screen.time_trend
        shown = ", ".join(
            f"{point.label}: {_format_number(point.mean)} (n={point.n})"
            for point in trend.points[-8:]
        )
        lines.append(
            f"Time trend on {trend.column} ({trend.period}, "
            f"{trend.first} to {trend.last}): {shown}"
        )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
