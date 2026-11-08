"""Candidate model-specification generation for automated model search.

The generator is deliberately data-light: it needs the target name and kind,
the usable feature names (with their kinds), and optionally relevance scores
and the training frame. Screening scores are produced by
:mod:`whack26.model_search.screen`; this module only consumes a ranking.

Ordering is deterministic. The baseline is always first, and when the pool is
capped candidates are taken round-robin by family so no family is starved.
"""

from __future__ import annotations

import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .types import FeatureKind, ModelSpec, TargetKind

_NUMERIC = "numeric"
_CATEGORICAL = "categorical"
_EXCLUDED = {"datetime", "id", "leakage_risk"}
_TARGET_TRANSFORMS = ("none", "log1p", "sqrt")
_FEATURE_TRANSFORMS = ("log1p", "sqrt")
_RIDGE_ALPHA = 1.0
_MAX_INTERACTION_PAIRS = 3
_MAX_CATEGORICAL_FEATURES = 3


def _is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_datetime64_any_dtype(series.dtype)
    )


def _infer_feature_kind(series: pd.Series) -> FeatureKind:
    if _is_numeric(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    return "categorical"


def _ranked(
    names: Iterable[str], scores: Mapping[str, float] | None
) -> list[str]:
    """Sort names by relevance score, keeping input order for ties."""

    ordered = list(dict.fromkeys(names))
    if not scores:
        return ordered

    def key(item: tuple[int, str]) -> tuple[float, int]:
        index, name = item
        try:
            value = float(scores.get(name, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        return (-abs(value), index)

    return [name for _, name in sorted(enumerate(ordered), key=key)]


def _usable_features(
    target: str,
    features: Sequence[str] | None,
    feature_kinds: Mapping[str, FeatureKind] | None,
    frame: pd.DataFrame | None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return ``(numeric, categorical, resolved_kinds)``."""

    resolved: dict[str, str] = {}
    if features is None:
        if feature_kinds:
            features = list(feature_kinds)
        elif frame is not None:
            features = [
                str(column) for column in frame.columns if str(column) != target
            ]
        else:
            features = []

    numeric: list[str] = []
    categorical: list[str] = []
    for name in features:
        if name == target:
            continue
        kind: str | None = None
        if feature_kinds and name in feature_kinds:
            kind = str(feature_kinds[name])
        elif frame is not None and name in frame.columns:
            kind = _infer_feature_kind(frame[name])
        else:
            kind = _NUMERIC
        if kind in _EXCLUDED:
            continue
        if kind == _CATEGORICAL:
            categorical.append(name)
            resolved[name] = _CATEGORICAL
        else:
            numeric.append(name)
            resolved[name] = _NUMERIC
    return numeric, categorical, resolved


def _non_negative_features(
    numeric: Sequence[str],
    frame: pd.DataFrame | None,
    explicit: Sequence[str] | None,
) -> set[str]:
    allowed = set(explicit or ())
    if frame is None:
        return {name for name in numeric if name in allowed}
    result: set[str] = set()
    for name in numeric:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if float(values.min()) >= 0.0:
            result.add(name)
    return result


def _target_transform_options(
    target_kind: TargetKind,
    frame: pd.DataFrame | None,
    target: str,
) -> list[str]:
    if target_kind in ("binary", "multiclass"):
        return ["none"]
    if target_kind not in ("continuous", "count", "time_series"):
        return ["none"]
    minimum: float | None = None
    if frame is not None and target in frame.columns:
        values = pd.to_numeric(frame[target], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        values = values[np.isfinite(values)]
        if values.size:
            minimum = float(values.min())
    options: list[str] = []
    for transform in _TARGET_TRANSFORMS:
        if minimum is None:
            options.append(transform)
        elif transform == "none":
            options.append(transform)
        elif transform == "log1p" and minimum > -1.0:
            options.append(transform)
        elif transform == "sqrt" and minimum >= 0.0:
            options.append(transform)
    return options or ["none"]


def _baseline(target_kind: TargetKind) -> ModelSpec:
    statistic = (
        "majority"
        if target_kind in ("binary", "multiclass")
        else "mean"
    )
    return ModelSpec(
        family="baseline",
        target_kind=target_kind,
        hyperparameters={"statistic": statistic},
    )


def _linear_specs(
    target_kind: TargetKind,
    target_transforms: Sequence[str],
    numeric: Sequence[str],
    categorical: Sequence[str],
    non_negative: set[str],
) -> list[ModelSpec]:
    if numeric:
        features = list(numeric)
    else:
        features = list(categorical)
    if not features:
        return []

    specs: list[ModelSpec] = []
    for transform in target_transforms:
        specs.append(
            ModelSpec(
                family="linear",
                target_kind=target_kind,
                features=features,
                target_transform=transform,  # type: ignore[arg-type]
                hyperparameters={"fit_intercept": 1},
            )
        )

    if len(features) > 3 and numeric:
        specs.append(
            ModelSpec(
                family="linear",
                target_kind=target_kind,
                features=list(numeric[:3]),
                target_transform="none",
                hyperparameters={"fit_intercept": 1, "subset": "top3"},
            )
        )

    if numeric:
        specs.append(
            ModelSpec(
                family="linear",
                target_kind=target_kind,
                features=list(numeric),
                target_transform="none",
                regularization=_RIDGE_ALPHA,
                hyperparameters={"penalty": "ridge"},
            )
        )

    transformable = [name for name in numeric if name in non_negative]
    for feature_transform in _FEATURE_TRANSFORMS:
        if not transformable:
            break
        specs.append(
            ModelSpec(
                family="linear",
                target_kind=target_kind,
                features=list(numeric),
                target_transform="none",
                feature_transforms={
                    name: feature_transform for name in transformable
                },
                hyperparameters={"feature_transform": feature_transform},
            )
        )
    return specs


def _interaction_pairs(numeric: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left in range(len(numeric)):
        for right in range(left + 1, len(numeric)):
            pairs.append((numeric[left], numeric[right]))
            if len(pairs) >= _MAX_INTERACTION_PAIRS:
                return pairs
    return pairs


def _interaction_specs(
    target_kind: TargetKind,
    numeric: Sequence[str],
) -> list[ModelSpec]:
    pairs = _interaction_pairs(numeric)
    if not pairs:
        return []
    specs: list[ModelSpec] = []
    used_pairs: list[tuple[str, str]] = []
    used_features: list[str] = []
    for pair in pairs:
        used_pairs.append(pair)
        for name in pair:
            if name not in used_features:
                used_features.append(name)
        specs.append(
            ModelSpec(
                family="interaction",
                target_kind=target_kind,
                features=list(used_features),
                interactions=list(used_pairs),
                target_transform="none",
                hyperparameters={"interaction_count": len(used_pairs)},
            )
        )
    return specs


def _classification_specs(
    family: str,
    target_kind: TargetKind,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> list[ModelSpec]:
    features = list(numeric) + list(categorical)
    if not features:
        return []
    specs = [
        ModelSpec(
            family=family,  # type: ignore[arg-type]
            target_kind=target_kind,
            features=features,
            target_transform="none",
        ),
        ModelSpec(
            family=family,  # type: ignore[arg-type]
            target_kind=target_kind,
            features=list(features),
            target_transform="none",
            regularization=_RIDGE_ALPHA,
            hyperparameters={"penalty": "ridge"},
        ),
        ModelSpec(
            family=family,  # type: ignore[arg-type]
            target_kind=target_kind,
            features=list(features),
            target_transform="none",
            hyperparameters={"class_weight": "balanced"},
        ),
    ]
    return specs


def _count_specs(
    target_kind: TargetKind,
    numeric: Sequence[str],
    categorical: Sequence[str],
    offset: str | None,
) -> list[ModelSpec]:
    features = list(numeric) + list(categorical)
    if not features:
        return []
    specs = [
        ModelSpec(
            family="poisson",
            target_kind=target_kind,
            features=list(features),
            target_transform="none",
            hyperparameters={"offset": offset} if offset else {},
        ),
        ModelSpec(
            family="poisson",
            target_kind=target_kind,
            features=list(features),
            target_transform="none",
            regularization=_RIDGE_ALPHA,
            hyperparameters={"penalty": "ridge"},
        ),
        ModelSpec(
            family="negative_binomial",
            target_kind=target_kind,
            features=list(features),
            target_transform="none",
            hyperparameters={"offset": offset} if offset else {},
        ),
    ]
    if offset:
        specs.append(
            ModelSpec(
                family="poisson",
                target_kind=target_kind,
                features=list(features),
                target_transform="none",
                hyperparameters={"offset": offset},
            )
        )
    return specs


def _group_specs(
    target_kind: TargetKind,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> list[ModelSpec]:
    statistic = "rate" if target_kind == "binary" else "mean"
    specs: list[ModelSpec] = []
    for name in categorical[:_MAX_CATEGORICAL_FEATURES]:
        specs.append(
            ModelSpec(
                family="group_comparison",
                target_kind=target_kind,
                features=[name],
                target_transform="none",
                hyperparameters={"statistic": statistic, "bins": 0},
            )
        )
    for name in numeric[:2]:
        specs.append(
            ModelSpec(
                family="group_comparison",
                target_kind=target_kind,
                features=[name],
                target_transform="none",
                hyperparameters={"statistic": statistic, "bins": 5},
            )
        )
    return specs


def _dedupe(specs: Sequence[ModelSpec]) -> list[ModelSpec]:
    seen: set[str] = set()
    unique: list[ModelSpec] = []
    for spec in specs:
        key = json.dumps(spec.model_dump(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _cap(specs: Sequence[ModelSpec], max_candidates: int) -> list[ModelSpec]:
    if max_candidates <= 0 or len(specs) <= max_candidates:
        return list(specs)
    baseline = [spec for spec in specs if spec.family == "baseline"]
    rest = [spec for spec in specs if spec.family != "baseline"]

    groups: dict[str, list[ModelSpec]] = {}
    order: list[str] = []
    for spec in rest:
        if spec.family not in groups:
            groups[spec.family] = []
            order.append(spec.family)
        groups[spec.family].append(spec)

    capped: list[ModelSpec] = baseline[:1]
    while len(capped) < max_candidates:
        progress = False
        for family in order:
            bucket = groups[family]
            if not bucket:
                continue
            capped.append(bucket.pop(0))
            progress = True
            if len(capped) >= max_candidates:
                break
        if not progress:
            break
    return capped


def generate_candidates(
    *,
    target: str,
    target_kind: TargetKind,
    features: Sequence[str] | None = None,
    feature_kinds: Mapping[str, FeatureKind] | None = None,
    feature_scores: Mapping[str, float] | None = None,
    frame: pd.DataFrame | None = None,
    non_negative_features: Sequence[str] | None = None,
    offset: str | None = None,
    max_features: int = 10,
    max_candidates: int = 100,
) -> list[ModelSpec]:
    """Generate a deterministic candidate pool for ``target``.

    Parameters
    ----------
    target:
        Target column name (never used as a feature).
    target_kind:
        Target type from :mod:`whack26.model_search.types`.
    features:
        Usable feature names. When omitted, inferred from ``frame`` or from
        the keys of ``feature_kinds``.
    feature_kinds:
        Feature typing produced by ``data.py``. ``id``, ``datetime``, and
        ``leakage_risk`` columns are excluded.
    feature_scores:
        Screening relevance per feature (higher is better, sign ignored).
        The ranking decides the feature subset and interaction pairs.
    frame:
        Optional training frame, used only to decide whether non-negative
        target/feature transforms are safe.
    non_negative_features:
        Extra features known to be non-negative when no frame is supplied.
    offset:
        Optional exposure/offset column for Poisson/NB candidates.
    max_features, max_candidates:
        Feature and candidate hard caps.

    Returns
    -------
    list[ModelSpec]
        Baseline first, then candidates. Never empty.
    """

    if target_kind not in (
        "continuous",
        "binary",
        "multiclass",
        "count",
        "time_series",
    ):
        raise ValueError(f"unknown target kind {target_kind!r}")

    numeric, categorical, _ = _usable_features(
        target, features, feature_kinds, frame
    )
    numeric = _ranked(numeric, feature_scores)[: max(int(max_features), 1)]
    categorical = _ranked(categorical, feature_scores)[
        : max(int(max_features), 1)
    ]

    non_negative = _non_negative_features(
        numeric, frame, non_negative_features
    )
    target_transforms = _target_transform_options(
        target_kind, frame, target
    )

    specs: list[ModelSpec] = [_baseline(target_kind)]

    if target_kind in ("continuous", "time_series", "count"):
        specs.extend(
            _linear_specs(
                target_kind,
                target_transforms,
                numeric,
                categorical,
                non_negative,
            )
        )
        specs.extend(_interaction_specs(target_kind, numeric))
        if target_kind == "count":
            specs.extend(
                _count_specs(target_kind, numeric, categorical, offset)
            )
    elif target_kind == "binary":
        specs.extend(
            _classification_specs(
                "logistic", target_kind, numeric, categorical
            )
        )
        specs.extend(_interaction_specs(target_kind, numeric))
    elif target_kind == "multiclass":
        specs.extend(
            _classification_specs(
                "multinomial", target_kind, numeric, categorical
            )
        )

    if target_kind in ("continuous", "time_series", "binary", "multiclass", "count"):
        specs.extend(_group_specs(target_kind, numeric, categorical))

    return _cap(_dedupe(specs), int(max_candidates))


build_candidates = generate_candidates


__all__ = ["build_candidates", "generate_candidates"]
