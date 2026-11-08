"""Candidate ranking, baseline margin, stability and top-N selection.

The selection rules mirror the spec (§4.8):

1. discard candidates that did not converge or produced non-finite metrics;
2. require a minimum improvement over the baseline;
3. rank by primary-metric mean, then stability, then parsimony;
4. apply a complexity penalty and an events-per-variable parameter cap;
5. keep the top ``top_n`` candidates;
6. report the selection confidence (``high``/``medium``/``low``).

Selection never looks at the final holdout and never uses a target-derived
column as a feature.  A candidate that references the target, a declared
leakage column, or an id column is rejected.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .types import (
    MetricName,
    ModelCandidate,
    ModelSearchConfig,
    ModelSearchResult,
    ModelSpec,
    TargetKind,
)

Confidence = Literal["high", "medium", "low"]

# Metrics where larger values are better.  Every other known metric is treated
# as "smaller is better".
_HIGHER_IS_BETTER = frozenset(
    {"r2", "accuracy", "balanced_accuracy", "auc"}
)
_LOWER_IS_BETTER = frozenset(
    {"rmse", "mae", "log_loss", "poisson_deviance", "aic", "bic"}
)

_PRIMARY_METRICS: dict[TargetKind, tuple[MetricName, ...]] = {
    "continuous": ("r2", "rmse", "mae"),
    "time_series": ("r2", "rmse", "mae"),
    "binary": ("auc", "balanced_accuracy", "accuracy", "log_loss"),
    "multiclass": ("balanced_accuracy", "accuracy", "auc", "log_loss"),
    "count": ("poisson_deviance", "rmse", "mae"),
}

# Families whose parameter count is checked against events-per-variable.
_EPV_FAMILIES = frozenset(
    {"linear", "logistic", "multinomial", "poisson", "negative_binomial", "interaction"}
)

# Candidate families that are models rather than the baseline.
_NON_BASELINE = frozenset(
    {
        "linear",
        "logistic",
        "multinomial",
        "poisson",
        "negative_binomial",
        "interaction",
        "group_comparison",
        "tree",
        "forest",
        "gradient_boosting",
        "sindy",
    }
)

_CONFIDENCE_PREFIX = "selection_confidence:"


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate plus the values used to rank it."""

    candidate: ModelCandidate
    score: float
    improvement: float | None
    parameters: int


def metric_direction(metric: str) -> int:
    """Return ``1`` when higher is better and ``-1`` when lower is better."""

    if metric in _HIGHER_IS_BETTER:
        return 1
    if metric in _LOWER_IS_BETTER:
        return -1
    # Unknown custom metrics are assumed to be error-like, which is the safe
    # default for a metric such as RMSE that is not in the frozen Literal.
    return -1


def primary_metric(
    target_kind: TargetKind,
    candidates: Sequence[ModelCandidate] = (),
    *,
    metric: MetricName | None = None,
) -> MetricName:
    """Pick the primary metric for a target kind.

    An explicit metric always wins.  Otherwise the first preferred metric that
    is actually present on at least one candidate is used, falling back to the
    target kind's default.
    """

    if metric is not None:
        if metric not in _HIGHER_IS_BETTER and metric not in _LOWER_IS_BETTER:
            raise ValueError(f"unknown metric: {metric!r}")
        return metric

    preferred = _PRIMARY_METRICS[target_kind]
    for name in preferred:
        if any(_metric_value(candidate, name)[0] is not None for candidate in candidates):
            return name
    return preferred[0]


def parameter_count(spec: ModelSpec) -> int:
    """Estimate a model's free parameters for parsimony and EPV checks."""

    override = spec.hyperparameters.get("parameter_count")
    if override is None:
        override = spec.hyperparameters.get("n_parameters")
    if override is not None:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            pass

    features = len(spec.features)
    interactions = len(spec.interactions)

    if spec.family == "baseline":
        return 1
    if spec.family in {"tree", "forest", "gradient_boosting", "sindy"}:
        n_terms = spec.hyperparameters.get("n_terms")
        try:
            return max(1, int(n_terms)) if n_terms is not None else max(1, features + 1)
        except (TypeError, ValueError):
            return max(1, features + 1)

    parameters = 1 + features + interactions
    if spec.family == "multinomial":
        try:
            classes = int(spec.hyperparameters.get("classes", 2))
        except (TypeError, ValueError):
            classes = 2
        if classes > 2:
            parameters = 1 + (classes - 1) * max(features, 1) + interactions
    return max(1, parameters)


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fold_values(
    candidate: ModelCandidate, metric: str, *, baseline: bool = False
) -> list[float]:
    key = "baseline_metrics" if baseline else "metrics"
    values: list[float] = []
    for fold in candidate.folds:
        raw = getattr(fold, key).get(metric)
        number = _finite(raw)
        if number is not None:
            values.append(number)
    return values


def _metric_value(candidate: ModelCandidate, metric: str) -> tuple[float | None, float]:
    """Return ``(mean, standard_deviation)`` for a candidate and metric."""

    raw_mean = candidate.metrics.get(metric)
    if metric in candidate.metrics and _finite(raw_mean) is None:
        # An explicitly recorded non-finite summary metric means the candidate
        # failed evaluation; do not resurrect it from individual folds.
        return None, 0.0
    mean = _finite(raw_mean)
    values = _fold_values(candidate, metric)
    if mean is None and values:
        mean = statistics.fmean(values)
    if mean is None:
        return None, 0.0

    std = _finite(candidate.metric_std.get(metric))
    if std is None:
        std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, max(0.0, std)


def _baseline_value(
    candidate: ModelCandidate,
    baseline: ModelCandidate | None,
    metric: str,
) -> float | None:
    """Find the baseline mean for a candidate.

    A candidate's own per-fold baseline metrics are preferred because they were
    measured on the same folds.  A separate baseline candidate is the fallback.
    """

    values = _fold_values(candidate, metric, baseline=True)
    if values:
        return statistics.fmean(values)
    if baseline is not None:
        return _metric_value(baseline, metric)[0]
    return None


def _leaked_features(
    spec: ModelSpec,
    *,
    target: str | None,
    leakage_columns: Iterable[str],
) -> list[str]:
    forbidden = {str(column) for column in leakage_columns}
    if target:
        forbidden.add(str(target))
    referenced: set[str] = set(spec.features)
    referenced.update(spec.feature_transforms)
    for left, right in spec.interactions:
        referenced.add(left)
        referenced.add(right)
    return sorted(referenced & forbidden)


def _nonconverged(candidate: ModelCandidate) -> bool:
    lowered = " ".join(candidate.warnings).lower()
    return any(
        phrase in lowered
        for phrase in (
            "did not converge",
            "did_not_converge",
            "failed to converge",
            "non-finite",
            "nonfinite",
        )
    )


def _infer_rows(candidates: Sequence[ModelCandidate]) -> int:
    best = 0
    for candidate in candidates:
        for fold in candidate.folds:
            best = max(best, int(fold.train_rows) + int(fold.test_rows))
    return best


def _parsimony_key(candidate: ModelCandidate, parameters: int) -> tuple[float, float, int]:
    aic = _finite(candidate.aic)
    bic = _finite(candidate.bic)
    return (
        aic if aic is not None else math.inf,
        bic if bic is not None else math.inf,
        parameters,
    )


def _rank(
    candidates: Sequence[ModelCandidate],
    *,
    target_kind: TargetKind,
    metric: MetricName,
    baseline: ModelCandidate | None,
    config: ModelSearchConfig,
    n_rows: int,
    target: str | None,
    leakage_columns: Iterable[str],
    complexity_penalty: float,
) -> tuple[list[RankedCandidate], list[tuple[ModelCandidate, str]], ModelCandidate | None, float | None]:
    baseline_candidate = baseline
    if baseline_candidate is None:
        baseline_candidate = next(
            (item for item in candidates if item.spec.family == "baseline"), None
        )

    pool = [item for item in candidates if item.spec.family != "baseline"]
    direction = metric_direction(metric)
    rows = n_rows if n_rows > 0 else _infer_rows(candidates)
    parameter_cap = max(1, rows // 10) if rows > 0 else 0

    ranked: list[RankedCandidate] = []
    rejected: list[tuple[ModelCandidate, str]] = []

    for candidate in pool:
        leaked = _leaked_features(
            candidate.spec, target=target, leakage_columns=leakage_columns
        )
        if leaked:
            rejected.append((candidate, f"target leakage: {', '.join(leaked)}"))
            continue

        if _nonconverged(candidate):
            rejected.append((candidate, "did not converge"))
            continue

        mean, std = _metric_value(candidate, metric)
        if mean is None:
            rejected.append((candidate, f"metric {metric!r} is missing or non-finite"))
            continue

        parameters = parameter_count(candidate.spec)
        if (
            parameter_cap > 0
            and candidate.spec.family in _EPV_FAMILIES
            and parameters > parameter_cap
        ):
            rejected.append(
                (
                    candidate,
                    f"parameter cap exceeded: {parameters} > {parameter_cap} (n/10, n={rows})",
                )
            )
            continue

        baseline_mean = _baseline_value(candidate, baseline_candidate, metric)
        improvement = (
            direction * (mean - baseline_mean) if baseline_mean is not None else None
        )
        if improvement is not None and improvement < config.margin:
            rejected.append(
                (
                    candidate,
                    f"baseline margin not met: {improvement:.4f} < {config.margin:.4f}",
                )
            )
            continue

        penalty = 0.0
        if complexity_penalty and rows > 0:
            penalty = complexity_penalty * parameters / rows
        score = direction * mean - penalty
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                score=score,
                improvement=improvement,
                parameters=parameters,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.score,
            _metric_value(item.candidate, metric)[1],
            *_parsimony_key(item.candidate, item.parameters),
            item.candidate.spec.family,
            ",".join(item.candidate.spec.features),
        )
    )

    global_baseline: float | None = None
    if baseline_candidate is not None:
        global_baseline = _metric_value(baseline_candidate, metric)[0]
    return ranked, rejected, baseline_candidate, global_baseline


def rank_candidates(
    candidates: Sequence[ModelCandidate],
    *,
    target_kind: TargetKind,
    metric: MetricName | None = None,
    baseline: ModelCandidate | None = None,
    config: ModelSearchConfig | None = None,
    n_rows: int = 0,
    target: str | None = None,
    leakage_columns: Iterable[str] = (),
    complexity_penalty: float = 0.01,
) -> list[ModelCandidate]:
    """Return the eligible candidates in rank order.

    Rank, selected flags, and no other attributes are mutated.  The returned
    list is not truncated; use :func:`select_models` for the top-N contract.
    """

    settings = config or ModelSearchConfig()
    chosen = primary_metric(target_kind, candidates, metric=metric)
    ranked, _rejected, _baseline, _baseline_mean = _rank(
        candidates,
        target_kind=target_kind,
        metric=chosen,
        baseline=baseline,
        config=settings,
        n_rows=n_rows,
        target=target,
        leakage_columns=leakage_columns,
        complexity_penalty=complexity_penalty,
    )

    output: list[ModelCandidate] = []
    for index, item in enumerate(ranked, start=1):
        item.candidate.rank = index
        item.candidate.selected = False
        output.append(item.candidate)
    return output


def selection_confidence(
    mean: float | None,
    std: float,
    *,
    folds: int,
    improvement: float | None,
    margin: float,
) -> Confidence:
    """Classify selection confidence from fold stability and baseline margin."""

    if mean is None:
        return "low"
    scale = abs(mean) if abs(mean) > 1e-12 else 1.0
    relative_std = std / scale
    delta = improvement if improvement is not None else 0.0

    if folds >= 3 and relative_std <= 0.10 and delta >= 2 * margin:
        return "high"
    if folds >= 2 and relative_std <= 0.30 and delta >= margin:
        return "medium"
    return "low"


def _confidence_for(
    ranked: Sequence[RankedCandidate], metric: MetricName, settings: ModelSearchConfig
) -> Confidence:
    if not ranked:
        return "low"
    top = ranked[0]
    mean, std = _metric_value(top.candidate, metric)
    fold_count = len(top.candidate.folds) or settings.folds
    return selection_confidence(
        mean,
        std,
        folds=fold_count,
        improvement=top.improvement,
        margin=settings.margin,
    )


def confidence_from_result(result: ModelSearchResult) -> Confidence:
    """Read the confidence recorded by :func:`select_models`."""

    for warning in result.warnings:
        if warning.startswith(_CONFIDENCE_PREFIX):
            value = warning.split(":", 1)[1].strip()
            if value in {"high", "medium", "low"}:
                return value  # type: ignore[return-value]
    return "low"


def select_models(
    candidates: Sequence[ModelCandidate],
    *,
    target_kind: TargetKind,
    metric: MetricName | None = None,
    baseline: ModelCandidate | None = None,
    config: ModelSearchConfig | None = None,
    n_rows: int = 0,
    target: str | None = None,
    leakage_columns: Iterable[str] = (),
    top_n: int | None = None,
    margin: float | None = None,
    complexity_penalty: float = 0.01,
    warnings: Iterable[str] = (),
) -> ModelSearchResult:
    """Rank candidates, keep the top N, and build a ``ModelSearchResult``.

    ``warnings`` are prepended to the result warnings (for upstream warnings
    such as a missing final holdout).  Selection confidence is recorded as
    ``selection_confidence: <level>`` in the result warnings.
    """

    settings = config or ModelSearchConfig()
    if top_n is not None:
        settings = settings.model_copy(update={"top_n": max(1, int(top_n))})
    if margin is not None:
        settings = settings.model_copy(update={"margin": float(margin)})

    chosen = primary_metric(target_kind, candidates, metric=metric)
    ranked, rejected, baseline_candidate, _baseline_mean = _rank(
        candidates,
        target_kind=target_kind,
        metric=chosen,
        baseline=baseline,
        config=settings,
        n_rows=n_rows,
        target=target,
        leakage_columns=leakage_columns,
        complexity_penalty=complexity_penalty,
    )

    for item in ranked:
        item.candidate.rank = None
        item.candidate.selected = False

    top = ranked[: settings.top_n]
    for index, item in enumerate(top, start=1):
        item.candidate.rank = index
        item.candidate.selected = index == 1
    selected = top[0].candidate if top else None

    output_warnings = list(warnings)
    confidence = _confidence_for(top, chosen, settings)
    output_warnings.append(f"{_CONFIDENCE_PREFIX} {confidence}")
    if selected is None:
        output_warnings.append("no_candidate_beats_baseline")
    if baseline_candidate is None:
        output_warnings.append("baseline_missing")
    for item, reason in rejected:
        output_warnings.append(
            f"rejected_candidate[{item.spec.family}]: {reason}"
        )

    return ModelSearchResult(
        target_kind=target_kind,
        metric=chosen,
        candidates=[item.candidate for item in top],
        selected=selected,
        baseline=baseline_candidate,
        folds=settings.folds,
        seed=settings.seed,
        backend=settings.backend,
        warnings=output_warnings,
    )


__all__ = [
    "Confidence",
    "RankedCandidate",
    "confidence_from_result",
    "metric_direction",
    "parameter_count",
    "primary_metric",
    "rank_candidates",
    "select_models",
    "selection_confidence",
]
