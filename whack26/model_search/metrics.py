"""Evaluation metrics for automated regression model search.

Every metric returns a plain ``float``. Genuinely undefined cases (an empty
input, a single class in an evaluation fold) return ``NaN`` so callers can drop
them rather than inventing a score. ``metrics_for_target`` removes non-finite
values before returning its dictionary, which keeps fold metrics JSON-safe.

Direction metadata lives in :data:`METRIC_DIRECTIONS`: ``"higher"`` means a
larger value is better (R², AUC, accuracy), ``"lower"`` means a smaller value
is better (RMSE, log-loss, Poisson deviance, AIC/BIC).
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

MetricDirection = Literal["higher", "lower"]

METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "r2": "higher",
    "rmse": "lower",
    "mae": "lower",
    "mape": "lower",
    "accuracy": "higher",
    "balanced_accuracy": "higher",
    "auc": "higher",
    "log_loss": "lower",
    "brier": "lower",
    "poisson_deviance": "lower",
    "overdispersion": "lower",
    "aic": "lower",
    "bic": "lower",
}

PRIMARY_METRIC_BY_TARGET: dict[str, str] = {
    "continuous": "r2",
    "time_series": "r2",
    "binary": "auc",
    "multiclass": "balanced_accuracy",
    "count": "poisson_deviance",
}

TARGET_METRICS: dict[str, tuple[str, ...]] = {
    "continuous": ("r2", "rmse", "mae", "mape"),
    "time_series": ("r2", "rmse", "mae", "mape"),
    "binary": ("accuracy", "balanced_accuracy", "auc", "log_loss", "brier"),
    "multiclass": ("accuracy", "balanced_accuracy", "auc", "log_loss"),
    "count": ("poisson_deviance", "rmse", "mae", "overdispersion"),
}


def metric_direction(name: str) -> MetricDirection:
    """Return whether ``name`` is maximized or minimized."""

    return METRIC_DIRECTIONS.get(name, "lower")


def is_higher_better(name: str) -> bool:
    return metric_direction(name) == "higher"


def primary_metric(target_kind: str) -> str:
    """Return the default headline metric for a target kind."""

    return PRIMARY_METRIC_BY_TARGET.get(target_kind, "r2")


def metric_names(target_kind: str) -> tuple[str, ...]:
    """Return the metric names reported for a target kind."""

    return TARGET_METRICS.get(target_kind, TARGET_METRICS["continuous"])


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _regression_arrays(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(y_pred, dtype=float).ravel()
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length ({truth.size} != "
            f"{prediction.size})"
        )
    keep = np.isfinite(truth) & np.isfinite(prediction)
    return truth[keep], prediction[keep]


def _classification_arrays(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    if truth.shape != prediction.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length ({truth.size} != "
            f"{prediction.size})"
        )
    keep = np.array(
        [_present(item) for item in truth], dtype=bool
    ) & np.array([_present(item) for item in prediction], dtype=bool)
    return truth[keep], prediction[keep]


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def r2_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Coefficient of determination (higher is better).

    A constant target follows the scikit-learn convention: perfect
    predictions score ``1.0`` and anything else scores ``0.0`` rather than
    leaking a non-finite value into fold summaries.
    """

    truth, prediction = _regression_arrays(y_true, y_pred)
    if truth.size < 2:
        return math.nan
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total <= 0.0:
        return 1.0 if residual <= 0.0 else 0.0
    return 1.0 - residual / total


def rmse(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    truth, prediction = _regression_arrays(y_true, y_pred)
    if truth.size == 0:
        return math.nan
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def mae(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    truth, prediction = _regression_arrays(y_true, y_pred)
    if truth.size == 0:
        return math.nan
    return float(np.mean(np.abs(truth - prediction)))


def mape(y_true: Sequence[Any], y_pred: Sequence[Any], *, eps: float = 1e-8) -> float:
    """Mean absolute percentage error, guarded against zero targets.

    The result is a percentage (0-100 scale). Rows whose true value is
    ``abs(value) <= eps`` are ignored; if none remain the metric is ``NaN``.
    """

    truth, prediction = _regression_arrays(y_true, y_pred)
    keep = np.abs(truth) > eps
    if not np.any(keep):
        return math.nan
    truth = truth[keep]
    prediction = prediction[keep]
    return float(np.mean(np.abs((truth - prediction) / truth)) * 100.0)


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    truth, prediction = _classification_arrays(y_true, y_pred)
    if truth.size == 0:
        return math.nan
    return float(np.mean(truth == prediction))


def balanced_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Macro-averaged recall over classes present in ``y_true``."""

    truth, prediction = _classification_arrays(y_true, y_pred)
    if truth.size == 0:
        return math.nan
    recalls: list[float] = []
    for label in np.unique(truth):
        mask = truth == label
        recalls.append(float(np.mean(prediction[mask] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def binary_auc(y_true: Sequence[Any], scores: Sequence[Any]) -> float:
    """Rank-based ROC AUC for a binary outcome.

    Ties in ``scores`` receive average ranks, so the result matches the
    Mann-Whitney U interpretation. Returns ``NaN`` unless ``y_true`` has
    exactly two classes.
    """

    truth, score = _classification_arrays(
        y_true, np.asarray(scores, dtype=float).ravel()
    )
    if truth.shape != score.shape:
        raise ValueError("y_true and scores must have the same length")
    labels = np.unique(truth)
    if labels.size != 2:
        return math.nan
    positive = truth == labels[1]
    n_positive = int(np.sum(positive))
    n_negative = int(truth.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return math.nan

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=float)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    rank_sum = float(np.sum(ranks[positive]))
    statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def macro_auc_ovr(
    y_true: Sequence[Any],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """Macro one-vs-rest AUC for multiclass probabilities."""

    matrix = np.asarray(probabilities, dtype=float)
    truth = np.asarray(y_true).ravel()
    if matrix.ndim != 2 or matrix.shape[0] != truth.size:
        raise ValueError("probabilities must be an (n, n_classes) matrix")
    classes = list(labels) if labels is not None else list(np.unique(truth))
    if len(classes) != matrix.shape[1]:
        raise ValueError("labels and probability columns do not match")
    scores: list[float] = []
    for index, label in enumerate(classes):
        binary = np.where(truth == label, 1, 0)
        value = binary_auc(binary, matrix[:, index])
        if math.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else math.nan


def log_loss(
    y_true: Sequence[Any],
    probabilities: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
    eps: float = 1e-15,
) -> float:
    """Mean negative log-likelihood.

    ``probabilities`` is either a 1-D vector of positive-class probabilities
    (binary) or an ``(n, n_classes)`` matrix. Labels are sorted for a 1-D
    vector: the larger label is the positive class, matching
    :func:`binary_auc`.
    """

    truth = np.asarray(y_true).ravel()
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        if matrix.size != truth.size:
            raise ValueError("y_true and probabilities must have the same length")
        classes = np.unique(truth)
        if classes.size != 2:
            return math.nan
        positive = truth == classes[1]
        clipped = np.clip(matrix, eps, 1.0 - eps)
        terms = np.where(positive, clipped, 1.0 - clipped)
        return float(-np.mean(np.log(terms)))

    if matrix.ndim != 2 or matrix.shape[0] != truth.size:
        raise ValueError("probabilities must be an (n, n_classes) matrix")
    classes = list(labels) if labels is not None else list(np.unique(truth))
    if len(classes) != matrix.shape[1]:
        raise ValueError("labels and probability columns do not match")
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        indices = np.array([lookup[item] for item in truth], dtype=int)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unexpected class label {exc.args[0]!r}") from exc
    clipped = np.clip(matrix[indices, indices], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier_score(
    y_true: Sequence[Any],
    probabilities: Sequence[Any] | np.ndarray,
) -> float:
    """Brier score for binary probabilities or a multiclass matrix."""

    truth = np.asarray(y_true).ravel()
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        classes = np.unique(truth)
        if classes.size != 2:
            return math.nan
        target = (truth == classes[1]).astype(float)
        return float(np.mean((matrix - target) ** 2))
    if matrix.ndim != 2 or matrix.shape[0] != truth.size:
        raise ValueError("probabilities must be an (n, n_classes) matrix")
    classes = list(np.unique(truth))
    if len(classes) != matrix.shape[1]:
        return math.nan
    lookup = {label: index for index, label in enumerate(classes)}
    target = np.zeros_like(matrix)
    for row, label in enumerate(truth):
        target[row, lookup[label]] = 1.0
    return float(np.mean(np.sum((matrix - target) ** 2, axis=1)))


def poisson_deviance(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    eps: float = 1e-12,
) -> float:
    """Mean Poisson deviance. Lower is better."""

    truth, prediction = _regression_arrays(y_true, y_pred)
    keep = (truth >= 0.0) & (prediction > 0.0)
    truth = truth[keep]
    prediction = prediction[keep]
    if truth.size == 0:
        return math.nan
    safe_prediction = np.maximum(prediction, eps)
    log_term = np.zeros_like(truth)
    positive = truth > 0.0
    log_term[positive] = truth[positive] * np.log(
        truth[positive] / safe_prediction[positive]
    )
    deviance = 2.0 * (log_term - (truth - safe_prediction))
    return float(np.mean(np.maximum(deviance, 0.0)))


def overdispersion_ratio(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    n_params: int = 1,
) -> float:
    """Pearson chi-square statistic divided by its residual degrees of freedom."""

    truth, prediction = _regression_arrays(y_true, y_pred)
    keep = prediction > 0.0
    truth = truth[keep]
    prediction = prediction[keep]
    if truth.size == 0:
        return math.nan
    dof = truth.size - max(int(n_params), 1)
    if dof <= 0:
        return math.nan
    chi_square = float(np.sum((truth - prediction) ** 2 / prediction))
    return chi_square / dof


def gaussian_log_likelihood(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> float:
    """MLE Gaussian log-likelihood using the mean squared residual."""

    truth, prediction = _regression_arrays(y_true, y_pred)
    if truth.size == 0:
        return math.nan
    variance = float(np.mean((truth - prediction) ** 2))
    variance = max(variance, np.finfo(float).tiny)
    return float(
        -0.5 * truth.size * (math.log(2.0 * math.pi * variance) + 1.0)
    )


def aic(log_likelihood: float, n_params: int) -> float:
    """Akaike information criterion (lower is better)."""

    ll = _finite(log_likelihood)
    if not math.isfinite(ll):
        return math.nan
    return float(2.0 * max(int(n_params), 0) - 2.0 * ll)


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Bayesian information criterion (lower is better)."""

    ll = _finite(log_likelihood)
    if not math.isfinite(ll) or n_obs <= 0:
        return math.nan
    return float(
        math.log(int(n_obs)) * max(int(n_params), 0) - 2.0 * ll
    )


def metrics_for_target(
    target_kind: str,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    probabilities: Sequence[Sequence[float]] | Sequence[float] | np.ndarray | None = None,
    labels: Sequence[Any] | None = None,
    n_params: int | None = None,
    log_likelihood: float | None = None,
    exposure: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Compute the standard metric block for a target kind.

    Metrics that are undefined or non-finite are omitted. AIC/BIC are added
    when both ``n_params`` and ``log_likelihood`` are supplied.
    """

    values: dict[str, float] = {}
    kind = target_kind

    if kind in ("continuous", "time_series"):
        truth, prediction = _regression_arrays(y_true, y_pred)
        values["r2"] = r2_score(truth, prediction)
        values["rmse"] = rmse(truth, prediction)
        values["mae"] = mae(truth, prediction)
        if np.any(np.abs(truth) > 1e-8):
            values["mape"] = mape(truth, prediction)
    elif kind == "binary":
        values["accuracy"] = accuracy(y_true, y_pred)
        values["balanced_accuracy"] = balanced_accuracy(y_true, y_pred)
        if probabilities is not None:
            matrix = np.asarray(probabilities, dtype=float)
            positive = matrix[:, 1] if matrix.ndim == 2 else matrix
            values["auc"] = binary_auc(y_true, positive)
            values["log_loss"] = log_loss(y_true, positive)
            values["brier"] = brier_score(y_true, positive)
    elif kind == "multiclass":
        values["accuracy"] = accuracy(y_true, y_pred)
        values["balanced_accuracy"] = balanced_accuracy(y_true, y_pred)
        if probabilities is not None:
            matrix = np.asarray(probabilities, dtype=float)
            values["auc"] = macro_auc_ovr(y_true, matrix, labels=labels)
            values["log_loss"] = log_loss(
                y_true, matrix, labels=labels
            )
    elif kind == "count":
        values["poisson_deviance"] = poisson_deviance(y_true, y_pred)
        values["rmse"] = rmse(y_true, y_pred)
        values["mae"] = mae(y_true, y_pred)
        if n_params is not None:
            values["overdispersion"] = overdispersion_ratio(
                y_true, y_pred, n_params=n_params
            )
    else:
        raise ValueError(f"unknown target kind {target_kind!r}")

    if n_params is not None and log_likelihood is not None:
        values["aic"] = aic(log_likelihood, n_params)
        values["bic"] = bic(log_likelihood, n_params, len(np.asarray(y_true)))

    return {
        name: float(value)
        for name, value in values.items()
        if math.isfinite(float(value))
    }


def auc(
    y_true: Sequence[Any],
    scores: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | None = None,
) -> float:
    """ROC AUC: binary for 1-D scores, macro one-vs-rest for a matrix."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 2:
        return macro_auc_ovr(y_true, matrix, labels=labels)
    return binary_auc(y_true, matrix)


r2 = r2_score
brier = brier_score
overdispersion = overdispersion_ratio
compute_metrics = metrics_for_target


__all__ = [
    "METRIC_DIRECTIONS",
    "PRIMARY_METRIC_BY_TARGET",
    "TARGET_METRICS",
    "accuracy",
    "aic",
    "auc",
    "balanced_accuracy",
    "bic",
    "binary_auc",
    "brier",
    "brier_score",
    "compute_metrics",
    "gaussian_log_likelihood",
    "is_higher_better",
    "log_loss",
    "mae",
    "macro_auc_ovr",
    "mape",
    "metric_direction",
    "metric_names",
    "metrics_for_target",
    "overdispersion",
    "overdispersion_ratio",
    "poisson_deviance",
    "primary_metric",
    "r2",
    "r2_score",
    "rmse",
]
