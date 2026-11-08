"""In-slice statistical validation of a selected model-search candidate.

This module is the default statistical backend. It consumes the
:class:`~whack26.model_search.types.ModelSearchResult` produced by model search
and reports baseline comparison, a permutation test, fold stability, and the
model-search diagnostics. It never re-runs model selection and never calls the
four-layer strategy engine; the latter is the ``stats_engine`` backend.

The context also knows how to recognise a return-like series. That gate is
deliberately conservative: a general continuous regression target is not a
return series, even when it is numerically compatible with the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from ..model_search.types import Diagnostic, ModelSearchResult
from .models import ValidationConfig, ValidationLayerReport, ValidationReport

StatisticalVerdict = Literal["pass", "fail", "inconclusive", "insufficient"]
StatisticalStatus = Literal["passed", "failed", "partial", "skipped", "error"]

IN_SLICE_TARGET_KINDS = frozenset(
    {"continuous", "binary", "multiclass", "count", "time_series"}
)
UNSUPPORTED_ENGINE_TARGET_KINDS = frozenset(
    {"binary", "multiclass", "count", "constant"}
)

# Metrics for which a larger value means a better model.
HIGHER_IS_BETTER: dict[str, bool] = {
    "r2": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "macro_f1": True,
    "mcc": True,
    "auc": True,
    "rmse": False,
    "mae": False,
    "log_loss": False,
    "brier": False,
    "poisson_deviance": False,
    "aic": False,
    "bic": False,
}

_RETURN_TARGET_HINTS = (
    "return",
    "ret_",
    "rets",
    "pct_change",
    "pctchange",
    "logret",
    "log_ret",
    "pnl",
    "sharpe",
    "strategy",
    "alpha",
)

_MAX_EXACT_PERMUTATIONS = 20_000


class StatisticalValidationResult(BaseModel):
    """One statistical validation result, independent of the artifact contract.

    ``report`` carries the existing :class:`ValidationReport` when the
    ``stats_engine`` backend ran, so callers that already embed that type can
    keep using it unchanged.
    """

    hypothesis_id: str
    backend: str = "in_slice"
    verdict: StatisticalVerdict = "inconclusive"
    status: StatisticalStatus = "partial"
    applicable: bool = True
    reason: str | None = None
    target: str | None = None
    target_kind: str | None = None
    metric: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_delta: float | None = None
    permutation_p_value: float | None = None
    permutation_source: str | None = None
    n_permutations: int = 0
    stability: dict[str, Any] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    attempts: int = 1
    engine_applicable: bool = False
    external_status: str | None = None
    external_validator: str | None = None
    library_version: str | None = None
    warnings: list[str] = Field(default_factory=list)
    supplementary: list[dict[str, Any]] = Field(default_factory=list)
    report: ValidationReport | None = None

    def to_validation_report(self) -> ValidationReport:
        """Return the existing validation-report shape for artifact embedding."""
        if self.report is not None:
            return self.report
        status: str
        verdict: str
        if self.verdict == "pass":
            status, verdict = "passed", "pass"
        elif self.verdict == "fail":
            status, verdict = "failed", "fail"
        elif self.verdict == "insufficient":
            status, verdict = "skipped", "inconclusive"
        else:
            status, verdict = "partial", "inconclusive"
        reasons = [self.reason] if self.reason else []
        report = ValidationReport(
            hypothesis_id=self.hypothesis_id,
            status=status,  # type: ignore[arg-type]
            verdict=verdict,  # type: ignore[arg-type]
            target=self.target,
            config={},
            layers=[
                ValidationLayerReport(
                    layer=f"{self.backend} statistical validation",
                    status=status,  # type: ignore[arg-type]
                    reason=self.reason,
                    metrics={
                        key: value
                        for key, value in {
                            "selected_score": self.metrics.get("selected_score"),
                            "baseline_score": self.metrics.get("baseline_score"),
                            "baseline_delta": self.baseline_delta,
                            "permutation_p_value": self.permutation_p_value,
                        }.items()
                        if value is not None
                    },
                )
            ],
            p_value=self.permutation_p_value,
            error="; ".join(reasons) if status == "error" and reasons else None,
            warnings=list(self.warnings),
        )
        return report


@dataclass
class ValidationContext:
    """Everything the statistical backends may need for one hypothesis.

    Only ``hypothesis_id`` is mandatory. The in-slice backend needs
    ``model_search``; the stats-engine backend needs an artifact/hypothesis pair
    or a run path, plus either ``target_values`` or a readable stored dataset.
    """

    hypothesis_id: str
    target: str | None = None
    target_kind: str | None = None
    model_search: ModelSearchResult | None = None
    attempts: int = 1
    seed: int = 0
    margin: float = 0.0
    max_p: float = 0.05
    n_permutations: int = 500
    predictions: Sequence[float] | Sequence[Sequence[float]] | None = None
    actuals: Sequence[float] | None = None
    target_values: Sequence[float] | None = None
    return_series: bool | None = None
    run_dir: Path | None = None
    artifact: Any = None
    hypothesis: Any = None
    run_path: Path | None = None
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    warnings: list[str] = field(default_factory=list)

    @property
    def resolved_target_kind(self) -> str | None:
        if self.target_kind:
            return str(self.target_kind).lower()
        if self.model_search is not None:
            return str(self.model_search.target_kind).lower()
        return None


# ---------------------------------------------------------------------------
# Result factories
# ---------------------------------------------------------------------------


def insufficient_result(
    hypothesis_id: str,
    reason: str,
    *,
    backend: str = "in_slice",
    target: str | None = None,
    target_kind: str | None = None,
    warnings: Sequence[str] | None = None,
) -> StatisticalValidationResult:
    return StatisticalValidationResult(
        hypothesis_id=hypothesis_id,
        backend=backend,
        verdict="insufficient",
        status="skipped",
        applicable=False,
        reason=reason,
        target=target,
        target_kind=target_kind,
        warnings=list(warnings or []),
    )


def error_result(
    hypothesis_id: str,
    reason: str,
    *,
    backend: str = "in_slice",
    warnings: Sequence[str] | None = None,
) -> StatisticalValidationResult:
    return StatisticalValidationResult(
        hypothesis_id=hypothesis_id,
        backend=backend,
        verdict="inconclusive",
        status="error",
        applicable=True,
        reason=reason,
        warnings=list(warnings or []),
    )


def validation_result_from_report(
    report: ValidationReport,
    *,
    backend: str,
) -> StatisticalValidationResult:
    """Wrap an existing :class:`ValidationReport` as a backend result."""
    status = report.status
    if status == "passed":
        verdict: StatisticalVerdict = "pass"
    elif status == "failed":
        verdict = "fail"
    elif status == "skipped":
        verdict = "insufficient"
    else:
        verdict = "inconclusive"
    metrics: dict[str, float] = {}
    for name in ("dsr_score", "p_value", "oos_score", "decay_rate", "tolerance"):
        value = getattr(report, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[name] = float(value)
    diagnostics = [
        Diagnostic(
            name=layer.layer,
            status=_layer_diagnostic_status(layer.status),
            message=layer.reason or "",
        )
        for layer in report.layers
    ]
    failed = report.failed_layers()
    reason = report.error
    if reason is None and failed:
        layer = report.layer(failed[0])
        reason = layer.reason if layer is not None else None
    if reason is None and report.skipped_layers():
        layer = report.layer(report.skipped_layers()[0])
        reason = layer.reason if layer is not None else None
    return StatisticalValidationResult(
        hypothesis_id=report.hypothesis_id,
        backend=backend,
        verdict=verdict,
        status=status,  # type: ignore[arg-type]
        applicable=status != "skipped",
        reason=reason,
        target=report.target,
        target_kind=None,
        metrics=metrics,
        baseline_delta=None,
        permutation_p_value=report.p_value,
        diagnostics=diagnostics,
        engine_applicable=backend == "stats_engine",
        warnings=list(report.warnings),
        report=report,
    )


def _layer_diagnostic_status(
    status: str,
) -> Literal["ok", "warn", "fail", "not_applicable"]:
    if status == "passed":
        return "ok"
    if status == "failed":
        return "fail"
    if status == "skipped":
        return "not_applicable"
    return "fail"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def metric_value(
    metric: str,
    actuals: Sequence[float] | np.ndarray,
    predictions: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Compute one supported metric from predictions, or ``nan``.

    ``predictions`` may be a 1-D vector or an ``(n, n_classes)`` probability
    matrix. Multiclass ``actuals`` are the class codes (integer labels); the
    sorted distinct values select the matrix column order, matching
    :mod:`whack26.model_search.metrics`.

    Binary metrics keep their historical 1-D behavior: probabilities are
    thresholded at ``0.5`` for accuracy and balanced accuracy, and
    ``log_loss`` treats the actuals as positive-class probabilities. Metrics
    that are undefined for the supplied shape return ``nan`` instead of
    raising.
    """
    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    if not np.all(np.isfinite(y_true)):
        return float("nan")
    if y_pred.ndim == 2:
        if y_pred.shape[0] != y_true.size or y_pred.shape[1] < 2:
            return float("nan")
        if not np.all(np.isfinite(y_pred)):
            return float("nan")
        return _metric_value_matrix(metric, y_true, y_pred)
    if y_pred.ndim != 1 or y_pred.shape != y_true.shape:
        return float("nan")
    if not np.all(np.isfinite(y_pred)):
        return float("nan")
    return _metric_value_vector(metric, y_true, y_pred)


def _integer_like(values: np.ndarray, *, atol: float = 1e-9) -> bool:
    return bool(np.all(np.abs(values - np.round(values)) <= atol))


def _metric_value_vector(
    metric: str, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    if metric == "r2":
        total = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if total == 0.0:
            return float("nan")
        residual = float(np.sum((y_true - y_pred) ** 2))
        return 1.0 - residual / total
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "poisson_deviance":
        if np.any(y_true < 0.0) or np.any(y_pred <= 0.0):
            return float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                y_true == 0.0,
                y_pred,
                y_true * np.log(y_true / y_pred) - (y_true - y_pred),
            )
        return float(2.0 * np.mean(terms))

    classes = np.unique(y_true)
    multiclass = classes.size > 2
    if metric == "accuracy":
        if multiclass:
            if not _integer_like(y_pred):
                return float("nan")
            return float(np.mean(y_pred == y_true))
        labels = (y_pred >= 0.5).astype(float)
        truth = (y_true >= 0.5).astype(float)
        return float(np.mean(labels == truth))
    if metric == "balanced_accuracy":
        if multiclass:
            if not _integer_like(y_pred):
                return float("nan")
            recalls = [
                float(np.mean(y_pred[y_true == label] == label))
                for label in classes
            ]
            return float(np.mean(recalls)) if recalls else float("nan")
        truth = (y_true >= 0.5).astype(bool)
        labels = (y_pred >= 0.5).astype(bool)
        recalls = []
        for value in (False, True):
            mask = truth == value
            if mask.any():
                recalls.append(float(np.mean(labels[mask] == value)))
        return float(np.mean(recalls)) if recalls else float("nan")
    if metric in {"macro_f1", "mcc"}:
        labels = _labels_from_vector(y_true, y_pred)
        if labels is None:
            return float("nan")
        if metric == "macro_f1":
            return _macro_f1(y_true, labels)
        return _mcc(y_true, labels)
    if metric == "auc":
        if multiclass:
            return float("nan")
        return _auc(y_true, y_pred)
    if metric == "brier":
        if multiclass or classes.size != 2:
            return float("nan")
        outcome = (y_true == classes[-1]).astype(float)
        return float(np.mean((y_pred - outcome) ** 2))
    if metric == "log_loss":
        if multiclass:
            return float("nan")
        clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(
            -np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
        )
    return float("nan")


def _metric_value_matrix(
    metric: str, y_true: np.ndarray, probabilities: np.ndarray
) -> float:
    classes = np.unique(y_true)
    if classes.size < 2 or classes.size != probabilities.shape[1]:
        return float("nan")
    codes = np.searchsorted(classes, y_true)
    if metric == "log_loss":
        clipped = np.clip(probabilities[np.arange(y_true.size), codes], 1e-15, 1.0)
        return float(-np.mean(np.log(clipped)))
    if metric == "brier":
        if classes.size == 2:
            outcome = (codes == 1).astype(float)
            return float(np.mean((probabilities[:, 1] - outcome) ** 2))
        one_hot = np.eye(classes.size)[codes]
        return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    predicted = classes[np.argmax(probabilities, axis=1)]
    if metric == "accuracy":
        return float(np.mean(predicted == y_true))
    if metric == "balanced_accuracy":
        recalls = [
            float(np.mean(predicted[y_true == label] == label))
            for label in classes
        ]
        return float(np.mean(recalls)) if recalls else float("nan")
    if metric == "macro_f1":
        return _macro_f1(y_true, predicted)
    if metric == "mcc":
        return _mcc(y_true, predicted)
    if metric == "auc":
        return _auc_ovr_macro(y_true, probabilities)
    return float("nan")


def _labels_from_vector(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray | None:
    """Resolve hard labels from a 1-D prediction vector.

    Binary vectors are assumed to be positive-class probabilities and are
    thresholded at ``0.5``. Three or more classes require integer class
    codes, because a single vector cannot carry one probability per class.
    """
    classes = np.unique(y_true)
    if classes.size < 2:
        return None
    if classes.size > 2:
        if not _integer_like(y_pred):
            return None
        return y_pred
    return np.where(y_pred >= 0.5, classes[-1], classes[0])


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    scores: list[float] = []
    for label in classes:
        true_positive = float(np.sum((y_pred == label) & (y_true == label)))
        false_positive = float(np.sum((y_pred == label) & (y_true != label)))
        false_negative = float(np.sum((y_pred != label) & (y_true == label)))
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    if classes.size < 2:
        return float("nan")
    lookup = {float(label): index for index, label in enumerate(classes)}
    true_codes = np.searchsorted(classes, y_true)
    pred_codes = np.asarray(
        [lookup.get(float(label), -1) for label in y_pred], dtype=int
    )
    size = int(classes.size)
    matrix = np.zeros((size, size), dtype=float)
    for true_code, pred_code in zip(true_codes, pred_codes):
        if pred_code >= 0:
            matrix[true_code, pred_code] += 1.0
    total = float(y_true.size)
    true_totals = matrix.sum(axis=1)
    pred_totals = matrix.sum(axis=0)
    correct = float(np.trace(matrix))
    numerator = correct * total - float(np.sum(true_totals * pred_totals))
    denominator = math.sqrt(
        max(total**2 - float(np.sum(pred_totals**2)), 0.0)
        * max(total**2 - float(np.sum(true_totals**2)), 0.0)
    )
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    truth = y_true >= 0.5
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    return float(
        (ranks[truth].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Tie-aware ROC AUC used by the one-vs-rest multiclass path."""
    values = np.asarray(scores, dtype=float).ravel()
    truth = np.asarray(positive, dtype=bool).ravel()
    if values.shape != truth.shape or values.size == 0:
        return float("nan")
    n_positive = int(truth.sum())
    n_negative = int(truth.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    statistic = float(ranks[truth].sum()) - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def _auc_ovr_macro(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    classes = np.unique(y_true)
    if classes.size < 2 or classes.size != probabilities.shape[1]:
        return float("nan")
    scores: list[float] = []
    for index, label in enumerate(classes):
        value = _rank_auc(probabilities[:, index], y_true == label)
        if _finite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")


def _metric_supported_for_permutation(metric: str) -> bool:
    return metric in {
        "r2",
        "rmse",
        "mae",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "auc",
        "log_loss",
        "brier",
        "poisson_deviance",
    }


def expected_calibration_error(
    probabilities: Sequence[Any] | np.ndarray,
    actuals: Sequence[Any] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over equally spaced confidence bins.

    Binary inputs may be a 1-D vector of positive-class probabilities.
    Multiclass inputs must be an ``(n, n_classes)`` matrix; the top-label
    confidence is used. Returns ``nan`` when the inputs cannot be interpreted
    as probabilistic classification predictions.
    """
    bins = int(n_bins)
    if bins < 2:
        return float("nan")
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(actuals, dtype=float).ravel()
    if truth.size == 0 or not np.all(np.isfinite(truth)):
        return float("nan")
    classes = np.unique(truth)
    if classes.size < 2:
        return float("nan")
    codes = np.searchsorted(classes, truth)
    if probs.ndim == 1:
        if classes.size != 2 or probs.size != truth.size:
            return float("nan")
        if not np.all(np.isfinite(probs)):
            return float("nan")
        predicted_codes = (probs >= 0.5).astype(int)
        confidence = np.maximum(probs, 1.0 - probs)
    elif probs.ndim == 2:
        if probs.shape != (truth.size, classes.size):
            return float("nan")
        if not np.all(np.isfinite(probs)):
            return float("nan")
        predicted_codes = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
    else:
        return float("nan")
    correct = (predicted_codes == codes).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = float(truth.size)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            accuracy = float(np.mean(correct[mask]))
            mean_confidence = float(np.mean(confidence[mask]))
            ece += float(np.sum(mask)) / total * abs(accuracy - mean_confidence)
    return float(ece) if _finite(ece) else float("nan")


def pearson_dispersion(
    predictions: Sequence[float] | np.ndarray,
    actuals: Sequence[float] | np.ndarray,
    *,
    n_params: int = 1,
) -> float:
    """Pearson chi-square dispersion estimate for count predictions.

    Values near ``1.0`` are consistent with Poisson variance; materially
    larger values indicate overdispersion. Returns ``nan`` when the inputs
    are not valid non-negative counts with positive fitted means.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return float("nan")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return float("nan")
    if np.any(acts < 0.0) or np.any(preds <= 0.0):
        return float("nan")
    degrees = acts.size - int(n_params)
    if degrees <= 0:
        return float("nan")
    statistic = float(np.sum((acts - preds) ** 2 / preds))
    return float(statistic / degrees) if _finite(statistic) else float("nan")


def _target_diagnostics(
    target_kind: str | None,
    actuals: Sequence[Any] | None,
    predictions: Sequence[Any] | None,
) -> tuple[list[Diagnostic], list[str]]:
    """Build calibration/dispersion diagnostics that never fail a result.

    Diagnostics are only added when the raw predictions and actuals are
    available and interpretable. Unavailable diagnostics produce a warning,
    not a failure.
    """
    kind = (target_kind or "").lower()
    if actuals is None or predictions is None:
        return [], []
    diagnostics: list[Diagnostic] = []
    warnings: list[str] = []
    if kind in {"binary", "multiclass"}:
        try:
            ece = expected_calibration_error(predictions, actuals)
        except (TypeError, ValueError):
            ece = float("nan")
        if _finite(ece):
            diagnostics.append(
                Diagnostic(
                    name="expected_calibration_error",
                    status="warn" if ece > 0.1 else "ok",
                    value=float(ece),
                    message=f"ECE={ece:.3f}",
                )
            )
        else:
            warnings.append(
                "expected_calibration_error_unavailable: probabilistic "
                "classification predictions are required"
            )
    elif kind == "count":
        try:
            dispersion = pearson_dispersion(predictions, actuals)
        except (TypeError, ValueError):
            dispersion = float("nan")
        if _finite(dispersion):
            diagnostics.append(
                Diagnostic(
                    name="pearson_dispersion",
                    status="warn" if dispersion > 1.5 else "ok",
                    value=float(dispersion),
                    message=f"Pearson dispersion={dispersion:.3f}",
                )
            )
        else:
            warnings.append(
                "pearson_dispersion_unavailable: non-negative counts and "
                "positive predicted means are required"
            )
    return diagnostics, warnings


def _fold_scores(selected: Any, metric: str) -> list[float]:
    scores: list[float] = []
    for fold in selected.folds:
        value = fold.metrics.get(metric)
        if value is not None and _finite(value):
            scores.append(float(value))
    return scores


def _baseline_fold_scores(selected: Any, metric: str) -> list[float]:
    scores: list[float] = []
    for fold in selected.folds:
        value = fold.baseline_metrics.get(metric)
        if value is not None and _finite(value):
            scores.append(float(value))
    return scores


def _delta(metric: str, selected_score: float, baseline_score: float) -> float:
    if HIGHER_IS_BETTER.get(metric, True):
        return selected_score - baseline_score
    return baseline_score - selected_score


def _pooled_permutation_p(
    selected_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    metric: str,
    seed: int,
    n_permutations: int,
) -> float:
    """One-sided permutation p-value over pooled fold scores.

    Under the null the selected and baseline fold scores are exchangeable, so
    the observed mean difference is compared with relabelled pools. When raw
    out-of-fold predictions are unavailable this is the permutation test.
    """
    higher = HIGHER_IS_BETTER.get(metric, True)
    selected = np.asarray(selected_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if selected.size == 0 or baseline.size == 0:
        return float("nan")
    if not higher:
        selected = -selected
        baseline = -baseline
    observed = float(np.mean(selected) - np.mean(baseline))
    pooled = np.concatenate([selected, baseline])
    count_selected = selected.size
    total = pooled.size
    if count_selected >= total:
        return 1.0
    exact = math.comb(total, count_selected)
    if exact <= _MAX_EXACT_PERMUTATIONS:
        count = 0
        total_sum = float(np.sum(pooled))
        for indices in combinations(range(total), count_selected):
            chosen = float(np.sum(pooled[list(indices)]))
            remaining = total - count_selected
            delta = chosen / count_selected - (total_sum - chosen) / remaining
            if delta >= observed - 1e-12:
                count += 1
        return (count + 1) / (exact + 1)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(max(1, n_permutations)):
        permuted = rng.permutation(pooled)
        delta = float(
            np.mean(permuted[:count_selected])
            - np.mean(permuted[count_selected:])
        )
        if delta >= observed - 1e-12:
            count += 1
    return (count + 1) / (max(1, n_permutations) + 1)


def _raw_permutation_p(
    metric: str,
    actuals: Sequence[float],
    predictions: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    seed: int,
    n_permutations: int,
) -> float:
    if not _metric_supported_for_permutation(metric):
        return float("nan")
    y_true = np.asarray(actuals, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim != 2:
        y_pred = y_pred.ravel()
    observed = metric_value(metric, y_true, y_pred)
    if not _finite(observed):
        return float("nan")
    higher = HIGHER_IS_BETTER.get(metric, True)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(max(1, n_permutations)):
        shuffled = rng.permutation(y_true)
        value = metric_value(metric, shuffled, y_pred)
        if not _finite(value):
            continue
        if (higher and value >= observed) or (not higher and value <= observed):
            count += 1
    return (count + 1) / (max(1, n_permutations) + 1)


def _non_finite_metric_values(model_search: ModelSearchResult) -> list[str]:
    names: list[str] = []
    candidates = [
        item
        for item in (model_search.selected, model_search.baseline)
        if item is not None
    ]
    for candidate in candidates:
        for name, value in candidate.metrics.items():
            if not _finite(value):
                names.append(f"{candidate.spec.family}:{name}")
        for name, value in candidate.metric_std.items():
            if not _finite(value):
                names.append(f"{candidate.spec.family}:std:{name}")
        for fold in candidate.folds:
            for name, value in fold.metrics.items():
                if not _finite(value):
                    names.append(f"{candidate.spec.family}:fold{fold.fold}:{name}")
            for name, value in fold.baseline_metrics.items():
                if not _finite(value):
                    names.append(
                        f"{candidate.spec.family}:fold{fold.fold}:baseline:{name}"
                    )
    return names


# ---------------------------------------------------------------------------
# Target gates
# ---------------------------------------------------------------------------


def in_slice_supported(context: ValidationContext) -> tuple[bool, str]:
    if context.model_search is None:
        return False, "model_search_unavailable"
    kind = context.resolved_target_kind
    if not kind:
        return False, "target_kind_unknown"
    if kind == "constant":
        return False, "target_kind:constant"
    if kind not in IN_SLICE_TARGET_KINDS:
        return False, f"target_kind:{kind}"
    return True, ""


def _looks_like_return_target(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in _RETURN_TARGET_HINTS)


def _series_contract_reason(values: Sequence[float] | None) -> str | None:
    if values is None:
        return "target_values_unavailable"
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 20:
        return "insufficient_rows"
    if not np.all(np.isfinite(array)):
        return "non_finite_target"
    if int(np.unique(array).size) < 3:
        return "too_few_distinct_values"
    if float(np.std(array, ddof=1)) == 0.0:
        return "constant_target"
    return None


def target_values_from_context(context: ValidationContext) -> list[float] | None:
    if context.target_values is not None:
        return [float(value) for value in context.target_values]
    if (
        context.artifact is None
        or context.run_dir is None
        or not context.target
    ):
        return None
    stored_path = getattr(context.artifact.dataset, "stored_path", None)
    if not stored_path:
        return None
    dataset_path = Path(context.run_dir) / stored_path
    if not dataset_path.is_file():
        return None
    try:
        import pandas as pd

        frame = pd.read_csv(dataset_path, usecols=[context.target])
    except (OSError, ValueError, ImportError):
        return None
    series = pd.to_numeric(frame[context.target], errors="coerce")
    return [float(value) for value in series.dropna().tolist()]


def return_series_applicable(
    *,
    target_kind: str | None,
    target_name: str | None,
    values: Sequence[float] | None,
    explicit: bool | None = None,
) -> tuple[bool, str]:
    """Gate the strategy engine to a return-like continuous series."""
    kind = (target_kind or "").lower()
    if kind in UNSUPPORTED_ENGINE_TARGET_KINDS:
        return False, f"target_kind:{kind}"
    if kind not in {"continuous", "time_series"}:
        return False, f"target_kind:{kind or 'unknown'}"
    if explicit is False:
        return False, "return_series_disabled"
    reason = _series_contract_reason(values)
    if reason is not None:
        if explicit is True and values is None:
            return True, "explicit_return_series"
        return False, reason
    if explicit is True:
        return True, ""
    if _looks_like_return_target(target_name):
        return True, ""
    return False, "not_return_like"


def stats_engine_supported(context: ValidationContext) -> tuple[bool, str]:
    return return_series_applicable(
        target_kind=context.resolved_target_kind,
        target_name=context.target,
        values=target_values_from_context(context),
        explicit=context.return_series,
    )


# ---------------------------------------------------------------------------
# In-slice backend
# ---------------------------------------------------------------------------


def validate_in_slice(context: ValidationContext) -> StatisticalValidationResult:
    """Validate a selected model against its baseline using fold-level data."""
    supported, reason = in_slice_supported(context)
    if not supported:
        return insufficient_result(
            context.hypothesis_id,
            reason,
            backend="in_slice",
            target=context.target,
            target_kind=context.resolved_target_kind,
            warnings=context.warnings,
        )

    model_search = context.model_search
    assert model_search is not None
    selected = model_search.selected
    if selected is None:
        return insufficient_result(
            context.hypothesis_id,
            "no_selected_model",
            backend="in_slice",
            target=context.target,
            target_kind=context.resolved_target_kind,
            warnings=context.warnings,
        )

    target_kind = context.resolved_target_kind
    metric = str(model_search.metric)
    warnings: list[str] = list(context.warnings) + list(model_search.warnings)

    non_finite = _non_finite_metric_values(model_search)
    if non_finite:
        return _failed_result(
            context,
            reason="non_finite_metrics: " + ", ".join(sorted(set(non_finite))),
            target_kind=target_kind,
            metric=metric,
            warnings=warnings,
        )

    folds = selected.folds
    if not folds:
        return _failed_result(
            context,
            reason="no_evaluation_split: the selected model has no evaluation folds",
            target_kind=target_kind,
            metric=metric,
            warnings=warnings,
        )
    bad_splits = [
        fold.fold
        for fold in folds
        if fold.train_rows <= 0 or fold.test_rows <= 0
    ]
    if bad_splits:
        return _failed_result(
            context,
            reason="no_evaluation_split: empty train/test rows in folds "
            + ", ".join(str(index) for index in bad_splits),
            target_kind=target_kind,
            metric=metric,
            warnings=warnings,
        )

    selected_scores = _fold_scores(selected, metric)
    baseline_scores = _baseline_fold_scores(selected, metric)
    selected_value = _mean(selected_scores) if selected_scores else None
    baseline_value = _mean(baseline_scores) if baseline_scores else None
    if selected_value is None:
        aggregate = selected.metrics.get(metric)
        if aggregate is not None and _finite(aggregate):
            selected_value = float(aggregate)
    if baseline_value is None and model_search.baseline is not None:
        aggregate = model_search.baseline.metrics.get(metric)
        if aggregate is not None and _finite(aggregate):
            baseline_value = float(aggregate)
    if selected_value is None:
        return insufficient_result(
            context.hypothesis_id,
            f"metric_unavailable:{metric}",
            backend="in_slice",
            target=context.target,
            target_kind=target_kind,
            warnings=warnings,
        )
    if baseline_value is None:
        return insufficient_result(
            context.hypothesis_id,
            f"baseline_metric_unavailable:{metric}",
            backend="in_slice",
            target=context.target,
            target_kind=target_kind,
            warnings=warnings,
        )

    delta = _delta(metric, selected_value, baseline_value)
    fold_deltas: list[float] = []
    for fold in folds:
        selected_score = fold.metrics.get(metric)
        baseline_score = fold.baseline_metrics.get(metric)
        if (
            selected_score is not None
            and baseline_score is not None
            and _finite(selected_score)
            and _finite(baseline_score)
        ):
            fold_deltas.append(
                _delta(metric, float(selected_score), float(baseline_score))
            )
    sign_consistency = (
        max(
            sum(1 for value in fold_deltas if value > 0.0),
            sum(1 for value in fold_deltas if value < 0.0),
        )
        / len(fold_deltas)
        if fold_deltas
        else 0.0
    )
    stable = bool(fold_deltas) and all(value > 0.0 for value in fold_deltas)
    metric_std = selected.metric_std.get(metric)
    if metric_std is None and selected_scores:
        metric_std = (
            float(np.std(np.asarray(selected_scores, dtype=float), ddof=1))
            if len(selected_scores) > 1
            else 0.0
        )
    coefficient_signs = {
        str(name): ("positive" if value > 0 else "negative" if value < 0 else "zero")
        for name, value in selected.coefficients.items()
    }

    permutation_source: str | None = None
    p_value: float | None = None
    n_permutations = 0
    if (
        context.predictions is not None
        and context.actuals is not None
        and _metric_supported_for_permutation(metric)
    ):
        p_value = _raw_permutation_p(
            metric,
            context.actuals,
            context.predictions,
            seed=context.seed,
            n_permutations=max(1, context.n_permutations),
        )
        if _finite(p_value):
            permutation_source = "predictions"
            n_permutations = max(1, context.n_permutations)
    if p_value is None and selected_scores and baseline_scores:
        p_value = _pooled_permutation_p(
            selected_scores,
            baseline_scores,
            metric=metric,
            seed=context.seed,
            n_permutations=max(1, context.n_permutations),
        )
        if _finite(p_value):
            permutation_source = "fold_scores"
            combinations_count = math.comb(
                len(selected_scores) + len(baseline_scores), len(selected_scores)
            )
            n_permutations = (
                combinations_count
                if combinations_count <= _MAX_EXACT_PERMUTATIONS
                else max(1, context.n_permutations)
            )
    if p_value is None or not _finite(p_value):
        p_value = None
        warnings.append(
            "permutation_unavailable: no out-of-fold predictions or paired "
            "fold metric scores were available"
        )

    diagnostics = list(selected.diagnostics)
    data_diagnostics, diagnostic_warnings = _target_diagnostics(
        target_kind, context.actuals, context.predictions
    )
    diagnostics.extend(data_diagnostics)
    warnings.extend(diagnostic_warnings)
    failing = [
        item
        for item in diagnostics
        if item.status == "fail"
        or (
            "leak" in f"{item.name} {item.message}".lower()
            and item.status == "warn"
        )
    ]
    leakage = [
        item
        for item in failing
        if "leak" in f"{item.name} {item.message}".lower()
    ]

    result_metrics: dict[str, float] = {
        "selected_score": float(selected_value),
        "baseline_score": float(baseline_value),
        "baseline_delta": float(delta),
        "folds": float(len(folds)),
        "attempts": float(max(1, context.attempts)),
    }
    if metric_std is not None and _finite(metric_std):
        result_metrics["score_std"] = float(metric_std)
    if fold_deltas:
        result_metrics["fold_sign_consistency"] = float(sign_consistency)

    result = StatisticalValidationResult(
        hypothesis_id=context.hypothesis_id,
        backend="in_slice",
        verdict="inconclusive",
        status="partial",
        applicable=True,
        target=context.target,
        target_kind=target_kind,
        metric=metric,
        metrics=result_metrics,
        baseline_metrics={"score": float(baseline_value)},
        baseline_delta=float(delta),
        permutation_p_value=p_value,
        permutation_source=permutation_source,
        n_permutations=n_permutations,
        stability={
            "stable": stable,
            "sign_consistency": sign_consistency,
            "fold_deltas": fold_deltas,
            "coefficient_signs": coefficient_signs,
            "fold_count": len(folds),
        },
        diagnostics=diagnostics,
        attempts=max(1, context.attempts),
        warnings=warnings,
    )

    if leakage:
        result.verdict = "fail"
        result.status = "failed"
        result.reason = "leakage_diagnostic: " + ", ".join(
            item.name for item in leakage
        )
        return result
    if failing:
        result.verdict = "fail"
        result.status = "failed"
        result.reason = "diagnostic_failed: " + ", ".join(
            item.name for item in failing
        )
        return result
    if delta <= context.margin:
        result.verdict = "fail"
        result.status = "failed"
        result.reason = (
            f"no_baseline_improvement: delta {delta:.4f} <= {context.margin:.4f}"
        )
        return result
    if p_value is None:
        result.reason = "permutation_unavailable"
        return result
    if p_value > context.max_p:
        result.verdict = "fail"
        result.status = "failed"
        result.reason = (
            f"permutation_not_significant: p {p_value:.4f} > {context.max_p:.4f}"
        )
        return result
    if not stable:
        result.reason = (
            "unstable_folds: the selected model did not beat the baseline in "
            "every fold"
        )
        return result
    result.verdict = "pass"
    result.status = "passed"
    result.reason = (
        f"baseline improvement {delta:.4f} with permutation p {p_value:.4f} "
        f"over {max(1, context.attempts)} attempt(s)"
    )
    return result


def _failed_result(
    context: ValidationContext,
    *,
    reason: str,
    target_kind: str | None,
    metric: str | None,
    warnings: Sequence[str],
) -> StatisticalValidationResult:
    return StatisticalValidationResult(
        hypothesis_id=context.hypothesis_id,
        backend="in_slice",
        verdict="fail",
        status="failed",
        applicable=True,
        reason=reason,
        target=context.target,
        target_kind=target_kind,
        metric=metric,
        attempts=max(1, context.attempts),
        warnings=list(warnings),
    )
