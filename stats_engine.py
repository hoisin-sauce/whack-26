"""Autonomous validation engine for continuous, classification, and count targets.

Entry points
------------
``validation_engine``
    The original return-series strategy validator. It keeps the strict
    four-layer contract: a continuous, return-like target and real trial
    Sharpe ratios are required. Existing callers are unaffected.

``validate_model``
    General-purpose validator for any model exposed as
    ``(params, X, y) -> predict``. It supports continuous, binary,
    multiclass, and count targets, selects an appropriate scoring rule,
    and marks layers ``not_applicable`` when their assumptions do not hold.

Target inference
----------------
``infer_target_type`` classifies a target vector as ``continuous``,
``binary``, ``multiclass``, ``count``, or ``constant``. Integer-valued
targets with more than ``max_classes`` distinct values are treated as
counts when non-negative; low-cardinality integer targets are treated as
class labels. Pass ``target_type=`` explicitly when a count variable has
few distinct values or a class label set is unusually large.

Method notes
------------
The general validator follows the model-validation literature rather than
forcing return-series machinery onto every target:

* Classification: proper scoring rules (log loss, Brier score) are used to
  score probabilistic predictions; accuracy alone is not a proper scoring
  rule. Discrimination is reported with ROC AUC (binary), balanced
  accuracy, Matthews correlation, and macro F1. Calibration is summarised
  with expected calibration error (ECE), following the reliability-diagram
  approach popularised by Guo et al. (2017). These are the standard tools
  described by Gneiting & Raftery (2007) for probabilistic forecasts.
* Counts: the Poisson deviance is a proper scoring rule for count targets
  (Czado, Gneiting & Held, 2009). Pearson dispersion and zero-inflation
  diagnostics test the Poisson mean-variance assumption (Cameron &
  Trivedi, 2013); overdispersion usually calls for a negative binomial or
  quasi-Poisson model.
* Temporal validation: the holdout is chronological, matching the original
  engine. Permutation tests use ordinary shuffling by default and support a
  circular block shuffle (``block_size > 1``) when observations are
  autocorrelated (Kunsch, 1989; Politis & Romano, 1994).
* Matthews correlation is reported alongside accuracy because it stays
  informative under class imbalance (Chicco & Jurman, 2020).
* Layer 1 (DSR) is return-series-only. For non-continuous targets it is
  reported as ``not_applicable`` rather than silently reusing strategy
  semantics. Multiple-testing correction remains the responsibility of the
  model search that generated the candidate set.

Metric contract
---------------
``metric_func(predictions, actuals) -> float`` must return a finite score.
When it is omitted, the engine selects a proper scoring rule for the target
type (R-squared, log loss, or Poisson deviance). When it is supplied, the
direction is inferred from a recognised loss name (log loss, Brier score,
Poisson deviance, MSE, MAE, RMSE); otherwise pass ``higher_is_better``
explicitly. The default direction is ``higher_is_better=True``.

Layer 2 evaluates held-out predictions, not training predictions. Layer 3 fits
the supplied builder on the chronological training split and scores the test
split. Layer 4 perturbs real model hyperparameters; validation thresholds are
never perturbed.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.stats import kurtosis as calc_kurtosis
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import skew as calc_skew

# Keys that configure the validation engine rather than the model. PSS must
# never perturb these.
ENGINE_PARAM_KEYS = frozenset(
    {
        "min_dsr_limit",
        "max_p_value",
        "train_split_ratio",
        "num_permutations",
        "sensitivity_range",
        "max_performance_drop",
        "max_decay_tolerance",
    }
)

Metric = Callable[[np.ndarray, np.ndarray], float]
ModelBuilder = Callable[[dict[str, Any], np.ndarray, np.ndarray], Callable]


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _failure(layer: str, message: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "passed": False,
        "failed_layer": layer,
        "message": message,
    }
    result.update(extra)
    return result


def r2_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Higher-is-better coefficient of determination."""
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.shape != acts.shape or preds.size == 0:
        return float("nan")
    residual = float(np.sum((acts - preds) ** 2))
    total = float(np.sum((acts - np.mean(acts)) ** 2))
    if not _finite(residual) or not _finite(total):
        return float("nan")
    if total <= 0.0:
        return 0.0
    return 1.0 - residual / total


# ==========================================
# GENERAL TARGET TYPES AND METRICS
# ==========================================
TARGET_CONTINUOUS = "continuous"
TARGET_BINARY = "binary"
TARGET_MULTICLASS = "multiclass"
TARGET_COUNT = "count"
TARGET_CONSTANT = "constant"

LOWER_IS_BETTER_METRICS = frozenset(
    {
        "log_loss",
        "brier_score",
        "poisson_deviance",
        "mean_absolute_error",
        "mean_squared_error",
        "root_mean_squared_error",
    }
)


def infer_target_type(targets: np.ndarray, *, max_classes: int = 10) -> str:
    """Infer the statistical target type from the observed values.

    ``continuous`` targets are non-integer or high-cardinality numeric
    targets. ``binary`` has two distinct values, ``multiclass`` has between
    three and ``max_classes`` distinct values, and ``count`` is a
    non-negative integer target with more than ``max_classes`` distinct
    values. Pass ``target_type=`` to :func:`validate_model` when a count has
    few distinct values or a label set is unusually large.
    """
    if int(max_classes) < 2:
        raise ValueError("max_classes must be at least 2.")
    raw = np.asarray(targets)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("targets must be a non-empty 1D array.")

    if raw.dtype.kind == "b":
        return TARGET_CONSTANT if np.unique(raw).size < 2 else TARGET_BINARY

    try:
        values = raw.astype(float)
    except (TypeError, ValueError):
        distinct = np.unique(raw)
        if distinct.size < 2:
            return TARGET_CONSTANT
        return TARGET_BINARY if distinct.size == 2 else TARGET_MULTICLASS

    if not np.all(np.isfinite(values)):
        raise ValueError("targets must contain only finite values.")
    distinct = np.unique(values)
    if distinct.size < 2:
        return TARGET_CONSTANT
    if distinct.size == 2:
        return TARGET_BINARY

    integer_like = np.all(np.isclose(values, np.round(values), atol=1e-9, rtol=0.0))
    if integer_like:
        if distinct.size <= int(max_classes):
            return TARGET_MULTICLASS
        if np.all(values >= 0.0):
            return TARGET_COUNT
    return TARGET_CONTINUOUS


def _as_finite_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D numeric array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _classes_and_codes(actuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, codes = np.unique(actuals, return_inverse=True)
    return classes, codes


def _looks_like_binary_probabilities(
    predictions: np.ndarray, classes: np.ndarray
) -> bool:
    if classes.size != 2:
        return False
    try:
        values = np.asarray(predictions, dtype=float).ravel()
    except (TypeError, ValueError):
        return False
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        return False
    try:
        class_values = np.asarray(classes, dtype=float)
    except (TypeError, ValueError):
        return True
    rounded = np.round(values, 12)
    class_values = np.round(class_values, 12)
    # If every prediction is itself one of the class values, these are hard
    # labels, not probabilities.
    return not bool(np.all(np.isin(rounded, class_values)))


def _predicted_labels(predictions: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    preds = np.asarray(predictions)
    acts = np.asarray(actuals).ravel()
    classes, _ = _classes_and_codes(acts)
    if preds.ndim == 2:
        if preds.shape[0] != acts.size:
            raise ValueError("predictions and actuals have incompatible lengths.")
        if preds.shape[1] != classes.size:
            raise ValueError(
                "2D predictions must have one column per observed class."
            )
        return classes[np.argmax(preds, axis=1)]
    if preds.ndim != 1 or preds.size != acts.size:
        raise ValueError("predictions and actuals must have the same 1D length.")
    if classes.size == 2 and _looks_like_binary_probabilities(preds, classes):
        values = np.asarray(preds, dtype=float)
        return np.where(values >= 0.5, classes[1], classes[0])
    return preds


def accuracy_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of correctly classified labels."""
    try:
        pred_labels = _predicted_labels(predictions, actuals)
    except (TypeError, ValueError):
        return float("nan")
    return float(np.mean(pred_labels == np.asarray(actuals).ravel()))


def balanced_accuracy_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Macro-averaged recall, useful for imbalanced classes."""
    try:
        pred_labels = _predicted_labels(predictions, actuals)
    except (TypeError, ValueError):
        return float("nan")
    acts = np.asarray(actuals).ravel()
    classes, codes = _classes_and_codes(acts)
    recalls: list[float] = []
    for index in range(classes.size):
        mask = codes == index
        if np.any(mask):
            recalls.append(float(np.mean(pred_labels[mask] == classes[index])))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores."""
    try:
        pred_labels = _predicted_labels(predictions, actuals)
    except (TypeError, ValueError):
        return float("nan")
    acts = np.asarray(actuals).ravel()
    classes, _ = _classes_and_codes(acts)
    scores: list[float] = []
    for label in classes:
        true_positive = float(np.sum((pred_labels == label) & (acts == label)))
        false_positive = float(np.sum((pred_labels == label) & (acts != label)))
        false_negative = float(np.sum((pred_labels != label) & (acts == label)))
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def mcc_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Matthews correlation coefficient for binary or multiclass labels."""
    try:
        pred_labels = _predicted_labels(predictions, actuals)
    except (TypeError, ValueError):
        return float("nan")
    acts = np.asarray(actuals).ravel()
    classes, true_codes = _classes_and_codes(acts)
    if classes.size < 2:
        return float("nan")
    lookup = {label: index for index, label in enumerate(classes)}
    pred_codes = np.asarray([lookup.get(label, -1) for label in pred_labels])
    size = int(classes.size)
    matrix = np.zeros((size, size), dtype=float)
    for true_code, pred_code in zip(true_codes, pred_codes):
        if pred_code >= 0:
            matrix[true_code, pred_code] += 1.0
    total = float(acts.size)
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


def log_loss(
    probabilities: np.ndarray, actuals: np.ndarray, eps: float = 1e-15
) -> float:
    """Mean negative log-likelihood for probabilistic predictions.

    Binary predictions may be a 1D vector of positive-class probabilities.
    Multiclass predictions must be an ``(n, n_classes)`` probability matrix.
    """
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actuals).ravel()
    if acts.size == 0 or eps <= 0.0:
        return float("nan")
    classes, codes = _classes_and_codes(acts)
    if classes.size < 2:
        return float("nan")
    try:
        if probs.ndim == 2:
            if probs.shape != (acts.size, classes.size):
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            true_probs = probs[np.arange(acts.size), codes]
        elif probs.ndim == 1:
            if classes.size != 2 or probs.size != acts.size:
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            positive = probs
            true_probs = np.where(codes == 1, positive, 1.0 - positive)
        else:
            return float("nan")
    except (IndexError, TypeError, ValueError):
        return float("nan")
    clipped = np.clip(np.asarray(true_probs, dtype=float), eps, 1.0 - eps)
    if not np.all(np.isfinite(clipped)):
        return float("nan")
    return float(-np.mean(np.log(clipped)))


def brier_score(probabilities: np.ndarray, actuals: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actuals).ravel()
    if acts.size == 0:
        return float("nan")
    classes, codes = _classes_and_codes(acts)
    if classes.size < 2:
        return float("nan")
    try:
        if probs.ndim == 1:
            if classes.size != 2 or probs.size != acts.size:
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            outcome = (codes == 1).astype(float)
            return float(np.mean((probs - outcome) ** 2))
        if probs.ndim == 2:
            if probs.shape != (acts.size, classes.size):
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            if classes.size == 2:
                outcome = (codes == 1).astype(float)
                return float(np.mean((probs[:, 1] - outcome) ** 2))
            one_hot = np.eye(classes.size)[codes]
            return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    except (IndexError, TypeError, ValueError):
        return float("nan")
    return float("nan")


def roc_auc_score(scores: np.ndarray, actuals: np.ndarray) -> float:
    """Rank-based ROC AUC for a binary target (ties handled by mid-ranks)."""
    values = np.asarray(scores, dtype=float)
    acts = np.asarray(actuals).ravel()
    classes, _ = _classes_and_codes(acts)
    if classes.size != 2 or acts.size == 0:
        return float("nan")
    if values.ndim == 2:
        if values.shape[0] != acts.size:
            return float("nan")
        if values.shape[1] == 2:
            values = values[:, 1]
        elif values.shape[1] == 1:
            values = values[:, 0]
        else:
            return float("nan")
    elif values.ndim == 1:
        if values.size != acts.size:
            return float("nan")
    else:
        return float("nan")
    if not np.all(np.isfinite(values)):
        return float("nan")
    positive = acts == classes[1]
    num_positive = int(np.sum(positive))
    num_negative = int(acts.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    auc = (
        float(np.sum(ranks[positive]))
        - num_positive * (num_positive + 1) / 2.0
    ) / (num_positive * num_negative)
    return float(auc) if _finite(auc) else float("nan")


def mean_absolute_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return float("nan")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return float("nan")
    return float(np.mean(np.abs(acts - preds)))


def mean_squared_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return float("nan")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return float("nan")
    return float(np.mean((acts - preds) ** 2))


def root_mean_squared_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    mse = mean_squared_error(predictions, actuals)
    return float(math.sqrt(mse)) if _finite(mse) and mse >= 0.0 else float("nan")


def poisson_deviance(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Poisson deviance for non-negative count targets.

    Lower is better. Predictions are the fitted Poisson means and must be
    strictly positive; observed counts may be zero.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return float("nan")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return float("nan")
    if np.any(acts < 0.0) or np.any(preds <= 0.0):
        return float("nan")
    log_terms = np.zeros_like(acts)
    positive = acts > 0.0
    log_terms[positive] = acts[positive] * np.log(acts[positive] / preds[positive])
    deviance = 2.0 * np.sum(log_terms - (acts - preds)) / acts.size
    return float(deviance) if _finite(deviance) else float("nan")


def expected_calibration_error(
    probabilities: np.ndarray, actuals: np.ndarray, n_bins: int = 10
) -> float:
    """Expected calibration error over equally spaced confidence bins.

    Binary inputs may be 1D positive-class probabilities. Multiclass inputs
    must be an ``(n, n_classes)`` matrix; the top-label confidence is used.
    """
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2.")
    probs = np.asarray(probabilities, dtype=float)
    acts = np.asarray(actuals).ravel()
    if acts.size == 0:
        return float("nan")
    classes, codes = _classes_and_codes(acts)
    if classes.size < 2:
        return float("nan")
    try:
        if probs.ndim == 1:
            if classes.size != 2 or probs.size != acts.size:
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            predicted_codes = (probs >= 0.5).astype(int)
            confidence = np.maximum(probs, 1.0 - probs)
        elif probs.ndim == 2:
            if probs.shape != (acts.size, classes.size):
                return float("nan")
            if not np.all(np.isfinite(probs)):
                return float("nan")
            predicted_codes = np.argmax(probs, axis=1)
            confidence = np.max(probs, axis=1)
        else:
            return float("nan")
    except (TypeError, ValueError):
        return float("nan")
    correct = (predicted_codes == codes).astype(float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    total = float(acts.size)
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if np.any(mask):
            accuracy = float(np.mean(correct[mask]))
            mean_confidence = float(np.mean(confidence[mask]))
            ece += float(np.sum(mask)) / total * abs(accuracy - mean_confidence)
    return float(ece) if _finite(ece) else float("nan")


def pearson_dispersion(
    predictions: np.ndarray, actuals: np.ndarray, *, n_params: int = 1
) -> float:
    """Pearson chi-square dispersion estimate for count predictions.

    Values near 1.0 are consistent with Poisson variance. Values materially
    above 1.0 indicate overdispersion; values below 1.0 indicate
    underdispersion.
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


def poisson_dispersion_test(
    predictions: np.ndarray, actuals: np.ndarray, *, n_params: int = 1
) -> dict[str, Any]:
    """Pearson dispersion statistic with one-sided p-values."""
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return {"dispersion": float("nan"), "p_value_overdispersion": float("nan")}
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return {"dispersion": float("nan"), "p_value_overdispersion": float("nan")}
    if np.any(acts < 0.0) or np.any(preds <= 0.0):
        return {"dispersion": float("nan"), "p_value_overdispersion": float("nan")}
    degrees = acts.size - int(n_params)
    if degrees <= 0:
        return {"dispersion": float("nan"), "p_value_overdispersion": float("nan")}
    statistic = float(np.sum((acts - preds) ** 2 / preds))
    return {
        "dispersion": float(statistic / degrees),
        "statistic": statistic,
        "degrees_of_freedom": int(degrees),
        "p_value_overdispersion": float(chi2.sf(statistic, degrees)),
        "p_value_underdispersion": float(chi2.cdf(statistic, degrees)),
    }


def zero_inflation_ratio(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Observed zero rate divided by the Poisson-implied zero rate."""
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.size != acts.size or preds.size == 0:
        return float("nan")
    if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(acts)):
        return float("nan")
    if np.any(acts < 0.0) or np.any(preds < 0.0):
        return float("nan")
    expected = float(np.mean(np.exp(-preds)))
    if expected <= 0.0:
        return float("nan")
    return float(np.mean(acts == 0.0) / expected)


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    values = np.asarray(returns, dtype=float).ravel()
    if values.size < 2 or not _finite(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    if not np.all(np.isfinite(values)):
        return float("nan")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if not _finite(mean) or not _finite(std) or std == 0.0:
        return float("nan")
    return mean / std * math.sqrt(periods_per_year)


def build_linear_model(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ridge-regularised linear model and return a predict callable.

    The builder signature is ``(params, X, y)`` so the engine owns the split
    and the caller never has to pre-train a model.
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float).ravel()
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero number of rows")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("X and y must contain only finite values")
    ridge = float(params.get("ridge", 0.0))
    if not _finite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative number")
    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0  # never penalise the intercept
    gram = design.T @ design + penalty
    coefficients, *_ = np.linalg.lstsq(gram, design.T @ targets, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("linear model produced non-finite coefficients")

    def predict(new_X: np.ndarray) -> np.ndarray:
        values = np.asarray(new_X, dtype=float)
        if values.ndim != 2 or values.shape[1] != features.shape[1]:
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        return np.column_stack([np.ones(values.shape[0]), values]) @ coefficients

    return predict


def build_logistic_model(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ridge-regularised binary logistic model.

    The returned callable predicts the probability of the second sorted class
    label, matching the binary convention used by :func:`log_loss`.
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y).ravel()
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero number of rows")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    classes = np.unique(targets)
    if classes.size != 2:
        raise ValueError("binary logistic regression requires exactly two classes")

    ridge = float(params.get("ridge", 1e-3))
    max_iter = int(params.get("max_iter", 100))
    tolerance = float(params.get("tol", 1e-8))
    if not _finite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative number")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    binary_target = (targets == classes[1]).astype(float)
    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.zeros(design.shape[1], dtype=float)

    for _ in range(max_iter):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        working = linear + (binary_target - probabilities) / weights
        gram = design.T @ (weights[:, None] * design) + penalty
        rhs = design.T @ (weights * working)
        updated, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(updated)):
            raise ValueError("logistic model produced non-finite coefficients")
        change = float(np.max(np.abs(updated - coefficients)))
        coefficients = updated
        if change <= tolerance:
            break

    def predict(new_X: np.ndarray) -> np.ndarray:
        values = np.asarray(new_X, dtype=float)
        if values.ndim != 2 or values.shape[1] != features.shape[1]:
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        linear = np.clip(
            np.column_stack([np.ones(values.shape[0]), values]) @ coefficients,
            -30.0,
            30.0,
        )
        return 1.0 / (1.0 + np.exp(-linear))

    return predict


def build_softmax_model(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ridge-regularised multinomial logistic model.

    The returned callable predicts an ``(n, n_classes)`` probability matrix
    with columns in sorted class-label order.
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y).ravel()
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero number of rows")
    if not np.all(np.isfinite(features)):
        raise ValueError("X must contain only finite values")
    classes, codes = np.unique(targets, return_inverse=True)
    num_classes = classes.size
    if num_classes < 2:
        raise ValueError("softmax regression requires at least two classes")

    ridge = float(params.get("ridge", 1e-3))
    max_iter = int(params.get("max_iter", 500))
    tolerance = float(params.get("tol", 1e-8))
    if not _finite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative number")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    regularization = max(ridge, 1e-9)

    design = np.column_stack([np.ones(features.shape[0]), features])
    num_rows, num_columns = design.shape
    one_hot = np.eye(num_classes)[codes]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat.reshape(num_columns, num_classes)
        logits = design @ coefficients
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        nll = -float(
            np.mean(
                np.log(
                    np.clip(probabilities[np.arange(num_rows), codes], 1e-15, 1.0)
                )
            )
        )
        penalty = 0.5 * regularization * float(
            np.sum(coefficients[1:] ** 2)
        )
        gradient = design.T @ (probabilities - one_hot) / num_rows
        gradient[1:] += regularization * coefficients[1:]
        return nll + penalty, gradient.ravel()

    result = minimize(
        objective,
        np.zeros(num_columns * num_classes, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    coefficients = np.asarray(result.x, dtype=float).reshape(
        num_columns, num_classes
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("softmax model produced non-finite coefficients")

    def predict(new_X: np.ndarray) -> np.ndarray:
        values = np.asarray(new_X, dtype=float)
        if values.ndim != 2 or values.shape[1] != features.shape[1]:
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        logits = np.column_stack([np.ones(values.shape[0]), values]) @ coefficients
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    return predict


def build_poisson_model(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a ridge-regularised Poisson regression (log link).

    The returned callable predicts strictly positive Poisson means.
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float).ravel()
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be 2D and y must be 1D")
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError("X and y must have the same non-zero number of rows")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("X and y must contain only finite values")
    if np.any(targets < 0.0):
        raise ValueError("Poisson regression requires non-negative targets")

    ridge = float(params.get("ridge", 1e-3))
    max_iter = int(params.get("max_iter", 100))
    tolerance = float(params.get("tol", 1e-8))
    if not _finite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be a finite non-negative number")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.zeros(design.shape[1], dtype=float)
    coefficients[0] = math.log(max(float(np.mean(targets)), 1e-6))

    for _ in range(max_iter):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        means = np.exp(linear)
        working = linear + (targets - means) / means
        gram = design.T @ (means[:, None] * design) + penalty
        rhs = design.T @ (means * working)
        updated, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(updated)):
            raise ValueError("Poisson model produced non-finite coefficients")
        change = float(np.max(np.abs(updated - coefficients)))
        coefficients = updated
        if change <= tolerance:
            break

    def predict(new_X: np.ndarray) -> np.ndarray:
        values = np.asarray(new_X, dtype=float)
        if values.ndim != 2 or values.shape[1] != features.shape[1]:
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        linear = np.clip(
            np.column_stack([np.ones(values.shape[0]), values]) @ coefficients,
            -30.0,
            30.0,
        )
        return np.exp(linear)

    return predict


def trial_sharpes(
    features: np.ndarray,
    target: np.ndarray,
    periods_per_year: float,
    model_predict: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[list[float], np.ndarray]:
    """Return real trial Sharpes and the winning strategy's return series.

    Each numeric feature contributes one momentum-style trial: position is the
    sign of the centred feature, and the strategy return is position * target.
    When ``model_predict`` is supplied, the fitted model contributes one more
    trial using the sign of its prediction.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).ravel()
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("features and target shapes do not match")
    sharpes: list[float] = []
    candidates: list[np.ndarray] = []
    for index in range(X.shape[1]):
        column = X[:, index]
        if not np.all(np.isfinite(column)):
            continue
        centred = column - float(np.mean(column))
        signal = np.sign(centred)
        strategy = signal * y
        sharpe = sharpe_ratio(strategy, periods_per_year)
        if _finite(sharpe):
            sharpes.append(sharpe)
            candidates.append(strategy)
    if model_predict is not None:
        predicted = np.asarray(model_predict(X), dtype=float).ravel()
        if predicted.shape == y.shape and np.all(np.isfinite(predicted)):
            strategy = np.sign(predicted) * y
            sharpe = sharpe_ratio(strategy, periods_per_year)
            if _finite(sharpe):
                sharpes.append(sharpe)
                candidates.append(strategy)
    if not sharpes:
        return [], np.array([], dtype=float)
    best = int(np.argmax(sharpes))
    return sharpes, candidates[best]


# ==========================================
# LAYER 1: Deflated Sharpe Ratio (DSR)
# ==========================================
def dsr_dynamic(
    winning_return_series: np.ndarray,
    all_trial_sharpes: list,
    periods_per_year: int,
) -> float:
    """DSR adjusted for annualisation and multiple testing.

    Returns 0.0 for insufficient or non-finite inputs. A valid DSR requires at
    least two real trials.
    """
    series = np.asarray(winning_return_series, dtype=float).ravel()
    trials = np.asarray(all_trial_sharpes, dtype=float).ravel()
    if series.size < 2 or trials.size < 2:
        return 0.0
    if not _finite(periods_per_year) or periods_per_year <= 0:
        return 0.0
    if not np.all(np.isfinite(series)) or not np.all(np.isfinite(trials)):
        return 0.0

    mean_val = float(np.mean(series))
    std_val = float(np.std(series, ddof=1))
    if not _finite(mean_val) or not _finite(std_val) or std_val == 0.0:
        return 0.0

    observed_sr = (mean_val / std_val) * math.sqrt(periods_per_year)
    if not _finite(observed_sr):
        return 0.0
    num_obs = series.size

    return_skew = float(calc_skew(series))
    return_kurt = float(calc_kurtosis(series, fisher=False))
    if not _finite(return_skew) or not _finite(return_kurt):
        return 0.0

    sr_std = float(np.std(trials, ddof=1))
    if not _finite(sr_std) or sr_std == 0.0:
        sr_std = 1e-6

    gamma = 0.5772156649  # Euler-Mascheroni constant
    num_trials = trials.size
    z_1 = norm.ppf(1.0 - 1.0 / num_trials)
    z_2 = norm.ppf(1.0 - 1.0 / (num_trials * np.e))
    expected_max_sr = sr_std * ((1.0 - gamma) * z_1 + gamma * z_2)

    sr_variance = (
        1.0
        - return_skew * observed_sr
        + ((return_kurt - 1.0) / 4.0) * (observed_sr**2)
    ) / max(num_obs - 1, 1)
    sr_std_error = math.sqrt(max(sr_variance, 1e-9))
    if not _finite(sr_std_error) or sr_std_error == 0.0:
        return 0.0

    dsr_z_score = (observed_sr - expected_max_sr) / sr_std_error
    if not _finite(dsr_z_score):
        return 0.0
    score = float(norm.cdf(dsr_z_score))
    return score if _finite(score) else 0.0


# ==========================================
# LAYER 2: Monte Carlo Permutation Testing
# ==========================================
def _permute_series(
    values: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle values, optionally in circular blocks to preserve dependence."""
    if block_size <= 1:
        permuted = values.copy()
        rng.shuffle(permuted)
        return permuted
    size = values.size
    block_size = min(int(block_size), size)
    num_blocks = int(math.ceil(size / block_size))
    starts = rng.integers(0, size, size=num_blocks)
    pieces = [
        values[(np.arange(start, start + block_size) % size)]
        for start in starts
    ]
    return np.concatenate(pieces)[:size]


def monte_carlo(
    predictions: np.ndarray,
    actuals: np.ndarray,
    num_permutations: int,
    metric_func: Metric | None = None,
    *,
    higher_is_better: bool = True,
    seed: int = 0,
    block_size: int = 1,
) -> float:
    """Permutation p-value for held-out predictions.

    ``metric_func`` must return a finite score. Non-finite permutation scores
    are skipped; if fewer than half of the permutations are finite, the result
    is ``nan`` and the engine fails the layer.

    Predictions may be 1D labels/scores or a 2D probability matrix. When
    observations are autocorrelated, ``block_size > 1`` performs a circular
    block shuffle instead of an i.i.d. permutation.
    """
    preds = np.asarray(predictions)
    acts = np.asarray(actuals)
    if acts.ndim != 1 or acts.size == 0:
        raise ValueError("actuals must be a non-empty 1D array.")
    if preds.ndim not in (1, 2) or preds.shape[0] != acts.size:
        raise ValueError("Predictions and actuals must have matching lengths.")
    if metric_func is None and preds.ndim == 2:
        raise ValueError("metric_func is required for 2D predictions.")
    if not _finite(num_permutations) or int(num_permutations) < 1:
        raise ValueError("num_permutations must be a positive integer.")
    if not _finite(block_size) or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer.")
    num_permutations = int(num_permutations)
    block_size = int(block_size)
    metric: Metric = metric_func or (
        lambda p, a: float(np.mean(p == a))
    )
    real_score = float(metric(preds, acts))
    if not _finite(real_score):
        return float("nan")

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(num_permutations):
        permuted = _permute_series(acts, block_size, rng)
        score = float(metric(preds, permuted))
        if _finite(score):
            scores.append(score)
    if len(scores) < max(1, num_permutations // 2):
        return float("nan")

    values = np.asarray(scores, dtype=float)
    if higher_is_better:
        count_better_or_equal = int(np.sum(values >= real_score))
    else:
        count_better_or_equal = int(np.sum(values <= real_score))
    return float((count_better_or_equal + 1) / (values.size + 1))


# ==========================================
# LAYER 3: Chronological OOS Decay Testing
# ==========================================
def oos_decay(
    base_model_builder: ModelBuilder,
    features: np.ndarray,
    targets: np.ndarray,
    base_params: dict,
    train_split_ratio: float,
    metric_func: Metric | None = None,
    *,
    higher_is_better: bool = True,
    allow_non_positive_baseline: bool = False,
    clamp_tolerance: bool = True,
) -> dict:
    """Fit on the chronological training split and score the test split.

    Returns a result dict. ``tolerance_clamped`` is true when the requested
    ``max_decay_tolerance`` was outside the supported [0.15, 0.50] range.

    The original return-series engine requires a positive in-sample score.
    Set ``allow_non_positive_baseline=True`` for metrics such as R-squared
    that may legitimately be zero or negative; the decay denominator then
    uses the larger absolute in-sample or out-of-sample score. Set
    ``clamp_tolerance=False`` to honour a tolerance outside [0.15, 0.50],
    as the general model validator does.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets).ravel()
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "error": "Features and targets have incompatible shapes.",
        }
    n_samples = len(X)
    if n_samples < 20:
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "error": "Dataset too small.",
        }
    if not _finite(train_split_ratio) or not (0.0 < float(train_split_ratio) < 1.0):
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "error": "train_split_ratio must be between 0 and 1.",
        }

    try:
        numeric_targets = np.asarray(y, dtype=float)
        target_variance = (
            float(np.var(numeric_targets))
            if np.all(np.isfinite(numeric_targets))
            else 0.0
        )
    except (TypeError, ValueError):
        # Class labels may be strings; they have no meaningful variance.
        target_variance = 0.0
    size_penalty_factor = min(0.2, 1000 / n_samples)
    raw_tolerance = base_params.get(
        "max_decay_tolerance",
        0.25 + (target_variance * 0.1) + size_penalty_factor,
    )
    try:
        raw_tolerance = float(raw_tolerance)
    except (TypeError, ValueError):
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "error": "max_decay_tolerance must be a number.",
        }
    if not _finite(raw_tolerance):
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "error": "max_decay_tolerance must be finite.",
        }
    dynamic_tolerance = float(np.clip(raw_tolerance, 0.15, 0.50))
    tolerance_clamped = not math.isclose(
        dynamic_tolerance, raw_tolerance, rel_tol=1e-9, abs_tol=1e-12
    )
    if not clamp_tolerance:
        dynamic_tolerance = max(raw_tolerance, 0.0)
        tolerance_clamped = False

    split_index = int(n_samples * float(train_split_ratio))
    if split_index <= 0 or split_index >= n_samples:
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": "Insufficient OOS test data.",
        }
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    metric: Metric = metric_func or (
        lambda p, a: float(np.mean(p == a))
    )
    try:
        model = base_model_builder(base_params, X_train, y_train)
        train_preds = np.asarray(model(X_train), dtype=float)
        test_preds = np.asarray(model(X_test), dtype=float)
        if train_preds.ndim == 1:
            train_preds = train_preds.ravel()
        if test_preds.ndim == 1:
            test_preds = test_preds.ravel()
    except Exception as exc:  # noqa: BLE001 - report builder failures
        return {
            "success": False,
            "oos_score": 0.0,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": f"Model training/prediction failed: {exc}",
        }

    is_score = float(metric(train_preds, y_train))
    oos_score = float(metric(test_preds, y_test))
    if not _finite(is_score) or not _finite(oos_score):
        return {
            "success": False,
            "oos_score": oos_score if _finite(oos_score) else 0.0,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": "In-sample or OOS score is non-finite.",
        }
    if is_score <= 0.0 and not allow_non_positive_baseline:
        return {
            "success": False,
            "oos_score": oos_score,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": "In-sample baseline score is non-positive.",
        }

    if allow_non_positive_baseline:
        denominator = max(abs(is_score), abs(oos_score), 1e-12)
    else:
        denominator = abs(is_score)
    if higher_is_better:
        decay_rate = (is_score - oos_score) / denominator
    else:
        decay_rate = (oos_score - is_score) / denominator
    if not _finite(decay_rate):
        return {
            "success": False,
            "oos_score": oos_score,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": "Decay rate is non-finite.",
        }

    result = {
        "success": True,
        "oos_score": oos_score,
        "decay_rate": decay_rate,
        "tolerance": dynamic_tolerance,
        "tolerance_clamped": tolerance_clamped,
    }
    if decay_rate > dynamic_tolerance:
        result["success"] = False
        result["error"] = (
            f"OOS decay ({decay_rate:.1%}) exceeded limit "
            f"({dynamic_tolerance:.1%})."
        )
    return result


# ==========================================
# LAYER 4: Parameter Sensitivity Testing (PSS)
# ==========================================
def pss(
    base_model_builder: ModelBuilder,
    features: np.ndarray,
    targets: np.ndarray,
    base_params: dict,
    sensitivity_range: float,
    max_performance_drop: float,
    metric_func: Metric | None = None,
    *,
    higher_is_better: bool = True,
    allow_non_positive_baseline: bool = False,
) -> tuple[bool, str]:
    """Perturb real model hyperparameters and check score stability.

    Set ``allow_non_positive_baseline=True`` for metrics such as R-squared
    that may legitimately be zero or negative.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets).ravel()
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        return False, "Features and targets have incompatible shapes."
    if not _finite(sensitivity_range) or sensitivity_range <= 0.0:
        return False, "sensitivity_range must be positive."
    if not _finite(max_performance_drop) or max_performance_drop < 0.0:
        return False, "max_performance_drop must be non-negative."

    metric: Metric = metric_func or (
        lambda p, a: float(np.mean(p == a))
    )
    try:
        base_model = base_model_builder(base_params, X, y)
        base_score = float(metric(base_model(X), y))
    except Exception as exc:  # noqa: BLE001 - report builder failures
        return False, f"Baseline model construction failed: {exc}"
    if not _finite(base_score):
        return False, "Baseline model score is non-finite."
    if base_score <= 0.0 and not allow_non_positive_baseline:
        return False, "Baseline model score is non-positive or non-finite."

    total_trials = 0
    successful_trials = 0
    failed_trials = 0

    for param_name, base_val in base_params.items():
        if param_name in ENGINE_PARAM_KEYS:
            continue
        if isinstance(base_val, bool) or not isinstance(base_val, (int, float)):
            continue
        if not _finite(base_val):
            continue

        for sign in (-1, 1):
            perturbed_val: float | int = float(base_val) * (
                1.0 + sign * float(sensitivity_range)
            )
            if isinstance(base_val, int):
                perturbed_val = int(round(perturbed_val))
                if perturbed_val == base_val:
                    perturbed_val += sign

            test_params = dict(base_params)
            test_params[param_name] = perturbed_val
            total_trials += 1
            try:
                variant_model = base_model_builder(test_params, X, y)
                variant_score = float(metric(variant_model(X), y))
            except Exception:  # noqa: BLE001 - count the failed trial
                failed_trials += 1
                continue
            if not _finite(variant_score):
                failed_trials += 1
                continue

            if allow_non_positive_baseline:
                denominator = max(abs(base_score), abs(variant_score), 1e-12)
            else:
                denominator = abs(base_score)
            if higher_is_better:
                drop = (base_score - variant_score) / denominator
            else:
                drop = (variant_score - base_score) / denominator
            if not _finite(drop):
                failed_trials += 1
                continue
            successful_trials += 1
            if drop > max_performance_drop:
                return (
                    False,
                    f"High sensitivity on '{param_name}' "
                    f"({base_val} -> {perturbed_val}). Drop: {drop:.1%}",
                )

    if total_trials == 0:
        return False, "No perturbable parameters were evaluated."
    if successful_trials == 0:
        return False, f"All {total_trials} parameter perturbation trials failed."
    if failed_trials / total_trials >= 0.5:
        return (
            False,
            f"Too many perturbation failures ({failed_trials}/{total_trials}); "
            "the model is unstable.",
        )
    return (
        True,
        "Parameter stability confirmed across "
        f"{successful_trials}/{total_trials} trials.",
    )


def has_perturbable_params(base_params: dict[str, Any]) -> bool:
    """Return whether any model hyperparameter can be perturbed by PSS."""
    for name, value in base_params.items():
        if name in ENGINE_PARAM_KEYS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if _finite(value):
            return True
    return False


def _default_metric(
    target_type: str, predictions: np.ndarray, actuals: np.ndarray
) -> tuple[Metric, bool]:
    """Choose a metric and direction appropriate for the target type."""
    if target_type == TARGET_CONTINUOUS:
        return r2_score, True
    if target_type == TARGET_COUNT:
        return poisson_deviance, False

    acts = np.asarray(actuals).ravel()
    classes, _ = _classes_and_codes(acts)
    preds = np.asarray(predictions)
    probabilistic = _is_probability_matrix(preds, classes)
    if target_type == TARGET_BINARY and preds.ndim == 1:
        probabilistic = _looks_like_binary_probabilities(preds, classes)
    if probabilistic:
        return log_loss, False
    return accuracy_score, True


def _is_probability_matrix(predictions: np.ndarray, classes: np.ndarray) -> bool:
    preds = np.asarray(predictions)
    return preds.ndim == 2 and preds.shape[1] == classes.size


def _classification_diagnostics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    calibration_bins: int,
) -> dict[str, Any]:
    preds = np.asarray(predictions)
    acts = np.asarray(actuals).ravel()
    classes, _ = _classes_and_codes(acts)
    diagnostics: dict[str, Any] = {
        "accuracy": accuracy_score(preds, acts),
        "balanced_accuracy": balanced_accuracy_score(preds, acts),
        "macro_f1": macro_f1_score(preds, acts),
        "mcc": mcc_score(preds, acts),
    }
    probabilistic = _is_probability_matrix(preds, classes)
    if classes.size == 2 and preds.ndim == 1:
        probabilistic = _looks_like_binary_probabilities(preds, classes)
    if probabilistic:
        diagnostics["log_loss"] = log_loss(preds, acts)
        diagnostics["brier_score"] = brier_score(preds, acts)
        diagnostics["expected_calibration_error"] = expected_calibration_error(
            preds, acts, n_bins=calibration_bins
        )
        if classes.size == 2:
            diagnostics["roc_auc"] = roc_auc_score(preds, acts)
    return {key: value for key, value in diagnostics.items() if _finite(value)}


def _count_diagnostics(
    predictions: np.ndarray, actuals: np.ndarray, n_params: int
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "mean_absolute_error": mean_absolute_error(predictions, actuals),
        "root_mean_squared_error": root_mean_squared_error(predictions, actuals),
        "zero_inflation_ratio": zero_inflation_ratio(predictions, actuals),
    }
    dispersion = poisson_dispersion_test(
        predictions, actuals, n_params=n_params
    )
    diagnostics.update(dispersion)
    return {key: value for key, value in diagnostics.items() if _finite(value)}


def _regression_diagnostics(
    predictions: np.ndarray, actuals: np.ndarray
) -> dict[str, Any]:
    diagnostics = {
        "mean_absolute_error": mean_absolute_error(predictions, actuals),
        "root_mean_squared_error": root_mean_squared_error(predictions, actuals),
    }
    return {key: value for key, value in diagnostics.items() if _finite(value)}


def validate_model(
    features: np.ndarray,
    targets: np.ndarray,
    base_model_builder: ModelBuilder,
    base_params: dict[str, Any] | None = None,
    *,
    predictions: np.ndarray | None = None,
    actuals: np.ndarray | None = None,
    target_type: str | None = None,
    metric_func: Metric | None = None,
    higher_is_better: bool | None = None,
    max_p_value: float = 0.05,
    train_split_ratio: float = 0.70,
    num_permutations: int = 500,
    sensitivity_range: float = 0.10,
    max_performance_drop: float = 0.20,
    max_decay_tolerance: float | None = None,
    require_parameter_sensitivity: bool = False,
    calibration_bins: int = 10,
    max_ece: float | None = None,
    max_dispersion: float | None = None,
    permutation_seed: int = 0,
    block_size: int = 1,
    winning_returns: np.ndarray | None = None,
    all_trial_sharpes: list | None = None,
    periods_per_year: float = 252.0,
    min_dsr_limit: float = 0.95,
    data_description: str | None = None,
) -> dict[str, Any]:
    """Validate a model on continuous, classification, or count targets.

    The engine mirrors the four-layer structure of :func:`validation_engine`
    but adapts each layer to the target type:

    * Layer 1 (DSR) runs only for continuous return-series targets with real
      trial Sharpes; otherwise it is ``not_applicable``.
    * Layer 2 runs a label-permutation test with the selected metric.
    * Layer 3 runs a chronological out-of-sample decay test.
    * Layer 4 perturbs model hyperparameters. If none are available it is
      ``not_applicable`` unless ``require_parameter_sensitivity=True``.

    ``predictions``/``actuals`` may be supplied when the model search owns the
    holdout split. If omitted, the engine fits ``base_model_builder`` on the
    chronological training split and evaluates the remaining rows.

    Classification predictions may be hard labels (1D), binary positive-class
    probabilities (1D), or an ``(n, n_classes)`` probability matrix. Proper
    scoring rules (log loss / Brier) are used automatically when probabilities
    are available. Count targets default to Poisson deviance.

    Optional gates: ``max_ece`` checks classification calibration and
    ``max_dispersion`` checks Poisson overdispersion. Both are reported in
    ``diagnostics`` even when no gate is set.
    """
    params = dict(base_params or {})
    if max_decay_tolerance is not None:
        try:
            tolerance_value = float(max_decay_tolerance)
        except (TypeError, ValueError):
            return _failure(
                "Configuration", "max_decay_tolerance must be a number."
            )
        if not _finite(tolerance_value) or tolerance_value < 0.0:
            return _failure(
                "Configuration", "max_decay_tolerance must be non-negative."
            )
        params["max_decay_tolerance"] = tolerance_value
    if (winning_returns is None) != (all_trial_sharpes is None):
        return _failure(
            "Input contract",
            "Provide both winning_returns and all_trial_sharpes, or neither.",
        )

    X = np.asarray(features, dtype=float)
    raw_targets = np.asarray(targets)
    if X.ndim != 2 or X.size == 0:
        return _failure("Input contract", "features must be a non-empty 2D array.")
    if raw_targets.ndim != 1 or raw_targets.size != X.shape[0]:
        return _failure(
            "Input contract",
            "targets must be a 1D array matching the number of feature rows.",
        )
    if not np.all(np.isfinite(X)):
        return _failure("Input contract", "features must contain only finite values.")

    requested_type = (
        None if target_type is None else str(target_type).strip().lower()
    )
    try:
        resolved_type = (
            infer_target_type(raw_targets)
            if requested_type is None
            else requested_type
        )
    except (TypeError, ValueError) as exc:
        return _failure("Target contract", str(exc))

    aliases = {
        "regression": TARGET_CONTINUOUS,
        "regressor": TARGET_CONTINUOUS,
        "classification": TARGET_BINARY,
        "classifier": TARGET_BINARY,
        "categorical": TARGET_MULTICLASS,
    }
    resolved_type = aliases.get(resolved_type, resolved_type)
    if requested_type in {"classification", "classifier"}:
        distinct = np.unique(raw_targets).size
        resolved_type = TARGET_BINARY if distinct == 2 else TARGET_MULTICLASS
    if resolved_type not in {
        TARGET_CONTINUOUS,
        TARGET_BINARY,
        TARGET_MULTICLASS,
        TARGET_COUNT,
        TARGET_CONSTANT,
    }:
        return _failure(
            "Target contract",
            "target_type must be continuous, binary, multiclass, or count.",
        )
    if resolved_type == TARGET_CONSTANT:
        return _failure(
            "Target contract",
            "The target is constant; at least two distinct values are required.",
        )

    if resolved_type in {TARGET_CONTINUOUS, TARGET_COUNT}:
        try:
            numeric_targets = _as_finite_1d(raw_targets, "targets")
        except ValueError as exc:
            return _failure("Target contract", str(exc))
    else:
        numeric_targets = None
        if raw_targets.dtype.kind in "ifc" and not np.all(
            np.isfinite(raw_targets.astype(float))
        ):
            return _failure("Target contract", "targets must be finite.")

    classes = np.unique(raw_targets)
    if resolved_type == TARGET_BINARY and classes.size != 2:
        return _failure(
            "Target contract",
            "A binary target must contain exactly two distinct values.",
        )
    if resolved_type == TARGET_MULTICLASS and classes.size < 3:
        return _failure(
            "Target contract",
            "A multiclass target must contain at least three distinct values.",
        )
    if resolved_type == TARGET_COUNT:
        assert numeric_targets is not None
        if np.any(numeric_targets < 0.0):
            return _failure(
                "Target contract",
                "A count target must contain only non-negative values.",
            )
    if resolved_type == TARGET_CONTINUOUS:
        assert numeric_targets is not None
        if np.unique(numeric_targets).size < 3:
            return _failure(
                "Target contract",
                "A continuous target must contain at least three distinct values.",
            )
        if float(np.std(numeric_targets, ddof=1)) == 0.0:
            return _failure(
                "Target contract",
                "A continuous target must have non-zero variance.",
            )

    n_samples = int(X.shape[0])
    if n_samples < 20:
        return _failure("Input contract", "At least 20 rows are required.")
    if not _finite(train_split_ratio) or not (0.0 < float(train_split_ratio) < 1.0):
        return _failure(
            "Configuration", "train_split_ratio must be between 0 and 1."
        )
    split_index = int(n_samples * float(train_split_ratio))
    if split_index < 5 or n_samples - split_index < 5:
        return _failure(
            "Configuration",
            "train_split_ratio leaves fewer than five rows in a split.",
        )
    if not _finite(max_p_value) or not (0.0 < float(max_p_value) <= 1.0):
        return _failure("Configuration", "max_p_value must be between 0 and 1.")
    if not _finite(num_permutations) or int(num_permutations) < 1:
        return _failure("Configuration", "num_permutations must be positive.")
    if not _finite(block_size) or int(block_size) < 1:
        return _failure("Configuration", "block_size must be a positive integer.")
    if not _finite(sensitivity_range) or float(sensitivity_range) <= 0.0:
        return _failure("Configuration", "sensitivity_range must be positive.")
    if not _finite(max_performance_drop) or float(max_performance_drop) < 0.0:
        return _failure(
            "Configuration", "max_performance_drop must be non-negative."
        )
    if not _finite(min_dsr_limit) or not (0.0 <= float(min_dsr_limit) <= 1.0):
        return _failure("Configuration", "min_dsr_limit must be between 0 and 1.")
    if int(calibration_bins) < 2:
        return _failure("Configuration", "calibration_bins must be at least 2.")
    if max_ece is not None and (not _finite(max_ece) or float(max_ece) < 0.0):
        return _failure("Configuration", "max_ece must be non-negative.")
    if max_dispersion is not None and (
        not _finite(max_dispersion) or float(max_dispersion) < 0.0
    ):
        return _failure("Configuration", "max_dispersion must be non-negative.")

    # -------------------------------------------------- held-out predictions
    if not callable(base_model_builder):
        return _failure("Input contract", "base_model_builder must be callable.")
    if (predictions is None) != (actuals is None):
        return _failure(
            "Input contract", "Provide both predictions and actuals, or neither."
        )
    if predictions is None:
        try:
            model = base_model_builder(
                params, X[:split_index], raw_targets[:split_index]
            )
            predictions = np.asarray(model(X[split_index:]))
        except Exception as exc:  # noqa: BLE001 - report to the user
            return _failure("Model fit", f"Model construction failed: {exc}")
        actuals = raw_targets[split_index:]
    else:
        predictions = np.asarray(predictions)
        if actuals is None:
            return _failure("Input contract", "actuals are required.")
    heldout_actuals = np.asarray(actuals).ravel()
    if heldout_actuals.size == 0:
        return _failure("Input contract", "actuals must be non-empty.")
    if predictions.ndim not in (1, 2):
        return _failure(
            "Input contract", "predictions must be a 1D or 2D array."
        )
    if predictions.shape[0] != heldout_actuals.size:
        return _failure(
            "Input contract",
            "predictions and actuals must have the same number of rows.",
        )
    if np.issubdtype(predictions.dtype, np.number) and not np.all(
        np.isfinite(predictions.astype(float))
    ):
        return _failure("Input contract", "predictions must be finite.")

    # -------------------------------------------------- metric selection
    if metric_func is not None:
        metric = metric_func
        if higher_is_better is None:
            metric_name = getattr(metric_func, "__name__", "custom")
            higher_is_better = metric_name not in LOWER_IS_BETTER_METRICS
    else:
        metric, default_direction = _default_metric(
            resolved_type, predictions, heldout_actuals
        )
        if higher_is_better is None:
            higher_is_better = default_direction
    metric_name = getattr(metric, "__name__", "custom")
    try:
        measured_score = float(metric(predictions, heldout_actuals))
    except Exception as exc:  # noqa: BLE001 - report metric failures
        return _failure(
            "Metric",
            f"Metric raised {type(exc).__name__}: {exc}",
            target_type=resolved_type,
            metric=metric_name,
        )
    if not _finite(measured_score):
        return _failure(
            "Metric",
            "The selected metric returned a non-finite held-out score.",
            target_type=resolved_type,
            metric=metric_name,
        )

    layers: dict[str, Any] = {}
    if resolved_type in {TARGET_BINARY, TARGET_MULTICLASS}:
        diagnostics = _classification_diagnostics(
            predictions, heldout_actuals, int(calibration_bins)
        )
    elif resolved_type == TARGET_COUNT:
        diagnostics = _count_diagnostics(
            predictions, heldout_actuals, n_params=X.shape[1] + 1
        )
    else:
        diagnostics = _regression_diagnostics(predictions, heldout_actuals)
    context: dict[str, Any] = {
        "target_type": resolved_type,
        "metric": metric_name,
        "higher_is_better": bool(higher_is_better),
        "diagnostics": diagnostics,
    }

    # -------------------------------------------------- Layer 1: optional DSR
    dsr_score: float | None = None
    if (
        resolved_type == TARGET_CONTINUOUS
        and winning_returns is not None
        and all_trial_sharpes is not None
    ):
        if len(all_trial_sharpes) < 2:
            return _failure(
                "Layer 1 (DSR)",
                "At least two real trials are required for the DSR penalty.",
                **context,
                layers=layers,
            )
        dsr_score = dsr_dynamic(
            winning_returns, all_trial_sharpes, periods_per_year
        )
        if not _finite(dsr_score) or dsr_score < float(min_dsr_limit):
            return _failure(
                "Layer 1 (DSR)",
                f"Layer 1 FAIL: DSR Probability {dsr_score:.4f} < {min_dsr_limit}",
                **context,
                dsr_score=dsr_score,
                layers=layers,
            )
        layers["dsr"] = {"status": "passed", "score": float(dsr_score)}
    else:
        layers["dsr"] = {
            "status": "not_applicable",
            "reason": "DSR is defined only for continuous return-series targets.",
        }

    # -------------------------------------------------- Layer 2: permutation
    try:
        p_value = monte_carlo(
            predictions,
            heldout_actuals,
            num_permutations=int(num_permutations),
            metric_func=metric,
            higher_is_better=bool(higher_is_better),
            seed=int(permutation_seed),
            block_size=int(block_size),
        )
    except ValueError as exc:
        return _failure(
            "Layer 2 (Monte Carlo)",
            f"Layer 2 FAIL: {exc}",
            **context,
            dsr_score=dsr_score,
            layers=layers,
        )
    if not _finite(p_value) or p_value > float(max_p_value):
        return _failure(
            "Layer 2 (Monte Carlo)",
            f"Layer 2 FAIL: Permutation p-value {p_value:.4f} > {max_p_value}",
            **context,
            dsr_score=dsr_score,
            p_value=p_value,
            layers=layers,
        )
    layers["permutation"] = {"status": "passed", "p_value": float(p_value)}

    # -------------------------------------------------- Layer 3: OOS decay
    oos_result = oos_decay(
        base_model_builder,
        X,
        raw_targets,
        params,
        train_split_ratio=float(train_split_ratio),
        metric_func=metric,
        higher_is_better=bool(higher_is_better),
        allow_non_positive_baseline=True,
        clamp_tolerance=False,
    )
    if not oos_result.get("success"):
        return _failure(
            "Layer 3 (OOS Decay)",
            f"Layer 3 FAIL: {oos_result.get('error', 'unknown error')}",
            **context,
            dsr_score=dsr_score,
            p_value=p_value,
            oos_score=oos_result.get("oos_score"),
            tolerance=oos_result.get("tolerance"),
            tolerance_clamped=oos_result.get("tolerance_clamped", False),
            layers=layers,
        )
    layers["oos_decay"] = {
        "status": "passed",
        "oos_score": float(oos_result["oos_score"]),
        "decay_rate": float(oos_result["decay_rate"]),
        "tolerance": float(oos_result.get("tolerance", 0.0)),
        "tolerance_clamped": bool(oos_result.get("tolerance_clamped", False)),
    }

    # -------------------------------------------------- Layer 4: sensitivity
    pss_status = "not_applicable"
    pss_message = "No perturbable model hyperparameters were supplied."
    if has_perturbable_params(params):
        pss_passed, pss_message = pss(
            base_model_builder,
            X,
            raw_targets,
            params,
            sensitivity_range=float(sensitivity_range),
            max_performance_drop=float(max_performance_drop),
            metric_func=metric,
            higher_is_better=bool(higher_is_better),
            allow_non_positive_baseline=True,
        )
        if not pss_passed:
            layers["parameter_sensitivity"] = {
                "status": "failed",
                "message": pss_message,
            }
            return _failure(
                "Layer 4 (Parameter Sensitivity)",
                f"Layer 4 FAIL: {pss_message}",
                **context,
                dsr_score=dsr_score,
                p_value=p_value,
                oos_score=oos_result["oos_score"],
                tolerance=oos_result.get("tolerance"),
                tolerance_clamped=oos_result.get("tolerance_clamped", False),
                layers=layers,
            )
        pss_status = "passed"
    elif require_parameter_sensitivity:
        layers["parameter_sensitivity"] = {
            "status": "failed",
            "message": pss_message,
        }
        return _failure(
            "Layer 4 (Parameter Sensitivity)",
            "Layer 4 FAIL: No perturbable model hyperparameters were supplied.",
            **context,
            dsr_score=dsr_score,
            p_value=p_value,
            oos_score=oos_result["oos_score"],
            tolerance=oos_result.get("tolerance"),
            tolerance_clamped=oos_result.get("tolerance_clamped", False),
            layers=layers,
        )
    layers["parameter_sensitivity"] = {
        "status": pss_status,
        "message": pss_message,
    }

    if max_ece is not None:
        ece = diagnostics.get("expected_calibration_error")
        if not _finite(ece):
            return _failure(
                "Calibration",
                "max_ece requires probabilistic classification predictions.",
                **context,
                dsr_score=dsr_score,
                p_value=p_value,
                oos_score=oos_result["oos_score"],
                layers=layers,
            )
        layers["calibration"] = {"status": "passed", "ece": float(ece)}
        if float(ece) > float(max_ece):
            layers["calibration"] = {"status": "failed", "ece": float(ece)}
            return _failure(
                "Calibration",
                f"Calibration FAIL: ECE {float(ece):.4f} > {max_ece}",
                **context,
                dsr_score=dsr_score,
                p_value=p_value,
                oos_score=oos_result["oos_score"],
                layers=layers,
            )

    if max_dispersion is not None:
        dispersion = diagnostics.get("dispersion")
        if not _finite(dispersion):
            return _failure(
                "Count dispersion",
                "max_dispersion requires count-target predictions.",
                **context,
                dsr_score=dsr_score,
                p_value=p_value,
                oos_score=oos_result["oos_score"],
                layers=layers,
            )
        layers["count_dispersion"] = {
            "status": "passed",
            "dispersion": float(dispersion),
        }
        if float(dispersion) > float(max_dispersion):
            layers["count_dispersion"] = {
                "status": "failed",
                "dispersion": float(dispersion),
            }
            return _failure(
                "Count dispersion",
                f"Count dispersion FAIL: {float(dispersion):.4f} > "
                f"{max_dispersion}",
                **context,
                dsr_score=dsr_score,
                p_value=p_value,
                oos_score=oos_result["oos_score"],
                layers=layers,
            )

    return {
        "passed": True,
        "failed_layer": None,
        "target_type": resolved_type,
        "metric": metric_name,
        "higher_is_better": bool(higher_is_better),
        "dsr_score": dsr_score,
        "p_value": float(p_value),
        "oos_score": float(oos_result["oos_score"]),
        "decay_rate": float(oos_result["decay_rate"]),
        "tolerance": float(oos_result.get("tolerance", 0.0)),
        "tolerance_clamped": bool(oos_result.get("tolerance_clamped", False)),
        "parameter_sensitivity": pss_status,
        "layers": layers,
        "diagnostics": diagnostics,
        "data_description": data_description,
        "message": "All applicable validation layers cleared successfully.",
    }


# ==========================================
# MASTER VALIDATION ENGINE (RETURN SERIES)
# ==========================================
def validation_engine(
    winning_returns: np.ndarray,
    all_trial_sharpes: list,
    predictions: np.ndarray,
    actuals: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    base_model_builder: ModelBuilder,
    base_params: dict,
    periods_per_year: int,
    *,
    metric_func: Metric | None = None,
    higher_is_better: bool = True,
    data_description: str | None = None,
) -> dict:
    """Execute the four validation layers and return a result dict.

    There is no ``model_predict_func`` parameter: the engine fits the model
    through ``base_model_builder`` so training and evaluation cannot diverge.
    """
    y = np.asarray(targets, dtype=float).ravel()
    if y.size < 20:
        return _failure(
            "Target contract",
            "At least 20 target rows are required.",
            data_description=data_description,
        )
    if not np.all(np.isfinite(y)):
        return _failure(
            "Target contract",
            "The target must contain only finite values.",
            data_description=data_description,
        )
    if np.std(y, ddof=1) == 0.0 or np.unique(y).size < 3:
        return _failure(
            "Target contract",
            "The target must be a continuous series with non-zero variance and "
            "at least three distinct values; binary and constant targets are "
            "not supported. Supply a continuous return series.",
            data_description=data_description,
        )

    min_dsr = float(base_params.get("min_dsr_limit", 0.95))
    max_p = float(base_params.get("max_p_value", 0.05))
    train_split_ratio = float(base_params.get("train_split_ratio", 0.70))
    num_permutations = int(base_params.get("num_permutations", 500))
    sensitivity_range = float(base_params.get("sensitivity_range", 0.10))
    max_performance_drop = float(base_params.get("max_performance_drop", 0.20))

    if not (0.0 <= min_dsr <= 1.0):
        return _failure("Configuration", "min_dsr_limit must be between 0 and 1.")
    if not (0.0 < max_p <= 1.0):
        return _failure("Configuration", "max_p_value must be between 0 and 1.")
    if not (0.0 < train_split_ratio < 1.0):
        return _failure("Configuration", "train_split_ratio must be between 0 and 1.")
    if num_permutations < 1:
        return _failure("Configuration", "num_permutations must be positive.")
    if sensitivity_range <= 0.0:
        return _failure("Configuration", "sensitivity_range must be positive.")
    if max_performance_drop < 0.0:
        return _failure("Configuration", "max_performance_drop must be non-negative.")

    # --- LAYER 1 ---
    if len(all_trial_sharpes) < 2:
        return _failure(
            "Layer 1 (DSR)",
            "At least two real trials are required for a multiple-testing "
            "penalty; the trial list was empty or a single trial.",
            data_description=data_description,
        )
    dsr_score = dsr_dynamic(winning_returns, all_trial_sharpes, periods_per_year)
    if not _finite(dsr_score) or dsr_score < min_dsr:
        return _failure(
            "Layer 1 (DSR)",
            f"Layer 1 FAIL: DSR Probability {dsr_score:.4f} < {min_dsr}",
            dsr_score=dsr_score,
            data_description=data_description,
        )

    # --- LAYER 2 ---
    try:
        p_value = monte_carlo(
            predictions,
            actuals,
            num_permutations=num_permutations,
            metric_func=metric_func,
            higher_is_better=higher_is_better,
        )
    except ValueError as exc:
        return _failure(
            "Layer 2 (Monte Carlo)",
            f"Layer 2 FAIL: {exc}",
            dsr_score=dsr_score,
            data_description=data_description,
        )
    if not _finite(p_value) or p_value > max_p:
        return _failure(
            "Layer 2 (Monte Carlo)",
            f"Layer 2 FAIL: Permutation p-value {p_value:.4f} > {max_p}",
            dsr_score=dsr_score,
            p_value=p_value,
            data_description=data_description,
        )

    # --- LAYER 3 ---
    oos_result = oos_decay(
        base_model_builder,
        features,
        targets,
        base_params,
        train_split_ratio=train_split_ratio,
        metric_func=metric_func,
        higher_is_better=higher_is_better,
    )
    if not oos_result["success"]:
        return _failure(
            "Layer 3 (OOS Decay)",
            f"Layer 3 FAIL: {oos_result.get('error', 'unknown error')}",
            dsr_score=dsr_score,
            p_value=p_value,
            oos_score=oos_result.get("oos_score"),
            tolerance=oos_result.get("tolerance"),
            tolerance_clamped=oos_result.get("tolerance_clamped", False),
            data_description=data_description,
        )
    if not _finite(oos_result.get("oos_score")) or not _finite(
        oos_result.get("decay_rate")
    ):
        return _failure(
            "Layer 3 (OOS Decay)",
            "Layer 3 FAIL: OOS score or decay rate is non-finite.",
            dsr_score=dsr_score,
            p_value=p_value,
            data_description=data_description,
        )

    # --- LAYER 4 ---
    pss_passed, pss_msg = pss(
        base_model_builder,
        features,
        targets,
        base_params,
        sensitivity_range=sensitivity_range,
        max_performance_drop=max_performance_drop,
        metric_func=metric_func,
        higher_is_better=higher_is_better,
    )
    if not pss_passed:
        return _failure(
            "Layer 4 (Parameter Sensitivity)",
            f"Layer 4 FAIL: {pss_msg}",
            dsr_score=dsr_score,
            p_value=p_value,
            oos_score=oos_result["oos_score"],
            tolerance=oos_result.get("tolerance"),
            tolerance_clamped=oos_result.get("tolerance_clamped", False),
            data_description=data_description,
        )

    return {
        "passed": True,
        "dsr_score": float(dsr_score),
        "p_value": float(p_value),
        "oos_score": float(oos_result["oos_score"]),
        "decay_rate": float(oos_result["decay_rate"]),
        "tolerance": float(oos_result.get("tolerance", 0.0)),
        "tolerance_clamped": bool(oos_result.get("tolerance_clamped", False)),
        "metric": getattr(metric_func, "__name__", "custom"),
        "higher_is_better": bool(higher_is_better),
        "data_description": data_description,
        "message": "All 4 validation layers cleared successfully.",
    }
