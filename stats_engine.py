"""Four-layer autonomous validation engine.

Target contract
---------------
The target series is a continuous, return-like series: finite, with non-zero
variance and at least three distinct values. Binary labels and constant targets
are rejected instead of being silently forced through incompatible layers.

Metric contract
---------------
``metric_func(predictions, actuals) -> float`` must return a finite score. Set
``higher_is_better=False`` for losses such as MSE; the default is
``higher_is_better=True`` and the UI passes an R-squared metric.

Layer 2 evaluates held-out predictions, not training predictions. Layer 3 fits
the supplied builder on the chronological training split and scores the test
split. Layer 4 perturbs real model hyperparameters; validation thresholds are
never perturbed.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.stats import kurtosis as calc_kurtosis
from scipy.stats import norm
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
def monte_carlo(
    predictions: np.ndarray,
    actuals: np.ndarray,
    num_permutations: int,
    metric_func: Metric | None = None,
    *,
    higher_is_better: bool = True,
    seed: int = 0,
) -> float:
    """Permutation p-value for held-out predictions.

    ``metric_func`` must return a finite score. Non-finite permutation scores
    are skipped; if fewer than half of the permutations are finite, the result
    is ``nan`` and the engine fails the layer.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    acts = np.asarray(actuals, dtype=float).ravel()
    if preds.shape != acts.shape or preds.size == 0:
        raise ValueError("Predictions and actuals must have the same non-empty shape.")
    if not _finite(num_permutations) or int(num_permutations) < 1:
        raise ValueError("num_permutations must be a positive integer.")
    num_permutations = int(num_permutations)
    metric: Metric = metric_func or (
        lambda p, a: float(np.mean(p == a))
    )
    real_score = float(metric(preds, acts))
    if not _finite(real_score):
        return float("nan")

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(num_permutations):
        permuted = acts.copy()
        rng.shuffle(permuted)
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
) -> dict:
    """Fit on the chronological training split and score the test split.

    Returns a result dict. ``tolerance_clamped`` is true when the requested
    ``max_decay_tolerance`` was outside the supported [0.15, 0.50] range.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
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

    target_variance = float(np.var(y))
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
        train_preds = np.asarray(model(X_train), dtype=float).ravel()
        test_preds = np.asarray(model(X_test), dtype=float).ravel()
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
    if is_score <= 0.0:
        return {
            "success": False,
            "oos_score": oos_score,
            "decay_rate": 1.0,
            "tolerance": dynamic_tolerance,
            "tolerance_clamped": tolerance_clamped,
            "error": "In-sample baseline score is non-positive.",
        }

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
) -> tuple[bool, str]:
    """Perturb real model hyperparameters and check score stability."""
    X = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
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
    if not _finite(base_score) or base_score <= 0.0:
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
        f"Parameter stability confirmed across {successful_trials}/{total_trials} trials.",
    )


# ==========================================
# MASTER VALIDATION ENGINE
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
