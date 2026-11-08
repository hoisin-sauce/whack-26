"""Deterministic checks for the final judge.

These checks are computed in code before any LLM call.  The branch's original
auditor was R²-specific; this port keeps the same arithmetic but accepts the
primary metric and its direction so binary, multiclass, count, and time-series
models are judged on the metric that selected them.
"""

from __future__ import annotations

import math
from typing import Any, Callable

try:  # optional: only used for deterministic p-value / CI checks
    from scipy.stats import chi2
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover - scipy is in the core requirements
    chi2 = None
    student_t = None

HIGHER_IS_BETTER = frozenset(
    {"r2", "accuracy", "balanced_accuracy", "auc", "macro_f1", "mcc"}
)
LOWER_IS_BETTER = frozenset(
    {"rmse", "mae", "log_loss", "poisson_deviance", "aic", "bic", "brier"}
)
_BOUNDED_HIGHER = frozenset({"r2", "accuracy", "balanced_accuracy", "auc"})


def metric_is_higher_better(metric: str, explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    if metric in LOWER_IS_BETTER:
        return False
    return metric in HIGHER_IS_BETTER or metric not in LOWER_IS_BETTER


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _empty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return True
    if isinstance(value, dict):
        return all(_empty(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_empty(item) for item in value)
    return False


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def check_overfitting(raw: dict) -> dict:
    metric = str(raw.get("metric") or "r2")
    higher = metric_is_higher_better(metric, raw.get("higher_is_better"))
    train = raw.get("train_score", raw.get("train_r2"))
    cv = raw.get("cv_mean", raw.get("cv_mean_r2"))
    sd = raw.get("cv_sd", raw.get("cv_sd_r2"))
    out: dict[str, Any] = {
        "metric": metric,
        "higher_is_better": higher,
        "learning_curve_present": bool(raw.get("learning_curve_present")),
    }
    if _num(train) and _num(cv):
        gap = (float(train) - float(cv)) if higher else (float(cv) - float(train))
        out["train_cv_gap"] = round(gap, 4)
        if _num(sd) and float(sd) > 0:
            out["gap_in_fold_sds"] = round(gap / float(sd), 2)
        if metric in _BOUNDED_HIGHER:
            out["gap_is_large"] = bool(gap > 0.10)
        else:
            denominator = max(abs(float(cv)), 1e-9)
            out["gap_ratio"] = round(gap / denominator, 4)
            out["gap_is_large"] = bool(gap > 0.0 and gap / denominator > 0.25)
    return out


def check_residuals(raw: dict) -> dict:
    stat = raw.get("statistic")
    lags = raw.get("lags")
    p_value = raw.get("p_value")
    out: dict[str, Any] = {}
    if _num(p_value):
        out["reported_p_value"] = round(float(p_value), 6)
    for name in (
        "durbin_watson",
        "autocorrelation_lag1",
        "breusch_pagan_p",
        "white_p",
    ):
        value = raw.get(name)
        if _num(value):
            out[name] = round(float(value), 6)
    if not all(_num(value) for value in (stat, lags, p_value)):
        return out
    out["statistic"] = round(float(stat), 6)
    out["lags"] = int(lags)
    if chi2 is None:
        out["note"] = "scipy not installed; Ljung-Box p-value not recomputed"
        return out
    model_df = raw.get("model_df", 0)
    try:
        degrees = int(lags) - int(model_df or 0)
    except (TypeError, ValueError):
        degrees = int(lags)
    if degrees <= 0:
        return out
    expected = float(chi2.sf(float(stat), degrees))
    out["ljung_box_p_recomputed"] = round(expected, 6)
    out["ljung_box_consistent"] = abs(expected - float(p_value)) <= 0.02
    return out


def check_leakage(raw: dict) -> dict:
    features = raw.get("features") or []
    suspects: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        reasons: list[str] = []
        if feature.get("derived_from_target"):
            reasons.append("derived from the target")
        if feature.get("known_at_prediction_time") is False:
            reasons.append("not available at prediction time")
        if feature.get("used_to_define_split"):
            reasons.append("used to define the split or target")
        importance = feature.get("importance_zscore")
        if _num(importance) and float(importance) > 3.0:
            reasons.append("implausibly high importance")
        if reasons:
            suspects.append({"name": feature.get("name"), "reasons": reasons})
    return {"n_features": len(features), "suspect_features": suspects}


def check_assumptions(raw: dict) -> dict:
    tests = raw.get("tests") or {}
    out: dict[str, Any] = {}
    adf_p = tests.get("adf_p")
    if _num(adf_p):
        out["adf_rejects_unit_root_5pct"] = bool(float(adf_p) < 0.05)
    durbin_watson = tests.get("durbin_watson")
    if _num(durbin_watson):
        out["durbin_watson_near_2"] = bool(1.5 <= float(durbin_watson) <= 2.5)
    out["linearity_test_supplied"] = any(
        key in tests for key in ("reset_p", "rainbow_p", "linearity_p")
    )
    for key in ("breusch_pagan_p", "white_p"):
        value = tests.get(key)
        if _num(value):
            out["heteroskedasticity_rejected_5pct"] = bool(float(value) < 0.05)
    vif_max = tests.get("vif_max")
    condition_number = tests.get("condition_number")
    if _num(vif_max) or _num(condition_number):
        out["multicollinearity_high"] = bool(
            (_num(vif_max) and float(vif_max) > 10.0)
            or (_num(condition_number) and float(condition_number) > 30.0)
        )
    ece = tests.get("calibration_ece")
    if _num(ece):
        out["calibration_poor"] = bool(float(ece) > 0.10)
    overdispersion = tests.get("overdispersion")
    if _num(overdispersion):
        out["overdispersion_high"] = bool(float(overdispersion) > 1.5)
    return out


def check_subgroups(raw: dict) -> dict:
    slices = raw.get("slices") or {}
    metric = str(raw.get("metric") or "r2")
    higher = metric_is_higher_better(metric, raw.get("higher_is_better"))
    values: dict[str, float] = {}
    for name, value in slices.items():
        if isinstance(value, dict):
            candidate = value.get(metric, value.get("score", value.get("value")))
        else:
            candidate = value
        if _num(candidate):
            values[str(name)] = float(candidate)
    if len(values) < 2:
        return {}
    best = max(values, key=values.get)
    worst = min(values, key=values.get) if higher else max(values, key=values.get)
    if not higher:
        best = min(values, key=values.get)
    spread = abs(values[best] - values[worst])
    return {
        "metric": metric,
        "worst_slice": worst,
        "best_slice": best,
        "metric_spread": round(spread, 4),
        "r2_spread": round(spread, 4) if metric == "r2" else None,
    }


def check_multiple_comparisons(raw: dict) -> dict:
    n_tests = raw.get("n_tests")
    min_p = raw.get("min_p")
    if not (_num(n_tests) and _num(min_p)) or int(n_tests) < 1:
        return {}
    corrected = bool(raw.get("p_is_corrected"))
    threshold = 0.05 if corrected else 0.05 / int(n_tests)
    return {
        "threshold_used": threshold,
        "survives_correction": bool(float(min_p) < threshold),
        "n_looks": raw.get("n_looks"),
    }


def check_extrapolation(raw: dict) -> dict:
    train = raw.get("train") or {}
    prediction = raw.get("pred") or {}
    values = (
        train.get("min"),
        train.get("max"),
        prediction.get("min"),
        prediction.get("max"),
    )
    if not all(_num(value) for value in values):
        return {}
    above = max(0.0, float(prediction["max"]) - float(train["max"]))
    below = max(0.0, float(train["min"]) - float(prediction["min"]))
    span = float(train["max"]) - float(train["min"])
    return {
        "outside_support": bool(above > 0 or below > 0),
        "excess_above_train_max": round(above, 6),
        "excess_below_train_min": round(below, 6),
        "excess_pct_of_train_range": (
            round(100.0 * max(above, below) / span, 1) if span > 0 else None
        ),
    }


def check_simplicity(raw: dict) -> dict:
    metric = str(raw.get("metric") or "r2")
    higher = metric_is_higher_better(metric, raw.get("higher_is_better"))
    baseline = raw.get("baseline_score", raw.get("baseline_cv"))
    model = raw.get("model_score", raw.get("model_cv"))
    out: dict[str, Any] = {"metric": metric, "higher_is_better": higher}
    if _num(baseline) and _num(model):
        out["delta_improvement"] = round(
            (float(model) - float(baseline))
            if higher
            else (float(baseline) - float(model)),
            4,
        )
    model_folds = raw.get("model_fold_scores", raw.get("model_fold_r2"))
    baseline_folds = raw.get("baseline_fold_scores", raw.get("baseline_fold_r2"))
    if (
        isinstance(model_folds, list)
        and isinstance(baseline_folds, list)
        and len(model_folds) == len(baseline_folds) >= 3
        and all(_num(value) for value in model_folds + baseline_folds)
    ):
        differences = [
            (float(model_value) - float(baseline_value))
            if higher
            else (float(baseline_value) - float(model_value))
            for model_value, baseline_value in zip(model_folds, baseline_folds)
        ]
        count = len(differences)
        mean = sum(differences) / count
        variance = sum((value - mean) ** 2 for value in differences) / (count - 1)
        # Nadeau-Bengio corrected resampled variance: inflate 1/n by
        # n_test/n_train.  For k-fold CV n_test/n_train = 1/(k-1).
        n_train = raw.get("n_train")
        n_test = raw.get("n_test")
        ratio = (
            float(n_test) / float(n_train)
            if _num(n_train) and _num(n_test) and float(n_train) > 0
            else 1 / (count - 1)
        )
        standard_error = math.sqrt((1 / count + ratio) * variance)
        out.update(
            paired_n_folds=count,
            paired_mean_diff=round(mean, 4),
            paired_sd_diff=round(math.sqrt(variance), 4),
            paired_se_corrected=round(standard_error, 5),
            folds_where_model_wins=sum(value > 0 for value in differences),
        )
        if student_t is not None and standard_error > 0:
            critical = float(student_t.ppf(0.975, count - 1))
            out["paired_95ci_corrected"] = [
                round(mean - critical * standard_error, 4),
                round(mean + critical * standard_error, 4),
            ]
        out["caveat"] = (
            "Variance-corrected paired interval (Nadeau & Bengio): a standard "
            "approximation that widens the naive interval because CV folds share "
            "training data. Assumes k-fold CV unless n_train/n_test are supplied. "
            "Still approximate. The model's own fold SD is not the yardstick for "
            "this comparison."
        )
    return out


def check_none(raw: dict) -> dict:
    return {}


CHECKS: dict[str, Callable[[dict], dict]] = {
    "overfitting_skeptic": check_overfitting,
    "residual_analyst": check_residuals,
    "leakage_detective": check_leakage,
    "assumptions_auditor": check_assumptions,
    "subgroup_critic": check_subgroups,
    "multiple_comparisons_skeptic": check_multiple_comparisons,
    "extrapolation_critic": check_extrapolation,
    "simplicity_advocate": check_simplicity,
    "domain_sanity_critic": check_none,
}


def compute_all(evidence: dict) -> dict[str, dict]:
    return {
        name: function(evidence[name])
        for name, function in CHECKS.items()
        if not _empty(evidence.get(name))
    }


def data_flags(computed: dict[str, dict]) -> list[str]:
    """Plain-language flags from code-verified facts alone."""
    flags: list[str] = []
    extrapolation = computed.get("extrapolation_critic", {})
    if extrapolation.get("outside_support"):
        flags.append(
            "Prediction range leaves the training support (above max by "
            f"{extrapolation['excess_above_train_max']}, below min by "
            f"{extrapolation['excess_below_train_min']}; "
            f"{extrapolation['excess_pct_of_train_range']}% of the training range)."
        )
    residuals = computed.get("residual_analyst", {})
    if residuals.get("ljung_box_consistent") is False:
        flags.append(
            "Reported Ljung-Box p-value is inconsistent with its statistic "
            f"(recomputed p={residuals['ljung_box_p_recomputed']}). The reported "
            "diagnostics cannot all be right."
        )
    overfitting = computed.get("overfitting_skeptic", {})
    if overfitting.get("gap_is_large"):
        flags.append(
            f"Train-CV gap is large ({overfitting.get('train_cv_gap')} on "
            f"{overfitting.get('metric')})."
        )
    leakage = computed.get("leakage_detective", {})
    if leakage.get("suspect_features"):
        flags.append(f"Leakage suspects: {leakage['suspect_features']}")
    subgroups = computed.get("subgroup_critic", {})
    spread = subgroups.get("metric_spread", subgroups.get("r2_spread", 0))
    if _num(spread) and float(spread) > 0.10:
        flags.append(
            f"Large subgroup spread in {subgroups.get('metric', 'the metric')} "
            f"({spread}; worst: {subgroups.get('worst_slice')})."
        )
    multiple = computed.get("multiple_comparisons_skeptic", {})
    if multiple.get("survives_correction") is False:
        flags.append("Minimum p-value does not survive multiple-comparison correction.")
    simplicity = computed.get("simplicity_advocate", {})
    interval = simplicity.get("paired_95ci_corrected")
    if interval and interval[0] <= 0:
        flags.append(
            "Corrected paired CI for the model-vs-baseline improvement includes "
            f"zero: {interval}."
        )
    assumptions = computed.get("assumptions_auditor", {})
    if assumptions.get("heteroskedasticity_rejected_5pct"):
        flags.append("Heteroskedasticity test rejects constant residual variance.")
    if assumptions.get("multicollinearity_high"):
        flags.append("Multicollinearity diagnostics exceed the configured threshold.")
    if assumptions.get("calibration_poor"):
        flags.append("Calibration error is above 0.10.")
    if assumptions.get("overdispersion_high"):
        flags.append("Count model is overdispersed (Pearson ratio > 1.5).")
    return flags


__all__ = [
    "CHECKS",
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "check_assumptions",
    "check_extrapolation",
    "check_leakage",
    "check_multiple_comparisons",
    "check_none",
    "check_overfitting",
    "check_residuals",
    "check_simplicity",
    "check_subgroups",
    "compute_all",
    "data_flags",
    "metric_is_higher_better",
]
