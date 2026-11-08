from __future__ import annotations

import inspect
import math

import numpy as np

from stats_engine import (
    build_linear_model,
    dsr_dynamic,
    monte_carlo,
    oos_decay,
    pss,
    r2_score,
    sharpe_ratio,
    trial_sharpes,
    validation_engine,
)


def make_returns(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    target = 0.02 + 0.1 * features[:, 0] + rng.normal(0.0, 0.05, n)
    return features, target


def make_engine_inputs(n: int = 300, seed: int = 0):
    features, target = make_returns(n, seed)
    split = int(n * 0.70)
    params = {"ridge": 1e-6}
    model = build_linear_model(params, features[:split], target[:split])
    predictions = model(features[split:])
    sharpes, winning_returns = trial_sharpes(
        features[:split], target[:split], 252, model_predict=model
    )
    return features, target, predictions, sharpes, winning_returns


def base_params(**overrides):
    params = {
        "ridge": 1e-6,
        "min_dsr_limit": 0.0,
        "max_p_value": 1.0,
        "train_split_ratio": 0.70,
        "num_permutations": 200,
        "sensitivity_range": 0.10,
        "max_performance_drop": 1.0,
        "max_decay_tolerance": 0.50,
    }
    params.update(overrides)
    return params


def test_builder_accepts_params_features_and_targets() -> None:
    features, target = make_returns(60)
    predict = build_linear_model({"ridge": 1e-6}, features[:40], target[:40])

    predictions = predict(features[40:])

    assert predictions.shape == (20,)
    assert np.all(np.isfinite(predictions))


def test_builder_rejects_non_finite_inputs() -> None:
    features, target = make_returns(30)
    features[0, 0] = np.nan

    try:
        build_linear_model({"ridge": 0.0}, features, target)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite features must be rejected")


def test_r2_score_perfect_constant_and_shape_mismatch() -> None:
    assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    assert r2_score(np.array([1.0, 2.0]), np.array([2.0, 2.0])) == 0.0
    assert math.isnan(r2_score(np.array([1.0, 2.0]), np.array([1.0])))


def test_sharpe_ratio_rejects_non_finite_inputs() -> None:
    assert math.isnan(sharpe_ratio(np.array([1.0, np.nan]), 252))
    assert math.isnan(sharpe_ratio(np.array([1.0, 1.0]), 252))
    assert math.isnan(sharpe_ratio(np.array([1.0, 2.0]), 0))


def test_trial_sharpes_are_real_and_aligned() -> None:
    features, target = make_returns(100)
    model = build_linear_model({"ridge": 1e-6}, features, target)

    sharpes, winning_returns = trial_sharpes(
        features, target, 252, model_predict=model
    )

    assert len(sharpes) >= 2
    assert winning_returns.shape == target.shape
    assert all(math.isfinite(value) for value in sharpes)
    # The trial list is derived from data, not from a user threshold.
    assert len(sharpes) == features.shape[1] + 1


def test_dsr_rejects_non_finite_and_single_trial_inputs() -> None:
    trials = [1.0, 2.0, 3.0]
    assert dsr_dynamic(np.array([1.0, np.nan, 2.0]), trials, 252) == 0.0
    assert dsr_dynamic(np.array([1.0, np.inf, 2.0]), trials, 252) == 0.0
    assert dsr_dynamic(np.array([1.0, 1.0, 1.0]), trials, 252) == 0.0
    assert dsr_dynamic(np.array([1.0, 2.0]), [1.0], 252) == 0.0
    assert dsr_dynamic(np.array([1.0, 2.0]), [np.nan, 2.0], 252) == 0.0


def test_monte_carlo_supports_both_metric_directions() -> None:
    actuals = np.linspace(-1.0, 1.0, 40)
    predictions = actuals.copy()

    higher_p = monte_carlo(
        predictions, actuals, 200, metric_func=r2_score, higher_is_better=True
    )
    lower_p = monte_carlo(
        predictions,
        actuals,
        200,
        metric_func=lambda p, a: float(np.mean((p - a) ** 2)),
        higher_is_better=False,
    )
    binary_p = monte_carlo(
        np.array([0, 1] * 20), np.array([0, 1] * 20), 200
    )

    assert higher_p < 0.05
    assert lower_p < 0.05
    assert binary_p < 0.05


def test_monte_carlo_non_finite_metric_returns_nan() -> None:
    result = monte_carlo(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
        10,
        metric_func=lambda p, a: float("nan"),
    )
    assert math.isnan(result)


def test_oos_decay_rejects_nan_oos_score() -> None:
    features, target = make_returns(200)

    def train_only_metric(predictions, actuals):
        return 1.0 if len(actuals) > 100 else float("nan")

    result = oos_decay(
        build_linear_model,
        features,
        target,
        {"ridge": 1e-6},
        train_split_ratio=0.70,
        metric_func=train_only_metric,
    )

    assert result["success"] is False
    assert "non-finite" in result["error"]


def test_oos_decay_reports_clamped_tolerance() -> None:
    features, target = make_returns(200)

    result = oos_decay(
        build_linear_model,
        features,
        target,
        {"ridge": 1e-6, "max_decay_tolerance": 0.80},
        train_split_ratio=0.70,
        metric_func=r2_score,
    )

    assert result["tolerance"] == 0.50
    assert result["tolerance_clamped"] is True


def test_oos_decay_supports_lower_is_better_metrics() -> None:
    features, target = make_returns(200)
    mse = lambda p, a: float(np.mean((p - a) ** 2))

    result = oos_decay(
        build_linear_model,
        features,
        target,
        {"ridge": 1e-6, "max_decay_tolerance": 0.50},
        train_split_ratio=0.70,
        metric_func=mse,
        higher_is_better=False,
    )

    assert result["success"] is True
    assert result["decay_rate"] < 0.50


def test_pss_fails_with_zero_trials() -> None:
    features, target = make_returns(100)

    passed, message = pss(
        build_linear_model,
        features,
        target,
        {"min_dsr_limit": 0.95, "max_decay_tolerance": 0.2},
        sensitivity_range=0.1,
        max_performance_drop=0.2,
        metric_func=r2_score,
    )

    assert passed is False
    assert "No perturbable parameters" in message


def test_pss_fails_with_partial_trial_failures() -> None:
    features, target = make_returns(100)

    def flaky_builder(params, X, y):
        if params.get("alpha") != 1.0:
            raise RuntimeError("variant failed")
        return build_linear_model(params, X, y)

    passed, message = pss(
        flaky_builder,
        features,
        target,
        {"ridge": 1e-6, "alpha": 1.0},
        sensitivity_range=0.1,
        max_performance_drop=1.0,
        metric_func=r2_score,
    )

    assert passed is False
    assert "2/4" in message


def test_pss_rejects_infinite_variant_scores() -> None:
    features, target = make_returns(100)

    def inf_builder(params, X, y):
        model = build_linear_model(params, X, y)

        def predict(new_X):
            values = np.asarray(model(new_X), dtype=float)
            if params.get("ridge") != 1e-6:
                values = values.copy()
                values[0] = np.inf
            return values

        return predict

    passed, message = pss(
        inf_builder,
        features,
        target,
        {"ridge": 1e-6},
        sensitivity_range=0.1,
        max_performance_drop=1.0,
        metric_func=r2_score,
    )

    assert passed is False
    assert "All 2" in message


def test_pss_passes_with_stable_real_trials() -> None:
    features, target = make_returns(150)

    passed, message = pss(
        build_linear_model,
        features,
        target,
        {"ridge": 1e-6},
        sensitivity_range=0.1,
        max_performance_drop=1.0,
        metric_func=r2_score,
    )

    assert passed is True
    assert "2/2" in message


def test_validation_engine_passes_on_continuous_target() -> None:
    features, target, predictions, sharpes, winning_returns = make_engine_inputs()

    result = validation_engine(
        winning_returns=winning_returns,
        all_trial_sharpes=sharpes,
        predictions=predictions,
        actuals=target[int(len(target) * 0.70) :],
        features=features,
        targets=target,
        base_model_builder=build_linear_model,
        base_params=base_params(),
        periods_per_year=252,
        metric_func=r2_score,
        higher_is_better=True,
        data_description="synthetic continuous returns",
    )

    assert result["passed"] is True
    assert result["metric"] == "r2_score"
    assert result["data_description"] == "synthetic continuous returns"
    assert result["tolerance_clamped"] is False


def test_validation_engine_rejects_binary_and_constant_targets() -> None:
    features, _ = make_returns(100)
    common = dict(
        winning_returns=np.array([0.1, -0.1] * 50),
        all_trial_sharpes=[1.0, 2.0],
        predictions=np.zeros(100),
        actuals=np.zeros(100),
        features=features,
        base_model_builder=build_linear_model,
        base_params=base_params(),
        periods_per_year=252,
        metric_func=r2_score,
    )

    binary = validation_engine(targets=np.array([0.0, 1.0] * 50), **common)
    constant = validation_engine(targets=np.ones(100), **common)

    assert binary["passed"] is False
    assert binary["failed_layer"] == "Target contract"
    assert "continuous" in binary["message"]
    assert constant["passed"] is False
    assert constant["failed_layer"] == "Target contract"


def test_validation_engine_uses_min_dsr_as_the_layer_one_threshold() -> None:
    features, target, predictions, sharpes, winning_returns = make_engine_inputs()

    result = validation_engine(
        winning_returns=winning_returns,
        all_trial_sharpes=sharpes,
        predictions=predictions,
        actuals=target[int(len(target) * 0.70) :],
        features=features,
        targets=target,
        base_model_builder=build_linear_model,
        base_params=base_params(min_dsr_limit=1.0),
        periods_per_year=252,
        metric_func=r2_score,
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Layer 1 (DSR)"


def test_validation_engine_has_no_dead_predict_parameter() -> None:
    parameters = inspect.signature(validation_engine).parameters

    assert "model_predict_func" not in parameters
    assert "base_model_builder" in parameters
