from __future__ import annotations

import math

import numpy as np

from stats_engine import (
    TARGET_BINARY,
    TARGET_CONTINUOUS,
    TARGET_COUNT,
    TARGET_MULTICLASS,
    accuracy_score,
    brier_score,
    build_logistic_model,
    build_poisson_model,
    build_softmax_model,
    expected_calibration_error,
    infer_target_type,
    log_loss,
    macro_f1_score,
    mcc_score,
    monte_carlo,
    pearson_dispersion,
    poisson_deviance,
    r2_score,
    roc_auc_score,
    validate_model,
)


def make_binary_data(n: int = 300, seed: int = 3):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    logits = 1.1 * features[:, 0] - 0.7 * features[:, 1] + 0.2 * features[:, 2]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = rng.binomial(1, probabilities)
    return features, targets


def make_multiclass_data(n: int = 300, seed: int = 4):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    centres = rng.normal(size=(3, 3))
    distances = np.stack(
        [np.sum((features - centre) ** 2, axis=1) for centre in centres], axis=1
    )
    targets = np.argmin(distances + rng.normal(0.0, 0.2, distances.shape), axis=1)
    return features, targets


def make_count_data(n: int = 300, seed: int = 5):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    means = np.exp(0.6 + 0.35 * features[:, 0] - 0.25 * features[:, 1])
    targets = rng.poisson(means)
    return features, targets


def loose_options() -> dict:
    return {
        "max_p_value": 1.0,
        "num_permutations": 100,
        "max_performance_drop": 10.0,
        "max_decay_tolerance": 2.0,
    }


def test_infer_target_type_distinguishes_supported_families() -> None:
    assert infer_target_type(np.array([0, 1, 1, 0])) == TARGET_BINARY
    assert infer_target_type(np.array([0, 1, 2, 2, 1, 0])) == TARGET_MULTICLASS
    assert infer_target_type(np.arange(25)) == TARGET_COUNT
    assert infer_target_type(np.array([0.1, 0.2, 0.3, 0.5])) == TARGET_CONTINUOUS
    assert (
        infer_target_type(np.array(["buy", "hold", "sell", "buy"]))
        == TARGET_MULTICLASS
    )
    assert infer_target_type(np.ones(10)) == "constant"


def test_probability_metrics_are_correct_for_clear_cases() -> None:
    actuals = np.array([0, 1, 1, 0, 1])
    probabilities = np.array([0.01, 0.99, 0.98, 0.02, 0.97])
    hard_labels = np.array([0, 1, 1, 0, 1])

    true_probabilities = np.array([0.99, 0.99, 0.98, 0.98, 0.97])
    expected_log_loss = float(-np.mean(np.log(true_probabilities)))
    expected_brier = float(np.mean((probabilities - actuals) ** 2))
    assert math.isclose(
        log_loss(probabilities, actuals), expected_log_loss, abs_tol=1e-12
    )
    assert math.isclose(
        brier_score(probabilities, actuals), expected_brier, abs_tol=1e-12
    )
    assert roc_auc_score(probabilities, actuals) == 1.0
    assert accuracy_score(hard_labels, actuals) == 1.0
    assert macro_f1_score(hard_labels, actuals) == 1.0
    assert mcc_score(hard_labels, actuals) == 1.0


def test_multiclass_probability_metrics_and_calibration() -> None:
    actuals = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[actuals]
    assert math.isclose(log_loss(probabilities, actuals), 0.0, abs_tol=1e-9)
    assert math.isclose(brier_score(probabilities, actuals), 0.0, abs_tol=1e-9)
    assert math.isclose(
        expected_calibration_error(probabilities, actuals), 0.0, abs_tol=1e-9
    )


def test_count_metrics_and_dispersion() -> None:
    actuals = np.array([0.0, 1.0, 2.0, 4.0])
    means = np.array([1.0, 1.0, 2.0, 2.0])
    expected_deviance = 2.0 * (
        np.sum(actuals * np.log(np.where(actuals > 0, actuals / means, 1.0)))
        - np.sum(actuals - means)
    ) / actuals.size
    assert math.isclose(
        poisson_deviance(means, actuals), expected_deviance, abs_tol=1e-12
    )
    assert poisson_deviance(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    dispersion = pearson_dispersion(np.full(100, 2.0), np.full(100, 2.0))
    assert dispersion == 0.0


def test_metric_builders_have_expected_output_shapes() -> None:
    features, binary_target = make_binary_data(60)
    logistic = build_logistic_model({"ridge": 0.1}, features[:40], binary_target[:40])
    binary_predictions = logistic(features[40:])
    assert binary_predictions.shape == (20,)
    assert np.all((binary_predictions >= 0.0) & (binary_predictions <= 1.0))

    multi_features, multi_target = make_multiclass_data(90)
    softmax = build_softmax_model(
        {"ridge": 0.1, "max_iter": 100}, multi_features[:70], multi_target[:70]
    )
    multi_predictions = softmax(multi_features[70:])
    assert multi_predictions.shape == (20, 3)
    assert np.allclose(multi_predictions.sum(axis=1), 1.0)

    count_features, count_target = make_count_data(90)
    poisson = build_poisson_model(
        {"ridge": 0.1}, count_features[:70], count_target[:70]
    )
    count_predictions = poisson(count_features[70:])
    assert count_predictions.shape == (20,)
    assert np.all(count_predictions > 0.0)


def test_monte_carlo_accepts_probability_matrices_and_blocks() -> None:
    _, targets = make_multiclass_data(80)
    probabilities = np.eye(3)[targets]
    p_value = monte_carlo(
        probabilities,
        targets,
        50,
        metric_func=log_loss,
        higher_is_better=False,
        block_size=5,
    )
    assert math.isfinite(p_value)
    assert 0.0 < p_value <= 1.0


def test_validate_model_supports_string_class_labels() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(240, 3))
    targets = np.where(
        features[:, 0] + rng.normal(0.0, 0.5, features.shape[0]) > 0,
        "up",
        "down",
    )

    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {"ridge": 0.1},
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["target_type"] == TARGET_BINARY
    assert "roc_auc" in result["diagnostics"]


def test_validate_model_binary_classification() -> None:
    features, targets = make_binary_data()
    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {"ridge": 0.1},
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["target_type"] == TARGET_BINARY
    assert result["metric"] == "log_loss"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert result["diagnostics"]["roc_auc"] > 0.6
    assert result["diagnostics"]["expected_calibration_error"] >= 0.0
    assert result["parameter_sensitivity"] == "passed"


def test_validate_model_multiclass_classification() -> None:
    features, targets = make_multiclass_data()
    result = validate_model(
        features,
        targets,
        build_softmax_model,
        {"ridge": 0.1, "max_iter": 200},
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["target_type"] == TARGET_MULTICLASS
    assert result["metric"] == "log_loss"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert "macro_f1" in result["diagnostics"]


def test_validate_model_count_target() -> None:
    features, targets = make_count_data()
    result = validate_model(
        features,
        targets,
        build_poisson_model,
        {"ridge": 0.1},
        target_type=TARGET_COUNT,
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["target_type"] == TARGET_COUNT
    assert result["metric"] == "poisson_deviance"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert "dispersion" in result["diagnostics"]


def test_validate_model_marks_missing_hyperparameters_not_applicable() -> None:
    features, targets = make_binary_data()
    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {},
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["parameter_sensitivity"] == "not_applicable"


def test_validate_model_requires_hyperparameters_when_requested() -> None:
    features, targets = make_binary_data()
    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {},
        require_parameter_sensitivity=True,
        **loose_options(),
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Layer 4 (Parameter Sensitivity)"


def test_validate_model_accepts_negative_r2_baseline() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(120, 2))
    targets = rng.normal(size=120)

    def bad_builder(params, X, y):
        return lambda new_X: np.full(new_X.shape[0], 100.0)

    result = validate_model(
        features,
        targets,
        bad_builder,
        {},
        target_type=TARGET_CONTINUOUS,
        metric_func=r2_score,
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["oos_score"] < 0.0
    assert result["layers"]["dsr"]["status"] == "not_applicable"


def test_validate_model_rejects_constant_target() -> None:
    features, _ = make_binary_data(60)
    result = validate_model(
        features,
        np.ones(60),
        build_logistic_model,
        {"ridge": 0.1},
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Target contract"


def test_validate_model_calibration_gate() -> None:
    features, targets = make_binary_data()
    overconfident = np.full(targets.shape, 0.95)

    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {"ridge": 0.1},
        predictions=overconfident,
        actuals=targets,
        max_ece=0.10,
        **loose_options(),
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Calibration"


def test_validate_model_dispersion_gate() -> None:
    features, targets = make_count_data()
    result = validate_model(
        features,
        targets,
        build_poisson_model,
        {"ridge": 0.1},
        target_type=TARGET_COUNT,
        max_dispersion=0.10,
        **loose_options(),
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Count dispersion"


def test_validate_model_honours_unclamped_decay_tolerance() -> None:
    features, targets = make_binary_data()
    result = validate_model(
        features,
        targets,
        build_logistic_model,
        {"ridge": 0.1},
        **loose_options(),
    )

    assert result["passed"] is True
    assert result["tolerance"] == 2.0
    assert result["tolerance_clamped"] is False


def test_validate_model_rejects_probability_matrix_with_wrong_width() -> None:
    features, targets = make_multiclass_data()
    probabilities = np.full((targets.size, 2), 0.5)

    result = validate_model(
        features,
        targets,
        build_softmax_model,
        {"ridge": 0.1},
        predictions=probabilities,
        actuals=targets,
        **loose_options(),
    )

    assert result["passed"] is False
    assert result["failed_layer"] == "Metric"
