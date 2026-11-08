"""Tests for in-slice metrics and diagnostics on non-continuous targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)
from whack26.validation.statistical import (
    HIGHER_IS_BETTER,
    ValidationContext,
    _metric_supported_for_permutation,
    _raw_permutation_p,
    expected_calibration_error,
    metric_value,
    pearson_dispersion,
    validate_in_slice,
)


def make_search_result(
    *,
    target_kind: str = "multiclass",
    metric: str = "balanced_accuracy",
    selected_scores: list[float] | None = None,
    baseline_scores: list[float] | None = None,
) -> ModelSearchResult:
    selected_scores = selected_scores or [0.82, 0.84, 0.80]
    baseline_scores = baseline_scores or [0.50, 0.52, 0.48]
    folds = [
        FoldResult(
            fold=index,
            train_rows=80,
            test_rows=40,
            metrics={metric: selected},
            baseline_metrics={metric: baseline},
        )
        for index, (selected, baseline) in enumerate(
            zip(selected_scores, baseline_scores)
        )
    ]
    candidate = ModelCandidate(
        spec=ModelSpec(family="logistic", target_kind=target_kind),
        folds=folds,
        metrics={metric: float(np.mean(selected_scores))},
        metric_std={metric: 0.02},
        diagnostics=[Diagnostic(name="residuals", status="ok")],
        coefficients={"x": 1.0},
        selected=True,
    )
    baseline = ModelCandidate(
        spec=ModelSpec(family="baseline", target_kind=target_kind),
        folds=[
            FoldResult(
                fold=index,
                train_rows=80,
                test_rows=40,
                metrics={metric: score},
                baseline_metrics={metric: score},
            )
            for index, score in enumerate(baseline_scores)
        ],
        metrics={metric: float(np.mean(baseline_scores))},
    )
    return ModelSearchResult(
        target_kind=target_kind,  # type: ignore[arg-type]
        metric=metric,  # type: ignore[arg-type]
        candidates=[candidate],
        selected=candidate,
        baseline=baseline,
    )


def make_context(**overrides) -> ValidationContext:
    values = {
        "hypothesis_id": "hyp_noncontinuous",
        "target": "y",
        "target_kind": "multiclass",
        "model_search": make_search_result(),
        "seed": 0,
    }
    values.update(overrides)
    return ValidationContext(**values)


# ---------------------------------------------------------------------------
# Binary metrics keep their historical behavior
# ---------------------------------------------------------------------------


def test_binary_metrics_keep_historical_behavior() -> None:
    actuals = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=float)
    probabilities = np.array([0.05, 0.35, 0.8, 0.9, 0.2, 0.6, 0.55, 0.45])

    labels = (probabilities >= 0.5).astype(float)
    assert metric_value("accuracy", actuals, probabilities) == pytest.approx(
        float(np.mean(labels == actuals))
    )
    assert metric_value(
        "balanced_accuracy", actuals, probabilities
    ) == pytest.approx(1.0)
    assert metric_value("auc", actuals, probabilities) == pytest.approx(1.0)

    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    expected_log_loss = -float(
        np.mean(
            actuals * np.log(clipped)
            + (1.0 - actuals) * np.log(1.0 - clipped)
        )
    )
    assert metric_value(
        "log_loss", actuals, probabilities
    ) == pytest.approx(expected_log_loss)

    expected_brier = float(np.mean((probabilities - actuals) ** 2))
    assert metric_value("brier", actuals, probabilities) == pytest.approx(
        expected_brier
    )


# ---------------------------------------------------------------------------
# Multiclass metric values
# ---------------------------------------------------------------------------


def test_multiclass_accuracy_balanced_accuracy_and_macro_f1() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
            [0.4, 0.3, 0.3],
        ]
    )

    assert metric_value("accuracy", truth, probabilities) == pytest.approx(5.0 / 6.0)
    assert metric_value(
        "balanced_accuracy", truth, probabilities
    ) == pytest.approx(5.0 / 6.0)

    expected_macro_f1 = float(np.mean([0.8, 1.0, 2.0 / 3.0]))
    assert metric_value(
        "macro_f1", truth, probabilities
    ) == pytest.approx(expected_macro_f1)


def test_multiclass_hard_label_codes() -> None:
    truth = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    labels = np.array([0, 1, 2, 0, 1, 1], dtype=float)

    assert metric_value("accuracy", truth, labels) == pytest.approx(5.0 / 6.0)
    expected_macro_f1 = float(np.mean([1.0, 0.8, 2.0 / 3.0]))
    assert metric_value("macro_f1", truth, labels) == pytest.approx(
        expected_macro_f1
    )
    assert math.isnan(metric_value("log_loss", truth, labels))


def test_multiclass_log_loss_and_brier() -> None:
    truth = np.array([0, 1, 2], dtype=float)
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
        ]
    )

    expected_log_loss = -float(np.mean(np.log([0.8, 0.7, 0.6])))
    assert metric_value(
        "log_loss", truth, probabilities
    ) == pytest.approx(expected_log_loss)

    one_hot = np.eye(3)
    expected_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    assert metric_value("brier", truth, probabilities) == pytest.approx(
        expected_brier
    )


def test_multiclass_mcc_and_one_vs_rest_auc() -> None:
    truth = np.repeat(np.arange(3), 4)
    perfect = np.eye(3)[truth]

    assert metric_value("mcc", truth, perfect) == pytest.approx(1.0)
    assert metric_value("auc", truth, perfect) == pytest.approx(1.0)

    flat = np.full((truth.size, 3), 1.0 / 3.0)
    assert metric_value("auc", truth, flat) == pytest.approx(0.5)

    binary_truth = np.array([0, 0, 1, 1], dtype=float)
    ordered = np.array([0.1, 0.2, 0.8, 0.9])
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert metric_value("mcc", binary_truth, ordered) == pytest.approx(1.0)
    assert metric_value("mcc", binary_truth, reversed_scores) == pytest.approx(
        -1.0
    )


# ---------------------------------------------------------------------------
# Count metric values
# ---------------------------------------------------------------------------


def test_count_metrics() -> None:
    actuals = np.array([0, 1, 2, 3, 4], dtype=float)
    predictions = np.array([0.5, 1.0, 1.5, 2.5, 3.5])

    assert metric_value("rmse", actuals, predictions) == pytest.approx(
        float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    )
    assert metric_value("mae", actuals, predictions) == pytest.approx(
        float(np.mean(np.abs(actuals - predictions)))
    )

    terms = []
    for actual, predicted in zip(actuals, predictions):
        if actual == 0.0:
            terms.append(predicted)
        else:
            terms.append(
                actual * math.log(actual / predicted) - (actual - predicted)
            )
    expected_deviance = 2.0 * float(np.mean(terms))
    assert metric_value(
        "poisson_deviance", actuals, predictions
    ) == pytest.approx(expected_deviance)


# ---------------------------------------------------------------------------
# Unsupported or malformed inputs
# ---------------------------------------------------------------------------


def test_unknown_metric_returns_nan() -> None:
    actuals = np.array([0, 1, 1, 0], dtype=float)
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])
    truth = np.array([0, 1, 2], dtype=float)
    matrix = np.eye(3)

    assert math.isnan(metric_value("mystery", actuals, probabilities))
    assert math.isnan(metric_value("mystery", truth, matrix))


def test_shape_and_non_finite_inputs_return_nan() -> None:
    assert math.isnan(metric_value("accuracy", [0, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(metric_value("log_loss", [0, 1, 2], [0.1, 0.2]))
    assert math.isnan(metric_value("accuracy", [0, float("nan")], [0.1, 0.2]))
    assert math.isnan(
        metric_value("log_loss", [0, 1], [[0.8, 0.2], [0.1, float("nan")]])
    )
    assert math.isnan(metric_value("rmse", [0, 1], np.eye(2)))


def test_metric_directions_and_permutation_support() -> None:
    for name in ("brier", "macro_f1", "mcc"):
        assert name in HIGHER_IS_BETTER
        assert _metric_supported_for_permutation(name)
    assert HIGHER_IS_BETTER["brier"] is False
    assert HIGHER_IS_BETTER["macro_f1"] is True
    assert HIGHER_IS_BETTER["mcc"] is True
    assert not _metric_supported_for_permutation("mystery")


# ---------------------------------------------------------------------------
# Permutation p-values for the new primary metrics
# ---------------------------------------------------------------------------


def test_permutation_p_value_for_macro_f1() -> None:
    truth = np.repeat(np.arange(3), 20)
    perfect = np.eye(3)[truth]

    p_value = _raw_permutation_p(
        "macro_f1", truth, perfect, seed=0, n_permutations=99
    )

    assert 0.0 < p_value <= 0.05


def test_permutation_p_value_for_poisson_deviance() -> None:
    rng = np.random.default_rng(12)
    predictions = rng.uniform(1.0, 8.0, size=80)
    actuals = predictions.copy()

    p_value = _raw_permutation_p(
        "poisson_deviance", actuals, predictions, seed=0, n_permutations=99
    )

    assert 0.0 < p_value <= 0.05


# ---------------------------------------------------------------------------
# Calibration and dispersion diagnostics
# ---------------------------------------------------------------------------


def test_expected_calibration_error_binary_and_multiclass() -> None:
    actuals = np.tile(np.array([0.0, 1.0]), 50)
    probabilities = np.where(actuals == 1.0, 0.9, 0.1)
    assert expected_calibration_error(
        probabilities, actuals
    ) == pytest.approx(0.1)

    truth = np.repeat(np.arange(3), 20)
    matrix = np.full((truth.size, 3), 0.1)
    matrix[np.arange(truth.size), truth] = 0.8
    assert expected_calibration_error(matrix, truth) == pytest.approx(0.2)

    assert math.isnan(expected_calibration_error(np.zeros(10), np.arange(10)))


def test_pearson_dispersion() -> None:
    predictions = np.full(8, 2.0)
    actuals = np.array([1, 2, 3, 4, 3, 2, 1, 2], dtype=float)
    expected = float(np.sum((actuals - predictions) ** 2 / predictions) / 7.0)

    assert pearson_dispersion(predictions, actuals) == pytest.approx(expected)
    assert pearson_dispersion(actuals, actuals) == pytest.approx(0.0)
    assert math.isnan(pearson_dispersion([1.0, -1.0], [0.0, 1.0]))


def test_in_slice_adds_calibration_diagnostic_for_binary() -> None:
    rng = np.random.default_rng(5)
    actuals = rng.integers(0, 2, size=240).astype(float)
    probabilities = np.where(actuals == 1.0, 0.85, 0.15)
    probabilities += rng.normal(0.0, 0.02, size=actuals.size)
    probabilities = np.clip(probabilities, 0.01, 0.99)
    search = make_search_result(
        target_kind="binary",
        metric="log_loss",
        selected_scores=[0.20, 0.22, 0.18],
        baseline_scores=[0.70, 0.70, 0.70],
    )

    result = validate_in_slice(
        make_context(
            target_kind="binary",
            model_search=search,
            predictions=probabilities,
            actuals=actuals,
            max_p=1.0,
            n_permutations=50,
        )
    )

    item = next(
        (
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.name == "expected_calibration_error"
        ),
        None,
    )
    assert item is not None
    assert item.value is not None
    assert 0.0 <= item.value <= 1.0


def test_in_slice_adds_calibration_diagnostic_for_multiclass() -> None:
    truth = np.repeat(np.arange(3), 40)
    probabilities = np.full((truth.size, 3), 0.1)
    probabilities[np.arange(truth.size), truth] = 0.8
    search = make_search_result(
        target_kind="multiclass",
        metric="balanced_accuracy",
        selected_scores=[0.80, 0.82, 0.78],
        baseline_scores=[0.45, 0.47, 0.43],
    )

    result = validate_in_slice(
        make_context(
            target_kind="multiclass",
            model_search=search,
            predictions=probabilities,
            actuals=truth,
            max_p=1.0,
            n_permutations=50,
        )
    )

    item = next(
        (
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.name == "expected_calibration_error"
        ),
        None,
    )
    assert item is not None
    assert item.value == pytest.approx(0.2)
    assert result.permutation_source == "predictions"


def test_in_slice_adds_dispersion_diagnostic_for_count() -> None:
    rng = np.random.default_rng(9)
    predictions = rng.uniform(1.0, 9.0, size=150)
    actuals = rng.poisson(predictions).astype(float)
    search = make_search_result(
        target_kind="count",
        metric="poisson_deviance",
        selected_scores=[0.70, 0.72, 0.68],
        baseline_scores=[1.40, 1.40, 1.40],
    )

    result = validate_in_slice(
        make_context(
            target_kind="count",
            model_search=search,
            predictions=predictions,
            actuals=actuals,
            max_p=1.0,
            n_permutations=50,
        )
    )

    item = next(
        (
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.name == "pearson_dispersion"
        ),
        None,
    )
    assert item is not None
    assert item.value is not None
    assert item.value > 0.0


def test_unavailable_calibration_diagnostic_only_warns() -> None:
    truth = np.repeat(np.arange(3), 10)
    hard_labels = truth.astype(float).copy()
    search = make_search_result(
        target_kind="multiclass",
        metric="balanced_accuracy",
        selected_scores=[0.80, 0.82, 0.78],
        baseline_scores=[0.40, 0.42, 0.38],
    )

    result = validate_in_slice(
        make_context(
            target_kind="multiclass",
            model_search=search,
            predictions=hard_labels,
            actuals=truth,
            max_p=1.0,
            n_permutations=50,
        )
    )

    assert result.verdict == "pass"
    assert not any(
        diagnostic.name == "expected_calibration_error"
        for diagnostic in result.diagnostics
    )
    assert any(
        "expected_calibration_error_unavailable" in warning
        for warning in result.warnings
    )


def test_unavailable_dispersion_diagnostic_only_warns() -> None:
    actuals = np.array([0.0, 1.0, 2.0] * 20)
    predictions = actuals.copy()
    search = make_search_result(
        target_kind="count",
        metric="mae",
        selected_scores=[0.20, 0.22, 0.18],
        baseline_scores=[0.70, 0.70, 0.70],
    )

    result = validate_in_slice(
        make_context(
            target_kind="count",
            model_search=search,
            predictions=predictions,
            actuals=actuals,
            max_p=1.0,
            n_permutations=20,
        )
    )

    assert result.verdict == "pass"
    assert not any(
        diagnostic.name == "pearson_dispersion"
        for diagnostic in result.diagnostics
    )
    assert any(
        "pearson_dispersion_unavailable" in warning
        for warning in result.warnings
    )


def test_no_diagnostics_without_raw_predictions() -> None:
    result = validate_in_slice(
        make_context(
            target_kind="multiclass",
            model_search=make_search_result(
                target_kind="multiclass",
                metric="balanced_accuracy",
            ),
            max_p=1.0,
        )
    )

    names = {diagnostic.name for diagnostic in result.diagnostics}
    assert "expected_calibration_error" not in names
    assert "pearson_dispersion" not in names
