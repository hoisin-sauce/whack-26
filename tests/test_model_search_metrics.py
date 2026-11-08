from __future__ import annotations

import math

import numpy as np
import pytest

from whack26.model_search import metrics


def test_regression_metrics_are_exact_on_perfect_predictions() -> None:
    y = [1.0, 2.0, 3.0, 4.0]
    prediction = [1.0, 2.0, 3.0, 4.0]

    assert metrics.r2_score(y, prediction) == pytest.approx(1.0)
    assert metrics.rmse(y, prediction) == pytest.approx(0.0)
    assert metrics.mae(y, prediction) == pytest.approx(0.0)
    assert metrics.mape(y, prediction) == pytest.approx(0.0)


def test_r2_is_zero_for_the_mean_baseline() -> None:
    y = np.array([1.0, 2.0, 3.0])
    prediction = np.full(3, 2.0)
    assert metrics.r2_score(y, prediction) == pytest.approx(0.0)


def test_mape_ignores_zero_and_near_zero_targets() -> None:
    y = np.array([0.0, 1e-12, 2.0, 4.0])
    prediction = np.array([1.0, 1.0, 3.0, 3.0])
    # Only 2 and 4 remain: |2-3|/2 and |4-3|/4 -> 37.5 percent.
    assert metrics.mape(y, prediction) == pytest.approx(37.5)


def test_binary_metrics_and_auc_with_ties() -> None:
    y = np.array([0, 0, 1, 1])
    prediction = np.array([0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert metrics.accuracy(y, prediction) == pytest.approx(0.75)
    assert metrics.balanced_accuracy(y, prediction) == pytest.approx(0.75)
    assert metrics.binary_auc(y, scores) == pytest.approx(0.75)
    assert 0.0 < metrics.log_loss(y, scores) < 1.0
    assert 0.0 <= metrics.brier_score(y, scores) <= 1.0

    # All scores tied -> chance-level AUC.
    assert metrics.binary_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_multiclass_metrics() -> None:
    y = np.array(["a", "b", "c", "a", "b", "c"])
    prediction = np.array(["a", "b", "c", "a", "c", "c"])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
        ]
    )

    assert metrics.accuracy(y, prediction) == pytest.approx(5 / 6)
    assert metrics.balanced_accuracy(y, prediction) == pytest.approx(5 / 6)
    assert 0.0 < metrics.macro_auc_ovr(y, probabilities) <= 1.0
    assert metrics.log_loss(y, probabilities) > 0.0


def test_poisson_deviance_and_overdispersion() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    perfect = y.copy()
    assert metrics.poisson_deviance(y, perfect) == pytest.approx(0.0)
    assert metrics.poisson_deviance(y, np.full(4, 1.5)) > 0.0
    assert metrics.overdispersion_ratio(y, np.full(4, 1.5)) >= 0.0


def test_aic_bic_and_log_likelihood() -> None:
    y = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.1, 1.9, 3.2])
    log_likelihood = metrics.gaussian_log_likelihood(y, prediction)
    assert math.isfinite(log_likelihood)
    assert metrics.aic(log_likelihood, 2) == pytest.approx(
        2 * 2 - 2 * log_likelihood
    )
    assert metrics.bic(log_likelihood, 2, 3) == pytest.approx(
        math.log(3) * 2 - 2 * log_likelihood
    )


def test_metrics_for_target_reports_the_expected_block() -> None:
    continuous = metrics.metrics_for_target(
        "continuous",
        [1.0, 2.0, 3.0],
        [1.1, 1.9, 3.2],
        n_params=2,
        log_likelihood=-2.0,
    )
    assert {"r2", "rmse", "mae", "aic", "bic"} <= set(continuous)

    binary = metrics.metrics_for_target(
        "binary",
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        probabilities=[0.1, 0.4, 0.35, 0.8],
    )
    assert {"accuracy", "balanced_accuracy", "auc", "log_loss", "brier"} <= set(
        binary
    )

    count = metrics.metrics_for_target(
        "count", [0, 1, 2, 3], [0.5, 1.2, 1.9, 3.1], n_params=2
    )
    assert {"poisson_deviance", "rmse", "mae", "overdispersion"} <= set(count)


def test_direction_metadata() -> None:
    assert metrics.metric_direction("r2") == "higher"
    assert metrics.metric_direction("rmse") == "lower"
    assert metrics.is_higher_better("auc")
    assert not metrics.is_higher_better("log_loss")
    assert metrics.primary_metric("continuous") == "r2"
    assert metrics.primary_metric("count") == "poisson_deviance"
    assert "accuracy" in metrics.metric_names("binary")


def test_short_aliases_cover_binary_and_multiclass_auc() -> None:
    y = np.array([0, 0, 1, 1])
    assert metrics.r2([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert metrics.auc(y, [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)

    labels = ["a", "b", "c"]
    truth = np.array(["a", "b", "c", "a"])
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )
    assert metrics.auc(truth, probabilities, labels=labels) == pytest.approx(1.0)


def test_undefined_metrics_are_nan_not_errors() -> None:
    assert metrics.r2_score([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert metrics.r2_score([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.0)
    assert math.isnan(metrics.binary_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(metrics.poisson_deviance([1.0], [0.0]))
