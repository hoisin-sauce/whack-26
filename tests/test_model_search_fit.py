from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from whack26.model_search import metrics
from whack26.model_search.fit import (
    BackendUnavailableError,
    FitError,
    apply_target_transform,
    fit_model,
    inverse_target_transform,
    register_fitter,
    unregister_fitter,
)
from whack26.model_search.types import ModelSpec


def _linear_frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.01, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_linear_coefficient_recovery() -> None:
    frame = _linear_frame()
    spec = ModelSpec(
        family="linear",
        target_kind="continuous",
        features=["x1", "x2"],
    )
    model = fit_model(spec, frame, "y")

    assert model.converged
    assert model.coefficients["intercept"] == pytest.approx(3.0, abs=0.01)
    assert model.coefficients["x1"] == pytest.approx(2.0, abs=0.01)
    assert model.coefficients["x2"] == pytest.approx(-1.5, abs=0.01)

    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert metrics.r2_score(frame["y"], predictions) > 0.999
    assert math.isfinite(model.log_likelihood)
    assert model.aic is not None and model.bic is not None


def test_linear_target_transform_inverts() -> None:
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n)
    z = 0.5 + 0.3 * x
    y = np.expm1(z)
    frame = pd.DataFrame({"x": x, "y": y})

    spec = ModelSpec(
        family="linear",
        target_kind="continuous",
        features=["x"],
        target_transform="log1p",
    )
    model = fit_model(spec, frame, "y")

    assert model.coefficients["intercept"] == pytest.approx(0.5, abs=0.001)
    assert model.coefficients["x"] == pytest.approx(0.3, abs=0.001)
    assert np.allclose(model.predict(frame), y, atol=1e-8)

    transformed = apply_target_transform(y, "log1p")
    assert np.allclose(inverse_target_transform(transformed, "log1p"), y)


def test_target_transform_domain_is_validated() -> None:
    with pytest.raises(FitError):
        apply_target_transform([-2.0, 1.0], "log1p")
    with pytest.raises(FitError):
        apply_target_transform([-1.0, 1.0], "sqrt")


def test_ridge_is_finite_with_collinear_features() -> None:
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    x2 = 2.0 * x1 + rng.normal(scale=1e-6, size=n)
    y = 1.0 + x1 + x2 + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    spec = ModelSpec(
        family="linear",
        target_kind="continuous",
        features=["x1", "x2"],
        regularization=1.0,
    )
    model = fit_model(spec, frame, "y")
    assert model.converged
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert metrics.r2_score(y, predictions) > 0.99


def test_interaction_coefficient_recovery() -> None:
    rng = np.random.default_rng(4)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 + 2.0 * x1 * x2 + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    spec = ModelSpec(
        family="interaction",
        target_kind="continuous",
        features=["x1", "x2"],
        interactions=[("x1", "x2")],
    )
    model = fit_model(spec, frame, "y")

    assert model.coefficients["x1:x2"] == pytest.approx(2.0, abs=0.01)
    assert metrics.r2_score(y, model.predict(frame)) > 0.999


def test_logistic_fit_and_probabilities() -> None:
    rng = np.random.default_rng(5)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(0.3 + 1.5 * x1 - 1.0 * x2)))
    y = rng.binomial(1, probability)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    spec = ModelSpec(
        family="logistic", target_kind="binary", features=["x1", "x2"]
    )
    model = fit_model(spec, frame, "y")

    assert model.converged
    assert model.coefficients["x1"] == pytest.approx(1.5, abs=0.25)
    assert model.coefficients["x2"] == pytest.approx(-1.0, abs=0.25)

    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (n, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

    labels = model.predict(frame)
    assert set(np.unique(labels)) <= {0, 1}
    block = metrics.metrics_for_target(
        "binary", y, labels, probabilities=probabilities, labels=model.classes
    )
    assert block["auc"] > 0.8
    assert math.isfinite(block["log_loss"])


def test_multinomial_fit_and_probabilities() -> None:
    rng = np.random.default_rng(6)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scores = np.column_stack(
        [0.5 * x1, -0.5 * x1 + x2, -x2]
    )
    y = np.argmax(scores + rng.normal(scale=0.5, size=(n, 3)), axis=1)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    spec = ModelSpec(
        family="multinomial",
        target_kind="multiclass",
        features=["x1", "x2"],
    )
    model = fit_model(spec, frame, "y")

    assert model.converged
    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (n, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    labels = model.predict(frame)
    assert set(np.unique(labels)) <= {0, 1, 2}
    block = metrics.metrics_for_target(
        "multiclass",
        y,
        labels,
        probabilities=probabilities,
        labels=model.classes,
    )
    assert block["accuracy"] > 0.65
    assert block["auc"] > 0.8


def test_poisson_fit_recovers_coefficients() -> None:
    rng = np.random.default_rng(7)
    n = 2500
    x1 = rng.normal(size=n)
    mean = np.exp(0.5 + 0.4 * x1)
    y = rng.poisson(mean)
    frame = pd.DataFrame({"x1": x1, "y": y})

    spec = ModelSpec(
        family="poisson", target_kind="count", features=["x1"]
    )
    model = fit_model(spec, frame, "y")

    assert model.converged
    assert model.coefficients["intercept"] == pytest.approx(0.5, abs=0.08)
    assert model.coefficients["x1"] == pytest.approx(0.4, abs=0.08)
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)
    assert math.isfinite(model.log_likelihood)
    assert metrics.poisson_deviance(y, predictions) < metrics.poisson_deviance(
        y, np.full(n, float(np.mean(y)))
    )


def test_negative_binomial_estimates_overdispersion() -> None:
    rng = np.random.default_rng(8)
    n = 3000
    x1 = rng.normal(size=n)
    mean = np.exp(0.3 + 0.5 * x1)
    shape = 2.0
    y = rng.negative_binomial(shape, shape / (shape + mean))
    frame = pd.DataFrame({"x1": x1, "y": y})

    spec = ModelSpec(
        family="negative_binomial",
        target_kind="count",
        features=["x1"],
    )
    model = fit_model(spec, frame, "y")

    assert model.converged
    assert model.coefficients["x1"] == pytest.approx(0.5, abs=0.15)
    assert 0.1 < model.estimates["dispersion"] < 1.5
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)


def test_group_comparison_recovers_group_means() -> None:
    frame = pd.DataFrame(
        {
            "g": np.repeat(["a", "b", "c"], 40),
            "y": np.concatenate(
                [np.full(40, 1.0), np.full(40, 2.0), np.full(40, 3.0)]
            ),
        }
    )
    spec = ModelSpec(
        family="group_comparison",
        target_kind="continuous",
        features=["g"],
    )
    model = fit_model(spec, frame, "y")

    predictions = model.predict(frame)
    assert np.allclose(predictions[:40], 1.0)
    assert np.allclose(predictions[40:80], 2.0)
    assert np.allclose(predictions[80:], 3.0)
    assert model.coefficients["g=a"] == pytest.approx(1.0)

    unseen = pd.DataFrame({"g": ["z"]})
    assert np.isfinite(model.predict(unseen)).all()


def test_group_comparison_rates_for_binary_targets() -> None:
    rng = np.random.default_rng(9)
    groups = np.repeat(["a", "b", "c"], 500)
    rates = {"a": 0.2, "b": 0.5, "c": 0.8}
    y = np.array([rng.binomial(1, rates[group]) for group in groups])
    frame = pd.DataFrame({"g": groups, "y": y})

    spec = ModelSpec(
        family="group_comparison",
        target_kind="binary",
        features=["g"],
    )
    model = fit_model(spec, frame, "y")
    probabilities = model.predict_proba(frame)
    assert probabilities.shape == (len(frame), 2)
    for group, rate in rates.items():
        mask = groups == group
        assert probabilities[mask, 1].mean() == pytest.approx(rate, abs=0.08)


def test_group_comparison_bins_a_numeric_factor() -> None:
    rng = np.random.default_rng(10)
    x = rng.uniform(0.0, 10.0, size=300)
    y = x + rng.normal(scale=0.1, size=300)
    frame = pd.DataFrame({"x": x, "y": y})

    spec = ModelSpec(
        family="group_comparison",
        target_kind="continuous",
        features=["x"],
        hyperparameters={"bins": 5, "statistic": "mean"},
    )
    model = fit_model(spec, frame, "y")
    assert len(model.coefficients) == 5
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert metrics.r2_score(y, predictions) > 0.95


def test_baseline_continuous_is_the_mean() -> None:
    frame = _linear_frame()
    spec = ModelSpec(family="baseline", target_kind="continuous")
    model = fit_model(spec, frame, "y")
    predictions = model.predict(frame)
    assert np.allclose(predictions, float(frame["y"].mean()))
    assert model.coefficients["intercept"] == pytest.approx(
        float(frame["y"].mean())
    )


def test_baseline_classification_is_the_majority_class() -> None:
    frame = pd.DataFrame({"y": ["a", "a", "a", "b"]})
    spec = ModelSpec(family="baseline", target_kind="binary")
    model = fit_model(spec, frame, "y")
    assert np.all(model.predict(frame) == "a")
    probabilities = model.predict_proba(frame)
    assert probabilities[0, 1] == pytest.approx(0.25)


def test_missing_numeric_features_are_imputed_from_training_data() -> None:
    frame = _linear_frame()
    frame.loc[frame.index[::11], "x1"] = np.nan
    spec = ModelSpec(
        family="linear",
        target_kind="continuous",
        features=["x1", "x2"],
    )
    model = fit_model(spec, frame, "y")
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))


def test_categorical_feature_is_one_hot_encoded() -> None:
    rng = np.random.default_rng(11)
    n = 300
    group = rng.choice(["a", "b", "c"], size=n)
    y = np.where(group == "a", 1.0, np.where(group == "b", 2.0, 3.0))
    y = y + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame({"g": group, "y": y})

    spec = ModelSpec(
        family="linear", target_kind="continuous", features=["g"]
    )
    model = fit_model(spec, frame, "y")
    assert "g=b" in model.coefficients
    assert "g=c" in model.coefficients
    assert metrics.r2_score(y, model.predict(frame)) > 0.99


def test_feature_transform_is_applied_before_fitting() -> None:
    rng = np.random.default_rng(12)
    n = 300
    x = rng.uniform(0.1, 10.0, size=n)
    y = 1.0 + 2.0 * np.log1p(x) + rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame({"x": x, "y": y})

    spec = ModelSpec(
        family="linear",
        target_kind="continuous",
        features=["x"],
        feature_transforms={"x": "log1p"},
    )
    model = fit_model(spec, frame, "y")
    assert model.coefficients["x"] == pytest.approx(2.0, abs=0.01)
    assert metrics.r2_score(y, model.predict(frame)) > 0.999


def test_poisson_supports_an_exposure_offset() -> None:
    rng = np.random.default_rng(13)
    n = 2500
    x = rng.normal(size=n)
    exposure = rng.uniform(1.0, 5.0, size=n)
    rate = np.exp(0.2 + 0.3 * x)
    y = rng.poisson(exposure * rate)
    frame = pd.DataFrame({"x": x, "exposure": exposure, "y": y})

    spec = ModelSpec(
        family="poisson",
        target_kind="count",
        features=["x"],
        hyperparameters={"offset": "exposure"},
    )
    model = fit_model(spec, frame, "y")
    assert model.coefficients["x"] == pytest.approx(0.3, abs=0.08)
    predictions = model.predict(frame)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions > 0.0)


def test_logistic_class_weights_are_applied() -> None:
    rng = np.random.default_rng(14)
    n = 2000
    x = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(-1.0 + 1.5 * x)))
    y = rng.binomial(1, probability)
    # Make the positive class rare.
    y[np.flatnonzero(y == 1)[: int(0.7 * np.sum(y))]] = 0
    frame = pd.DataFrame({"x": x, "y": y})

    spec = ModelSpec(
        family="logistic",
        target_kind="binary",
        features=["x"],
        hyperparameters={"class_weight": "balanced"},
    )
    model = fit_model(spec, frame, "y")
    assert model.converged
    probabilities = model.predict_proba(frame)
    assert np.all(np.isfinite(probabilities))
    assert model.coefficients["x"] > 0.0


def test_baseline_probabilities_follow_the_prediction_frame_length() -> None:
    frame = pd.DataFrame({"y": ["a", "a", "a", "b"]})
    spec = ModelSpec(family="baseline", target_kind="binary")
    model = fit_model(spec, frame, "y")
    probabilities = model.predict_proba(frame.iloc[:2])
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_tree_family_requires_an_optional_backend() -> None:
    frame = _linear_frame()
    spec = ModelSpec(family="tree", target_kind="continuous", features=["x1"])
    with pytest.raises(BackendUnavailableError):
        fit_model(spec, frame, "y")


def test_optional_backend_can_be_registered() -> None:
    from whack26.model_search.fit import FittedModel

    def fake_tree(spec, frame, target, *, sample_weight=None):
        return FittedModel(
            spec=spec,
            target=target,
            coefficients={"constant": 1.0},
            n_params=1,
            n_train=len(frame),
            _predict=lambda new_frame: np.ones(len(new_frame)),
        )

    register_fitter("tree", fake_tree)
    try:
        frame = _linear_frame()
        spec = ModelSpec(
            family="tree", target_kind="continuous", features=["x1"]
        )
        model = fit_model(spec, frame, "y")
        assert np.all(model.predict(frame) == 1.0)
    finally:
        unregister_fitter("tree")
