"""Tests for the optional sklearn and pysindy backends."""

from __future__ import annotations

import numpy as np
import pytest

from whack26.model_search import sindy, trees
from whack26.model_search.types import ModelSpec


def test_sklearn_backend_reports_availability() -> None:
    status = trees.tree_backend_status()

    assert status["backend"] == "sklearn"
    assert status["available"] is trees.sklearn_available()
    assert sorted(trees.TREE_FAMILIES) == status["families"]
    if not trees.sklearn_available():
        assert status["reason"]


def test_tree_backend_skips_when_sklearn_is_missing() -> None:
    if trees.sklearn_available():
        pytest.skip("scikit-learn is installed")

    spec = ModelSpec(family="tree", features=["x1", "x2"])
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    assert trees.skip_reason()
    assert trees.fit_tree(spec, X, y) is None


def test_tree_backend_fits_when_sklearn_is_available() -> None:
    if not trees.sklearn_available():
        pytest.skip("scikit-learn is not installed")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = 3.0 * X[:, 0] - X[:, 1]
    fit = trees.fit_tree(ModelSpec(family="tree", features=["x1", "x2"]), X, y)

    assert fit is not None
    predictions = fit.predict(X)
    assert predictions.shape == (80,)
    assert np.all(np.isfinite(predictions))
    assert set(fit.feature_importances()) == {"x1", "x2"}


def test_tree_backend_can_be_forced_off(monkeypatch) -> None:
    monkeypatch.setattr(trees, "SKLEARN_AVAILABLE", False)
    spec = ModelSpec(family="forest", features=["x1"])
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)

    assert trees.fit_tree(spec, X, y) is None
    assert trees.skip_reason()


def test_tree_backend_rejects_non_tree_family() -> None:
    with pytest.raises(ValueError):
        trees.fit_tree(ModelSpec(family="linear"), np.zeros((4, 1)), np.zeros(4))


def test_pysindy_backend_reports_availability() -> None:
    status = sindy.sindy_backend_status()

    assert status["backend"] == "pysindy"
    assert status["available"] is sindy.pysindy_available()
    if not sindy.pysindy_available():
        assert status["reason"]


def test_sindy_backend_skips_when_pysindy_is_missing() -> None:
    if sindy.pysindy_available():
        pytest.skip("pysindy is installed")

    spec = ModelSpec(family="sindy", features=["x"])
    X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, 20)

    assert sindy.skip_reason()
    assert sindy.fit_sindy(spec, X, t) is None


def test_sindy_backend_fits_when_pysindy_is_available() -> None:
    if not sindy.pysindy_available():
        pytest.skip("pysindy is not installed")

    t = np.linspace(0.0, 2.0, 50)
    X = np.exp(-t).reshape(-1, 1)
    spec = ModelSpec(
        family="sindy",
        features=["x"],
        hyperparameters={"degree": 2, "threshold": 0.01},
    )
    fit = sindy.fit_sindy(spec, X, t)

    assert fit is not None
    predictions = fit.predict(initial=X[0], times=t)
    assert predictions.shape[0] == t.size
    assert np.all(np.isfinite(predictions))
    assert fit.coefficient_table()


def test_sindy_backend_can_be_forced_off(monkeypatch) -> None:
    monkeypatch.setattr(sindy, "PYSINDY_AVAILABLE", False)
    spec = ModelSpec(family="sindy", features=["x"])
    X = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, 20)

    assert sindy.fit_sindy(spec, X, t) is None
    assert sindy.skip_reason()


def test_sindy_backend_rejects_non_sindy_family() -> None:
    with pytest.raises(ValueError):
        sindy.fit_sindy(
            ModelSpec(family="linear"), np.zeros((4, 1)), np.zeros(4)
        )
