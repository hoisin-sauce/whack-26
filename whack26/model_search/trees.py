"""Optional scikit-learn tree/forest/boosting backend.

The backend is entirely optional.  When scikit-learn is not installed,
:func:`fit_tree` returns ``None`` and :func:`skip_reason` explains why; nothing
else in the model search imports scikit-learn or fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import ModelSpec

TREE_FAMILIES = frozenset({"tree", "forest", "gradient_boosting"})

try:  # optional dependency
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    SKLEARN_AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised when sklearn is absent
    SKLEARN_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass
class TreeFit:
    """A fitted optional-backend estimator with a small predict interface."""

    family: str
    estimator: Any
    feature_names: list[str]

    def predict(self, X: Any) -> np.ndarray:
        matrix = _matrix(X, self.feature_names)
        return np.asarray(self.estimator.predict(matrix), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = _matrix(X, self.feature_names)
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.family} estimator has no predict_proba")
        return np.asarray(self.estimator.predict_proba(matrix), dtype=float)

    def feature_importances(self) -> dict[str, float]:
        values = getattr(self.estimator, "feature_importances_", None)
        if values is None:
            return {}
        return {
            name: float(value)
            for name, value in zip(self.feature_names, np.asarray(values).reshape(-1))
        }


def sklearn_available() -> bool:
    """Return whether the optional sklearn backend can be used."""

    return SKLEARN_AVAILABLE


def skip_reason() -> str | None:
    """Return the import error when sklearn is unavailable, else ``None``."""

    return None if SKLEARN_AVAILABLE else _IMPORT_ERROR or "scikit-learn is not installed"


def _feature_names(X: Any, spec: ModelSpec) -> list[str]:
    if hasattr(X, "columns"):
        columns = [str(column) for column in X.columns]
        if spec.features:
            missing = [name for name in spec.features if name not in columns]
            if missing:
                raise KeyError(f"missing feature columns: {', '.join(missing)}")
            return list(spec.features)
        return columns
    width = np.asarray(X).shape[1] if np.asarray(X).ndim == 2 else 1
    if spec.features:
        if len(spec.features) != width:
            raise ValueError("spec.features does not match the design matrix width")
        return list(spec.features)
    return [f"x{index}" for index in range(width)]


def _matrix(X: Any, feature_names: Sequence[str]) -> np.ndarray:
    if hasattr(X, "columns"):
        frame = X[list(feature_names)]
        return np.asarray(frame, dtype=float)
    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] != len(feature_names):
        raise ValueError("design matrix width does not match feature names")
    return matrix


def _classification(target_kind: str) -> bool:
    return target_kind in {"binary", "multiclass"}


def _numeric(value: Any, default: float | int) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return int(number) if float(number).is_integer() else number


def _estimator_for(spec: ModelSpec, *, random_state: int) -> Any:
    hyperparameters = spec.hyperparameters
    classification = _classification(spec.target_kind)

    if spec.family == "tree":
        cls = DecisionTreeClassifier if classification else DecisionTreeRegressor
        return cls(
            max_depth=(
                int(hyperparameters["max_depth"])
                if hyperparameters.get("max_depth") is not None
                else 4
            ),
            min_samples_leaf=max(1, int(_numeric(hyperparameters.get("min_samples_leaf"), 1))),
            random_state=random_state,
        )

    if spec.family == "forest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(
            n_estimators=max(1, int(_numeric(hyperparameters.get("n_estimators"), 100))),
            max_depth=(
                int(hyperparameters["max_depth"])
                if hyperparameters.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=max(1, int(_numeric(hyperparameters.get("min_samples_leaf"), 1))),
            random_state=random_state,
            n_jobs=1,
        )

    if spec.family == "gradient_boosting":
        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        return cls(
            n_estimators=max(1, int(_numeric(hyperparameters.get("n_estimators"), 100))),
            learning_rate=float(_numeric(hyperparameters.get("learning_rate"), 0.1)),
            max_depth=int(_numeric(hyperparameters.get("max_depth"), 3)),
            min_samples_leaf=max(1, int(_numeric(hyperparameters.get("min_samples_leaf"), 1))),
            random_state=random_state,
        )

    raise ValueError(f"{spec.family!r} is not a tree backend family")


def fit_tree(
    spec: ModelSpec,
    X: Any,
    y: Any,
    *,
    random_state: int = 0,
) -> TreeFit | None:
    """Fit a tree/forest/boosting spec, or return ``None`` when unavailable."""

    if spec.family not in TREE_FAMILIES:
        raise ValueError(f"{spec.family!r} is not a tree backend family")
    if not SKLEARN_AVAILABLE:
        return None

    feature_names = _feature_names(X, spec)
    matrix = _matrix(X, feature_names)
    target = np.asarray(y, dtype=float).reshape(-1)
    estimator = _estimator_for(spec, random_state=random_state)
    estimator.fit(matrix, target)
    return TreeFit(family=spec.family, estimator=estimator, feature_names=feature_names)


def tree_backend_status() -> dict[str, Any]:
    """Small machine-readable status payload for reports."""

    return {
        "backend": "sklearn",
        "available": SKLEARN_AVAILABLE,
        "families": sorted(TREE_FAMILIES),
        "reason": skip_reason(),
    }


__all__ = [
    "SKLEARN_AVAILABLE",
    "TREE_FAMILIES",
    "TreeFit",
    "fit_tree",
    "sklearn_available",
    "skip_reason",
    "tree_backend_status",
]
