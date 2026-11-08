"""Core numpy/scipy fitters for automated model search.

The module implements the core backend: least-squares linear/ridge,
logistic, multinomial, Poisson, negative-binomial, group-comparison, and
baseline models. Tree-style families are optional and delegated through
:func:`register_fitter` so :mod:`whack26.model_search.trees` can plug in a
scikit-learn backend without a hard dependency here.

Design matrices are built inside the fitter: numeric features get median
imputation and the configured ``none``/``log1p``/``sqrt`` transform,
categorical features are one-hot encoded (first level dropped), and declared
interaction pairs are multiplied after transformation. All preprocessing
state is learned on the training frame and reused by ``predict``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special

from . import metrics
from .types import ModelSpec, TargetKind

_TREE_FAMILIES = {"tree", "forest", "gradient_boosting", "sindy"}
_SUPPORTED_FAMILIES = {
    "baseline",
    "linear",
    "logistic",
    "multinomial",
    "poisson",
    "negative_binomial",
    "interaction",
    "group_comparison",
}
_FEATURE_TRANSFORMS = {"none", "log1p", "sqrt"}
_TARGET_TRANSFORMS = {"none", "log1p", "sqrt"}
_ETA_CLIP = 30.0
_MIN_VARIANCE = 1e-12


class FitError(RuntimeError):
    """Raised when a candidate cannot be fitted or produces invalid output."""


class UnsupportedFamilyError(FitError):
    """Raised for a family the core backend does not implement."""


class BackendUnavailableError(FitError):
    """Raised when an optional backend is not installed or registered."""


# ---------------------------------------------------------------------------
# Optional backend registry
# ---------------------------------------------------------------------------

OptionalFitter = Callable[..., "FittedModel"]
_OPTIONAL_FITTERS: dict[str, OptionalFitter] = {}


def register_fitter(family: str, fitter: OptionalFitter) -> None:
    """Register an optional fitter (used by ``trees.py``/``sindy.py``).

    The callable receives the same arguments as :func:`fit_model` minus the
    backend gate and must return a :class:`FittedModel`.
    """

    _OPTIONAL_FITTERS[str(family)] = fitter


def unregister_fitter(family: str) -> None:
    _OPTIONAL_FITTERS.pop(str(family), None)


# ---------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------


@dataclass
class FittedModel:
    """A fitted candidate with prediction closures and fit statistics."""

    spec: ModelSpec
    target: str
    coefficients: dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    converged: bool = True
    log_likelihood: float | None = None
    n_train: int = 0
    classes: list[Any] | None = None
    estimates: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _predict: Callable[[Any], np.ndarray] = field(repr=False, default=None)  # type: ignore[assignment]
    _predict_proba: Callable[[Any], np.ndarray] | None = field(
        repr=False, default=None
    )

    def predict(self, frame: Any) -> np.ndarray:
        """Predict labels (classification) or numeric values (regression)."""

        if self._predict is None:  # pragma: no cover - defensive
            raise FitError("fitted model has no predictor")
        values = np.asarray(self._predict(_as_frame(frame)))
        if self.classes is None:
            values = values.astype(float)
            if not np.all(np.isfinite(values)):
                raise FitError(
                    f"{self.spec.family} produced non-finite predictions"
                )
            return values
        for value in values.tolist():
            if value is None:
                raise FitError("classification produced a missing prediction")
            if isinstance(value, float) and not math.isfinite(value):
                raise FitError("classification produced a non-finite prediction")
        return values

    def predict_proba(self, frame: Any) -> np.ndarray:
        """Predict class probabilities as an ``(n, n_classes)`` matrix."""

        if self._predict_proba is None:
            raise FitError(
                f"{self.spec.family} does not expose class probabilities"
            )
        matrix = np.asarray(self._predict_proba(_as_frame(frame)), dtype=float)
        if matrix.ndim == 1:
            matrix = np.column_stack([1.0 - matrix, matrix])
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise FitError("class probabilities have an invalid shape")
        if not np.all(np.isfinite(matrix)):
            raise FitError("class probabilities contain non-finite values")
        matrix = np.clip(matrix, 0.0, 1.0)
        row_sums = matrix.sum(axis=1)
        if np.any(row_sums <= 0.0):
            raise FitError("class probability row sums to zero")
        return matrix / row_sums[:, None]

    @property
    def aic(self) -> float | None:
        if self.log_likelihood is None:
            return None
        value = metrics.aic(self.log_likelihood, self.n_params)
        return value if math.isfinite(value) else None

    @property
    def bic(self) -> float | None:
        if self.log_likelihood is None:
            return None
        value = metrics.bic(
            self.log_likelihood, self.n_params, self.n_train
        )
        return value if math.isfinite(value) else None


def predict(model: FittedModel, frame: Any) -> np.ndarray:
    return model.predict(frame)


def predict_proba(model: FittedModel, frame: Any) -> np.ndarray:
    return model.predict_proba(frame)


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------


def apply_target_transform(values: Any, transform: str) -> np.ndarray:
    """Transform a target before fitting, validating the domain."""

    array = np.asarray(values, dtype=float).ravel()
    if transform not in _TARGET_TRANSFORMS:
        raise FitError(f"unknown target transform {transform!r}")
    if transform == "none":
        return array
    if not np.all(np.isfinite(array)):
        raise FitError("target contains non-finite values")
    if transform == "log1p":
        if np.any(array <= -1.0):
            raise FitError("log1p target transform requires values > -1")
        return np.log1p(array)
    if np.any(array < 0.0):
        raise FitError("sqrt target transform requires non-negative values")
    return np.sqrt(array)


def inverse_target_transform(values: Any, transform: str) -> np.ndarray:
    """Invert :func:`apply_target_transform` on predictions."""

    array = np.asarray(values, dtype=float).ravel()
    if transform not in _TARGET_TRANSFORMS:
        raise FitError(f"unknown target transform {transform!r}")
    if transform == "none":
        return array
    if transform == "log1p":
        return np.expm1(np.clip(array, -700.0, 700.0))
    return np.square(np.clip(array, 0.0, 1e154))


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


@dataclass
class _DesignState:
    kinds: dict[str, str] = field(default_factory=dict)
    fills: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    transforms: dict[str, str] = field(default_factory=dict)


def _as_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(frame)


def _is_numeric(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_datetime64_any_dtype(series.dtype)
    )


def _numeric_series(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=float, na_value=np.nan)


def _feature_transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform not in _FEATURE_TRANSFORMS:
        raise FitError(f"unknown feature transform {transform!r}")
    if transform == "none":
        return values
    clipped = np.maximum(values, 0.0)
    if transform == "log1p":
        return np.log1p(clipped)
    return np.sqrt(clipped)


def _build_design(
    frame: pd.DataFrame,
    spec: ModelSpec,
    state: _DesignState | None = None,
) -> tuple[np.ndarray, list[str], _DesignState]:
    """Build ``[intercept, features..., interactions...]`` in spec order."""

    state = state or _DesignState()
    n_rows = len(frame)
    columns: list[np.ndarray] = [np.ones(n_rows)]
    names: list[str] = ["intercept"]
    numeric_values: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    for feature in spec.features:
        if feature in seen:
            continue
        seen.add(feature)
        if feature not in frame.columns:
            raise FitError(f"feature {feature!r} is not in the frame")
        series = frame[feature]
        if feature in state.kinds:
            kind = state.kinds[feature]
        else:
            kind = "numeric" if _is_numeric(series) else "categorical"
            state.kinds[feature] = kind

        if kind == "numeric":
            values = _numeric_series(series)
            if feature in state.fills:
                fill = state.fills[feature]
            else:
                finite = values[np.isfinite(values)]
                fill = float(np.median(finite)) if finite.size else 0.0
                state.fills[feature] = fill
            values = np.where(np.isfinite(values), values, fill)
            transform = str(spec.feature_transforms.get(feature, "none"))
            state.transforms[feature] = transform
            values = _feature_transform(values, transform)
            numeric_values[feature] = values
            columns.append(values)
            names.append(feature)
            continue

        text = series.astype("object")
        text = text.where(pd.notna(text), "__missing__").astype(str)
        if feature in state.categories:
            categories = state.categories[feature]
        else:
            categories = sorted(text.unique().tolist())
            state.categories[feature] = categories
        for category in categories[1:]:
            columns.append(
                (text.to_numpy() == category).astype(float)
            )
            names.append(f"{feature}={category}")

    for left, right in spec.interactions:
        if left not in numeric_values or right not in numeric_values:
            raise FitError(
                "interactions require two numeric features; "
                f"got {left!r}:{right!r}"
            )
        columns.append(numeric_values[left] * numeric_values[right])
        names.append(f"{left}:{right}")

    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise FitError("design matrix contains non-finite values")
    return matrix, names, state


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _effective_target_kind(spec: ModelSpec) -> TargetKind:
    if spec.family == "logistic":
        return "binary"
    if spec.family == "multinomial":
        return "multiclass"
    if spec.family in ("poisson", "negative_binomial"):
        return "count"
    return spec.target_kind


def _train_mask(
    frame: pd.DataFrame, target: str, kind: TargetKind
) -> np.ndarray:
    if target not in frame.columns:
        raise FitError(f"target column {target!r} is not in the frame")
    if kind in ("continuous", "count", "time_series"):
        values = pd.to_numeric(frame[target], errors="coerce")
        return (
            values.notna() & np.isfinite(values.to_numpy(dtype=float))
        ).to_numpy()
    return frame[target].notna().to_numpy()


def _training_frame(
    frame: pd.DataFrame, target: str, kind: TargetKind
) -> pd.DataFrame:
    mask = _train_mask(frame, target, kind)
    train = frame.loc[mask]
    if len(train) < 2:
        raise FitError("at least two complete target rows are required")
    return train


def _weights(
    sample_weight: Sequence[float] | np.ndarray | None, n_rows: int
) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float).ravel()
    if weights.size != n_rows:
        raise FitError("sample_weight length does not match the training rows")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise FitError("sample_weight must be finite and non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise FitError("sample_weight must have a positive sum")
    return weights


def _class_target(series: pd.Series) -> tuple[np.ndarray, list[Any]]:
    values = series.astype("object").to_numpy()
    unique = list(pd.unique(values))
    try:
        classes = sorted(unique)
    except TypeError:  # mixed types
        classes = sorted(unique, key=str)
    if len(classes) < 2:
        raise FitError("classification requires at least two classes")
    lookup = {label: index for index, label in enumerate(classes)}
    indices = np.array([lookup[value] for value in values], dtype=int)
    return indices, classes


def _class_weights(
    y_index: np.ndarray, n_classes: int, mode: str | None
) -> np.ndarray | None:
    if mode is None:
        return None
    if str(mode).lower() != "balanced":
        raise FitError(f"unsupported class_weight {mode!r}")
    counts = np.bincount(y_index, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        raise FitError("class_weight requires every class in the training fold")
    weights = np.empty_like(counts, dtype=float)
    total = float(np.sum(counts))
    for index, count in enumerate(counts):
        weights[index] = total / (n_classes * count)
    return weights[y_index]


def _ensure_finite(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(values)):
        raise FitError(f"{label} contains non-finite values")
    return values


def _solve_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
    ridge: float | None,
) -> np.ndarray:
    if weights is not None:
        root = np.sqrt(weights)
        design = matrix * root[:, None]
        response = target * root
    else:
        design = matrix
        response = target
    if ridge is not None and float(ridge) > 0.0:
        penalty = np.eye(design.shape[1]) * math.sqrt(float(ridge))
        penalty[0, 0] = 0.0
        design = np.vstack([design, penalty])
        response = np.concatenate([response, np.zeros(design.shape[1])])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return _ensure_finite(coefficients, "linear coefficients")


def _gaussian_log_likelihood(
    residuals: np.ndarray,
    weights: np.ndarray | None,
    n_params: int,
) -> tuple[float, float]:
    if weights is not None:
        squared = float(np.sum(weights * residuals**2))
        effective_n = float(np.sum(weights))
    else:
        squared = float(np.sum(residuals**2))
        effective_n = float(residuals.size)
    variance = max(squared / max(effective_n, 1.0), _MIN_VARIANCE)
    log_likelihood = -0.5 * effective_n * (
        math.log(2.0 * math.pi * variance) + 1.0
    )
    return log_likelihood, variance


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def _fit_baseline(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    kind: TargetKind,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    train = _training_frame(frame, target, kind)
    statistic = str(spec.hyperparameters.get("statistic", "mean"))

    if kind in ("continuous", "count", "time_series"):
        values = _ensure_finite(
            pd.to_numeric(train[target], errors="coerce").to_numpy(
                dtype=float
            ),
            "target",
        )
        if statistic == "median":
            constant = float(np.median(values))
        else:
            constant = float(np.mean(values))
        weights = (
            sample_weight[_train_mask(frame, target, kind)]
            if sample_weight is not None
            else None
        )
        residuals = values - constant
        if kind == "count":
            log_likelihood = _poisson_log_likelihood(
                values, np.full(values.size, max(constant, 1e-15)), weights
            )
        else:
            log_likelihood, _ = _gaussian_log_likelihood(
                residuals, weights, 1
            )
        return FittedModel(
            spec=spec,
            target=target,
            coefficients={"intercept": constant},
            n_params=1,
            converged=True,
            log_likelihood=log_likelihood,
            n_train=len(train),
            _predict=lambda new_frame: np.full(len(_as_frame(new_frame)), constant),
        )

    y_index, classes = _class_target(train[target])
    counts = np.bincount(y_index, minlength=len(classes)).astype(float)
    probabilities = counts / float(np.sum(counts))
    majority = int(np.argmax(probabilities))
    majority_label = classes[majority]

    if kind == "binary":
        positive_rate = float(probabilities[1])
        log_likelihood = float(
            np.sum(
                y_index * math.log(max(positive_rate, 1e-15))
                + (1 - y_index) * math.log(max(1.0 - positive_rate, 1e-15))
            )
        )
    else:
        log_likelihood = float(
            np.sum(np.log(np.maximum(probabilities[y_index], 1e-15)))
        )

    def predict(new_frame: Any) -> np.ndarray:
        return np.full(
            len(_as_frame(new_frame)), majority_label, dtype=object
        )

    def predict_proba(new_frame: Any) -> np.ndarray:
        return np.tile(
            probabilities, (len(_as_frame(new_frame)), 1)
        )

    return FittedModel(
        spec=spec,
        target=target,
        n_params=1,
        converged=True,
        log_likelihood=log_likelihood,
        n_train=len(train),
        classes=list(classes),
        _predict=predict,
        _predict_proba=predict_proba,
    )


# ---------------------------------------------------------------------------
# Linear / ridge
# ---------------------------------------------------------------------------


def _fit_linear(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    kind: TargetKind,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    train = _training_frame(frame, target, kind)
    design, names, state = _build_design(train, spec)
    raw_target = _ensure_finite(
        pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float),
        "target",
    )
    transformed = apply_target_transform(raw_target, spec.target_transform)

    n_train = len(train)
    weights = None
    if sample_weight is not None:
        # sample_weight refers to the original frame; restrict to train rows.
        weights = sample_weight[_train_mask(frame, target, kind)]

    coefficients = _solve_least_squares(
        design, transformed, weights, spec.regularization
    )
    fitted = design @ coefficients
    residuals = transformed - fitted
    log_likelihood, _ = _gaussian_log_likelihood(
        residuals, weights, design.shape[1]
    )

    def predict(new_frame: Any) -> np.ndarray:
        matrix, _, _ = _build_design(_as_frame(new_frame), spec, state)
        return inverse_target_transform(
            matrix @ coefficients, spec.target_transform
        )

    return FittedModel(
        spec=spec,
        target=target,
        coefficients=dict(zip(names, coefficients.tolist())),
        n_params=design.shape[1],
        converged=True,
        log_likelihood=log_likelihood,
        n_train=n_train,
        _predict=predict,
    )


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------


def _fit_logistic(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    if spec.target_transform != "none":
        raise FitError("logistic candidates do not use a target transform")
    train = _training_frame(frame, target, "binary")
    design, names, state = _build_design(train, spec)
    y_index, classes = _class_target(train[target])
    if len(classes) != 2:
        raise FitError("logistic regression requires exactly two classes")
    response = y_index.astype(float)

    base_weights = np.ones(len(train), dtype=float)
    if sample_weight is not None:
        base_weights = sample_weight[_train_mask(frame, target, "binary")]
    class_weight = _class_weights(
        y_index,
        len(classes),
        spec.hyperparameters.get("class_weight"),
    )
    if class_weight is not None:
        base_weights = base_weights * class_weight
    scale = float(np.sum(base_weights))
    ridge = float(spec.regularization or 0.0)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        probabilities = special.expit(eta)
        nll = float(
            np.sum(
                base_weights
                * (np.logaddexp(0.0, eta) - response * eta)
            )
            / scale
        )
        nll += 0.5 * ridge * float(np.sum(beta[1:] ** 2))
        gradient = design.T @ (base_weights * (probabilities - response)) / scale
        gradient[1:] += ridge * beta[1:]
        return nll, gradient

    result = optimize.minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    coefficients = _ensure_finite(result.x, "logistic coefficients")
    fitted_probability = np.clip(special.expit(design @ coefficients), 1e-15, 1 - 1e-15)
    log_likelihood = float(
        np.sum(
            base_weights
            * (
                response * np.log(fitted_probability)
                + (1.0 - response) * np.log(1.0 - fitted_probability)
            )
        )
    )

    def predict(new_frame: Any) -> np.ndarray:
        matrix, _, _ = _build_design(_as_frame(new_frame), spec, state)
        probability = special.expit(matrix @ coefficients)
        return np.where(probability >= 0.5, classes[1], classes[0])

    def predict_proba(new_frame: Any) -> np.ndarray:
        matrix, _, _ = _build_design(_as_frame(new_frame), spec, state)
        probability = special.expit(matrix @ coefficients)
        return np.column_stack([1.0 - probability, probability])

    return FittedModel(
        spec=spec,
        target=target,
        coefficients=dict(zip(names, coefficients.tolist())),
        n_params=design.shape[1],
        converged=bool(result.success),
        log_likelihood=log_likelihood,
        n_train=len(train),
        classes=list(classes),
        _predict=predict,
        _predict_proba=predict_proba,
    )


# ---------------------------------------------------------------------------
# Multinomial
# ---------------------------------------------------------------------------


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -745.0, 0.0))
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _fit_multinomial(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    if spec.target_transform != "none":
        raise FitError("multinomial candidates do not use a target transform")
    train = _training_frame(frame, target, "multiclass")
    design, names, state = _build_design(train, spec)
    y_index, classes = _class_target(train[target])
    n_classes = len(classes)
    if n_classes < 2:
        raise FitError("multinomial regression requires at least two classes")
    one_hot = np.zeros((len(train), n_classes))
    one_hot[np.arange(len(train)), y_index] = 1.0

    base_weights = np.ones(len(train), dtype=float)
    if sample_weight is not None:
        base_weights = sample_weight[_train_mask(frame, target, "multiclass")]
    scale = float(np.sum(base_weights))
    ridge = (
        float(spec.regularization)
        if spec.regularization is not None
        else 1e-6
    )
    ridge = max(ridge, 1e-8)
    shape = (design.shape[1], n_classes)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        matrix = flat.reshape(shape)
        probabilities = _softmax(design @ matrix)
        chosen = probabilities[np.arange(len(train)), y_index]
        nll = float(
            -np.sum(base_weights * np.log(np.maximum(chosen, 1e-15)))
            / scale
        )
        nll += 0.5 * ridge * float(np.sum(matrix[1:] ** 2))
        gradient = design.T @ (
            (probabilities - one_hot) * base_weights[:, None]
        ) / scale
        gradient[1:] += ridge * matrix[1:]
        return nll, gradient.ravel()

    result = optimize.minimize(
        objective,
        np.zeros(shape[0] * shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    coefficient_matrix = _ensure_finite(
        result.x.reshape(shape), "multinomial coefficients"
    )
    probabilities = _softmax(design @ coefficient_matrix)
    log_likelihood = float(
        np.sum(
            base_weights
            * np.log(
                np.maximum(
                    probabilities[np.arange(len(train)), y_index], 1e-15
                )
            )
        )
    )

    coefficients: dict[str, float] = {}
    for class_index, label in enumerate(classes):
        for name, value in zip(names, coefficient_matrix[:, class_index]):
            coefficients[f"{label}:{name}"] = float(value)

    def predict(new_frame: Any) -> np.ndarray:
        matrix, _, _ = _build_design(_as_frame(new_frame), spec, state)
        return np.asarray(classes, dtype=object)[np.argmax(matrix @ coefficient_matrix, axis=1)]

    def predict_proba(new_frame: Any) -> np.ndarray:
        matrix, _, _ = _build_design(_as_frame(new_frame), spec, state)
        return _softmax(matrix @ coefficient_matrix)

    return FittedModel(
        spec=spec,
        target=target,
        coefficients=coefficients,
        n_params=shape[0] * shape[1],
        converged=bool(result.success),
        log_likelihood=log_likelihood,
        n_train=len(train),
        classes=list(classes),
        _predict=predict,
        _predict_proba=predict_proba,
    )


# ---------------------------------------------------------------------------
# Poisson / negative binomial
# ---------------------------------------------------------------------------


def _offset_values(
    frame: pd.DataFrame, spec: ModelSpec
) -> np.ndarray | None:
    name = spec.hyperparameters.get("offset")
    if not name:
        return None
    if name not in frame.columns:
        raise FitError(f"offset column {name!r} is not in the frame")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise FitError("offset values must be finite and positive")
    return np.log(values)


def _poisson_log_likelihood(
    response: np.ndarray, mean: np.ndarray, weights: np.ndarray | None
) -> float:
    mean = np.maximum(mean, 1e-15)
    terms = response * np.log(mean) - mean - special.gammaln(response + 1.0)
    if weights is not None:
        terms = terms * weights
    return float(np.sum(terms))


def _fit_poisson(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    if spec.target_transform != "none":
        raise FitError("Poisson candidates do not use a target transform")
    train = _training_frame(frame, target, "count")
    design, names, state = _build_design(train, spec)
    response = _ensure_finite(
        pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float),
        "target",
    )
    if np.any(response < 0.0):
        raise FitError("Poisson regression requires a non-negative target")
    offset = _offset_values(train, spec)

    base_weights = np.ones(len(train), dtype=float)
    if sample_weight is not None:
        base_weights = sample_weight[_train_mask(frame, target, "count")]
    scale = float(np.sum(base_weights))
    ridge = float(spec.regularization or 0.0)

    initial = np.zeros(design.shape[1], dtype=float)
    average = float(np.mean(response))
    if offset is not None and len(offset):
        average = float(np.mean(response / np.exp(offset)))
    initial[0] = math.log(max(average, 1e-6))

    def linear_predictor(beta: np.ndarray, new_design: np.ndarray, new_offset: np.ndarray | None) -> np.ndarray:
        eta = new_design @ beta
        if new_offset is not None:
            eta = eta + new_offset
        return np.clip(eta, -_ETA_CLIP, _ETA_CLIP)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = linear_predictor(beta, design, offset)
        mean = np.exp(eta)
        nll = float(np.sum(base_weights * (mean - response * eta)) / scale)
        nll += 0.5 * ridge * float(np.sum(beta[1:] ** 2))
        gradient = design.T @ (base_weights * (mean - response)) / scale
        gradient[1:] += ridge * beta[1:]
        return nll, gradient

    result = optimize.minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
    )
    coefficients = _ensure_finite(result.x, "Poisson coefficients")
    fitted_mean = np.exp(linear_predictor(coefficients, design, offset))
    log_likelihood = _poisson_log_likelihood(
        response, fitted_mean, base_weights
    )

    def predict(new_frame: Any) -> np.ndarray:
        new = _as_frame(new_frame)
        matrix, _, _ = _build_design(new, spec, state)
        new_offset = _offset_values(new, spec)
        return np.exp(linear_predictor(coefficients, matrix, new_offset))

    return FittedModel(
        spec=spec,
        target=target,
        coefficients=dict(zip(names, coefficients.tolist())),
        n_params=design.shape[1],
        converged=bool(result.success),
        log_likelihood=log_likelihood,
        n_train=len(train),
        _predict=predict,
    )


def _fit_negative_binomial(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    if spec.target_transform != "none":
        raise FitError(
            "negative-binomial candidates do not use a target transform"
        )
    train = _training_frame(frame, target, "count")
    design, names, state = _build_design(train, spec)
    response = _ensure_finite(
        pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float),
        "target",
    )
    if np.any(response < 0.0):
        raise FitError(
            "negative-binomial regression requires a non-negative target"
        )
    offset = _offset_values(train, spec)

    base_weights = np.ones(len(train), dtype=float)
    if sample_weight is not None:
        base_weights = sample_weight[_train_mask(frame, target, "count")]
    scale = float(np.sum(base_weights))
    ridge = float(spec.regularization or 0.0)

    mean_response = float(np.mean(response))
    variance = float(np.var(response))
    initial_dispersion = max(
        (variance - mean_response) / max(mean_response**2, 1e-8), 0.01
    )
    initial = np.zeros(design.shape[1] + 1, dtype=float)
    initial[0] = math.log(max(mean_response, 1e-6))
    initial[-1] = math.log(initial_dispersion)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, float]:
        beta = params[:-1]
        dispersion = float(np.exp(np.clip(params[-1], -20.0, 20.0)))
        return beta, max(dispersion, 1e-8)

    def eta_for(beta: np.ndarray, new_design: np.ndarray,
                new_offset: np.ndarray | None) -> np.ndarray:
        eta = new_design @ beta
        if new_offset is not None:
            eta = eta + new_offset
        return np.clip(eta, -_ETA_CLIP, _ETA_CLIP)

    def negative_log_likelihood(params: np.ndarray) -> float:
        beta, dispersion = unpack(params)
        eta = eta_for(beta, design, offset)
        mean = np.exp(eta)
        r = 1.0 / dispersion
        log_probability = (
            special.gammaln(response + r)
            - special.gammaln(r)
            - special.gammaln(response + 1.0)
            + r * np.log(1.0 / (1.0 + dispersion * mean))
            + response
            * np.log(
                dispersion * mean / (1.0 + dispersion * mean)
            )
        )
        nll = float(-np.sum(base_weights * log_probability) / scale)
        nll += 0.5 * ridge * float(np.sum(beta[1:] ** 2))
        return nll

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        beta, dispersion = unpack(params)
        eta = eta_for(beta, design, offset)
        mean = np.exp(eta)
        r = 1.0 / dispersion
        log_probability = (
            special.gammaln(response + r)
            - special.gammaln(r)
            - special.gammaln(response + 1.0)
            + r * np.log(1.0 / (1.0 + dispersion * mean))
            + response
            * np.log(
                dispersion * mean / (1.0 + dispersion * mean)
            )
        )
        nll = float(-np.sum(base_weights * log_probability) / scale)
        nll += 0.5 * ridge * float(np.sum(beta[1:] ** 2))

        derivative = (
            -(r + response) * dispersion * mean / (1.0 + dispersion * mean)
            + response
        )
        gradient = np.empty_like(params)
        gradient[:-1] = -design.T @ (base_weights * derivative) / scale
        gradient[1:-1] += ridge * beta[1:]
        epsilon = 1e-5
        plus = params.copy()
        plus[-1] += epsilon
        minus = params.copy()
        minus[-1] -= epsilon
        gradient[-1] = (
            negative_log_likelihood(plus) - negative_log_likelihood(minus)
        ) / (2.0 * epsilon)
        return nll, gradient

    result = optimize.minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-10, "gtol": 1e-7},
    )
    params = _ensure_finite(result.x, "negative-binomial parameters")
    coefficients_beta, dispersion = unpack(params)
    fitted_mean = np.exp(eta_for(coefficients_beta, design, offset))
    log_likelihood = float(
        np.sum(
            base_weights
            * (
                special.gammaln(response + 1.0 / dispersion)
                - special.gammaln(1.0 / dispersion)
                - special.gammaln(response + 1.0)
                + (1.0 / dispersion)
                * np.log(1.0 / (1.0 + dispersion * fitted_mean))
                + response
                * np.log(
                    dispersion
                    * fitted_mean
                    / (1.0 + dispersion * fitted_mean)
                )
            )
        )
    )

    def predict(new_frame: Any) -> np.ndarray:
        new = _as_frame(new_frame)
        matrix, _, _ = _build_design(new, spec, state)
        new_offset = _offset_values(new, spec)
        return np.exp(eta_for(coefficients_beta, matrix, new_offset))

    return FittedModel(
        spec=spec,
        target=target,
        coefficients=dict(zip(names, coefficients_beta.tolist())),
        n_params=design.shape[1] + 1,
        converged=bool(result.success),
        log_likelihood=log_likelihood,
        n_train=len(train),
        estimates={"dispersion": dispersion},
        _predict=predict,
    )


# ---------------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------------


def _group_keys(
    series: pd.Series,
    *,
    bins: int,
    edges: np.ndarray | None,
    fill: float | None,
    training: bool,
) -> tuple[list[str], np.ndarray | None, float | None]:
    """Return string keys plus the learned bin edges and median fill."""

    if bins > 0 and _is_numeric(series):
        values = _numeric_series(series)
        finite = values[np.isfinite(values)]
        if training:
            fill = float(np.median(finite)) if finite.size else 0.0
            if finite.size:
                quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
                edges = np.unique(np.quantile(finite, quantiles))
            else:
                edges = np.array([], dtype=float)
        values = np.where(np.isfinite(values), values, fill)
        if edges is None:
            edges = np.array([], dtype=float)
        indices = np.digitize(values, edges)
        return [f"bin:{int(index)}" for index in indices], edges, fill

    text = series.astype("object")
    text = text.where(pd.notna(text), "__missing__").astype(str)
    return text.tolist(), edges, fill


def _fit_group_comparison(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    kind: TargetKind,
    sample_weight: np.ndarray | None,
) -> FittedModel:
    if not spec.features:
        raise FitError("group_comparison requires a factor feature")
    factor = spec.features[0]
    if factor not in frame.columns:
        raise FitError(f"feature {factor!r} is not in the frame")
    train = _training_frame(frame, target, kind)
    bins = int(spec.hyperparameters.get("bins", 0) or 0)
    train_keys, edges, fill = _group_keys(
        train[factor], bins=bins, edges=None, fill=None, training=True
    )

    if kind in ("continuous", "count", "time_series"):
        response = _ensure_finite(
            pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float),
            "target",
        )
        global_value = float(np.mean(response))
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, value in zip(train_keys, response):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
        values = {
            key: sums[key] / counts[key] for key in sums
        }
        fitted = np.array([values[key] for key in train_keys])
        residuals = response - fitted
        weights = None
        if sample_weight is not None:
            weights = sample_weight[_train_mask(frame, target, kind)]
        if kind == "count":
            log_likelihood = _poisson_log_likelihood(
                response, np.maximum(fitted, 1e-15), weights
            )
        else:
            log_likelihood, _ = _gaussian_log_likelihood(
                residuals, weights, max(len(values), 1)
            )

        def predict(new_frame: Any) -> np.ndarray:
            new = _as_frame(new_frame)
            keys, _, _ = _group_keys(
                new[factor],
                bins=bins,
                edges=edges,
                fill=fill,
                training=False,
            )
            return np.array(
                [values.get(key, global_value) for key in keys], dtype=float
            )

        return FittedModel(
            spec=spec,
            target=target,
            coefficients={
                f"{factor}={key}": value for key, value in values.items()
            },
            n_params=max(len(values), 1),
            converged=True,
            log_likelihood=log_likelihood,
            n_train=len(train),
            _predict=predict,
        )

    y_index, classes = _class_target(train[target])
    n_classes = len(classes)
    if kind == "binary" and n_classes != 2:
        raise FitError("binary group comparison requires two classes")
    per_key: dict[str, np.ndarray] = {}
    for key, index in zip(train_keys, y_index):
        if key not in per_key:
            per_key[key] = np.zeros(n_classes, dtype=float)
        per_key[key][index] += 1.0
    global_counts = np.bincount(y_index, minlength=n_classes).astype(float)
    global_probabilities = global_counts / float(np.sum(global_counts))
    probabilities_by_key = {
        key: (class_counts / float(np.sum(class_counts)))
        for key, class_counts in per_key.items()
    }

    if kind == "binary":
        log_likelihood = 0.0
        for key, index in zip(train_keys, y_index):
            probability = probabilities_by_key[key][1]
            log_likelihood += math.log(
                max(
                    probability if index == 1 else 1.0 - probability,
                    1e-15,
                )
            )
    else:
        log_likelihood = 0.0
        for key, index in zip(train_keys, y_index):
            log_likelihood += math.log(
                max(probabilities_by_key[key][index], 1e-15)
            )

    def predict(new_frame: Any) -> np.ndarray:
        new = _as_frame(new_frame)
        keys, _, _ = _group_keys(
            new[factor],
            bins=bins,
            edges=edges,
            fill=fill,
            training=False,
        )
        majority = [
            classes[
                int(
                    np.argmax(
                        probabilities_by_key.get(key, global_probabilities)
                    )
                )
            ]
            for key in keys
        ]
        return np.asarray(majority, dtype=object)

    def predict_proba(new_frame: Any) -> np.ndarray:
        new = _as_frame(new_frame)
        keys, _, _ = _group_keys(
            new[factor],
            bins=bins,
            edges=edges,
            fill=fill,
            training=False,
        )
        return np.vstack(
            [
                probabilities_by_key.get(key, global_probabilities)
                for key in keys
            ]
        )

    return FittedModel(
        spec=spec,
        target=target,
        coefficients={
            f"{factor}={key}": float(probability[1]) if n_classes == 2 else float(np.max(probability))
            for key, probability in probabilities_by_key.items()
        },
        n_params=max(len(per_key), 1),
        converged=True,
        log_likelihood=log_likelihood,
        n_train=len(train),
        classes=list(classes),
        _predict=predict,
        _predict_proba=predict_proba,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def fit_model(
    spec: ModelSpec,
    frame: Any,
    target: str,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> FittedModel:
    """Fit ``spec`` on ``frame`` and return a reusable :class:`FittedModel`.

    Parameters
    ----------
    spec:
        Candidate specification.
    frame:
        Training frame (DataFrame or mapping of columns).
    target:
        Target column name. Rows with a missing target are dropped.
    sample_weight:
        Optional row weights aligned with ``frame``.
    """

    if spec.family in _TREE_FAMILIES or spec.family not in _SUPPORTED_FAMILIES:
        fitter = _OPTIONAL_FITTERS.get(spec.family)
        if fitter is None:
            raise BackendUnavailableError(
                f"{spec.family} candidates require an optional backend; "
                "register one with model_search.fit.register_fitter"
            )
        return fitter(spec, frame, target, sample_weight=sample_weight)

    data = _as_frame(frame)
    kind = _effective_target_kind(spec)

    prechecked_weights = (
        _weights(sample_weight, len(data)) if sample_weight is not None else None
    )

    if spec.family == "baseline":
        return _fit_baseline(spec, data, target, spec.target_kind, prechecked_weights)
    if spec.family == "group_comparison":
        return _fit_group_comparison(
            spec, data, target, spec.target_kind, prechecked_weights
        )
    if spec.family == "linear":
        if kind not in ("continuous", "count", "time_series"):
            raise UnsupportedFamilyError(
                f"linear candidates cannot model a {kind} target"
            )
        return _fit_linear(spec, data, target, kind, prechecked_weights)
    if spec.family == "logistic":
        return _fit_logistic(spec, data, target, prechecked_weights)
    if spec.family == "multinomial":
        return _fit_multinomial(spec, data, target, prechecked_weights)
    if spec.family == "poisson":
        return _fit_poisson(spec, data, target, prechecked_weights)
    if spec.family == "negative_binomial":
        return _fit_negative_binomial(
            spec, data, target, prechecked_weights
        )
    if spec.family == "interaction":
        if kind in ("continuous", "time_series", "count"):
            if kind == "count":
                return _fit_poisson(
                    spec, data, target, prechecked_weights
                )
            return _fit_linear(spec, data, target, kind, prechecked_weights)
        if kind == "binary":
            return _fit_logistic(spec, data, target, prechecked_weights)
        if kind == "multiclass":
            return _fit_multinomial(
                spec, data, target, prechecked_weights
            )
        raise UnsupportedFamilyError(
            f"interaction candidates cannot model a {kind} target"
        )
    raise UnsupportedFamilyError(f"unknown family {spec.family!r}")


fit = fit_model


__all__ = [
    "BackendUnavailableError",
    "FitError",
    "FittedModel",
    "UnsupportedFamilyError",
    "apply_target_transform",
    "fit",
    "fit_model",
    "inverse_target_transform",
    "predict",
    "predict_proba",
    "register_fitter",
    "unregister_fitter",
]
