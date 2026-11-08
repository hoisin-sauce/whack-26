"""Bridge between the project validation matrix and ``stats_engine``.

The general-purpose :func:`stats_engine.validate_model` validator understands
continuous, binary, multiclass, and count targets. This module translates the
project's target-kind names and
:class:`whack26.validation.models.ValidationConfig` thresholds into that
engine's vocabulary, and translates the engine's layer results back into
:class:`~whack26.validation.models.ValidationLayerReport` objects.

The matrix is duck-typed: :func:`run_engine` only reads ``X``, ``y``,
``target_kind``, and ``periods_per_year``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from stats_engine import (
    TARGET_BINARY,
    TARGET_CONTINUOUS,
    TARGET_COUNT,
    TARGET_MULTICLASS,
    build_linear_model,
    build_logistic_model,
    build_poisson_model,
    build_softmax_model,
    log_loss,
    poisson_deviance,
    r2_score,
    validate_model,
)

from .models import LayerStatus, ValidationConfig, ValidationLayerReport

# Project target-kind names mapped to the names accepted by stats_engine.
# ``time_series`` is the legacy alias for a continuous series; the bridge can
# validate it, but the runner only routes non-continuous kinds here.
_TARGET_TYPES: dict[str, str] = {
    "continuous": TARGET_CONTINUOUS,
    "time_series": TARGET_CONTINUOUS,
    "regression": TARGET_CONTINUOUS,
    "regressor": TARGET_CONTINUOUS,
    "binary": TARGET_BINARY,
    "multiclass": TARGET_MULTICLASS,
    "count": TARGET_COUNT,
}

_ENGINE_BUILDERS: dict[str, Callable[..., Callable[[np.ndarray], np.ndarray]]] = {
    TARGET_CONTINUOUS: build_linear_model,
    TARGET_BINARY: build_logistic_model,
    TARGET_MULTICLASS: build_softmax_model,
    TARGET_COUNT: build_poisson_model,
}

_PRIMARY_METRICS: dict[
    str, tuple[str, Callable[[np.ndarray, np.ndarray], float] | None, bool]
] = {
    TARGET_CONTINUOUS: ("r2", r2_score, True),
    TARGET_BINARY: ("log_loss", log_loss, False),
    TARGET_MULTICLASS: ("log_loss", log_loss, False),
    TARGET_COUNT: ("poisson_deviance", poisson_deviance, False),
}

# Engine layer key -> project layer name. The current engine calls Layer 2
# "permutation"; "monte_carlo" is accepted as the public bridge name.
_LAYER_KEYS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("dsr",), "Layer 1 (DSR)"),
    (("monte_carlo", "permutation"), "Layer 2 (Monte Carlo)"),
    (("oos_decay",), "Layer 3 (OOS Decay)"),
    (("parameter_sensitivity",), "Layer 4 (Parameter Sensitivity)"),
)

_GATE_LAYERS: tuple[tuple[str, str], ...] = (
    ("calibration", "Calibration"),
    ("count_dispersion", "Count dispersion"),
)

_SUMMARY_METRICS: dict[str, tuple[str, ...]] = {
    "Layer 1 (DSR)": ("dsr_score",),
    "Layer 2 (Monte Carlo)": ("p_value",),
    "Layer 3 (OOS Decay)": ("oos_score", "decay_rate", "tolerance"),
    "Layer 4 (Parameter Sensitivity)": (),
}

_STATUSES: dict[str, LayerStatus] = {
    "passed": "passed",
    "failed": "failed",
    "error": "error",
    "skipped": "skipped",
    "not_applicable": "skipped",
}


def engine_target_type(kind: str) -> str:
    """Return the ``stats_engine`` target type for a project target kind."""
    normalized = str(kind).strip().lower()
    try:
        return _TARGET_TYPES[normalized]
    except KeyError:
        raise ValueError(f"unsupported target kind: {kind!r}") from None


def engine_supported(kind: str) -> tuple[bool, str]:
    """Return whether the general engine can validate this target kind.

    The reason string is empty when supported and a stable
    ``target_kind:<kind>`` code otherwise.
    """
    normalized = str(kind).strip().lower() if kind is not None else ""
    if normalized not in _TARGET_TYPES:
        return False, f"target_kind:{normalized or 'unknown'}"
    return True, ""


def build_engine_model(
    kind: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit the engine's kind-specific model and return a predict callable."""
    target_type = engine_target_type(kind)
    builder = _ENGINE_BUILDERS[target_type]
    return builder(params, X, y)


def engine_metric(
    kind: str,
) -> tuple[str, Callable[[np.ndarray, np.ndarray], float] | None, bool]:
    """Return the primary metric name, function, and direction for a kind."""
    return _PRIMARY_METRICS[engine_target_type(kind)]


def run_engine(
    *,
    matrix: Any,
    config: ValidationConfig,
    description: str | None = None,
    base_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a duck-typed matrix with :func:`stats_engine.validate_model`.

    ``matrix`` must expose ``X``, ``y``, ``target_kind``, and
    ``periods_per_year``. Splits, permutation count, sensitivity settings,
    calibration/dispersion gates, and the random seed all come from
    ``config``. ``base_params`` carries the generated module's positive
    hyperparameters so Layer 4 can perturb them.
    """
    kind = str(getattr(matrix, "target_kind", "continuous") or "continuous")
    target_type = engine_target_type(kind)
    builder = _ENGINE_BUILDERS[target_type]
    _, metric_func, higher_is_better = engine_metric(kind)

    periods_per_year = getattr(matrix, "periods_per_year", 1.0)
    try:
        periods_per_year = float(periods_per_year)
    except (TypeError, ValueError):
        periods_per_year = 1.0
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        periods_per_year = 1.0

    def model_builder(
        params: dict[str, Any],
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return builder(params, features, targets)

    return validate_model(
        features=matrix.X,
        targets=matrix.y,
        base_model_builder=model_builder,
        base_params=config.engine_params(dict(base_params or {})),
        target_type=target_type,
        metric_func=metric_func,
        higher_is_better=higher_is_better,
        periods_per_year=periods_per_year,
        data_description=description,
        **config.engine_options(),
    )


def _status(value: Any) -> LayerStatus:
    return _STATUSES.get(str(value or "").strip().lower(), "skipped")


def _metrics(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in entry.items() if key not in {"status", "reason"}
    }


def _reason(
    status: LayerStatus, entry: dict[str, Any], fallback: str | None
) -> str | None:
    if status == "passed":
        return None
    value = entry.get("reason") or entry.get("message")
    if value:
        return str(value)
    return fallback if status in {"failed", "error"} else None


def _summary_metrics(result: dict[str, Any], layer_name: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in _SUMMARY_METRICS.get(layer_name, ()):
        if result.get(key) is not None:
            metrics[key] = result[key]
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        if layer_name == "Calibration":
            value = diagnostics.get("expected_calibration_error")
            if value is not None:
                metrics["ece"] = value
        elif layer_name == "Count dispersion":
            value = diagnostics.get("dispersion")
            if value is not None:
                metrics["dispersion"] = value
    return metrics


def engine_result_to_layers(
    result: dict[str, Any],
) -> list[ValidationLayerReport]:
    """Map a ``stats_engine.validate_model`` result to project layer reports.

    The four engine layers always appear, in order. A gate failure
    (``Calibration`` or ``Count dispersion``) is appended as an extra failed
    layer so callers do not lose the reason, and an early configuration or
    contract error is appended under its own engine layer name.
    """
    if not isinstance(result, dict):
        raise TypeError("engine result must be a dict")

    raw_layers = result.get("layers")
    if not isinstance(raw_layers, dict):
        raw_layers = {}
    failed_layer = result.get("failed_layer")
    message = result.get("message")
    canonical_names = [name for _, name in _LAYER_KEYS]
    failure_index = (
        canonical_names.index(failed_layer)
        if failed_layer in canonical_names
        else None
    )
    failed = bool(failed_layer) or result.get("passed") is False

    reports: list[ValidationLayerReport] = []
    for index, (keys, name) in enumerate(_LAYER_KEYS):
        entry: dict[str, Any] | None = None
        for key in keys:
            candidate = raw_layers.get(key)
            if isinstance(candidate, dict):
                entry = candidate
                break

        if entry is not None:
            status = _status(entry.get("status"))
            reason = _reason(status, entry, None)
            metrics = _metrics(entry)
            if failed_layer == name and status != "failed":
                status = "failed"
                reason = str(message or "layer failed")
        elif failed_layer == name:
            status = "failed"
            reason = str(message or "layer failed")
            metrics = _summary_metrics(result, name)
        elif failure_index is not None and index > failure_index:
            status = "skipped"
            reason = "engine stopped at an earlier layer"
            metrics = {}
        elif failed:
            status = "skipped"
            reason = "engine stopped before this layer"
            metrics = {}
        else:
            status = "skipped"
            reason = "layer result was not reported"
            metrics = {}

        reports.append(
            ValidationLayerReport(
                layer=name,
                status=status,
                reason=reason,
                metrics=metrics,
            )
        )

    gate_names = {name for _, name in _GATE_LAYERS}
    for key, name in _GATE_LAYERS:
        entry = raw_layers.get(key)
        if not isinstance(entry, dict):
            if failed_layer != name:
                continue
            entry = {}
        status = _status(entry.get("status"))
        if status != "failed" and failed_layer != name:
            continue
        metrics = _metrics(entry) or _summary_metrics(result, name)
        reports.append(
            ValidationLayerReport(
                layer=name,
                status="failed",
                reason=str(
                    entry.get("message")
                    or entry.get("reason")
                    or message
                    or "layer failed"
                ),
                metrics=metrics,
            )
        )

    if (
        failed_layer
        and failed_layer not in canonical_names
        and failed_layer not in gate_names
    ):
        reports.append(
            ValidationLayerReport(
                layer=str(failed_layer),
                status="failed",
                reason=str(message or "layer failed"),
                metrics=_summary_metrics(result, str(failed_layer)),
            )
        )

    return reports


__all__ = [
    "build_engine_model",
    "engine_metric",
    "engine_result_to_layers",
    "engine_supported",
    "engine_target_type",
    "run_engine",
]
