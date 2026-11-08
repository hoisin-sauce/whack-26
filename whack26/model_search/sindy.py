"""Optional pysindy backend for sparse dynamics discovery.

Like :mod:`whack26.model_search.trees`, this module is import-safe: pysindy is
loaded inside a guarded import and :func:`fit_sindy` returns ``None`` when the
package is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .types import ModelSpec

try:  # optional dependency
    import pysindy as ps

    PYSINDY_AVAILABLE = True
    _IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised when pysindy is absent
    ps = None  # type: ignore[assignment]
    PYSINDY_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass
class SindyFit:
    """A fitted SINDy model and the metadata needed to report it."""

    model: Any
    feature_names: list[str]
    sample_times: np.ndarray
    library: str

    def predict(self, *, initial: Any | None = None, times: Any | None = None) -> np.ndarray:
        sample_times = (
            np.asarray(times, dtype=float) if times is not None else self.sample_times
        )
        initial_state = (
            np.asarray(initial, dtype=float).reshape(-1)
            if initial is not None
            else np.asarray(self.model.history_, dtype=float)[0]
            if hasattr(self.model, "history_")
            else np.zeros(len(self.feature_names), dtype=float)
        )
        return np.asarray(self.model.simulate(initial_state, sample_times), dtype=float)

    def coefficients(self) -> np.ndarray:
        return np.asarray(self.model.coefficients(), dtype=float)

    def coefficient_table(self) -> list[dict[str, Any]]:
        """Return one row per identified term, with per-state coefficients."""

        coefficients = self.coefficients()
        names: list[str] = []
        try:
            names = list(self.model.get_feature_names())
        except Exception:
            names = [f"term_{index}" for index in range(coefficients.shape[1])]
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            row: dict[str, Any] = {"term": name}
            for state_index, state_name in enumerate(self.feature_names):
                if index < coefficients.shape[1] and state_index < coefficients.shape[0]:
                    row[state_name] = float(coefficients[state_index, index])
            rows.append(row)
        return rows


def pysindy_available() -> bool:
    """Return whether the optional pysindy backend can be used."""

    return PYSINDY_AVAILABLE


def skip_reason() -> str | None:
    """Return the import error when pysindy is unavailable, else ``None``."""

    return None if PYSINDY_AVAILABLE else _IMPORT_ERROR or "pysindy is not installed"


def _library(spec: ModelSpec):
    name = str(spec.hyperparameters.get("library", "polynomial")).lower()
    degree = int(spec.hyperparameters.get("degree", 2) or 2)
    if name in {"fourier", "trig", "trigonometric"}:
        return ps.FourierLibrary(n_frequencies=max(1, degree)), name
    return ps.PolynomialLibrary(degree=max(1, degree)), name


def fit_sindy(
    spec: ModelSpec,
    X: Any,
    t: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> SindyFit | None:
    """Fit a SINDy model, or return ``None`` when pysindy is unavailable.

    ``X`` is the state matrix ``(n_samples, n_states)`` and ``t`` the matching
    sample times.  The candidate's ``hyperparameters`` may set ``degree``,
    ``threshold``, and ``library`` (``polynomial`` or ``fourier``).
    """

    if spec.family != "sindy":
        raise ValueError(f"{spec.family!r} is not the SINDy family")
    if not PYSINDY_AVAILABLE:
        return None

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    times = np.asarray(t, dtype=float).reshape(-1)
    if times.size != matrix.shape[0]:
        raise ValueError("t must have one value per row of X")

    if feature_names is not None:
        names = [str(name) for name in feature_names]
    elif spec.features:
        names = list(spec.features)
    else:
        names = [f"x{index}" for index in range(matrix.shape[1])]
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names does not match the state width")

    library, library_name = _library(spec)
    threshold_value = spec.hyperparameters.get("threshold", 0.1)
    try:
        threshold = float(threshold_value)
    except (TypeError, ValueError):
        threshold = 0.1
    threshold = threshold if np.isfinite(threshold) else 0.1
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=library,
        differentiation_method=ps.FiniteDifference(),
        feature_names=names,
    )
    model.fit(matrix, t=times)
    return SindyFit(
        model=model,
        feature_names=names,
        sample_times=times,
        library=library_name,
    )


def sindy_backend_status() -> dict[str, Any]:
    """Small machine-readable status payload for reports."""

    return {
        "backend": "pysindy",
        "available": PYSINDY_AVAILABLE,
        "reason": skip_reason(),
    }


__all__ = [
    "PYSINDY_AVAILABLE",
    "SindyFit",
    "fit_sindy",
    "pysindy_available",
    "sindy_backend_status",
    "skip_reason",
]
