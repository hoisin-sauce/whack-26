"""Model diagnostics for the automated model search.

Every routine accepts plain numpy arrays (or pandas objects that numpy can
coerce) and returns :class:`~whack26.model_search.types.Diagnostic` values.
Diagnostics describe a model; they never reject it on their own.

The core backend is numpy/scipy only.  ``statsmodels`` is used for ADF/KPSS
when it is installed, with a deterministic core fallback for ADF.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from .types import Diagnostic, TargetKind

try:  # optional backend
    from statsmodels.tsa.stattools import adfuller as _sm_adfuller
    from statsmodels.tsa.stattools import kpss as _sm_kpss

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when statsmodels is absent
    _sm_adfuller = None
    _sm_kpss = None
    STATSMODELS_AVAILABLE = False


class Diagnostics(BaseModel):
    """Typed collection of diagnostics with small query helpers."""

    items: list[Diagnostic] = Field(default_factory=list)

    def add(
        self,
        name: str,
        status: str = "ok",
        value: float | None = None,
        message: str = "",
    ) -> Diagnostic:
        diagnostic = Diagnostic(name=name, status=status, value=value, message=message)  # type: ignore[arg-type]
        self.items.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.items.extend(diagnostics)

    def append(self, diagnostic: Diagnostic) -> None:
        self.items.append(diagnostic)

    def get(self, name: str) -> Diagnostic | None:
        for item in self.items:
            if item.name == name:
                return item
        return None

    def status(self, name: str) -> str:
        item = self.get(name)
        return item.status if item is not None else "not_applicable"

    def flagged(self) -> list[Diagnostic]:
        return [item for item in self.items if item.status in {"warn", "fail"}]

    def has_flags(self) -> bool:
        return bool(self.flagged())

    def worst_status(self) -> str:
        statuses = {item.status for item in self.items}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        if statuses and statuses <= {"not_applicable"}:
            return "not_applicable"
        return "ok"

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _aligned(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    size = min(left.size, right.size)
    left, right = left[:size], right[:size]
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _design_matrix(design: Any, n_rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if n_rows is not None and matrix.shape[0] != n_rows:
        matrix = matrix[:n_rows]
    return matrix


def _ols_r2(target: np.ndarray, design: np.ndarray) -> float:
    if design.size == 0 or target.size == 0:
        return 0.0
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    total = float(np.sum((target - target.mean()) ** 2))
    if total <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(np.sum(residual**2)) / total)


def _with_intercept(*columns: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(columns[0]), *columns])


def residual_summary(
    residuals: Any,
    fitted: Any | None = None,
    *,
    prefix: str = "residual",
) -> list[Diagnostic]:
    """Summarize the residual distribution and residual-vs-fitted shape."""

    values = _array(residuals)
    if values.size < 2:
        return [
            Diagnostic(
                name=f"{prefix}_summary",
                status="not_applicable",
                message="fewer than two finite residuals",
            )
        ]

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    skew = float(stats.skew(values, bias=False)) if values.size > 2 else 0.0
    kurtosis = float(stats.kurtosis(values, fisher=True, bias=False)) if values.size > 3 else 0.0
    maximum = float(np.max(np.abs(values)))
    tolerance = max(1e-8, 0.25 * std)

    items = [
        Diagnostic(
            name=f"{prefix}_mean",
            status="warn" if abs(mean) > tolerance else "ok",
            value=mean,
            message="" if abs(mean) <= tolerance else "residual mean is not near zero",
        ),
        Diagnostic(
            name=f"{prefix}_std",
            status="ok",
            value=std,
            message="residual standard deviation",
        ),
        Diagnostic(
            name=f"{prefix}_skew",
            status="warn" if abs(skew) > 2.0 else "ok",
            value=skew,
            message="" if abs(skew) <= 2.0 else "strong residual skew",
        ),
        Diagnostic(
            name=f"{prefix}_kurtosis",
            status="warn" if abs(kurtosis) > 7.0 else "ok",
            value=kurtosis,
            message="" if abs(kurtosis) <= 7.0 else "heavy residual tails",
        ),
        Diagnostic(
            name=f"{prefix}_max_abs",
            status="warn" if maximum > 6.0 * std and std > 0 else "ok",
            value=maximum,
            message="" if maximum <= 6.0 * std or std == 0 else "large residual outlier",
        ),
    ]

    if fitted is not None:
        paired_residual, paired_fitted = _aligned(residuals, fitted)
        if paired_residual.size >= 3 and np.std(paired_fitted) > 0:
            correlation = float(np.corrcoef(paired_residual, paired_fitted)[0, 1])
            if math.isfinite(correlation):
                items.append(
                    Diagnostic(
                        name=f"{prefix}_vs_fitted_corr",
                        status="warn" if abs(correlation) > 0.3 else "ok",
                        value=correlation,
                        message=""
                        if abs(correlation) <= 0.3
                        else "residuals are correlated with fitted values",
                    )
                )
    return items


def heteroskedasticity(
    residuals: Any,
    fitted: Any | None = None,
    *,
    design: Any | None = None,
) -> list[Diagnostic]:
    """Breusch–Pagan and White tests on squared residuals."""

    residual_values, fitted_values = _aligned(residuals, fitted) if fitted is not None else (_array(residuals), None)
    n_rows = residual_values.size
    if n_rows < 5:
        return [
            Diagnostic(
                name="breusch_pagan",
                status="not_applicable",
                message="fewer than five observations",
            )
        ]

    squared = residual_values**2
    if fitted_values is not None and fitted_values.size >= n_rows:
        regressor = fitted_values[:n_rows]
        bp_design = _with_intercept(regressor)
        white_design = _with_intercept(regressor, regressor**2)
        bp_df = 1
        white_df = 2
    elif design is not None:
        matrix = _design_matrix(design, n_rows)
        if not np.any(np.all(np.isclose(matrix, 1.0), axis=0)):
            matrix = np.column_stack([np.ones(n_rows), matrix])
        bp_design = matrix
        white_design = np.column_stack(
            [matrix[:, [0]], matrix[:, 1:] ** 2]
        ) if matrix.shape[1] > 1 else matrix
        bp_df = max(1, matrix.shape[1] - 1)
        white_df = max(1, white_design.shape[1] - 1)
    else:
        return [
            Diagnostic(
                name="breusch_pagan",
                status="not_applicable",
                message="fitted values or design matrix required",
            )
        ]

    bp_r2 = _ols_r2(squared, bp_design)
    bp_statistic = n_rows * bp_r2
    bp_pvalue = float(stats.chi2.sf(bp_statistic, bp_df))
    white_r2 = _ols_r2(squared, white_design)
    white_statistic = n_rows * white_r2
    white_pvalue = float(stats.chi2.sf(white_statistic, white_df))

    return [
        Diagnostic(
            name="breusch_pagan",
            status="warn" if bp_pvalue < 0.05 else "ok",
            value=bp_pvalue,
            message=f"LM={bp_statistic:.3f}, df={bp_df}",
        ),
        Diagnostic(
            name="white",
            status="warn" if white_pvalue < 0.05 else "ok",
            value=white_pvalue,
            message=f"LM={white_statistic:.3f}, df={white_df}",
        ),
    ]


def vif(design: Any, feature_names: Sequence[str] | None = None) -> list[Diagnostic]:
    """Variance inflation factors plus the design condition number."""

    matrix = _design_matrix(design)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return [
            Diagnostic(
                name="vif",
                status="not_applicable",
                message="empty design matrix",
            )
        ]

    names = list(feature_names or [f"x{index}" for index in range(matrix.shape[1])])
    if len(names) != matrix.shape[1]:
        names = [f"x{index}" for index in range(matrix.shape[1])]

    # Drop an intercept column from VIF calculations: it is collinear with
    # nothing in a well-formed design and VIF is not meaningful for it.
    columns = [
        index
        for index in range(matrix.shape[1])
        if not np.allclose(matrix[:, index], 1.0)
    ]
    if not columns:
        return [
            Diagnostic(
                name="condition_number",
                status="ok",
                value=1.0,
                message="intercept-only design",
            )
        ]

    items: list[Diagnostic] = []
    maximum = 0.0
    for index in columns:
        target = matrix[:, index]
        others = matrix[:, [column for column in columns if column != index]]
        if others.size == 0:
            value = 1.0
        else:
            centered = target - target.mean()
            if float(np.sum(centered**2)) <= 1e-20:
                value = math.inf
            else:
                r2 = _ols_r2(target, np.column_stack([np.ones(matrix.shape[0]), others]))
                value = 1.0 / max(1e-12, 1.0 - min(r2, 1.0 - 1e-12))
        maximum = max(maximum, value)
        items.append(
            Diagnostic(
                name=f"vif_{names[index]}",
                status="warn" if value > 10.0 else "ok",
                value=None if not math.isfinite(value) else value,
                message="" if value <= 10.0 else "high collinearity",
            )
        )

    scaled = matrix[:, columns]
    norms = np.linalg.norm(scaled, axis=0)
    norms = np.where(norms > 1e-12, norms, 1.0)
    scaled = scaled / norms
    centered = scaled - scaled.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    nonzero = singular_values[singular_values > 1e-12]
    condition = (
        float(nonzero.max() / nonzero.min())
        if nonzero.size > 1 and nonzero.min() > 0
        else 1.0
    )
    items.append(
        Diagnostic(
            name="condition_number",
            status="warn" if condition > 30.0 else "ok",
            value=condition,
            message="" if condition <= 30.0 else "ill-conditioned design",
        )
    )
    items.append(
        Diagnostic(
            name="max_vif",
            status="warn" if maximum > 10.0 else "ok",
            value=None if not math.isfinite(maximum) else maximum,
            message="" if maximum <= 10.0 else "at least one high-VIF feature",
        )
    )
    return items


def durbin_watson(residuals: Any) -> Diagnostic:
    """Durbin–Watson statistic for first-order residual autocorrelation."""

    values = _array(residuals)
    if values.size < 3:
        return Diagnostic(
            name="durbin_watson",
            status="not_applicable",
            message="fewer than three residuals",
        )
    denominator = float(np.sum(values**2))
    statistic = (
        float(np.sum(np.diff(values) ** 2)) / denominator
        if denominator > 0
        else 2.0
    )
    return Diagnostic(
        name="durbin_watson",
        status="ok" if 1.5 <= statistic <= 2.5 else "warn",
        value=statistic,
        message="" if 1.5 <= statistic <= 2.5 else "residual autocorrelation",
    )


def autocorrelation(residuals: Any, *, max_lag: int = 10) -> list[Diagnostic]:
    """Lag-1 autocorrelation and the Ljung–Box portmanteau test."""

    values = _array(residuals)
    n_rows = values.size
    if n_rows < 4:
        return [
            Diagnostic(
                name="autocorrelation",
                status="not_applicable",
                message="fewer than four residuals",
            )
        ]

    lag = max(1, min(max_lag, n_rows - 2))
    centered = values - values.mean()
    variance = float(np.sum(centered**2))
    if variance <= 0:
        return [
            Diagnostic(name="autocorrelation_lag1", status="ok", value=0.0),
            Diagnostic(name="ljung_box", status="ok", value=1.0),
        ]

    lag1 = float(np.sum(centered[:-1] * centered[1:]) / variance)
    q_statistic = 0.0
    for offset in range(1, lag + 1):
        rho = float(np.sum(centered[:-offset] * centered[offset:]) / variance)
        q_statistic += rho**2 / max(1, n_rows - offset)
    q_statistic *= n_rows * (n_rows + 2)
    pvalue = float(stats.chi2.sf(q_statistic, lag))

    return [
        Diagnostic(
            name="autocorrelation_lag1",
            status="warn" if abs(lag1) > 0.3 else "ok",
            value=lag1,
            message="" if abs(lag1) <= 0.3 else "strong lag-1 autocorrelation",
        ),
        Diagnostic(
            name="ljung_box",
            status="warn" if pvalue < 0.05 else "ok",
            value=pvalue,
            message=f"Q={q_statistic:.3f}, lag={lag}",
        ),
    ]


def overdispersion(y: Any, mu: Any, *, parameters: int = 1) -> Diagnostic:
    """Pearson overdispersion ratio for count models."""

    observed, expected = _aligned(y, mu)
    mask = expected > 0
    observed, expected = observed[mask], expected[mask]
    if observed.size < 2:
        return Diagnostic(
            name="overdispersion",
            status="not_applicable",
            message="fewer than two positive fitted means",
        )

    degrees = max(1, observed.size - max(1, parameters))
    pearson = float(np.sum((observed - expected) ** 2 / expected) / degrees)
    return Diagnostic(
        name="overdispersion",
        status="warn" if pearson > 1.5 else "ok",
        value=pearson,
        message="" if pearson <= 1.5 else "variance exceeds the Poisson mean",
    )


def _core_adf(series: np.ndarray) -> tuple[float, float]:
    """A deterministic Dickey–Fuller fallback with a left-tail p-value."""

    differenced = np.diff(series)
    lagged = series[:-1]
    design = np.column_stack([np.ones(lagged.size), lagged])
    coefficients, *_ = np.linalg.lstsq(design, differenced, rcond=None)
    residual = differenced - design @ coefficients
    degrees = max(1, lagged.size - 2)
    mse = float(np.sum(residual**2) / degrees)
    covariance = mse * np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(max(covariance[1, 1], 1e-20)))
    statistic = float(coefficients[1] / standard_error)
    pvalue = float(stats.t.cdf(statistic, degrees))
    return statistic, pvalue


def stationarity(series: Any) -> list[Diagnostic]:
    """ADF stationarity test (KPSS too when statsmodels is available)."""

    values = _array(series)
    if values.size < 8:
        return [
            Diagnostic(
                name="adf",
                status="not_applicable",
                message="fewer than eight observations",
            )
        ]

    if STATSMODELS_AVAILABLE and _sm_adfuller is not None:
        try:
            statistic, pvalue, *_ = _sm_adfuller(values, autolag="AIC")
        except Exception:
            statistic, pvalue = _core_adf(values)
    else:
        statistic, pvalue = _core_adf(values)

    items = [
        Diagnostic(
            name="adf",
            status="warn" if pvalue > 0.05 else "ok",
            value=pvalue,
            message=f"statistic={statistic:.3f}"
            + ("" if pvalue <= 0.05 else "; cannot reject a unit root"),
        )
    ]

    if STATSMODELS_AVAILABLE and _sm_kpss is not None:
        try:
            kpss_statistic, kpss_pvalue, *_ = _sm_kpss(values, regression="c", nlags="auto")
        except Exception:
            return items
        items.append(
            Diagnostic(
                name="kpss",
                status="warn" if kpss_pvalue < 0.05 else "ok",
                value=float(kpss_pvalue),
                message=f"statistic={kpss_statistic:.3f}",
            )
        )
    return items


def seasonality(series: Any, *, period: int | None = None) -> list[Diagnostic]:
    """Periodogram check for a dominant seasonal cycle."""

    values = _array(series)
    if values.size < 8:
        return [
            Diagnostic(
                name="seasonality",
                status="not_applicable",
                message="fewer than eight observations",
            )
        ]

    time = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(time, values, 1)
    detrended = values - (slope * time + intercept)
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    if spectrum.size < 3:
        return [
            Diagnostic(
                name="seasonality",
                status="not_applicable",
                message="series is too short for a periodogram",
            )
        ]

    power = spectrum[1:]
    total = float(np.sum(power))
    if total <= 0:
        return [Diagnostic(name="seasonality", status="ok", value=0.0)]

    peak_index = int(np.argmax(power)) + 1
    ratio = float(power[peak_index - 1] / total)
    dominant_period = float(values.size / peak_index)
    status = "warn" if ratio > 0.25 and dominant_period > 2 else "ok"
    message = f"dominant period ~{dominant_period:.1f}"
    if period is not None:
        message += f"; tested period {period}"
    return [
        Diagnostic(
            name="seasonality",
            status=status,
            value=ratio,
            message="" if status == "ok" else message,
        )
    ]


def influence(residuals: Any, design: Any | None = None, *, fitted: Any | None = None) -> list[Diagnostic]:
    """Cook's distance summary for detecting influential rows."""

    residual_values = np.asarray(residuals, dtype=float).reshape(-1)
    if design is not None:
        matrix = _design_matrix(design, residual_values.size)
    elif fitted is not None:
        fitted_values = np.asarray(fitted, dtype=float).reshape(-1)
        size = min(fitted_values.size, residual_values.size)
        residual_values = residual_values[:size]
        matrix = _with_intercept(fitted_values[:size])
    else:
        return [
            Diagnostic(
                name="cooks_distance",
                status="not_applicable",
                message="design matrix or fitted values required",
            )
        ]

    size = min(matrix.shape[0], residual_values.size)
    matrix = matrix[:size]
    residual_values = residual_values[:size]
    mask = np.isfinite(residual_values)
    matrix = matrix[mask]
    residual_values = residual_values[mask]
    n_rows, n_columns = matrix.shape
    if n_rows <= n_columns + 1:
        return [
            Diagnostic(
                name="cooks_distance",
                status="not_applicable",
                message="not enough observations for leverage",
            )
        ]

    covariance = np.linalg.pinv(matrix.T @ matrix)
    leverage = np.einsum("ij,jk,ik->i", matrix, covariance, matrix)
    leverage = np.clip(leverage, 0.0, 1.0 - 1e-9)
    mse = float(np.sum(residual_values**2) / max(1, n_rows - n_columns))
    if mse <= 0:
        return [Diagnostic(name="cooks_distance", status="ok", value=0.0)]
    distances = (
        residual_values**2
        / (max(1, n_columns) * mse)
        * leverage
        / (1.0 - leverage) ** 2
    )
    maximum = float(np.max(distances))
    count = int(np.sum(distances > 1.0))
    return [
        Diagnostic(
            name="cooks_distance",
            status="warn" if maximum > 1.0 else "ok",
            value=maximum,
            message=f"{count} row(s) with distance > 1" if count else "",
        )
    ]


def calibration(y_true: Any, probabilities: Any, *, n_bins: int = 10) -> list[Diagnostic]:
    """Expected calibration error and Brier score for binary probabilities."""

    observed, predicted = _aligned(y_true, probabilities)
    if observed.size < 5:
        return [
            Diagnostic(
                name="calibration",
                status="not_applicable",
                message="fewer than five observations",
            )
        ]

    predicted = np.clip(predicted, 0.0, 1.0)
    brier = float(np.mean((predicted - observed) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    expected_calibration_error = 0.0
    for index in range(n_bins):
        lower, upper = bins[index], bins[index + 1]
        if index == n_bins - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        if not np.any(mask):
            continue
        expected_calibration_error += (
            float(np.mean(mask))
            * abs(float(np.mean(predicted[mask])) - float(np.mean(observed[mask])))
        )

    return [
        Diagnostic(
            name="calibration",
            status="warn" if expected_calibration_error > 0.1 else "ok",
            value=expected_calibration_error,
            message=f"ECE={expected_calibration_error:.3f}, Brier={brier:.3f}",
        )
    ]


def coefficient_stability(
    coefficients: Sequence[dict[str, float]],
    *,
    max_terms: int = 20,
) -> list[Diagnostic]:
    """Report coefficient sign and magnitude stability across folds."""

    folds = [fold for fold in coefficients if fold]
    if len(folds) < 2:
        return [
            Diagnostic(
                name="coefficient_stability",
                status="not_applicable",
                message="at least two folds are required",
            )
        ]

    names: list[str] = []
    for fold in folds:
        for name in fold:
            if name not in names:
                names.append(name)

    items: list[Diagnostic] = []
    for name in names[:max_terms]:
        values = np.asarray(
            [fold.get(name, np.nan) for fold in folds], dtype=float
        )
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        unstable = abs(mean) > 1e-8 and std > 0.5 * abs(mean)
        sign_flip = bool(np.any(values > 0) and np.any(values < 0))
        items.append(
            Diagnostic(
                name=f"coefficient_stability_{name}",
                status="warn" if unstable or sign_flip else "ok",
                value=std,
                message=f"mean={mean:.4f}, std={std:.4f}"
                + ("; sign flips across folds" if sign_flip else ""),
            )
        )
    if not items:
        items.append(
            Diagnostic(
                name="coefficient_stability",
                status="not_applicable",
                message="no coefficients reported across folds",
            )
        )
    return items


def run_diagnostics(
    *,
    residuals: Any | None = None,
    fitted: Any | None = None,
    predicted: Any | None = None,
    design: Any | None = None,
    feature_names: Sequence[str] | None = None,
    y: Any | None = None,
    mu: Any | None = None,
    series: Any | None = None,
    probabilities: Any | None = None,
    y_true: Any | None = None,
    coefficient_folds: Sequence[dict[str, float]] | None = None,
    target_kind: TargetKind = "continuous",
    max_lag: int = 10,
    n_parameters: int = 1,
) -> Diagnostics:
    """Run every applicable diagnostic and return a typed container.

    ``predicted`` is an alias for ``fitted``.  When ``residuals`` is omitted it
    is derived from ``y - predicted`` (or ``y - mu`` for count models).
    """

    diagnostics = Diagnostics()
    fitted_values = fitted if fitted is not None else predicted
    residual_values = residuals
    if residual_values is None:
        if y is not None and fitted_values is not None:
            residual_values = np.asarray(y, dtype=float) - np.asarray(
                fitted_values, dtype=float
            )
        elif y is not None and mu is not None:
            residual_values = np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)

    if residual_values is not None:
        diagnostics.extend(residual_summary(residual_values, fitted_values))
        diagnostics.extend(heteroskedasticity(residual_values, fitted_values, design=design))
        diagnostics.append(durbin_watson(residual_values))
        diagnostics.extend(autocorrelation(residual_values, max_lag=max_lag))
        diagnostics.extend(influence(residual_values, design, fitted=fitted_values))

    if design is not None:
        diagnostics.extend(vif(design, feature_names))

    if target_kind == "count" and y is not None and mu is not None:
        diagnostics.append(overdispersion(y, mu, parameters=n_parameters))

    if target_kind == "time_series" and series is None and y is not None:
        series = y
    if target_kind == "time_series" and series is not None:
        diagnostics.extend(stationarity(series))
        diagnostics.extend(seasonality(series))

    if probabilities is not None and y_true is not None:
        diagnostics.extend(calibration(y_true, probabilities))

    if coefficient_folds:
        diagnostics.extend(coefficient_stability(coefficient_folds))

    return diagnostics


__all__ = [
    "Diagnostics",
    "STATSMODELS_AVAILABLE",
    "autocorrelation",
    "calibration",
    "coefficient_stability",
    "durbin_watson",
    "heteroskedasticity",
    "influence",
    "overdispersion",
    "residual_summary",
    "run_diagnostics",
    "seasonality",
    "stationarity",
    "vif",
]
