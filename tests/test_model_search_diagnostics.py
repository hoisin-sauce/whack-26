"""Tests for model-search diagnostics."""

from __future__ import annotations

import numpy as np

from whack26.model_search.diagnostics import (
    Diagnostics,
    autocorrelation,
    calibration,
    coefficient_stability,
    durbin_watson,
    heteroskedasticity,
    influence,
    overdispersion,
    residual_summary,
    run_diagnostics,
    seasonality,
    stationarity,
    vif,
)


def test_residual_summary_reports_shape() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 1.0, 200)

    items = residual_summary(residuals)
    names = {item.name for item in items}

    assert {"residual_mean", "residual_std", "residual_skew"} <= names
    assert all(item.status == "ok" for item in items)


def test_residual_summary_flags_outliers_and_trend() -> None:
    rng = np.random.default_rng(1)
    fitted = np.linspace(-3.0, 3.0, 100)
    residuals = rng.normal(0.0, 0.2, 100) + 0.8 * fitted
    residuals[0] = 12.0

    items = {item.name: item for item in residual_summary(residuals, fitted)}

    assert items["residual_max_abs"].status == "warn"
    assert items["residual_vs_fitted_corr"].status == "warn"


def test_breusch_pagan_and_white_flag_heteroskedasticity() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0.5, 4.0, 400)
    residuals = rng.normal(0.0, 1.0, x.size) * x
    fitted = 2.0 * x

    items = {item.name: item for item in heteroskedasticity(residuals, fitted)}

    assert items["breusch_pagan"].status == "warn"
    assert items["breusch_pagan"].value is not None
    assert items["breusch_pagan"].value < 0.05
    assert items["white"].status == "warn"


def test_vif_and_condition_number_flag_collinearity() -> None:
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=200)
    x2 = x1 + rng.normal(scale=0.001, size=200)
    x3 = rng.normal(size=200)
    design = np.column_stack([np.ones(200), x1, x2, x3])

    items = {item.name: item for item in vif(design, ["intercept", "x1", "x2", "x3"])}

    assert items["vif_x1"].status == "warn"
    assert items["vif_x2"].status == "warn"
    assert items["max_vif"].status == "warn"
    assert items["condition_number"].status == "warn"
    assert items["condition_number"].value is not None
    assert items["condition_number"].value > 30


def test_durbin_watson_and_autocorrelation_flag_serial_dependence() -> None:
    rng = np.random.default_rng(4)
    n = 300
    residuals = np.zeros(n)
    for index in range(1, n):
        residuals[index] = 0.85 * residuals[index - 1] + rng.normal()

    dw = durbin_watson(residuals)
    acf = {item.name: item for item in autocorrelation(residuals)}

    assert dw.status == "warn"
    assert dw.value is not None and dw.value < 1.5
    assert acf["autocorrelation_lag1"].status == "warn"
    assert acf["ljung_box"].status == "warn"


def test_overdispersion_flags_extra_poisson_variance() -> None:
    rng = np.random.default_rng(5)
    means = np.full(500, 2.0)
    y = rng.poisson(rng.gamma(shape=0.5, scale=4.0, size=means.size))

    item = overdispersion(y, means)

    assert item.status == "warn"
    assert item.value is not None and item.value > 1.5


def test_stationarity_distinguishes_random_walk_from_white_noise() -> None:
    rng = np.random.default_rng(6)
    random_walk = np.cumsum(rng.normal(size=250))
    white_noise = rng.normal(size=250)

    walk_items = {item.name: item for item in stationarity(random_walk)}
    noise_items = {item.name: item for item in stationarity(white_noise)}

    assert walk_items["adf"].status == "warn"
    assert noise_items["adf"].status == "ok"


def test_seasonality_finds_dominant_period() -> None:
    time = np.arange(240, dtype=float)
    series = np.sin(2.0 * np.pi * time / 12.0) + np.random.default_rng(7).normal(
        scale=0.05, size=time.size
    )

    item = seasonality(series)[0]

    assert item.status == "warn"
    assert item.value is not None and item.value > 0.25
    assert "period" in item.message


def test_influence_flags_a_single_extreme_row() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=60)
    y = 2.0 * x + rng.normal(scale=0.2, size=x.size)
    y[0] += 100.0
    design = np.column_stack([np.ones(x.size), x])
    residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]

    item = influence(residuals, design)[0]

    assert item.status == "warn"
    assert item.value is not None and item.value > 1.0


def test_calibration_flags_miscalibration() -> None:
    y_true = np.zeros(100)
    probabilities = np.full(100, 0.9)

    item = calibration(y_true, probabilities)[0]

    assert item.status == "warn"
    assert item.value is not None and item.value > 0.1


def test_coefficient_stability_flags_sign_flips() -> None:
    folds = [{"x1": 1.0, "x2": -0.5}, {"x1": -1.0, "x2": -0.45}]

    items = {item.name: item for item in coefficient_stability(folds)}

    assert items["coefficient_stability_x1"].status == "warn"
    assert items["coefficient_stability_x2"].status == "ok"


def test_run_diagnostics_returns_typed_container() -> None:
    rng = np.random.default_rng(9)
    x = rng.uniform(0.5, 3.0, 300)
    fitted = 1.5 * x
    y = fitted + rng.normal(size=x.size) * x
    design = np.column_stack([np.ones(x.size), x])

    diagnostics = run_diagnostics(
        y=y,
        predicted=fitted,
        design=design,
        feature_names=["intercept", "x"],
        target_kind="continuous",
    )

    assert isinstance(diagnostics, Diagnostics)
    assert len(diagnostics) > 5
    assert diagnostics.get("breusch_pagan") is not None
    assert diagnostics.has_flags()
    assert diagnostics.worst_status() in {"warn", "fail"}
    assert diagnostics.status("adf") == "not_applicable"

    count = run_diagnostics(
        y=y, mu=np.full(y.size, y.mean()), target_kind="count"
    )
    assert count.get("overdispersion") is not None

    time_series = run_diagnostics(
        y=rng.normal(size=120).cumsum(), target_kind="time_series"
    )
    assert time_series.get("adf") is not None
    assert time_series.get("seasonality") is not None


def test_empty_diagnostics_container() -> None:
    diagnostics = Diagnostics()
    assert len(diagnostics) == 0
    assert diagnostics.worst_status() == "ok"
    assert diagnostics.has_flags() is False
