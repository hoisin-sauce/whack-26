"""Deterministic-check tests for the final judge."""

from __future__ import annotations

from whack26.final_judge.checks import (
    check_assumptions,
    check_extrapolation,
    check_leakage,
    check_multiple_comparisons,
    check_none,
    check_overfitting,
    check_residuals,
    check_simplicity,
    check_subgroups,
    compute_all,
    data_flags,
    metric_is_higher_better,
)


def test_overfitting_detects_train_cv_gap() -> None:
    result = check_overfitting(
        {
            "metric": "r2",
            "train_score": 0.95,
            "cv_mean": 0.80,
            "cv_sd": 0.02,
            "n_folds": 3,
        }
    )

    assert result["train_cv_gap"] == 0.15
    assert result["gap_in_fold_sds"] == 7.5
    assert result["gap_is_large"] is True


def test_residual_check_recomputes_ljung_box_p_value() -> None:
    result = check_residuals(
        {"statistic": 9.2, "lags": 10, "p_value": 0.41}
    )

    assert result["ljung_box_consistent"] is False
    assert 0.4 < result["ljung_box_p_recomputed"] < 0.6


def test_leakage_detective_flags_target_derived_feature() -> None:
    result = check_leakage(
        {
            "features": [
                {"name": "y", "derived_from_target": True},
                {"name": "x", "known_at_prediction_time": True},
            ]
        }
    )

    assert result["n_features"] == 2
    assert result["suspect_features"] == [
        {"name": "y", "reasons": ["derived from the target"]}
    ]


def test_multiple_comparisons_threshold_uses_uncorrected_count() -> None:
    result = check_multiple_comparisons(
        {"n_tests": 10, "min_p": 0.01, "p_is_corrected": False}
    )

    assert result["threshold_used"] == 0.005
    assert result["survives_correction"] is False


def test_extrapolation_reports_outside_support() -> None:
    result = check_extrapolation(
        {"train": {"min": 0, "max": 10}, "pred": {"min": 0, "max": 12}}
    )

    assert result["outside_support"] is True
    assert result["excess_above_train_max"] == 2.0
    assert result["excess_pct_of_train_range"] == 20.0


def test_simplicity_uses_corrected_paired_interval() -> None:
    result = check_simplicity(
        {
            "metric": "r2",
            "higher_is_better": True,
            "model_fold_scores": [0.81, 0.99, 0.99, 0.85, 0.91],
            "baseline_fold_scores": [0.74, 0.93, 0.93, 0.80, 0.85],
            "n_train": 80,
            "n_test": 20,
        }
    )

    assert result["paired_n_folds"] == 5
    assert result["folds_where_model_wins"] == 5
    assert result["paired_95ci_corrected"][0] > 0
    assert "Nadeau" in result["caveat"]


def test_assumptions_check_flags_failed_tests() -> None:
    result = check_assumptions(
        {
            "tests": {
                "adf_p": 0.01,
                "durbin_watson": 1.0,
                "breusch_pagan_p": 0.02,
                "vif_max": 20.0,
                "calibration_ece": 0.2,
                "overdispersion": 2.0,
            }
        }
    )

    assert result["adf_rejects_unit_root_5pct"] is True
    assert result["durbin_watson_near_2"] is False
    assert result["linearity_test_supplied"] is False
    assert result["heteroskedasticity_rejected_5pct"] is True
    assert result["multicollinearity_high"] is True
    assert result["calibration_poor"] is True
    assert result["overdispersion_high"] is True


def test_assumptions_check_without_tests_only_reports_missing_linearity() -> None:
    assert check_assumptions({}) == {"linearity_test_supplied": False}


def test_subgroups_uses_the_metric_direction() -> None:
    higher = check_subgroups(
        {
            "metric": "r2",
            "slices": {"a": {"r2": 0.8}, "b": {"r2": 0.5}},
        }
    )
    lower = check_subgroups(
        {
            "metric": "rmse",
            "slices": {"a": {"rmse": 1.0}, "b": {"rmse": 2.0}},
        }
    )

    assert higher["best_slice"] == "a"
    assert higher["worst_slice"] == "b"
    assert higher["metric_spread"] == 0.3
    assert lower["best_slice"] == "a"
    assert lower["worst_slice"] == "b"
    assert lower["metric_spread"] == 1.0


def test_subgroups_requires_two_numeric_slices() -> None:
    assert check_subgroups({"slices": {"a": {"r2": 0.8}}}) == {}
    assert (
        check_subgroups(
            {"slices": {"a": {"r2": 0.8}, "b": {"r2": "not-a-number"}}}
        )
        == {}
    )


def test_none_check_is_empty() -> None:
    assert check_none({"anything": [1, 2, 3]}) == {}


def test_metric_direction_defaults_and_override() -> None:
    assert metric_is_higher_better("r2") is True
    assert metric_is_higher_better("auc") is True
    assert metric_is_higher_better("rmse") is False
    assert metric_is_higher_better("mae") is False
    assert metric_is_higher_better("rmse", explicit=True) is True
    assert metric_is_higher_better("r2", explicit=False) is False


def test_compute_all_skips_empty_evidence() -> None:
    assert compute_all({}) == {}
    assert compute_all({"domain_sanity_critic": {}}) == {}


def test_data_flags_include_computed_findings() -> None:
    evidence = {
        "overfitting_skeptic": {
            "metric": "r2",
            "train_score": 0.99,
            "cv_mean": 0.80,
            "cv_sd": 0.01,
        },
        "extrapolation_critic": {
            "train": {"min": 0, "max": 10},
            "pred": {"min": 0, "max": 11},
        },
        "leakage_detective": {
            "features": [{"name": "y", "derived_from_target": True}]
        },
        "multiple_comparisons_skeptic": {
            "n_tests": 10,
            "min_p": 0.01,
            "p_is_corrected": False,
        },
    }

    flags = data_flags(compute_all(evidence))

    assert any("Train-CV gap" in flag for flag in flags)
    assert any("Prediction range" in flag for flag in flags)
    assert any("Leakage suspects" in flag for flag in flags)
    assert any("multiple-comparison" in flag for flag in flags)
