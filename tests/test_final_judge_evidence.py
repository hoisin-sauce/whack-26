"""Evidence-adapter tests for the final judge."""

from __future__ import annotations

import pytest

from artifact_fixtures import make_artifact

from whack26.artifact import ModelSearchSection
from whack26.final_judge.evidence import build_evidence_bundle
from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)
from whack26.validation.models import ValidationLayerReport, ValidationReport


def make_model_search() -> ModelSearchResult:
    folds = [
        FoldResult(
            fold=index,
            train_rows=80,
            test_rows=20,
            metrics={"r2": 0.80 + index * 0.01},
            baseline_metrics={"r2": 0.10},
            train_metrics={"r2": 0.95},
            train_target_min=0.0,
            train_target_max=10.0,
            prediction_min=0.0,
            prediction_max=12.0,
        )
        for index in range(3)
    ]
    candidate = ModelCandidate(
        spec=ModelSpec(family="linear", features=["x"]),
        folds=folds,
        metrics={"r2": 0.81},
        metric_std={"r2": 0.01},
        diagnostics=[
            Diagnostic(
                name="ljung_box",
                status="ok",
                value=0.41,
                metadata={"statistic": 9.2, "lags": 10},
            ),
            Diagnostic(name="durbin_watson", status="ok", value=2.0),
            Diagnostic(name="breusch_pagan", status="ok", value=0.20),
        ],
        coefficients={"x": 1.0},
        selected=True,
    )
    return ModelSearchResult(
        target_kind="continuous",
        metric="r2",
        candidates=[candidate],
        selected=candidate,
        dataset_info={
            "time_column": None,
            "features": [{"name": "x", "kind": "numeric", "reason": ""}],
            "excluded": {"y": "target"},
        },
        screen={
            "features": [
                {"feature": "x", "p_value": 0.01, "adjusted_p": 0.02}
            ]
        },
    )


def test_bundle_maps_model_search_and_validation_evidence() -> None:
    report = ValidationReport(
        hypothesis_id="hyp_test",
        status="passed",
        verdict="pass",
        layers=[ValidationLayerReport(layer="Layer 1", status="passed")],
    )

    bundle = build_evidence_bundle(
        model_search=make_model_search(),
        validation_reports=[report],
    )

    assert bundle.evidence["overfitting_skeptic"]["train_score"] == pytest.approx(0.95)
    assert bundle.evidence["residual_analyst"]["statistic"] == 9.2
    assert bundle.evidence["residual_analyst"]["lags"] == 10
    assert bundle.evidence["assumptions_auditor"]["tests"]["durbin_watson"] == 2.0
    assert bundle.evidence["extrapolation_critic"]["pred"]["max"] == 12.0
    assert len(bundle.evidence["simplicity_advocate"]["model_fold_scores"]) == 3
    assert bundle.evidence["multiple_comparisons_skeptic"]["n_tests"] == 1
    assert bundle.evidence["leakage_detective"]["features"][0]["name"] == "x"
    assert bundle.model_summary["selected_family"] == "linear"
    assert bundle.model_summary["validation_verdict"] == "pass"
    assert "subgroup_slices" in bundle.missing_evidence
    assert "statistical_validation" not in bundle.missing_evidence


def test_bundle_without_model_search_reports_missing_evidence() -> None:
    bundle = build_evidence_bundle()

    assert bundle.evidence == {}
    assert "selected_model" in bundle.missing_evidence
    assert "fold_metrics" in bundle.missing_evidence
    assert "statistical_validation" in bundle.missing_evidence
    assert bundle.model_summary["selected_family"] is None


def test_bundle_falls_back_to_the_artifact_model_search_section(
    tmp_path,
) -> None:
    artifact, _, _ = make_artifact(tmp_path)
    artifact.model_search = ModelSearchSection(
        target_kind="continuous",
        metric="r2",
        selected_family="linear",
        selected_features=["x"],
        selected_metrics={"r2": 0.81},
        candidate_count=1,
        candidates=[
            {
                "family": "linear",
                "features": ["x"],
                "rank": 1,
                "selected": True,
                "metrics": {"r2": 0.81},
                "metric_std": {"r2": 0.01},
                "diagnostics": [
                    {
                        "name": "ljung_box",
                        "status": "warn",
                        "value": 0.41,
                        "message": "Q=9.2, lag=10, df=2",
                    },
                    {
                        "name": "durbin_watson",
                        "status": "ok",
                        "value": 2.0,
                        "message": "",
                    },
                ],
            }
        ],
        warnings=["artifact_warning"],
    )

    bundle = build_evidence_bundle(artifact=artifact)

    assert bundle.model_summary["selected_family"] == "linear"
    assert bundle.model_summary["metric"] == "r2"
    assert bundle.evidence["residual_analyst"]["p_value"] == 0.41
    assert bundle.evidence["residual_analyst"]["statistic"] == 9.2
    assert bundle.evidence["residual_analyst"]["lags"] == 10
    assert bundle.evidence["leakage_detective"]["features"][0]["name"] == "x"
    assert "extrapolation_critic" not in bundle.evidence
    assert "fold_metrics" in bundle.missing_evidence
    assert "train_metrics" in bundle.missing_evidence
    assert "prediction_range" in bundle.missing_evidence


def test_bundle_uses_dataset_target_and_reports_missing_model_search(
    tmp_path,
) -> None:
    artifact, _, _ = make_artifact(tmp_path)
    artifact.model_search = None

    bundle = build_evidence_bundle(artifact=artifact)

    assert bundle.model_summary["target"] == "y"
    assert "overfitting_skeptic" not in bundle.evidence
    assert "domain_sanity_critic" in bundle.evidence
    assert "selected_model" in bundle.missing_evidence
