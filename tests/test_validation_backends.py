from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from whack26.artifact import (
    Artifact,
    ClassificationSection,
    CriticSummary,
    DatasetSection,
    ExecutionSection,
    HypothesisSection,
    Provenance,
    ResearchSection,
    RunInfo,
    ScreenSection,
    TargetSection,
)
from whack26.config import Settings
from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)
from whack26.validation import backends
from whack26.validation.backends import (
    validate_statistical,
    validate_with_settings,
)
from whack26.validation.models import ValidationConfig, ValidationReport
from whack26.validation.statistical import (
    StatisticalValidationResult,
    ValidationContext,
    validation_result_from_report,
    validate_in_slice,
)

TIMESTAMP = "2026-11-08T10:00:00Z"


def make_search_result(
    *,
    target_kind: str = "continuous",
    metric: str = "r2",
    selected_scores: list[float] | None = None,
    baseline_scores: list[float] | None = None,
    diagnostics: list[Diagnostic] | None = None,
    folds: bool = True,
    selected_metrics: dict[str, float] | None = None,
) -> ModelSearchResult:
    selected_scores = selected_scores or [0.8, 0.82, 0.78, 0.81, 0.79]
    baseline_scores = baseline_scores or [0.0, 0.0, 0.0, 0.0, 0.0]
    fold_results = (
        [
            FoldResult(
                fold=index,
                train_rows=70,
                test_rows=30,
                metrics={metric: selected},
                baseline_metrics={metric: baseline},
            )
            for index, (selected, baseline) in enumerate(
                zip(selected_scores, baseline_scores)
            )
        ]
        if folds
        else []
    )
    metrics = (
        dict(selected_metrics)
        if selected_metrics is not None
        else {metric: float(np.mean(selected_scores))}
    )
    candidate = ModelCandidate(
        spec=ModelSpec(  # type: ignore[arg-type]
            family="linear", target_kind=target_kind
        ),
        folds=fold_results,
        metrics=metrics,
        metric_std={metric: 0.02},
        diagnostics=list(diagnostics or [Diagnostic(name="residuals", status="ok")]),
        coefficients={"x": 1.0, "intercept": 0.1},
        selected=True,
    )
    baseline = ModelCandidate(
        spec=ModelSpec(family="baseline"),
        folds=[
            FoldResult(
                fold=index,
                train_rows=70,
                test_rows=30,
                metrics={metric: baseline},
                baseline_metrics={metric: baseline},
            )
            for index, baseline in enumerate(baseline_scores)
        ],
        metrics={metric: float(np.mean(baseline_scores))},
    )
    return ModelSearchResult(
        target_kind=target_kind,  # type: ignore[arg-type]
        metric=metric,  # type: ignore[arg-type]
        candidates=[candidate],
        selected=candidate,
        baseline=baseline,
    )


def context(**overrides) -> ValidationContext:
    values = {
        "hypothesis_id": "hyp_test",
        "target": "y",
        "target_kind": "continuous",
        "model_search": make_search_result(),
        "seed": 0,
    }
    values.update(overrides)
    return ValidationContext(**values)


# ---------------------------------------------------------------------------
# In-slice validation
# ---------------------------------------------------------------------------


def test_in_slice_clean_model_passes() -> None:
    result = validate_in_slice(context())

    assert result.backend == "in_slice"
    assert result.verdict == "pass", result
    assert result.status == "passed"
    assert result.permutation_p_value is not None
    assert result.permutation_p_value <= 0.05
    assert result.permutation_source == "fold_scores"
    assert result.baseline_delta is not None and result.baseline_delta > 0
    assert result.stability["stable"] is True
    assert result.stability["coefficient_signs"] == {
        "x": "positive",
        "intercept": "positive",
    }
    assert result.diagnostics[0].name == "residuals"


def test_in_slice_reports_baseline_and_multiple_attempts() -> None:
    result = validate_in_slice(context(attempts=4))

    assert result.attempts == 4
    assert result.metrics["attempts"] == 4.0
    assert result.metrics["folds"] == 5.0
    assert result.baseline_metrics["score"] == pytest.approx(0.0)


def test_in_slice_leakage_diagnostic_fails() -> None:
    result = validate_in_slice(
        context(
            model_search=make_search_result(
                diagnostics=[
                    Diagnostic(
                        name="jumbled_label_leak",
                        status="fail",
                        message="jumbled-label performance exceeded the null",
                    )
                ]
            )
        )
    )

    assert result.verdict == "fail"
    assert result.status == "failed"
    assert "leakage" in (result.reason or "")


def test_in_slice_non_finite_metrics_fail() -> None:
    result = validate_in_slice(
        context(
            model_search=make_search_result(
                selected_metrics={"r2": float("nan")}
            )
        )
    )

    assert result.verdict == "fail"
    assert "non_finite_metrics" in (result.reason or "")


def test_in_slice_without_folds_fails() -> None:
    result = validate_in_slice(
        context(model_search=make_search_result(folds=False))
    )

    assert result.verdict == "fail"
    assert "no_evaluation_split" in (result.reason or "")


def test_in_slice_unstable_folds_are_inconclusive() -> None:
    result = validate_in_slice(
        context(
            model_search=make_search_result(
                selected_scores=[0.9, 0.8, -0.05],
                baseline_scores=[0.0, 0.0, 0.0],
            ),
            max_p=1.0,
        )
    )

    assert result.verdict == "inconclusive", result
    assert result.stability["stable"] is False
    assert "unstable_folds" in (result.reason or "")


def test_in_slice_constant_target_is_insufficient() -> None:
    result = validate_in_slice(context(target_kind="constant"))

    assert result.verdict == "insufficient"
    assert result.status == "skipped"
    assert "constant" in (result.reason or "")


def test_in_slice_without_model_search_is_insufficient() -> None:
    result = validate_in_slice(context(model_search=None))

    assert result.verdict == "insufficient"
    assert result.reason == "model_search_unavailable"


def test_in_slice_uses_raw_predictions_for_permutation() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)
    actuals = 2.0 * x + rng.normal(0.0, 0.1, size=200)
    predictions = 2.0 * x

    result = validate_in_slice(
        context(
            predictions=predictions.tolist(),
            actuals=actuals.tolist(),
            n_permutations=100,
        )
    )

    assert result.verdict == "pass"
    assert result.permutation_source == "predictions"
    assert result.n_permutations == 100


# ---------------------------------------------------------------------------
# Target-kind gating: the engine is never called for unsupported data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["binary", "multiclass", "count", "constant"])
def test_stats_engine_never_called_for_unsupported_targets(
    kind: str, monkeypatch
) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("the stats engine must not be called")

    monkeypatch.setattr(backends, "validate_hypothesis", _boom)
    monkeypatch.setattr(backends, "validate_run", _boom)

    result = validate_statistical(
        context(target_kind=kind),
        backend="stats_engine",
    )

    assert result.verdict == "insufficient"
    assert result.engine_applicable is False
    assert result.backend == "stats_engine"
    assert f"target_kind:{kind}" in (result.reason or "")


def test_stats_engine_not_used_for_general_regression(monkeypatch) -> None:
    calls: list[str] = []

    def _record(*args, **kwargs):
        calls.append("called")
        raise AssertionError("the stats engine must not be called")

    monkeypatch.setattr(backends, "validate_hypothesis", _record)
    monkeypatch.setattr(backends, "validate_run", _record)

    result = validate_statistical(context(target="outcome"), backend="auto")

    assert result.backend == "in_slice"
    assert result.verdict == "pass"
    assert calls == []


def test_auto_uses_in_slice_for_general_data(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("the stats engine must not be called")

    monkeypatch.setattr(backends, "validate_hypothesis", _boom)
    monkeypatch.setattr(backends, "validate_run", _boom)

    resolution = backends.resolve_backend("auto", context(target="outcome"))

    assert resolution.backend.name == "in_slice"


# ---------------------------------------------------------------------------
# External backend seam
# ---------------------------------------------------------------------------


def _write_external_module(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(
        "from whack26.validation.statistical import StatisticalValidationResult\n\n"
        + body,
        encoding="utf-8",
    )


def test_external_fake_adapter_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "_whack26_fake_external_validator"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_external_module(
        tmp_path,
        module_name,
        """
calls = 0


class FakeValidator:
    version = "9.9.9"

    def validate(self, context):
        global calls
        calls += 1
        return StatisticalValidationResult(
            hypothesis_id=context.hypothesis_id,
            backend="fake",
            verdict="pass",
            status="passed",
            applicable=True,
            metrics={"score": 1.0},
        )


validator = FakeValidator()
""",
    )

    result = validate_statistical(
        context(),
        backend="external",
        external_validator=f"{module_name}:validator",
    )

    assert result.backend == "external"
    assert result.verdict == "pass"
    assert result.external_status == "ok"
    assert result.external_validator == f"{module_name}:validator"
    assert result.library_version == "9.9.9"
    assert importlib.import_module(module_name).calls == 1


def test_external_validator_may_return_validation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "_whack26_fake_report_validator"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / f"{module_name}.py").write_text(
        """
from whack26.validation.models import ValidationLayerReport, ValidationReport


class FakeValidator:
    version = "0.1.0"

    def validate(self, context):
        return ValidationReport(
            hypothesis_id=context.hypothesis_id,
            status="passed",
            verdict="pass",
            layers=[
                ValidationLayerReport(layer="Layer 1 (DSR)", status="passed")
            ],
        )


validator = FakeValidator()
""",
        encoding="utf-8",
    )

    result = validate_statistical(
        context(),
        backend="external",
        external_validator=f"{module_name}:validator",
    )

    assert result.backend == "external"
    assert isinstance(result.report, ValidationReport)
    assert result.report.status == "passed"
    assert result.verdict == "pass"


def test_absent_external_falls_back_to_in_slice_with_warning(
    monkeypatch,
) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("the stats engine must not be called")

    monkeypatch.setattr(backends, "validate_hypothesis", _boom)
    monkeypatch.setattr(backends, "validate_run", _boom)

    result = validate_statistical(
        context(),
        backend="external",
        external_validator="_whack26_missing_validator_module",
    )

    assert result.backend == "in_slice"
    assert result.verdict == "pass"
    assert result.external_status == "unavailable"
    assert any(
        "external_validator_unavailable" in warning for warning in result.warnings
    )


def test_throwing_external_falls_back_to_in_slice(tmp_path: Path, monkeypatch) -> None:
    module_name = "_whack26_throwing_external_validator"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_external_module(
        tmp_path,
        module_name,
        """
class FakeValidator:
    def validate(self, context):
        raise RuntimeError("library is still in progress")


validator = FakeValidator()
""",
    )

    result = validate_statistical(
        context(),
        backend="external",
        external_validator=f"{module_name}:validator",
    )

    assert result.backend == "in_slice"
    assert result.external_status == "error"
    assert any(
        "external_validator_failed" in warning for warning in result.warnings
    )


# ---------------------------------------------------------------------------
# No applicable backend
# ---------------------------------------------------------------------------


def test_none_backend_is_insufficient() -> None:
    result = validate_statistical(context(model_search=None), backend="none")

    assert result.verdict == "insufficient"
    assert result.status == "skipped"
    assert result.reason == "statistical_backend_unavailable"
    assert any("statistical_backend_disabled" in item for item in result.warnings)


def test_auto_without_a_supporting_backend_is_insufficient() -> None:
    result = validate_statistical(context(model_search=None), backend="auto")

    assert result.verdict == "insufficient"
    assert result.backend == "none"
    assert result.reason == "statistical_backend_unavailable"


def test_unknown_backend_name_is_insufficient() -> None:
    result = validate_statistical(context(), backend="mystery")

    assert result.verdict == "insufficient"
    assert any("unknown_statistical_backend" in item for item in result.warnings)


def test_validate_with_settings_reads_configuration() -> None:
    result = validate_with_settings(
        context(model_search=None),
        Settings(statistical_backend="none"),
    )

    assert result.verdict == "insufficient"


# ---------------------------------------------------------------------------
# stats_engine backend on a synthetic return series
# ---------------------------------------------------------------------------

RETURN_MODULE = '''import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    return {"x", "returns"}.issubset(df.columns)


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    params = params or PARAMETERS
    ridge = float(params.get("ridge", 1e-3))
    x = df["x"].to_numpy(dtype=float)
    y = df["returns"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(2) * ridge
    penalty[0, 0] = 0.0
    gram = design.T @ design + penalty
    coefficients, *_ = np.linalg.lstsq(gram, design.T @ y, rcond=None)
    return {"coefficients": coefficients.tolist()}


def _predict(model, df: pd.DataFrame):
    coefficients = np.asarray(model["coefficients"], dtype=float)
    x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    return np.column_stack([np.ones(len(x)), x]) @ coefficients


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    model = _fit(df, seed, PARAMETERS)
    return {
        "predictions": _predict(model, df.drop(columns=["returns"])),
        "target": "returns",
    }
'''


def _make_return_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    (run_dir / "hypotheses").mkdir(parents=True)
    (run_dir / "predictions").mkdir(parents=True)
    module_path = run_dir / "hypotheses" / "hyp_test.py"
    module_path.write_text(RETURN_MODULE, encoding="utf-8")

    rng = np.random.default_rng(11)
    x = rng.normal(size=180)
    returns = 0.001 + 0.02 * x + rng.normal(0.0, 0.001, size=180)
    frame = pd.DataFrame({"x": x, "returns": returns})
    frame.to_csv(run_dir / "dataset.csv", index=False)
    prediction = 0.001 + 0.02 * x
    pd.DataFrame(
        {"row_index": range(len(frame)), "prediction": prediction}
    ).to_csv(run_dir / "predictions" / "hyp_test.csv", index=False)

    artifact = Artifact(
        run=RunInfo(
            run_id="run_20261108T100000Z_aaaaaaaaaaaa",
            started_at=TIMESTAMP,
            finished_at=TIMESTAMP,
            mode="mock",
            seed=0,
            tool_version="0.1.0",
            models={"llm": "mock", "embedding": "mock@16"},
        ),
        dataset=DatasetSection(
            id="ds_test",
            source_path="dataset.csv",
            stored_path="dataset.csv",
            sha256="0" * 64,
            format="csv",
            rows=len(frame),
            target=TargetSection(name="returns", source="manual", confidence=1.0),
        ),
        classification=ClassificationSection(),
        research=ResearchSection(),
        screen=ScreenSection(
            target_summary={"name": "returns", "kind": "continuous"}
        ),
        critic_summary=CriticSummary(),
        provenance=Provenance(python="3.12"),
    )
    hypothesis = HypothesisSection(
        id="hyp_test",
        title="Return-series hypothesis",
        method_family="linear_model",
        claim_type="prediction",
        target="returns",
        required_columns=["x"],
        min_rows=20,
        code_path="hypotheses/hyp_test.py",
        code_sha256="0" * 64,
        execution=ExecutionSection(
            status="ok",
            n_predictions=len(frame),
            predictions_path="predictions/hyp_test.csv",
        ),
    )
    config = ValidationConfig(
        min_dsr=0.0,
        max_p=1.0,
        num_permutations=30,
        max_performance_drop=1.0,
        max_decay_tolerance=0.50,
        timeout_seconds=60.0,
    )
    return run_dir, artifact, hypothesis, config, list(returns)


def test_return_series_uses_stats_engine(tmp_path: Path) -> None:
    run_dir, artifact, hypothesis, config, returns = _make_return_run(tmp_path)

    result = validate_statistical(
        ValidationContext(
            hypothesis_id="hyp_test",
            target="returns",
            target_kind="continuous",
            run_dir=run_dir,
            artifact=artifact,
            hypothesis=hypothesis,
            validation_config=config,
            target_values=returns,
        ),
        backend="stats_engine",
    )

    assert result.backend == "stats_engine"
    assert result.engine_applicable is True
    assert result.verdict == "pass", result
    assert result.status == "passed"
    assert isinstance(result.report, ValidationReport)
    assert result.report.status == "passed"
    assert [layer.layer for layer in result.report.layers] == [
        "Layer 1 (DSR)",
        "Layer 2 (Monte Carlo)",
        "Layer 3 (OOS Decay)",
        "Layer 4 (Parameter Sensitivity)",
    ]
    assert result.to_validation_report() is result.report


def test_auto_return_series_prefers_engine_and_keeps_in_slice(
    tmp_path: Path,
) -> None:
    run_dir, artifact, hypothesis, config, returns = _make_return_run(tmp_path)

    result = validate_statistical(
        ValidationContext(
            hypothesis_id="hyp_test",
            target="returns",
            target_kind="continuous",
            run_dir=run_dir,
            artifact=artifact,
            hypothesis=hypothesis,
            validation_config=config,
            target_values=returns,
            model_search=make_search_result(),
        ),
        backend="auto",
    )

    assert result.backend == "stats_engine"
    assert result.verdict == "pass"
    assert result.supplementary
    assert result.supplementary[0]["backend"] == "in_slice"


def test_return_series_name_is_not_enough_without_values() -> None:
    result = validate_statistical(
        context(target="returns", target_kind="continuous"),
        backend="stats_engine",
    )

    assert result.verdict == "insufficient"
    assert "target_values_unavailable" in (result.reason or "")


# ---------------------------------------------------------------------------
# Existing ValidationReport compatibility
# ---------------------------------------------------------------------------


def test_in_slice_result_maps_to_existing_validation_report() -> None:
    result = validate_in_slice(context())

    report = result.to_validation_report()

    assert isinstance(report, ValidationReport)
    assert report.hypothesis_id == "hyp_test"
    assert report.status == "passed"
    assert report.verdict == "pass"
    assert report.p_value is not None
    assert report.layers[0].layer == "in_slice statistical validation"


def test_validation_report_wrapper_round_trips() -> None:
    original = ValidationReport(
        hypothesis_id="hyp_test",
        status="failed",
        verdict="fail",
        layers=[],
        warnings=["kept"],
    )

    result = validation_result_from_report(original, backend="stats_engine")

    assert result.backend == "stats_engine"
    assert result.verdict == "fail"
    assert result.status == "failed"
    assert result.report is original
    assert result.to_validation_report() is original
    assert result.warnings == ["kept"]


def test_unsupported_backend_does_not_pass() -> None:
    result = StatisticalValidationResult(
        hypothesis_id="hyp_test",
        verdict="insufficient",
        status="skipped",
        applicable=False,
        reason="statistical_backend_unavailable",
    )

    report = result.to_validation_report()

    assert report.status == "skipped"
    assert report.verdict == "inconclusive"
