"""WI-4: generated module -> statistical validator -> artifact contract.

Three real mock/offline pipeline runs (continuous, binary, count) exercise the
whole chain:

    make_dataset -> run_pipeline(validate=True) -> generated module
    -> isolated validation runner -> artifact.validation -> preview.md

Every run validates its three generated hypotheses with a small permutation
budget so the slice stays fast. Two extra validator passes reuse copies of the
continuous run to prove the legacy in-sample fallback and the advisory
failure path. No network access and no optional stages (final judge, scrape,
model search, variants, generation validation or field pitfalls) are enabled.
"""

from __future__ import annotations

import importlib.util
import inspect
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from integration_fixtures import (
    TARGET,
    load_artifact,
    make_dataset,
    make_settings,
    read_json,
    run_mock_pipeline,
)

from whack26.validation.adapter import build_feature_matrix, infer_target_kind
from whack26.validation.engine_bridge import engine_metric
from whack26.validation.models import (
    ValidationConfig,
    ValidationReport,
    summary_counts,
)
from whack26.validation.pipeline import validate_hypothesis

pytestmark = pytest.mark.integration

VALIDATION_CONFIG = ValidationConfig(
    num_permutations=10,
    timeout_seconds=60.0,
    min_dsr=0.0,
    max_p=1.0,
    max_performance_drop=1.0,
)

LAYERS = (
    "Layer 1 (DSR)",
    "Layer 2 (Monte Carlo)",
    "Layer 3 (OOS Decay)",
    "Layer 4 (Parameter Sensitivity)",
)

LEGACY_MODULE = '''"""Legacy hypothesis module without validation helpers."""

import numpy as np
import pandas as pd


def applies(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) > 0


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    df = df.copy()
    np.random.seed(seed)
    numeric = [
        name for name in df.columns
        if pd.api.types.is_numeric_dtype(df[name])
    ]
    if numeric:
        values = pd.to_numeric(df[numeric[0]], errors="coerce").fillna(0.0)
        predictions = values.to_numpy(dtype=float)
    else:
        predictions = np.zeros(len(df), dtype=float)
    return {"predictions": predictions, "target": "y"}
'''

FAILING_FIT_MODULE = '''"""Hypothesis module whose documented _fit helper always fails."""

import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) > 0


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    raise RuntimeError("wi4 deliberate _fit failure")


def _predict(model, df: pd.DataFrame):
    return np.zeros(len(df), dtype=float)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    return {"predictions": np.zeros(len(df), dtype=float), "target": "y"}
'''


# ---------------------------------------------------------------------------
# Fixtures: three pipeline executions, one per target kind
# ---------------------------------------------------------------------------


def _run_kind(tmp_path: Path, kind: str) -> Any:
    dataset = make_dataset(tmp_path / f"input_{kind}.csv", kind=kind, rows=80)
    return run_mock_pipeline(
        tmp_path,
        dataset,
        settings=make_settings(tmp_path),
        out_name=kind,
        validate=True,
        validation_config=VALIDATION_CONFIG,
    )


@pytest.fixture(scope="module")
def continuous_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_kind(tmp_path_factory.mktemp("continuous"), "continuous")


@pytest.fixture(scope="module")
def binary_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_kind(tmp_path_factory.mktemp("binary"), "binary")


@pytest.fixture(scope="module")
def count_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return _run_kind(tmp_path_factory.mktemp("count"), "count")


@pytest.fixture(scope="module")
def legacy_run(
    tmp_path_factory: pytest.TempPathFactory, continuous_run: Any
) -> tuple[Path, Any, Any]:
    destination = tmp_path_factory.mktemp("legacy-copy") / "run"
    shutil.copytree(continuous_run.run_dir, destination)
    artifact = load_artifact(destination)
    hypothesis = artifact.hypotheses[0]
    predictions = destination / hypothesis.execution.predictions_path
    assert predictions.is_file(), "the completed run has no saved predictions"
    (destination / hypothesis.code_path).write_text(
        LEGACY_MODULE, encoding="utf-8"
    )
    return destination, artifact, hypothesis


@pytest.fixture(scope="module")
def failing_fit_run(
    tmp_path_factory: pytest.TempPathFactory, continuous_run: Any
) -> tuple[Path, Any, Any]:
    destination = tmp_path_factory.mktemp("failing-copy") / "run"
    shutil.copytree(continuous_run.run_dir, destination)
    artifact = load_artifact(destination)
    hypothesis = artifact.hypotheses[0]
    (destination / hypothesis.code_path).write_text(
        FAILING_FIT_MODULE, encoding="utf-8"
    )
    return destination, artifact, hypothesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_module(path: Path) -> Any:
    name = f"_wi4_contract_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, (
        f"could not build an import spec for {path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_with_documented_signature(
    fit: Any, frame: pd.DataFrame, seed: int, params: dict[str, Any]
) -> Any:
    """Call ``_fit`` the way the isolated validation runner does."""
    signature = inspect.signature(fit)
    parameters = signature.parameters
    has_var_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "seed" in parameters or has_var_kwargs:
        kwargs["seed"] = seed
    if "params" in parameters or has_var_kwargs:
        kwargs["params"] = dict(params)
    if kwargs:
        return fit(frame, **kwargs)
    positional = [
        item
        for item in parameters.values()
        if item.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) <= 1:
        return fit(frame)
    if len(positional) == 2:
        return fit(frame, seed)
    return fit(frame, seed, dict(params))


def _predictions(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        result = result.get("predictions")
    assert result is not None, "`_predict` returned no predictions"
    return np.asarray(result, dtype=float).ravel()


def _assert_finite_uniform_predictions(
    module: Any,
    model: Any,
    frame: pd.DataFrame,
    *,
    label: str,
) -> None:
    predictions = _predictions(module._predict(model, frame))
    assert predictions.shape == (len(frame),), (
        f"{label}: `_predict` returned {predictions.shape} predictions for "
        f"{len(frame)} rows"
    )
    assert np.all(np.isfinite(predictions)), (
        f"{label}: `_predict` returned non-finite predictions"
    )


def _assert_four_continuous_layers(report: ValidationReport) -> None:
    names = [layer.layer for layer in report.layers]
    assert set(LAYERS) <= set(names), (
        f"{report.hypothesis_id}: expected all four layers, got {names}"
    )
    for layer in report.layers:
        if layer.status == "skipped":
            assert layer.reason, (
                f"{report.hypothesis_id}: {layer.layer} was skipped without "
                "an explicit reason"
            )


def _assert_persisted_summary(
    run_dir: Path, artifact: Any, reports: list[ValidationReport]
) -> None:
    summary = artifact.validation_summary
    assert summary is not None, "artifact has no validation summary"
    assert summary_counts(reports) == summary.counts
    assert sum(summary.counts.values()) == 3

    payload = read_json(run_dir / "validation.json")
    assert payload["schema_version"] == "1.0.0"
    assert payload["run_id"] == artifact.run.run_id
    assert payload["config"]["num_permutations"] == 10
    assert payload["config"]["timeout_seconds"] == 60.0
    assert payload["config"]["min_dsr"] == 0.0
    assert payload["config"]["max_p"] == 1.0
    assert payload["config"]["max_performance_drop"] == 1.0
    assert payload["summary"] == summary.counts
    assert len(payload["reports"]) == 3
    assert [item["hypothesis_id"] for item in payload["reports"]] == [
        report.hypothesis_id for report in reports
    ]
    assert [item["status"] for item in payload["reports"]] == [
        report.status for report in reports
    ]


# ---------------------------------------------------------------------------
# 1. Generated module contract
# ---------------------------------------------------------------------------


def test_generated_modules_expose_the_validation_helpers(request: Any) -> None:
    run_fixtures = {
        "continuous_run": "continuous",
        "binary_run": "binary",
        "count_run": "count",
    }
    checked = 0
    for fixture_name, kind in run_fixtures.items():
        run = request.getfixturevalue(fixture_name)
        artifact = load_artifact(run.run_dir)
        frame = pd.read_csv(run.run_dir / artifact.dataset.stored_path)
        for hypothesis in artifact.hypotheses:
            label = f"{kind}/{hypothesis.id}"
            module_path = run.run_dir / hypothesis.code_path
            module = _load_module(module_path)

            assert callable(getattr(module, "_fit", None)), (
                f"{label}: module has no callable _fit"
            )
            assert callable(getattr(module, "_predict", None)), (
                f"{label}: module has no callable _predict"
            )
            parameters = getattr(module, "PARAMETERS", None)
            assert isinstance(parameters, dict), (
                f"{label}: PARAMETERS is {type(parameters).__name__}, not a dict"
            )
            assert parameters, f"{label}: PARAMETERS is empty"
            for name, value in parameters.items():
                assert isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ), f"{label}: PARAMETERS[{name!r}] is not numeric"
                assert np.isfinite(float(value)), (
                    f"{label}: PARAMETERS[{name!r}] is not finite"
                )

            declared = set(hypothesis.required_columns) | set(
                hypothesis.optional_columns
            )
            assert TARGET not in declared, (
                f"{label}: the target was declared as a feature column"
            )

            matrix = build_feature_matrix(
                frame,
                target=TARGET,
                required_columns=hypothesis.required_columns,
                optional_columns=hypothesis.optional_columns,
                time_column=artifact.dataset.time_column,
                target_kind=kind,
            )
            prediction_frame = matrix.frame_from_matrix(matrix.X)
            assert TARGET not in prediction_frame.columns, (
                f"{label}: the prediction frame contains the target"
            )
            # Mock modules declare no columns, so the adapter auto-selects the
            # numeric features. When declarations exist, the frame must be a
            # subset of them.
            allowed = declared or (set(frame.columns) - {TARGET})
            assert set(prediction_frame.columns) <= allowed, (
                f"{label}: prediction frame columns "
                f"{sorted(prediction_frame.columns)} are outside {sorted(allowed)}"
            )
            assert set(prediction_frame.columns) == set(matrix.feature_columns)

            fit_frame = matrix.frame_from_matrix(matrix.X, matrix.y)
            model = _fit_with_documented_signature(
                module._fit, fit_frame, seed=VALIDATION_CONFIG.seed,
                params=parameters,
            )
            assert model is not None, f"{label}: `_fit` returned no model"
            _assert_finite_uniform_predictions(
                module, model, prediction_frame, label=label
            )
            checked += 1

    assert checked == 9, f"expected nine generated modules, checked {checked}"


# ---------------------------------------------------------------------------
# 2. Continuous run: reports, artifact and preview
# ---------------------------------------------------------------------------


def test_continuous_reports_flow_into_artifact_and_preview(
    continuous_run: Any,
) -> None:
    run_dir = continuous_run.run_dir
    artifact = load_artifact(run_dir)
    hypothesis_ids = [item.id for item in artifact.hypotheses]
    reports = list(continuous_run.validations.values())

    assert len(reports) == 3
    assert [report.hypothesis_id for report in reports] == hypothesis_ids

    for report in reports:
        assert report.status != "error", report
        assert all(layer.status != "error" for layer in report.layers), report
        assert report.target == TARGET
        _assert_four_continuous_layers(report)

    # The documented continuous primary metric is R2. The bridge reports the
    # same mapping and the artifact preview renders the OOS score as "OOS R2".
    metric_name, metric_func, higher_is_better = engine_metric("continuous")
    assert metric_name == "r2"
    assert metric_func is not None and higher_is_better is True

    with_oos = [report for report in reports if report.oos_score is not None]
    assert with_oos, "no continuous report captured an out-of-sample R2"
    for report in with_oos:
        assert np.isfinite(report.oos_score)
        layer = report.layer("Layer 3 (OOS Decay)")
        assert layer is not None
        assert layer.metrics.get("oos_score") is not None

    for hypothesis in artifact.hypotheses:
        assert hypothesis.validation is not None, (
            f"{hypothesis.id}: artifact has no embedded validation report"
        )

    _assert_persisted_summary(run_dir, artifact, reports)

    preview = (run_dir / "preview.md").read_text(encoding="utf-8")
    assert "## Validation summary" in preview
    assert "Statistical validation:" in preview
    assert "OOS R2" in preview


# ---------------------------------------------------------------------------
# 3. Binary and count runs validate end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture_name", "expected_kind"),
    [("binary_run", "binary"), ("count_run", "count")],
)
def test_binary_and_count_targets_validate_end_to_end(
    request: Any, fixture_name: str, expected_kind: str
) -> None:
    run = request.getfixturevalue(fixture_name)
    run_dir = run.run_dir
    artifact = load_artifact(run_dir)
    frame = pd.read_csv(run_dir / artifact.dataset.stored_path)

    # The validator resolves the screen's generic numeric/categorical kind by
    # inferring the concrete target kind from the stored dataset.
    assert infer_target_kind(frame[TARGET]) == expected_kind
    assert artifact.screen.target_summary.get("name") == TARGET

    reports = list(run.validations.values())
    assert len(reports) == 3
    for report in reports:
        assert report.status != "error", report
        assert report.status in {"partial", "passed"}, report
        assert all(layer.status != "error" for layer in report.layers), report
        dsr = report.layer("Layer 1 (DSR)")
        assert dsr is not None
        assert dsr.status == "skipped", (
            f"{report.hypothesis_id}: DSR should be not_applicable for "
            f"{expected_kind}, got {dsr.status}"
        )
        assert dsr.reason, "the skipped DSR layer has no explicit reason"

    summary = artifact.validation_summary
    assert summary is not None
    assert summary_counts(reports) == summary.counts
    assert sum(summary.counts.values()) == 3
    assert summary.counts.get("error", 0) == 0

    _assert_persisted_summary(run_dir, artifact, reports)
    # The artifact still passes handoff validation with embedded reports.
    load_artifact(run_dir)

    preview = (run_dir / "preview.md").read_text(encoding="utf-8")
    assert "## Validation summary" in preview
    assert all(
        f"Statistical validation: {report.status}" in preview
        for report in reports
    )


# ---------------------------------------------------------------------------
# 4. Legacy module without helpers
# ---------------------------------------------------------------------------


def test_legacy_module_without_helpers_falls_back(
    legacy_run: tuple[Path, Any, Any],
) -> None:
    run_dir, artifact, hypothesis = legacy_run
    predictions_path = run_dir / hypothesis.execution.predictions_path
    assert predictions_path.is_file(), "copied run lost the saved predictions"

    # Must not raise: validation is advisory even without fit/predict helpers.
    report = validate_hypothesis(
        hypothesis,
        run_dir=run_dir,
        artifact=artifact,
        config=VALIDATION_CONFIG,
    )

    assert report.hypothesis_id == hypothesis.id
    assert report.in_sample_fallback is True
    assert report.has_fit_predict is False
    assert report.status == "partial", report
    assert report.layer("Layer 3 (OOS Decay)").status == "skipped"
    assert report.layer("Layer 4 (Parameter Sensitivity)").status == "skipped"
    assert any(
        "missing_fit_predict" in warning for warning in report.warnings
    ), report.warnings
    assert predictions_path.is_file(), "validation consumed saved predictions"


# ---------------------------------------------------------------------------
# 5. A failing _fit is reported, never raised
# ---------------------------------------------------------------------------


def test_validation_failure_is_reported_not_raised(
    failing_fit_run: tuple[Path, Any, Any],
) -> None:
    run_dir, artifact, hypothesis = failing_fit_run

    # Must not raise out of the advisory validation stage.
    report = validate_hypothesis(
        hypothesis,
        run_dir=run_dir,
        artifact=artifact,
        config=VALIDATION_CONFIG,
    )

    assert report.status == "error", report
    # The engine fallback records the failure per layer: layers 1-2 surface it
    # as ``error`` while OOS decay and PSS convert builder failures to
    # ``failed`` results. At least one layer must carry the error status.
    assert any(layer.status == "error" for layer in report.layers), report
    texts = (
        [report.error or ""]
        + [layer.reason or "" for layer in report.layers]
        + list(report.warnings)
    )
    combined = "\n".join(texts)
    assert "RuntimeError" in combined, combined
    assert "wi4 deliberate _fit failure" in combined, combined
