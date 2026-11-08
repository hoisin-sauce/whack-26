"""Runner routing for non-continuous targets.

Synthetic numpy/pandas data only. These tests exercise
``whack26.validation.runner._full_engine`` and ``_validate`` directly: binary,
multiclass, and count matrices must go through ``engine_bridge`` while
continuous/time-series matrices keep the legacy ``validation_engine`` path.
Bridge failures must become error reports, and unsupported kinds must be
skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from stats_engine import r2_score
from whack26.validation import engine_bridge, runner
from whack26.validation.adapter import FeatureMatrix
from whack26.validation.models import (
    ValidationConfig,
    ValidationLayerReport,
    ValidationReport,
)

LAYER_NAMES = [
    "Layer 1 (DSR)",
    "Layer 2 (Monte Carlo)",
    "Layer 3 (OOS Decay)",
    "Layer 4 (Parameter Sensitivity)",
]

MODULE_SOURCE = '''import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    return {"x", "z", "y"}.issubset(df.columns)


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    x = df[["x", "z"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"coefficients": coefficients.tolist()}


def _predict(model, df: pd.DataFrame):
    coefficients = np.asarray(model["coefficients"], dtype=float)
    x = df[["x", "z"]].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(x)), x]) @ coefficients
'''


def _dummy_builder(
    params: dict[str, Any], X: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    def predict(new_X: np.ndarray) -> np.ndarray:
        return np.zeros(len(new_X), dtype=float)

    return predict


def _feature_matrix(kind: str, rows: int = 90, seed: int = 3) -> FeatureMatrix:
    """Hand-built matrix so routing can be tested without the adapter."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 2))
    if kind in {"continuous", "time_series"}:
        y = 1.2 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(0.0, 0.2, rows)
    elif kind == "binary":
        y = (X[:, 0] + 0.4 * X[:, 1] > 0.0).astype(float)
    elif kind == "multiclass":
        y = np.mod(np.arange(rows), 3).astype(float)
    elif kind == "count":
        y = rng.poisson(2.0, size=rows).astype(float)
    else:  # values are irrelevant for an unsupported kind
        y = np.arange(rows, dtype=float)
    return FeatureMatrix(
        X=X,
        y=y,
        specs=[],
        feature_columns=["x", "z"],
        retained_index=list(range(rows)),
        dropped_rows=0,
        target_name="y",
        target_dtype="float64",
        periods_per_year=1.0,
        periods_per_year_source="default",
        target_kind=kind,
    )


def _dataset(kind: str, rows: int = 160, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    if kind == "binary":
        y = (x + 0.5 * z > 0.0).astype(int)
    elif kind == "multiclass":
        y = np.digitize(x + 0.3 * z, [-0.6, 0.6])
    elif kind == "count":
        y = rng.poisson(np.exp(0.4 + 0.5 * x))
    else:
        y = 2.0 * x - 0.5 * z + rng.normal(0.0, 0.2, rows)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _prepare(tmp_path: Path, kind: str) -> tuple[Path, Path]:
    module_path = tmp_path / "hypothesis.py"
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    dataset_path = tmp_path / "dataset.csv"
    _dataset(kind).to_csv(dataset_path, index=False)
    return module_path, dataset_path


def _payload(module_path: Path, dataset_path: Path, kind: str) -> dict[str, Any]:
    return {
        "hypothesis_id": f"hyp_{kind}",
        "module_path": str(module_path),
        "dataset_path": str(dataset_path),
        "predictions_path": None,
        "target": "y",
        "required_columns": ["x", "z"],
        "optional_columns": [],
        "time_column": None,
        "target_kind": kind,
        "description": f"synthetic {kind}",
        "config": {
            "min_dsr": 0.0,
            "max_p": 1.0,
            "num_permutations": 20,
            "timeout_seconds": 60.0,
        },
    }


def _success_result() -> dict[str, Any]:
    """A successful general-engine result with non-applicable layers."""
    return {
        "passed": True,
        "failed_layer": None,
        "message": "All applicable validation layers cleared successfully.",
        "dsr_score": None,
        "p_value": 0.01,
        "oos_score": 0.55,
        "decay_rate": 0.10,
        "tolerance": 0.30,
        "layers": {
            "dsr": {
                "status": "not_applicable",
                "reason": "DSR applies only to continuous targets.",
            },
            "permutation": {"status": "passed", "p_value": 0.01},
            "oos_decay": {
                "status": "passed",
                "oos_score": 0.55,
                "decay_rate": 0.10,
                "tolerance": 0.30,
            },
            "parameter_sensitivity": {
                "status": "not_applicable",
                "message": "No perturbable model hyperparameters were supplied.",
            },
        },
    }


# ---------------------------------------------------------------------------
# Bridge routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["binary", "multiclass", "count"])
def test_noncontinuous_targets_route_to_the_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module_path, dataset_path = _prepare(tmp_path, kind)
    payload = _payload(module_path, dataset_path, kind)
    captured: dict[str, Any] = {}

    def fake_run_engine(
        *,
        matrix: FeatureMatrix,
        config: ValidationConfig,
        description: str | None = None,
        base_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        captured["matrix"] = matrix
        captured["config"] = config
        captured["description"] = description
        captured["base_params"] = dict(base_params or {})
        return _success_result()

    def legacy_should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("legacy engine used for a non-continuous target")

    monkeypatch.setattr(engine_bridge, "run_engine", fake_run_engine)
    monkeypatch.setattr(runner, "validation_engine", legacy_should_not_run)

    report = runner._validate(payload)

    matrix = captured["matrix"]
    assert isinstance(matrix, FeatureMatrix)
    assert matrix.target_kind == kind
    assert isinstance(captured["config"], ValidationConfig)
    assert captured["description"] == f"synthetic {kind}"
    assert captured["base_params"].get("ridge") == 1e-3

    assert report.status == "partial"
    assert report.verdict == "inconclusive"
    assert [layer.layer for layer in report.layers] == LAYER_NAMES
    assert [layer.status for layer in report.layers] == [
        "skipped",
        "passed",
        "passed",
        "skipped",
    ]
    assert "DSR" in (report.layers[0].reason or "")
    assert report.layers[1].metrics["p_value"] == pytest.approx(0.01)
    assert report.layers[2].metrics["oos_score"] == pytest.approx(0.55)
    assert report.dsr_score is None
    assert report.p_value == pytest.approx(0.01)
    assert report.oos_score == pytest.approx(0.55)
    assert report.decay_rate == pytest.approx(0.10)
    assert report.tolerance == pytest.approx(0.30)
    assert report.in_sample_fallback is False
    assert report.has_fit_predict is True
    assert not any(
        warning.startswith("engine_fallback") for warning in report.warnings
    )


def test_continuous_target_uses_the_legacy_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path, dataset_path = _prepare(tmp_path, "continuous")
    payload = _payload(module_path, dataset_path, "continuous")
    captured: dict[str, Any] = {}
    bridge_calls: list[dict[str, Any]] = []

    def fake_trial_sharpes(*args: Any, **kwargs: Any):
        return np.array([0.4, 0.8, 1.2]), np.array([0.01, 0.02, 0.03])

    def fake_validation_engine(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "passed": True,
            "dsr_score": 0.97,
            "p_value": 0.02,
            "oos_score": 0.50,
            "decay_rate": 0.10,
            "tolerance": 0.20,
            "message": "All 4 validation layers cleared successfully.",
        }

    def bridge_should_not_run(**kwargs: Any) -> dict[str, Any]:
        bridge_calls.append(kwargs)
        raise AssertionError("bridge used for a continuous target")

    monkeypatch.setattr(runner, "trial_sharpes", fake_trial_sharpes)
    monkeypatch.setattr(runner, "validation_engine", fake_validation_engine)
    monkeypatch.setattr(engine_bridge, "run_engine", bridge_should_not_run)

    report = runner._validate(payload)

    assert bridge_calls == []
    assert captured["metric_func"] is r2_score
    assert captured["higher_is_better"] is True
    assert captured["data_description"] == "synthetic continuous"
    assert report.status == "passed"
    assert report.verdict == "pass"
    assert [layer.status for layer in report.layers] == ["passed"] * 4
    assert report.dsr_score == pytest.approx(0.97)


@pytest.mark.parametrize("kind", ["continuous", "time_series"])
def test_time_series_alias_keeps_the_legacy_path(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    matrix = _feature_matrix(kind)
    config = ValidationConfig(min_dsr=0.0, max_p=1.0, num_permutations=10)
    captured: dict[str, Any] = {}

    def fake_trial_sharpes(*args: Any, **kwargs: Any):
        return np.array([0.2, 0.5, 0.9]), np.array([0.01, 0.02, 0.03])

    def fake_validation_engine(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"passed": True, "message": "legacy"}

    def bridge_should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("bridge used for a continuous-like target")

    monkeypatch.setattr(runner, "trial_sharpes", fake_trial_sharpes)
    monkeypatch.setattr(runner, "validation_engine", fake_validation_engine)
    monkeypatch.setattr(engine_bridge, "run_engine", bridge_should_not_run)

    outcome = runner._full_engine(
        builder=_dummy_builder,
        matrix=matrix,
        base_params=config.engine_params({}),
        config=config,
        description="legacy",
    )

    assert captured["metric_func"] is r2_score
    assert captured["higher_is_better"] is True
    assert captured["data_description"] == "legacy"
    assert outcome == {"passed": True, "message": "legacy"}
    assert not isinstance(outcome.get("layers"), list)


@pytest.mark.parametrize("kind", ["ordinal", "constant", "classifier"])
def test_unsupported_kind_is_skipped(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    matrix = _feature_matrix(kind)
    config = ValidationConfig()
    calls: list[str] = []

    def legacy_should_not_run(**kwargs: Any) -> dict[str, Any]:
        calls.append("legacy")
        return {"passed": True}

    def bridge_should_not_run(**kwargs: Any) -> dict[str, Any]:
        calls.append("bridge")
        return {"passed": True}

    monkeypatch.setattr(runner, "validation_engine", legacy_should_not_run)
    monkeypatch.setattr(engine_bridge, "run_engine", bridge_should_not_run)

    outcome = runner._full_engine(
        builder=_dummy_builder,
        matrix=matrix,
        base_params={},
        config=config,
        description=None,
    )

    assert calls == []
    layers = outcome["layers"]
    assert isinstance(layers, list)
    assert len(layers) == 1
    assert isinstance(layers[0], ValidationLayerReport)
    assert layers[0].layer == "Target contract"
    assert layers[0].status == "skipped"
    assert kind in (layers[0].reason or "")

    report = runner._finish(
        ValidationReport(hypothesis_id="unsupported", layers=layers)
    )
    assert report.status == "skipped"
    assert report.verdict == "inconclusive"


# ---------------------------------------------------------------------------
# Bridge outcomes and failures
# ---------------------------------------------------------------------------


def test_bridge_failure_maps_to_a_failed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path, dataset_path = _prepare(tmp_path, "binary")
    payload = _payload(module_path, dataset_path, "binary")

    def fake_run_engine(**kwargs: Any) -> dict[str, Any]:
        return {
            "passed": False,
            "failed_layer": "Layer 2 (Monte Carlo)",
            "message": "Layer 2 FAIL: Permutation p-value 0.9000 > 0.05",
            "p_value": 0.90,
            "layers": {
                "dsr": {
                    "status": "not_applicable",
                    "reason": "DSR applies only to continuous targets.",
                }
            },
        }

    monkeypatch.setattr(engine_bridge, "run_engine", fake_run_engine)

    report = runner._validate(payload)

    assert report.status == "failed"
    assert report.verdict == "fail"
    layer2 = report.layer("Layer 2 (Monte Carlo)")
    assert layer2 is not None
    assert layer2.status == "failed"
    assert layer2.reason == "Layer 2 FAIL: Permutation p-value 0.9000 > 0.05"
    assert report.p_value == pytest.approx(0.90)
    assert [layer.status for layer in report.layers] == [
        "skipped",
        "failed",
        "skipped",
        "skipped",
    ]


def test_bridge_exception_becomes_an_error_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path, dataset_path = _prepare(tmp_path, "binary")
    payload = _payload(module_path, dataset_path, "binary")

    def explode(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("bridge exploded")

    def legacy_should_not_run(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("legacy engine used after a bridge failure")

    monkeypatch.setattr(engine_bridge, "run_engine", explode)
    monkeypatch.setattr(runner, "validation_engine", legacy_should_not_run)

    report = runner._validate(payload)

    assert report.status == "error"
    assert report.verdict == "inconclusive"
    assert report.error == "RuntimeError: bridge exploded"
    assert [layer.layer for layer in report.layers] == ["Validation"]
    assert report.layers[0].status == "error"
    assert "RuntimeError: bridge exploded" in (report.layers[0].reason or "")
    assert not any(
        warning.startswith("engine_fallback") for warning in report.warnings
    )


def test_layer_mapping_exception_becomes_an_error_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path, dataset_path = _prepare(tmp_path, "count")
    payload = _payload(module_path, dataset_path, "count")

    monkeypatch.setattr(
        engine_bridge,
        "run_engine",
        lambda **kwargs: None,  # engine_result_to_layers rejects non-dicts
    )

    report = runner._validate(payload)

    assert report.status == "error"
    assert report.verdict == "inconclusive"
    assert "TypeError" in (report.error or "")
    assert report.layers[0].status == "error"
