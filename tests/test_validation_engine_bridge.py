from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stats_engine import (
    TARGET_BINARY,
    TARGET_CONTINUOUS,
    TARGET_COUNT,
    TARGET_MULTICLASS,
    log_loss,
    poisson_deviance,
    r2_score,
)
from whack26.validation import engine_bridge
from whack26.validation.engine_bridge import (
    build_engine_model,
    engine_metric,
    engine_result_to_layers,
    engine_supported,
    engine_target_type,
    run_engine,
)
from whack26.validation.models import ValidationConfig

LAYER_NAMES = [
    "Layer 1 (DSR)",
    "Layer 2 (Monte Carlo)",
    "Layer 3 (OOS Decay)",
    "Layer 4 (Parameter Sensitivity)",
]


@dataclass
class DummyMatrix:
    """Duck-typed stand-in for ``FeatureMatrix``."""

    X: np.ndarray
    y: np.ndarray
    target_kind: str
    periods_per_year: float = 1.0


def make_binary_data(n: int = 240, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    logits = 1.1 * features[:, 0] - 0.7 * features[:, 1] + 0.2 * features[:, 2]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    targets = rng.binomial(1, probabilities)
    return features, targets


def make_multiclass_data(n: int = 240, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    centres = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]])
    distances = np.stack(
        [np.sum((features - centre) ** 2, axis=1) for centre in centres], axis=1
    )
    targets = np.argmin(distances + rng.normal(0.0, 0.2, distances.shape), axis=1)
    return features, targets


def make_count_data(n: int = 240, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    means = np.exp(0.6 + 0.35 * features[:, 0] - 0.25 * features[:, 1])
    targets = rng.poisson(means)
    return features, targets


def make_continuous_data(
    n: int = 180, seed: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 3))
    targets = (
        1.5 * features[:, 0]
        - 0.5 * features[:, 1]
        + rng.normal(0.0, 0.3, n)
    )
    return features, targets


def loose_config(**overrides: object) -> ValidationConfig:
    values: dict[str, object] = {
        "max_p": 1.0,
        "num_permutations": 50,
        "max_performance_drop": 10.0,
        "max_decay_tolerance": 2.0,
        "seed": 7,
    }
    values.update(overrides)
    return ValidationConfig(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Target-kind mapping
# ---------------------------------------------------------------------------


def test_engine_target_type_maps_project_kinds() -> None:
    assert engine_target_type("continuous") == TARGET_CONTINUOUS
    assert engine_target_type("binary") == TARGET_BINARY
    assert engine_target_type("multiclass") == TARGET_MULTICLASS
    assert engine_target_type("count") == TARGET_COUNT
    assert engine_target_type("time_series") == TARGET_CONTINUOUS
    assert engine_target_type(" Binary ") == TARGET_BINARY


def test_engine_target_type_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        engine_target_type("ordinal")
    with pytest.raises(ValueError):
        engine_target_type("constant")


def test_engine_supported_flags_unsupported_kinds() -> None:
    for kind in ("continuous", "binary", "multiclass", "count", "time_series"):
        assert engine_supported(kind) == (True, "")
    assert engine_supported("constant") == (False, "target_kind:constant")
    assert engine_supported("ordinal") == (False, "target_kind:ordinal")
    assert engine_supported("") == (False, "target_kind:unknown")


# ---------------------------------------------------------------------------
# Builder and metric contract
# ---------------------------------------------------------------------------


def test_build_engine_model_prediction_shapes() -> None:
    features, targets = make_binary_data(120)
    predict = build_engine_model(
        "binary", {"ridge": 0.1, "max_iter": 100}, features[:90], targets[:90]
    )
    probabilities = predict(features[90:])
    assert probabilities.shape == (30,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    features, targets = make_multiclass_data(150)
    assert np.unique(targets[:120]).size == 3
    predict = build_engine_model(
        "multiclass",
        {"ridge": 0.1, "max_iter": 200},
        features[:120],
        targets[:120],
    )
    matrix = predict(features[120:])
    assert matrix.shape == (30, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)

    features, targets = make_count_data(120)
    predict = build_engine_model(
        "count", {"ridge": 0.1}, features[:90], targets[:90]
    )
    means = predict(features[90:])
    assert means.shape == (30,)
    assert np.all(means > 0.0)

    features, targets = make_continuous_data(120)
    predict = build_engine_model(
        "continuous", {"ridge": 0.1}, features[:90], targets[:90]
    )
    assert predict(features[90:]).shape == (30,)

    with pytest.raises(ValueError):
        build_engine_model("ordinal", {}, features, targets)


def test_engine_metric_reports_primary_metric() -> None:
    assert engine_metric("continuous") == ("r2", r2_score, True)
    assert engine_metric("time_series") == ("r2", r2_score, True)
    assert engine_metric("binary") == ("log_loss", log_loss, False)
    assert engine_metric("multiclass") == ("log_loss", log_loss, False)
    assert engine_metric("count") == ("poisson_deviance", poisson_deviance, False)

    with pytest.raises(ValueError):
        engine_metric("ordinal")


# ---------------------------------------------------------------------------
# Engine routing
# ---------------------------------------------------------------------------


def test_run_engine_binary_end_to_end() -> None:
    features, targets = make_binary_data()
    matrix = DummyMatrix(features, targets, "binary")

    result = run_engine(matrix=matrix, config=loose_config(), description="binary")

    assert result["passed"] is True
    assert result["target_type"] == TARGET_BINARY
    assert result["metric"] == "log_loss"
    assert result["data_description"] == "binary"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert result["layers"]["permutation"]["status"] == "passed"
    assert result["layers"]["oos_decay"]["status"] == "passed"
    assert result["diagnostics"]["expected_calibration_error"] >= 0.0


def test_run_engine_multiclass_end_to_end() -> None:
    features, targets = make_multiclass_data()
    matrix = DummyMatrix(features, targets, "multiclass")

    result = run_engine(matrix=matrix, config=loose_config())

    assert result["passed"] is True
    assert result["target_type"] == TARGET_MULTICLASS
    assert result["metric"] == "log_loss"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert "macro_f1" in result["diagnostics"]


def test_run_engine_count_end_to_end() -> None:
    features, targets = make_count_data()
    matrix = DummyMatrix(features, targets, "count")

    result = run_engine(matrix=matrix, config=loose_config())

    assert result["passed"] is True
    assert result["target_type"] == TARGET_COUNT
    assert result["metric"] == "poisson_deviance"
    assert result["layers"]["dsr"]["status"] == "not_applicable"
    assert "dispersion" in result["diagnostics"]


def test_run_engine_continuous_end_to_end() -> None:
    features, targets = make_continuous_data()
    matrix = DummyMatrix(features, targets, "continuous", periods_per_year=252.0)

    result = run_engine(matrix=matrix, config=loose_config())

    assert result["passed"] is True
    assert result["target_type"] == TARGET_CONTINUOUS
    assert result["metric"] == r2_score.__name__
    assert result["layers"]["dsr"]["status"] == "not_applicable"


def test_run_engine_forwards_config_and_matrix_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_validate_model(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(engine_bridge, "validate_model", fake_validate_model)
    features, targets = make_count_data(80)
    matrix = DummyMatrix(features, targets, "count", periods_per_year=12.0)
    config = ValidationConfig(
        max_ece=0.25,
        max_dispersion=1.5,
        num_permutations=7,
        seed=11,
        max_decay_tolerance=0.4,
    )

    result = engine_bridge.run_engine(
        matrix=matrix, config=config, description="counts"
    )

    assert result == {"passed": True}
    assert captured["features"] is matrix.X
    assert captured["targets"] is matrix.y
    assert captured["target_type"] == TARGET_COUNT
    assert captured["metric_func"] is poisson_deviance
    assert captured["higher_is_better"] is False
    assert captured["max_ece"] == 0.25
    assert captured["max_dispersion"] == 1.5
    assert captured["max_p_value"] == config.max_p
    assert captured["num_permutations"] == 7
    assert captured["permutation_seed"] == 11
    assert captured["train_split_ratio"] == config.train_split_ratio
    assert captured["max_decay_tolerance"] == 0.4
    assert captured["periods_per_year"] == 12.0
    assert captured["data_description"] == "counts"
    assert captured["base_params"]["max_p_value"] == config.max_p
    assert callable(captured["base_model_builder"])


def test_run_engine_rejects_unsupported_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_validate_model(**kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {"passed": True}

    monkeypatch.setattr(engine_bridge, "validate_model", fake_validate_model)
    matrix = DummyMatrix(np.zeros((30, 2)), np.zeros(30), "constant")

    with pytest.raises(ValueError):
        run_engine(matrix=matrix, config=ValidationConfig())
    assert called is False


# ---------------------------------------------------------------------------
# Layer translation
# ---------------------------------------------------------------------------


def test_engine_result_to_layers_maps_success() -> None:
    result = {
        "passed": True,
        "failed_layer": None,
        "layers": {
            "dsr": {
                "status": "not_applicable",
                "reason": "DSR is defined only for continuous targets.",
            },
            "permutation": {"status": "passed", "p_value": 0.01},
            "oos_decay": {
                "status": "passed",
                "oos_score": 0.62,
                "decay_rate": 0.05,
                "tolerance": 0.25,
                "tolerance_clamped": False,
            },
            "parameter_sensitivity": {
                "status": "not_applicable",
                "message": "No perturbable model hyperparameters were supplied.",
            },
        },
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES
    assert [layer.status for layer in layers] == [
        "skipped",
        "passed",
        "passed",
        "skipped",
    ]
    assert layers[0].reason is not None
    assert "DSR" in layers[0].reason
    assert layers[1].metrics["p_value"] == pytest.approx(0.01)
    assert layers[2].metrics["oos_score"] == pytest.approx(0.62)
    assert layers[2].metrics["tolerance_clamped"] is False
    assert layers[3].reason == "No perturbable model hyperparameters were supplied."


def test_engine_result_to_layers_accepts_monte_carlo_key() -> None:
    result = {
        "passed": True,
        "failed_layer": None,
        "layers": {
            "dsr": {"status": "not_applicable", "reason": "not applicable"},
            "monte_carlo": {"status": "passed", "p_value": 0.02},
            "oos_decay": {"status": "passed", "oos_score": 0.5},
            "parameter_sensitivity": {"status": "passed", "message": "stable"},
        },
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES
    assert layers[1].status == "passed"
    assert layers[1].metrics["p_value"] == pytest.approx(0.02)


def test_engine_result_to_layers_marks_failed_layer_and_stops() -> None:
    result = {
        "passed": False,
        "failed_layer": "Layer 2 (Monte Carlo)",
        "message": "Layer 2 FAIL: Permutation p-value 0.9000 > 0.05",
        "p_value": 0.9,
        "layers": {
            "dsr": {"status": "not_applicable", "reason": "not applicable"}
        },
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES
    assert [layer.status for layer in layers] == [
        "skipped",
        "failed",
        "skipped",
        "skipped",
    ]
    assert layers[1].reason == result["message"]
    assert layers[1].metrics["p_value"] == pytest.approx(0.9)
    assert layers[2].reason == "engine stopped at an earlier layer"
    assert layers[3].reason == "engine stopped at an earlier layer"


def test_engine_result_to_layers_appends_calibration_gate_failure() -> None:
    result = {
        "passed": False,
        "failed_layer": "Calibration",
        "message": "Calibration FAIL: ECE 0.2000 > 0.1",
        "layers": {
            "dsr": {"status": "not_applicable", "reason": "not applicable"},
            "permutation": {"status": "passed", "p_value": 0.02},
            "oos_decay": {"status": "passed", "oos_score": 0.5},
            "parameter_sensitivity": {"status": "not_applicable", "message": "none"},
            "calibration": {"status": "failed", "ece": 0.2},
        },
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES + ["Calibration"]
    assert layers[-1].status == "failed"
    assert layers[-1].reason == result["message"]
    assert layers[-1].metrics["ece"] == pytest.approx(0.2)


def test_engine_result_to_layers_appends_count_dispersion_gate_failure() -> None:
    result = {
        "passed": False,
        "failed_layer": "Count dispersion",
        "message": "Count dispersion FAIL: 1.5000 > 1.1",
        "diagnostics": {"dispersion": 1.5},
        "layers": {
            "dsr": {"status": "not_applicable", "reason": "not applicable"},
            "permutation": {"status": "passed", "p_value": 0.02},
            "oos_decay": {"status": "passed", "oos_score": 0.5},
            "parameter_sensitivity": {"status": "passed", "message": "stable"},
            "count_dispersion": {"status": "failed", "dispersion": 1.5},
        },
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES + ["Count dispersion"]
    assert layers[-1].status == "failed"
    assert layers[-1].metrics["dispersion"] == pytest.approx(1.5)


def test_engine_result_to_layers_appends_unmapped_early_failure() -> None:
    result = {
        "passed": False,
        "failed_layer": "Configuration",
        "message": "max_dispersion must be non-negative.",
    }

    layers = engine_result_to_layers(result)

    assert [layer.layer for layer in layers] == LAYER_NAMES + ["Configuration"]
    assert all(layer.status == "skipped" for layer in layers[:4])
    assert layers[-1].status == "failed"
    assert layers[-1].reason == result["message"]


def test_engine_result_to_layers_requires_a_dict() -> None:
    with pytest.raises(TypeError):
        engine_result_to_layers(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Configuration gates
# ---------------------------------------------------------------------------


def test_validation_config_gate_defaults_and_engine_options() -> None:
    config = ValidationConfig()

    assert config.max_ece is None
    assert config.max_dispersion is None
    options = config.engine_options()
    assert options["max_ece"] is None
    assert options["max_dispersion"] is None
    assert options["max_p_value"] == config.max_p
    assert options["permutation_seed"] == config.seed
    assert config.as_dict()["max_ece"] is None
    assert config.as_dict()["max_dispersion"] is None

    configured = ValidationConfig(max_ece=1.0, max_dispersion=0.0)
    assert configured.engine_options()["max_ece"] == 1.0
    assert configured.engine_options()["max_dispersion"] == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_ece": 0.0},
        {"max_ece": -0.1},
        {"max_ece": 1.01},
        {"max_dispersion": -0.1},
        {"max_dispersion": float("nan")},
        {"max_dispersion": float("inf")},
    ],
)
def test_validation_config_rejects_invalid_gates(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ValidationConfig(**overrides)
