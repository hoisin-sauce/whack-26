from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from whack26.validation.execute import run_validation
from whack26.validation.models import ValidationConfig, ValidationRequest

LINEAR_MODULE = '''import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    return {"x", "y"}.issubset(df.columns)


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    params = params or PARAMETERS
    ridge = float(params.get("ridge", 1e-3))
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
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
    return {"predictions": _predict(model, df.drop(columns=["y"])), "target": "y"}
'''

ZERO_PARAMETER_MODULE = LINEAR_MODULE.replace(
    'PARAMETERS = {"ridge": 1e-3}', 'PARAMETERS = {"ridge": 0.0}'
)

SLOW_MODULE = '''import time
import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    return {"x", "y"}.issubset(df.columns)


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    time.sleep(5.0)
    return {"coefficients": [0.0, 1.0]}


def _predict(model, df: pd.DataFrame):
    return [0.0] * len(df)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    return {"predictions": [0.0] * len(df), "target": "y"}
'''

NO_HELPERS_MODULE = '''import numpy as np
import pandas as pd


def applies(df: pd.DataFrame) -> bool:
    return {"x", "y"}.issubset(df.columns)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    x = df["x"].to_numpy(dtype=float)
    return {"predictions": x, "target": "y"}
'''


def make_dataset(path: Path, *, binary: bool = False) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=160)
    y = 2.0 * x + rng.normal(0.0, 0.15, size=160)
    if binary:
        y = (y > np.median(y)).astype(float)
    frame = pd.DataFrame({"x": x, "y": y})
    frame.to_csv(path, index=False)


def prepare(tmp_path: Path, code: str = LINEAR_MODULE, *, binary: bool = False):
    run_dir = tmp_path / "run"
    (run_dir / "hypotheses").mkdir(parents=True)
    (run_dir / "predictions").mkdir(parents=True)
    module_path = run_dir / "hypotheses" / "hyp_test.py"
    module_path.write_text(code, encoding="utf-8")
    dataset_path = run_dir / "dataset.csv"
    make_dataset(dataset_path, binary=binary)
    predictions_path = run_dir / "predictions" / "hyp_test.csv"
    if binary:
        prediction = np.zeros(160)
    else:
        frame = pd.read_csv(dataset_path)
        prediction = 2.0 * frame["x"].to_numpy(dtype=float)
    pd.DataFrame(
        {"row_index": range(160), "prediction": prediction}
    ).to_csv(predictions_path, index=False)
    return run_dir, module_path, dataset_path, predictions_path


def request(module_path: Path, dataset_path: Path, predictions_path: Path) -> ValidationRequest:
    return ValidationRequest(
        hypothesis_id="hyp_test",
        module_path=str(module_path),
        dataset_path=str(dataset_path),
        predictions_path=str(predictions_path),
        target="y",
        required_columns=["x"],
        optional_columns=[],
    )


def config(**overrides) -> ValidationConfig:
    values = {
        "min_dsr": 0.0,
        "max_p": 1.0,
        "train_split_ratio": 0.70,
        "num_permutations": 40,
        "sensitivity_range": 0.10,
        "max_performance_drop": 1.0,
        "max_decay_tolerance": 0.50,
        "timeout_seconds": 60.0,
    }
    values.update(overrides)
    return ValidationConfig(**values)


def test_full_four_layer_validation_passes(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path, predictions_path = prepare(tmp_path)

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(),
        run_dir=run_dir,
    )

    assert report.status == "passed", report
    assert report.has_fit_predict is True
    assert [layer.layer for layer in report.layers] == [
        "Layer 1 (DSR)",
        "Layer 2 (Monte Carlo)",
        "Layer 3 (OOS Decay)",
        "Layer 4 (Parameter Sensitivity)",
    ]
    assert all(layer.status == "passed" for layer in report.layers)


def test_zero_parameter_skips_layer_four(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path, predictions_path = prepare(
        tmp_path, ZERO_PARAMETER_MODULE
    )

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(),
        run_dir=run_dir,
    )

    layer4 = report.layer("Layer 4 (Parameter Sensitivity)")
    assert layer4 is not None
    assert layer4.status == "skipped"
    assert "non-degenerate" in (layer4.reason or "")
    assert report.status == "partial"


def test_binary_target_routes_through_general_engine(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path, predictions_path = prepare(
        tmp_path, binary=True
    )

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(),
        run_dir=run_dir,
    )

    # Layer 1 is DSR-only, so classification runs are partial by design.
    assert report.status == "partial"
    assert report.layer("Layer 1 (DSR)").status == "skipped"
    assert report.layer("Layer 2 (Monte Carlo)").status == "passed"
    assert report.layer("Layer 3 (OOS Decay)").status == "passed"
    assert report.layer("Layer 4 (Parameter Sensitivity)").status == "passed"


def test_applies_false_is_skipped(tmp_path: Path) -> None:
    code = LINEAR_MODULE.replace(
        'return {"x", "y"}.issubset(df.columns)', "return False"
    )
    run_dir, module_path, dataset_path, predictions_path = prepare(tmp_path, code)

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(),
        run_dir=run_dir,
    )

    assert report.status == "skipped"
    assert report.layers[0].layer == "Applicability"
    assert "applies" in (report.layers[0].reason or "")


def test_timeout_is_reported_as_error(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path, predictions_path = prepare(
        tmp_path, SLOW_MODULE
    )

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(timeout_seconds=0.2),
        run_dir=run_dir,
    )

    assert report.status == "error"
    assert "timeout" in (report.error or "").lower()


def test_missing_helpers_use_saved_predictions(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path, predictions_path = prepare(
        tmp_path, NO_HELPERS_MODULE
    )

    report = run_validation(
        request(module_path, dataset_path, predictions_path),
        config(),
        run_dir=run_dir,
    )

    assert report.in_sample_fallback is True
    assert report.has_fit_predict is False
    assert report.layer("Layer 3 (OOS Decay)").status == "skipped"
    assert report.layer("Layer 4 (Parameter Sensitivity)").status == "skipped"
    assert report.status in {"partial", "failed"}
