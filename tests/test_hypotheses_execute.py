from __future__ import annotations

from pathlib import Path

from whack26.hypotheses.execute import run_module

DATASET = "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n"

VALID_MODULE = '''import numpy as np
import pandas as pd


def applies(df: pd.DataFrame) -> bool:
    return {"x", "y"}.issubset(set(df.columns))


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    np.random.seed(seed)
    df = df.copy()
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {
        "predictions": design @ coefficients,
        "target": "y",
        "model_summary": {"coefficients": coefficients.tolist()},
        "notes": "ok",
    }
'''


def prepare(tmp_path: Path, code: str = VALID_MODULE) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    (run_dir / "hypotheses").mkdir(parents=True)
    module_path = run_dir / "hypotheses" / "hyp_test.py"
    module_path.write_text(code, encoding="utf-8")
    dataset_path = run_dir / "dataset.csv"
    dataset_path.write_text(DATASET, encoding="utf-8")
    return run_dir, module_path, dataset_path


def test_runner_executes_and_writes_predictions(tmp_path: Path) -> None:
    run_dir, module_path, dataset_path = prepare(tmp_path)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "ok", result.error
    assert result.applies is True
    assert result.n_predictions == 5
    assert result.target == "y"
    assert result.predictions_path == "predictions/hyp_test.csv"
    assert (run_dir / result.predictions_path).is_file()
    assert result.predictions_sha256


def test_runner_reports_applies_false(tmp_path: Path) -> None:
    code = VALID_MODULE.replace(
        'return {"x", "y"}.issubset(set(df.columns))', "return False"
    )
    run_dir, module_path, dataset_path = prepare(tmp_path, code)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "applies_false"
    assert result.applies is False


def test_runner_reports_runtime_error(tmp_path: Path) -> None:
    code = VALID_MODULE.replace(
        '    x = df["x"].to_numpy(dtype=float)',
        '    raise ValueError("deliberate failure")',
    )
    run_dir, module_path, dataset_path = prepare(tmp_path, code)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "runtime_error"
    assert "deliberate failure" in (result.error or "")
    assert result.traceback


def test_runner_reports_invalid_row_index_length(tmp_path: Path) -> None:
    code = VALID_MODULE.replace(
        '        "target": "y",',
        '        "target": "y",\n        "row_index": [1, 2],',
    )
    run_dir, module_path, dataset_path = prepare(tmp_path, code)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "invalid_output"
    assert "different lengths" in (result.error or "")


def test_runner_blocks_network_access(tmp_path: Path) -> None:
    code = VALID_MODULE.replace(
        '    np.random.seed(seed)',
        '    import socket\n    socket.create_connection(("example.test", 80))',
    )
    run_dir, module_path, dataset_path = prepare(tmp_path, code)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "runtime_error"
    assert "network access is blocked" in (result.error or "")


def test_runner_accepts_relative_paths(tmp_path: Path, monkeypatch) -> None:
    run_dir, module_path, dataset_path = prepare(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_module(
        Path("run") / "hypotheses" / "hyp_test.py",
        Path("run") / "dataset.csv",
        0,
        run_dir=Path("run"),
    )

    assert result.status == "ok", result.error
    assert result.n_predictions == 5
    assert (run_dir / result.predictions_path).is_file()
    assert not (run_dir / "run").exists()


def test_runner_keeps_generated_code_output_out_of_the_envelope(tmp_path: Path) -> None:
    code = VALID_MODULE.replace(
        '    np.random.seed(seed)', '    print("hello from generated code")\n    np.random.seed(seed)'
    )
    run_dir, module_path, dataset_path = prepare(tmp_path, code)

    result = run_module(module_path, dataset_path, 0, run_dir=run_dir)

    assert result.status == "ok"
    assert result.code_stdout is not None
    assert "hello from generated code" in result.code_stdout
    assert not (run_dir / "hypotheses" / "__pycache__").exists()
