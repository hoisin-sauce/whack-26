"""Isolated child-process runner for one generated hypothesis.

Invocation:

    python -I runner.py <module.py> <dataset.csv> <seed>

The runner writes ``predictions/<module stem>.csv`` beside the module's run
directory and prints one JSON envelope to real stdout. Generated code's own
stdout and stderr are captured and never mixed with that envelope.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Keep run directories clean: importing generated modules must not leave
# __pycache__ directories inside the handoff artifact.
sys.dont_write_bytecode = True


def _block_network() -> None:
    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is blocked in the hypothesis runner")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _predictions_from(result: Any) -> tuple[list[Any], Any]:
    if not isinstance(result, dict):
        raise TypeError(
            f"run() must return a dict, got {type(result).__name__}"
        )
    if "predictions" not in result:
        raise ValueError("run() result is missing the required `predictions` key")
    predictions = result["predictions"]
    if isinstance(predictions, pd.Series):
        return predictions.tolist(), predictions.index.tolist()
    if isinstance(predictions, np.ndarray):
        return predictions.tolist(), None
    if isinstance(predictions, (list, tuple)):
        return list(predictions), None
    raise TypeError(
        "`predictions` must be a list, pandas Series, or numpy array; "
        f"got {type(predictions).__name__}"
    )


def _write_predictions(
    *,
    module_path: Path,
    predictions: list[Any],
    row_index: list[Any],
) -> tuple[Path, str]:
    run_dir = module_path.resolve().parent.parent
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    destination = predictions_dir / f"{module_path.stem}.csv"
    frame = pd.DataFrame(
        {
            "row_index": row_index,
            "prediction": predictions,
        }
    )
    frame.to_csv(destination, index=False, lineterminator="\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination, digest


def _main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: runner.py <module.py> <dataset.csv> <seed>",
            file=sys.stderr,
        )
        return 2
    module_path = Path(argv[1]).resolve()
    dataset_path = Path(argv[2]).resolve()
    try:
        seed = int(argv[3])
    except ValueError:
        print(f"seed must be an integer, got {argv[3]!r}", file=sys.stderr)
        return 2

    _block_network()

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    envelope: dict[str, Any] = {"applies": None}
    try:
        frame = pd.read_csv(dataset_path)
        module_name = f"_whack26_hypothesis_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load a module from {module_path.name!r}")
        module = importlib.util.module_from_spec(spec)
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            spec.loader.exec_module(module)
            applies = getattr(module, "applies", None)
            run = getattr(module, "run", None)
            if not callable(applies) or not callable(run):
                raise AttributeError(
                    "the module must define callable `applies(df)` and `run(df, seed=0)`"
                )
            applicable = bool(applies(frame))
            if not applicable:
                envelope = {"applies": False}
                return 0
            result = run(frame.copy(), seed)
        predictions, series_index = _predictions_from(result)
        row_index = result.get("row_index")
        if row_index is None:
            row_index = series_index if series_index is not None else frame.index.tolist()
        elif isinstance(row_index, pd.Series):
            row_index = row_index.tolist()
        elif isinstance(row_index, np.ndarray):
            row_index = row_index.tolist()
        elif not isinstance(row_index, (list, tuple)):
            raise TypeError(
                "`row_index` must be a list, pandas Series, or numpy array"
            )
        row_index = list(row_index)
        if len(row_index) != len(predictions):
            raise ValueError(
                "predictions and row_index have different lengths: "
                f"{len(predictions)} != {len(row_index)}"
            )
        if series_index is not None and result.get("row_index") is not None:
            # An explicit row_index wins; the Series index is only a fallback.
            pass
        target = result.get("target")
        if target is not None and not isinstance(target, str):
            raise TypeError("`target` must be a string or None")
        model_summary = result.get("model_summary")
        if model_summary is not None and not isinstance(model_summary, dict):
            raise TypeError("`model_summary` must be a dict when present")
        notes = result.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise TypeError("`notes` must be a string when present")

        destination, digest = _write_predictions(
            module_path=module_path,
            predictions=predictions,
            row_index=row_index,
        )
        envelope = {
            "applies": True,
            "n_predictions": len(predictions),
            "target": target,
            "model_summary": _json_safe(model_summary),
            "notes": notes,
            "row_index_provided": result.get("row_index") is not None
            or series_index is not None,
            "predictions_path": str(
                destination.relative_to(module_path.resolve().parent.parent)
            ).replace("\\", "/"),
            "predictions_sha256": digest,
        }
        return 0
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        traceback.print_exc(file=sys.stderr)
        envelope = {
            "applies": envelope.get("applies"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
        return 1
    finally:
        code_stdout = captured_stdout.getvalue()[-2000:]
        code_stderr = captured_stderr.getvalue()[-2000:]
        if code_stdout:
            envelope["code_stdout"] = code_stdout
        if code_stderr:
            envelope["code_stderr"] = code_stderr
        sys.__stdout__.write(json.dumps(envelope, allow_nan=False) + "\n")
        sys.__stdout__.flush()


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
