"""Parent-side orchestration for the isolated hypothesis runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .runner import _json_safe

RUNNER_PATH = Path(__file__).with_name("runner.py")
TIMEOUT_SECONDS = 60.0
MAX_STDOUT_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024

ExecutionStatus = Literal[
    "ok",
    "applies_false",
    "runtime_error",
    "timeout",
    "invalid_output",
    "runner_error",
]

_ENV_KEYS = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
)


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    applies: bool | None = None
    runtime_ms: int = 0
    n_predictions: int | None = None
    target: str | None = None
    model_summary: dict[str, Any] | None = None
    notes: str | None = None
    predictions_path: str | None = None
    predictions_sha256: str | None = None
    error: str | None = None
    traceback: str | None = None
    code_stdout: str | None = None
    code_stderr: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_artifact(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "runtime_ms": self.runtime_ms,
            "n_predictions": self.n_predictions,
            "predictions_path": self.predictions_path,
            "error": self.error,
        }


def _environment() -> dict[str, str]:
    return {
        key: value
        for key in _ENV_KEYS
        if (value := os.environ.get(key)) is not None
    }


def _last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _classify_failure(error: str | None) -> ExecutionStatus:
    text = (error or "").lower()
    shape_markers = (
        "predictions and row_index have different lengths",
        "missing the required `predictions` key",
        "`predictions` must be",
        "`row_index` must be",
        "`target` must be",
        "`model_summary` must be",
        "`notes` must be",
        "run() must return a dict",
    )
    if any(marker in text for marker in shape_markers):
        return "invalid_output"
    return "runtime_error"


def _verify_predictions(path: Path, expected: int | None) -> tuple[int, str | None]:
    if not path.is_file():
        return 0, f"predictions file was not written: {path.name}"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return 0, f"predictions file is unreadable: {type(exc).__name__}: {exc}"
    if list(frame.columns) != ["row_index", "prediction"]:
        return 0, (
            "predictions file must have columns row_index,prediction; got "
            + ",".join(str(column) for column in frame.columns)
        )
    if expected is not None and len(frame) != expected:
        return len(frame), (
            f"predictions file has {len(frame)} rows; expected {expected}"
        )
    return len(frame), None


def run_module(
    module_path: Path,
    dataset_path: Path,
    seed: int,
    *,
    run_dir: Path,
    timeout: float = TIMEOUT_SECONDS,
) -> ExecutionResult:
    """Execute one module in a subprocess and validate its envelope."""
    module_path = Path(module_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    run_dir = Path(run_dir).resolve()
    started = time.perf_counter()
    command = [
        sys.executable,
        "-I",
        str(RUNNER_PATH),
        str(module_path),
        str(dataset_path),
        str(seed),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(run_dir),
            env=_environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ExecutionResult(
            status="timeout",
            runtime_ms=int(timeout * 1000),
            error=f"timeout after {timeout:g}s",
            code_stdout=stdout[-2000:] or None,
            code_stderr=stderr[-2000:] or None,
        )
    except OSError as exc:
        return ExecutionResult(
            status="runner_error",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            error=f"could not start the runner: {type(exc).__name__}: {exc}",
        )

    runtime_ms = int((time.perf_counter() - started) * 1000)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if len(stdout.encode("utf-8", errors="replace")) > MAX_STDOUT_BYTES:
        stdout = stdout.encode("utf-8", errors="replace")[:MAX_STDOUT_BYTES].decode(
            "utf-8", errors="replace"
        )
    if len(stderr.encode("utf-8", errors="replace")) > MAX_STDERR_BYTES:
        stderr = stderr.encode("utf-8", errors="replace")[:MAX_STDERR_BYTES].decode(
            "utf-8", errors="replace"
        )
    envelope = _last_json_line(stdout)
    if envelope is None:
        return ExecutionResult(
            status="runner_error",
            runtime_ms=runtime_ms,
            error=(
                f"runner exited with code {completed.returncode} without a JSON "
                "envelope"
            ),
            code_stderr=stderr[-4000:] or None,
        )

    if envelope.get("applies") is False:
        return ExecutionResult(
            status="applies_false",
            applies=False,
            runtime_ms=runtime_ms,
            code_stdout=envelope.get("code_stdout"),
            code_stderr=envelope.get("code_stderr"),
        )

    if completed.returncode != 0 or envelope.get("error"):
        error = str(envelope.get("error") or f"runner exited with code {completed.returncode}")
        return ExecutionResult(
            status=_classify_failure(error),
            applies=bool(envelope.get("applies")) if envelope.get("applies") is not None else None,
            runtime_ms=runtime_ms,
            error=error,
            traceback=envelope.get("traceback"),
            code_stdout=envelope.get("code_stdout"),
            code_stderr=envelope.get("code_stderr"),
        )

    n_predictions = envelope.get("n_predictions")
    if not isinstance(n_predictions, int) or n_predictions <= 0:
        return ExecutionResult(
            status="invalid_output",
            applies=True,
            runtime_ms=runtime_ms,
            error=f"runner reported an invalid prediction count: {n_predictions!r}",
            code_stdout=envelope.get("code_stdout"),
            code_stderr=envelope.get("code_stderr"),
        )

    relative_path = envelope.get("predictions_path")
    predictions_path = (
        Path(relative_path) if isinstance(relative_path, str) else None
    )
    if predictions_path is None:
        return ExecutionResult(
            status="invalid_output",
            applies=True,
            runtime_ms=runtime_ms,
            n_predictions=n_predictions,
            error="runner did not report a predictions path",
        )
    actual_count, problem = _verify_predictions(run_dir / predictions_path, n_predictions)
    if problem is not None:
        return ExecutionResult(
            status="invalid_output",
            applies=True,
            runtime_ms=runtime_ms,
            n_predictions=actual_count,
            error=problem,
        )

    return ExecutionResult(
        status="ok",
        applies=True,
        runtime_ms=runtime_ms,
        n_predictions=n_predictions,
        target=envelope.get("target"),
        model_summary=_json_safe(envelope.get("model_summary")),
        notes=envelope.get("notes"),
        predictions_path=str(predictions_path).replace("\\", "/"),
        predictions_sha256=envelope.get("predictions_sha256"),
        code_stdout=envelope.get("code_stdout"),
        code_stderr=envelope.get("code_stderr"),
    )
