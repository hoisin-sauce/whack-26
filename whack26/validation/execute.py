"""Parent-side orchestration for the isolated validation runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .models import (
    ValidationConfig,
    ValidationRequest,
    ValidationReport,
    error_report,
)

VALIDATION_RUNNER_PATH = Path(__file__).with_name("runner.py")
TIMEOUT_SECONDS = 120.0
MAX_STDOUT_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024

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


def _environment() -> dict[str, str]:
    return {
        key: value
        for key in _ENV_KEYS
        if (value := os.environ.get(key)) is not None
    }


def _truncate(text: str, limit: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="replace")


def run_validation(
    request: ValidationRequest,
    config: ValidationConfig,
    *,
    run_dir: Path,
    timeout: float | None = None,
) -> ValidationReport:
    """Execute one hypothesis in the isolated validation runner."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(config.timeout_seconds if timeout is None else timeout)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".validation-", dir=run_dir) as temp:
        temp_dir = Path(temp)
        config_path = temp_dir / "request.json"
        report_path = temp_dir / "report.json"
        payload = request.as_dict()
        payload["config"] = config.as_dict()
        config_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-I",
            str(VALIDATION_RUNNER_PATH),
            str(config_path),
            str(report_path),
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
            report = error_report(
                request.hypothesis_id,
                f"timeout after {timeout:g}s",
                runtime_ms=int(timeout * 1000),
            )
            if stdout or stderr:
                report.warnings.append(
                    "runner_output: "
                    + _truncate((stdout + "\n" + stderr).strip(), 2000)
                )
            return report
        except OSError as exc:
            return error_report(
                request.hypothesis_id,
                f"could not start the validation runner: "
                f"{type(exc).__name__}: {exc}",
                runtime_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # noqa: BLE001 - validation is advisory
            return error_report(
                request.hypothesis_id,
                f"validation runner failed: {type(exc).__name__}: {exc}",
                runtime_ms=int((time.perf_counter() - started) * 1000),
            )

        runtime_ms = int((time.perf_counter() - started) * 1000)
        stderr = _truncate(completed.stderr or "", MAX_STDERR_BYTES)
        if not report_path.is_file():
            message = (
                f"validation runner exited with code {completed.returncode} "
                "without a report"
            )
            report = error_report(
                request.hypothesis_id,
                message,
                runtime_ms=runtime_ms,
            )
            if stderr.strip():
                report.warnings.append(
                    "runner_stderr: " + stderr.strip()[-4000:]
                )
            return report
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = ValidationReport.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report = error_report(
                request.hypothesis_id,
                f"validation report is unreadable: {type(exc).__name__}: {exc}",
                runtime_ms=runtime_ms,
            )
            if stderr.strip():
                report.warnings.append(
                    "runner_stderr: " + stderr.strip()[-4000:]
                )
            return report
        if runtime_ms > 0:
            report.runtime_ms = runtime_ms
        if completed.returncode != 0 and report.status != "error":
            report.warnings.append(
                f"validation runner exited with code {completed.returncode}"
            )
        if stderr.strip() and "Traceback" in stderr:
            report.warnings.append("runner_stderr: " + stderr.strip()[-4000:])
        return report
