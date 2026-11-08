"""Run-level statistical validation over a generated artifact."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..artifact import (
    Artifact,
    HypothesisSection,
    ValidationSummary,
    validate_artifact,
    write_artifact as write_artifact_file,
)
from ..common import utc_now_iso
from ..preview import write_preview
from .execute import run_validation
from .models import (
    ValidationConfig,
    ValidationReport,
    ValidationRequest,
    error_report,
    skipped_report,
    summary_counts,
)


@dataclass
class ValidationRunResult:
    run_dir: Path
    artifact_path: Path
    artifact: Artifact
    reports: list[ValidationReport]
    report_path: Path


def _resolve_artifact(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "artifact.json"
        if not candidate.is_file():
            raise ValueError(f"no artifact.json found in {path}")
        return candidate
    if not path.is_file():
        raise ValueError(f"artifact not found: {path}")
    return path


def _skip_for_execution(
    hypothesis: HypothesisSection,
    reason: str,
) -> ValidationReport:
    return skipped_report(
        hypothesis.id,
        reason,
        target=hypothesis.target,
    )


def validate_hypothesis(
    hypothesis: HypothesisSection,
    *,
    run_dir: Path,
    artifact: Artifact,
    config: ValidationConfig,
) -> ValidationReport:
    """Validate one artifact hypothesis without raising on model failures."""
    run_dir = Path(run_dir).resolve()
    if hypothesis.execution.status != "ok":
        return _skip_for_execution(
            hypothesis,
            f"execution_status: {hypothesis.execution.status}",
        )
    if not hypothesis.execution.predictions_path:
        return _skip_for_execution(
            hypothesis,
            "no_predictions: the hypothesis has no saved predictions",
        )
    module_path = run_dir / hypothesis.code_path
    dataset_path = run_dir / artifact.dataset.stored_path
    predictions_path = run_dir / hypothesis.execution.predictions_path
    if not module_path.is_file():
        return error_report(
            hypothesis.id,
            f"hypothesis code is missing: {hypothesis.code_path}",
        )
    if not dataset_path.is_file():
        return error_report(
            hypothesis.id,
            f"stored dataset is missing: {artifact.dataset.stored_path}",
        )
    if not predictions_path.is_file():
        return error_report(
            hypothesis.id,
            "predictions are missing: "
            f"{hypothesis.execution.predictions_path}",
        )
    target_kind = None
    screen_target = artifact.screen.target_summary or {}
    if screen_target.get("name") == hypothesis.target:
        target_kind = screen_target.get("kind")
    request = ValidationRequest(
        hypothesis_id=hypothesis.id,
        module_path=str(module_path),
        dataset_path=str(dataset_path),
        predictions_path=str(predictions_path),
        target=hypothesis.target,
        required_columns=list(hypothesis.required_columns),
        optional_columns=list(hypothesis.optional_columns),
        time_column=artifact.dataset.time_column,
        target_kind=str(target_kind) if target_kind else None,
        description=artifact.dataset.description,
    )
    return run_validation(request, config, run_dir=run_dir)


def validate_run(
    path: Path,
    *,
    config: ValidationConfig | None = None,
    out: Path | None = None,
    write_artifact: bool = False,
) -> ValidationRunResult:
    """Validate every hypothesis in a run directory or artifact."""
    config = config or ValidationConfig()
    artifact_path = _resolve_artifact(Path(path)).resolve()
    run_dir = artifact_path.parent
    artifact, errors = validate_artifact(artifact_path)
    if artifact is None or errors:
        raise ValueError(
            "artifact is not valid: " + "; ".join(errors[:5] or ["unknown error"])
        )

    reports: list[ValidationReport] = []
    for hypothesis in artifact.hypotheses:
        try:
            reports.append(
                validate_hypothesis(
                    hypothesis,
                    run_dir=run_dir,
                    artifact=artifact,
                    config=config,
                )
            )
        except Exception as exc:  # noqa: BLE001 - validation is advisory
            reports.append(
                error_report(
                    hypothesis.id,
                    f"validation failed: {type(exc).__name__}: {exc}",
                )
            )

    report_path = Path(out) if out is not None else run_dir / "validation.json"
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": artifact.run.run_id,
        "generated_at": utc_now_iso(),
        "config": config.as_dict(),
        "summary": summary_counts(reports),
        "reports": [report.model_dump(mode="json") for report in reports],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    if write_artifact:
        by_id = {report.hypothesis_id: report for report in reports}
        for hypothesis in artifact.hypotheses:
            hypothesis.validation = by_id.get(hypothesis.id)
        artifact.validation_summary = ValidationSummary(counts=summary_counts(reports))
        artifact.run.finished_at = utc_now_iso()
        write_artifact_file(artifact_path, artifact)
        write_preview(
            artifact_path.with_name("preview.md"),
            artifact,
            base_dir=run_dir,
        )

    return ValidationRunResult(
        run_dir=run_dir,
        artifact_path=artifact_path,
        artifact=artifact,
        reports=reports,
        report_path=report_path,
    )
