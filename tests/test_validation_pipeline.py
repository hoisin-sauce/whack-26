from __future__ import annotations

from pathlib import Path

import pytest

from artifact_fixtures import make_artifact

from whack26.artifact import validate_artifact, write_artifact
from whack26.cli import main
from whack26.validation.models import ValidationConfig
from whack26.validation.pipeline import validate_run


def prepare(tmp_path: Path):
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    return artifact, run_dir, artifact_path


def test_validate_run_writes_report_without_mutating_artifact(
    tmp_path: Path,
) -> None:
    artifact, run_dir, artifact_path = prepare(tmp_path)

    result = validate_run(
        run_dir,
        config=ValidationConfig(num_permutations=20, timeout_seconds=60.0),
    )

    assert result.report_path.is_file()
    assert len(result.reports) == 3
    assert all(report.status == "skipped" for report in result.reports)
    # The artifact is only rewritten on request.
    loaded, errors = validate_artifact(artifact_path)
    assert errors == []
    assert loaded is not None
    assert loaded.validation_summary is None


def test_validate_run_write_artifact_embeds_reports(tmp_path: Path) -> None:
    artifact, run_dir, artifact_path = prepare(tmp_path)

    validate_run(
        artifact_path,
        config=ValidationConfig(num_permutations=20, timeout_seconds=60.0),
        write_artifact=True,
    )

    loaded, errors = validate_artifact(artifact_path)
    assert errors == []
    assert loaded is not None
    assert loaded.validation_summary is not None
    assert sum(loaded.validation_summary.counts.values()) == 3
    assert all(item.validation is not None for item in loaded.hypotheses)
    preview = (run_dir / "preview.md").read_text(encoding="utf-8")
    assert "## Validation summary" in preview


def test_validate_run_rejects_invalid_artifact(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact is not valid"):
        validate_run(path)


def test_cli_validate_models(tmp_path: Path, capsys) -> None:
    artifact, run_dir, _ = prepare(tmp_path)

    code = main(
        [
            "validate-models",
            str(run_dir),
            "--num-permutations",
            "20",
            "--timeout",
            "60",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "run:" in captured.out
    assert "validation json:" in captured.out
    assert "summary:" in captured.out
    assert (run_dir / "validation.json").is_file()
