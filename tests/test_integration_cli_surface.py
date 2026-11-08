"""WI-7: the CLI command surface composes into one valid run.

One module-scoped CLI pipeline execution enables every optional stage at once
(``run --full --judge``), then a chain of downstream CLI commands
(``validate-models --generation``, ``judge``, ``validate-artifact``,
``preview``, ``schema``) reuses that run directory and must leave its artifact
contract intact.

A second, small CLI run proves the advisory final judge: a whole-judge failure
is recorded as a warning and never turns the CLI exit code into a failure.

Everything goes through ``whack26.cli.main`` with the cache, runs, and data
directories redirected under temporary roots. No subprocesses, network access,
live credentials, or UI imports are involved.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from integration_fixtures import (
    EXPECTED_FULL_FILES,
    load_artifact,
    make_dataset,
    read_json,
)
from whack26.artifact import artifact_json_schema
from whack26.cli import main

pytestmark = pytest.mark.integration

FULL_DATASET_ROWS = 80
FULL_DATASET_SEED = 0
DEGRADED_DATASET_ROWS = 60
DEGRADED_DATASET_SEED = 7

COLLECTION_DESIGN = "observational"
COLLECTION_UNIT_SENTINEL = "CLI-SENTINEL"

JUDGE_FILES = (
    "final_judge.json",
    "final_judge.md",
    "final_judge.html",
    "final_judge.fragment.html",
)


@dataclass(frozen=True)
class CliRun:
    """A captured ``main`` invocation and the run directory it produced."""

    run_dir: Path
    returncode: int
    stdout: str
    stderr: str


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    """Call ``main`` in-process and capture its output."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def _missing(run_dir: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (run_dir / name).is_file()]


@pytest.fixture(scope="module")
def cli_dirs(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Point every CLI settings directory under one temporary root.

    The patch stays active for the whole module so the downstream commands in
    the chain (which call ``Settings.from_env`` themselves) use the same
    isolated cache/runs/data directories as the initial run.
    """
    root = tmp_path_factory.mktemp("cli-surface")
    patch = pytest.MonkeyPatch()
    for variable, name in (
        ("WHACK26_CACHE_DIR", "cache"),
        ("WHACK26_RUNS_DIR", "runs"),
        ("WHACK26_DATA_DIR", "data"),
    ):
        patch.setenv(variable, str(root / name))
    try:
        yield root
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def full_cli_run(cli_dirs: Path) -> CliRun:
    """Execute the full CLI pipeline exactly once for this module."""
    dataset = make_dataset(
        cli_dirs / "full-input.csv",
        kind="continuous",
        rows=FULL_DATASET_ROWS,
        seed=FULL_DATASET_SEED,
    )
    run_dir = cli_dirs / "full-run"
    code, stdout, stderr = _invoke(
        [
            "run",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--out",
            str(run_dir),
            "--full",
            "--judge",
            "--mock",
            "--offline",
            "--validation-num-permutations",
            "10",
            "--validation-timeout",
            "60",
            "--collection-field",
            f"design={COLLECTION_DESIGN}",
            "--collection-field",
            f"unit_of_analysis={COLLECTION_UNIT_SENTINEL}",
        ]
    )
    return CliRun(
        run_dir=run_dir,
        returncode=code,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# 1. One CLI run composes every optional stage into a valid artifact
# ---------------------------------------------------------------------------


def test_cli_full_run_produces_one_valid_artifact(full_cli_run: CliRun) -> None:
    run_dir = full_cli_run.run_dir

    assert full_cli_run.returncode == 0, full_cli_run.stderr
    assert "Traceback" not in full_cli_run.stderr

    missing = _missing(run_dir, EXPECTED_FULL_FILES)
    assert not missing, f"the full CLI run is missing stage artifacts: {missing}"

    artifact = load_artifact(run_dir)
    assert artifact.model_search is not None
    assert artifact.variants is not None
    assert artifact.collection is not None
    assert artifact.collection.get("design") == COLLECTION_DESIGN
    assert (
        artifact.collection.get("unit_of_analysis")
        == COLLECTION_UNIT_SENTINEL
    )
    assert artifact.generation_validation_summary is not None
    assert artifact.field_pitfalls is not None
    assert artifact.validation_summary is not None
    assert artifact.final_judge is not None
    assert artifact.research.semantic_scraping is not None

    # The CLI's own report lines must name the pipeline outputs it produced.
    stdout = full_cli_run.stdout
    assert f"run: {run_dir}" in stdout
    assert f"artifact: {run_dir / 'artifact.json'}" in stdout
    assert f"preview: {run_dir / 'preview.md'}" in stdout
    assert stdout.count("\nvalidation: ") == len(artifact.hypotheses)
    assert "\ngeneration validation: " in stdout
    assert "\nmodel search: " in stdout
    assert "\nvariants: " in stdout
    assert "\nfield pitfalls: " in stdout
    assert "\nscrape: " in stdout
    assert "\nfinal judge: " in stdout


# ---------------------------------------------------------------------------
# 2. Downstream commands reuse the run without breaking its contract
# ---------------------------------------------------------------------------


def test_downstream_cli_commands_preserve_the_run_contract(
    full_cli_run: CliRun, tmp_path: Path
) -> None:
    run_dir = full_cli_run.run_dir
    artifact_path = run_dir / "artifact.json"

    code, stdout, stderr = _invoke(
        [
            "validate-models",
            str(run_dir),
            "--generation",
            "--num-permutations",
            "10",
            "--timeout",
            "60",
        ]
    )
    assert code == 0, stderr
    assert "validation json:" in stdout
    assert "generation validation:" in stdout
    assert (run_dir / "validation.json").is_file()
    artifact = load_artifact(run_dir)
    assert artifact.generation_validation_summary is not None

    code, stdout, stderr = _invoke(
        ["judge", str(run_dir), "--mock", "--offline"]
    )
    assert code == 0, stderr
    assert "final judge:" in stdout
    assert not _missing(run_dir, JUDGE_FILES), "judge command lost report files"
    load_artifact(run_dir)

    code, stdout, stderr = _invoke(
        ["validate-artifact", str(artifact_path)]
    )
    assert code == 0, stderr
    assert "OK" in stdout

    preview_path = run_dir / "preview.md"
    preview_path.unlink()
    assert not preview_path.exists()
    code, stdout, stderr = _invoke(["preview", str(artifact_path)])
    assert code == 0, stderr
    assert preview_path.is_file(), "preview was not regenerated"
    assert "wrote" in stdout

    schema_path = tmp_path / "schema.json"
    code, stdout, stderr = _invoke(["schema", "--out", str(schema_path)])
    assert code == 0, stderr
    generated = read_json(schema_path)
    expected = artifact_json_schema()
    assert generated == expected
    assert "final_judge" in expected["properties"]
    assert "FinalJudgeReport" in expected["$defs"]
    assert {
        "$ref": "#/$defs/FinalJudgeReport"
    } in expected["properties"]["final_judge"]["anyOf"]

    # Existing regression: the checked-in schema is the model's schema.
    checked_in = (
        Path(__file__).resolve().parent.parent
        / "schemas"
        / "artifact.schema.json"
    )
    assert read_json(checked_in) == expected


# ---------------------------------------------------------------------------
# 3. An advisory judge failure stays a warning, not a CLI failure
# ---------------------------------------------------------------------------


def test_cli_judge_failure_is_advisory(
    cli_dirs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        tmp_path / "degraded-input.csv",
        kind="continuous",
        rows=DEGRADED_DATASET_ROWS,
        seed=DEGRADED_DATASET_SEED,
    )
    run_dir = tmp_path / "degraded-run"
    attempts: list[dict[str, Any]] = []

    def _boom(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        raise RuntimeError("cli judge boom")

    # run.py imports the stage at call time, so the module attribute is the seam.
    monkeypatch.setattr(
        "whack26.final_judge.pipeline.run_final_judge", _boom
    )

    code, _stdout, stderr = _invoke(
        [
            "run",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--out",
            str(run_dir),
            "--judge",
            "--mock",
            "--offline",
        ]
    )

    assert code == 0, stderr
    assert attempts, "the CLI did not attempt the advisory judge stage"
    artifact = load_artifact(run_dir)
    assert artifact.final_judge is None
    assert any(
        warning.startswith("final_judge_failed:")
        for warning in artifact.warnings
    ), artifact.warnings

    log_text = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "final_judge_failed" in log_text
