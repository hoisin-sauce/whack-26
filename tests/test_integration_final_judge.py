"""WI-5: producer evidence -> final judge -> report -> artifact/preview handoff.

One module-scoped real mock/offline pipeline run enables every producer the
final judge consumes (model search, variants, generation validation,
statistical validation, method critics, collection metadata) and proves the
judge's report is reproduced byte-for-byte by the artifact and preview.

A second, small pipeline run proves the judge is advisory: a whole-judge
failure is recorded as a warning and never blocks artifact emission.

No network access, CLI reuse path, or UI imports are involved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from integration_fixtures import (
    load_artifact,
    make_collection_metadata,
    make_dataset,
    make_settings,
    read_json,
    run_mock_pipeline,
)

from whack26.artifact import validate_artifact
from whack26.final_judge.prompts import CRITICS
from whack26.final_judge.report import render_fragment
from whack26.final_judge.types import FinalJudgeReport, JudgeFinding
from whack26.run import RunOutcome
from whack26.validation.models import ValidationConfig

pytestmark = pytest.mark.integration

DATASET_ROWS = 100
DATASET_SEED = 11
SENTINEL = "JUDGE-SENTINEL"
REPORT_FILES = (
    "final_judge.json",
    "final_judge.md",
    "final_judge.html",
    "final_judge.fragment.html",
)
SPECIALIST_NAMES = frozenset(spec.name for spec in CRITICS)
_UI_FORBIDDEN = re.compile(
    r"<\s*/?\s*(?:html|head|body|style|script|link|iframe|img)\b",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def judge_run(tmp_path_factory: pytest.TempPathFactory) -> RunOutcome:
    """Run every judge producer once and reuse the outcome across tests."""
    root = tmp_path_factory.mktemp("final-judge-integration")
    dataset = make_dataset(
        root / "input.csv",
        kind="continuous",
        rows=DATASET_ROWS,
        seed=DATASET_SEED,
    )
    outcome = run_mock_pipeline(
        root,
        dataset,
        out_name="judge-run",
        settings=make_settings(root),
        model_search=True,
        variants=True,
        validate_generation=True,
        validate=True,
        final_judge=True,
        collection=make_collection_metadata(unit_of_analysis=SENTINEL),
        validation_config=ValidationConfig(
            num_permutations=10,
            timeout_seconds=60.0,
            min_dsr=0.0,
            max_p=1.0,
            max_performance_drop=1.0,
        ),
    )
    # Honest postconditions for the tests below.
    assert outcome.artifact.dataset.rows == DATASET_ROWS
    assert outcome.artifact.run.mode == "mock"
    assert outcome.final_judge is not None
    return outcome


def _report_from_disk(run_dir: Path) -> FinalJudgeReport:
    return FinalJudgeReport.model_validate_json(
        (run_dir / "final_judge.json").read_text(encoding="utf-8")
    )


def _evidence_context(report: FinalJudgeReport) -> dict[str, Any]:
    context = json.loads(report.model_description)
    assert isinstance(context, dict), "model_description is not a JSON object"
    return context


# ---------------------------------------------------------------------------
# 1. The judge report consumes every producer and never fakes a pass
# ---------------------------------------------------------------------------


def test_judge_report_uses_every_producer(judge_run: RunOutcome) -> None:
    run_dir = judge_run.run_dir
    artifact = load_artifact(run_dir)
    raw_text = (run_dir / "final_judge.json").read_text(encoding="utf-8")
    report = FinalJudgeReport.model_validate_json(raw_text)

    # The file is exactly the typed report, not a partial sidecar.
    assert report.model_dump(mode="json") == read_json(run_dir / "final_judge.json")

    # The selected model family travels from the model-search artifact section.
    assert artifact.model_search is not None
    assert artifact.model_search.selected_family
    assert (
        report.model_summary.get("selected_family")
        == artifact.model_search.selected_family
    )
    assert (
        report.model_summary.get("candidate_count")
        == artifact.model_search.candidate_count
    )

    # Critic accounting is internally consistent and uses the real roster.
    assert report.n_critics_evaluated == len(report.critics_run)
    assert report.min_required > 0
    assert set(report.critics_run) <= SPECIALIST_NAMES
    assert "devils_advocate" not in report.critics_run
    assert set(report.unevaluated_critics).isdisjoint(report.critics_run)
    assert set(report.unevaluated_critics) <= SPECIALIST_NAMES | {
        "devils_advocate"
    }

    # Mock mode must report missing evidence, never a fabricated clean pass.
    assert report.decision == "insufficient_evidence"
    assert report.note is not None
    assert "not a pass" in report.note.lower()

    # Every producer's evidence is represented in the report's run context.
    context = _evidence_context(report)
    assert (
        context["selected_model"]["family"]
        == artifact.model_search.selected_family
    )
    assert context["validation"], (
        "statistical validation evidence is absent from the judge report"
    )
    assert context["method_critic"], (
        "method-critic evidence is absent from the judge report"
    )
    assert context["generation_validation"], (
        "generation-validation evidence is absent from the judge report"
    )
    assert {
        item["hypothesis_id"] for item in context["generation_validation"]
    } == {hypothesis.id for hypothesis in artifact.hypotheses}
    variants_context = context["variants"]
    assert variants_context, (
        "variants evidence is absent from the judge report"
    )
    assert set(variants_context["variants"]) == {
        "original",
        "anonymized",
        "jumbled",
    }
    assert context["collection"]["unit_of_analysis"] == SENTINEL


# ---------------------------------------------------------------------------
# 2. JSON, artifact, preview, and sidecar files agree
# ---------------------------------------------------------------------------


def test_judge_files_artifact_and_preview_agree(judge_run: RunOutcome) -> None:
    run_dir = judge_run.run_dir

    missing = [name for name in REPORT_FILES if not (run_dir / name).is_file()]
    assert not missing, f"the judge did not write every report file: {missing}"

    artifact, errors = validate_artifact(judge_run.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.final_judge is not None

    payload = read_json(run_dir / "final_judge.json")
    assert artifact.final_judge.model_dump(mode="json") == payload
    assert FinalJudgeReport.model_validate(payload) == artifact.final_judge

    for attribute in ("report_path", "fragment_path"):
        value = getattr(artifact.final_judge, attribute)
        assert value is not None, f"artifact.final_judge.{attribute} is unset"
        relative = Path(value)
        assert not relative.is_absolute(), f"{attribute} is absolute: {value}"
        assert relative.name == value, (
            f"{attribute} is not a bare relative filename: {value}"
        )
        assert ".." not in relative.parts, (
            f"{attribute} escapes the run directory: {value}"
        )
        assert (run_dir / relative).is_file(), (
            f"{attribute} points at a missing file: {value}"
        )

    preview = judge_run.preview_path.read_text(encoding="utf-8")
    assert "## Final model judgement" in preview
    assert "Insufficient evidence" in preview


# ---------------------------------------------------------------------------
# 3. The UI fragment is safe, contract-bearing, and deterministic
# ---------------------------------------------------------------------------


def test_fragment_is_ui_safe_and_carries_the_contract(
    judge_run: RunOutcome,
) -> None:
    run_dir = judge_run.run_dir
    report = _report_from_disk(run_dir)
    fragment = (run_dir / "final_judge.fragment.html").read_text(encoding="utf-8")

    assert _UI_FORBIDDEN.search(fragment) is None, (
        "the UI fragment contains a document-level or resource-loading tag"
    )
    assert "javascript:" not in fragment.lower()
    assert "Final model judgement" in fragment
    assert "Insufficient evidence" in fragment
    expected_counts = f"{report.n_critics_evaluated}/{report.min_required}"
    assert expected_counts in fragment

    # Model-generated text is escaped, not injected as markup.
    tainted = report.model_copy(deep=True)
    tainted.concerns = [
        JudgeFinding(
            critic="overfitting_skeptic",
            severity="medium",
            finding="<script>alert('judge')</script>",
            evidence="model text",
            confidence="medium",
        )
    ]
    escaped = render_fragment(tainted)
    assert "<script" not in escaped.lower()
    assert "&lt;script&gt;" in escaped

    # The on-disk fragment is exactly what the persisted report renders, and
    # rendering is byte-identical on repetition.
    first = render_fragment(report).encode("utf-8")
    second = render_fragment(report).encode("utf-8")
    assert first == second
    assert first == fragment.encode("utf-8")


# ---------------------------------------------------------------------------
# 4. A whole-judge failure is advisory and does not block the artifact
# ---------------------------------------------------------------------------


def test_judge_failure_is_advisory_and_does_not_block_the_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = make_dataset(
        tmp_path / "input.csv",
        kind="continuous",
        rows=60,
        seed=7,
    )

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("judge boom")

    # run.py imports run_final_judge at call time, so patching the module
    # attribute is the real failure seam.
    monkeypatch.setattr(
        "whack26.final_judge.pipeline.run_final_judge", _boom
    )

    outcome = run_mock_pipeline(
        tmp_path,
        dataset,
        out_name="judge-failure",
        final_judge=True,
    )

    assert outcome.final_judge is None

    artifact, errors = validate_artifact(outcome.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.final_judge is None

    assert any(
        warning.startswith("final_judge_failed:")
        for warning in artifact.warnings
    ), artifact.warnings

    # The failed stage wrote nothing; no final_judge.* file is required.
    run_dir = outcome.run_dir
    written = [name for name in REPORT_FILES if (run_dir / name).exists()]
    assert not written, f"a failed judge still wrote report files: {written}"

    log_text = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "final_judge_failed" in log_text
    assert "judge boom" in log_text
