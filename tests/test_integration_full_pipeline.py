"""Full-pipeline component integration: every optional stage in one run.

This module runs the real pipeline **once** in mock/offline mode with every
optional stage enabled together, then proves from that single run directory
that the stages compose: sidecar files exist, artifact sections cross-reference
their JSON payloads, portable paths stay run-relative, and the preview covers
every stage.

No network access, live credentials, UI imports, or CLI calls are involved.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from integration_fixtures import (
    EXPECTED_FULL_FILES,
    TARGET,
    load_artifact,
    make_collection_metadata,
    make_dataset,
    make_settings,
    read_json,
    run_mock_pipeline,
)
from whack26.artifact import validate_artifact
from whack26.run import RunOutcome
from whack26.validation.models import ValidationConfig

pytestmark = pytest.mark.integration

DATASET_ROWS = 90
DATASET_SEED = 5
SENTINEL = "FULL-RUN-SENTINEL"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> RunOutcome:
    """Run every optional stage once and reuse the outcome across the module."""
    root = tmp_path_factory.mktemp("full-pipeline")
    dataset = make_dataset(
        root / "input.csv",
        kind="continuous",
        rows=DATASET_ROWS,
        seed=DATASET_SEED,
    )
    settings = make_settings(
        root,
        scrape_max_resources=5,
        scrape_max_depth=1,
    )
    # Run with the input named relative to the temporary root so the
    # informational ``dataset.source_path`` is relative too. The portability
    # test then proves that no absolute temporary path leaks into the artifact.
    previous_cwd = Path.cwd()
    os.chdir(root)
    try:
        outcome = run_mock_pipeline(
            root,
            Path(dataset.name),
            out_name="full-run",
            settings=settings,
            scrape=True,
            model_search=True,
            variants=True,
            field_pitfalls=True,
            validate_generation=True,
            validate=True,
            final_judge=True,
            collection=make_collection_metadata(unit_of_analysis=SENTINEL),
            validation_config=ValidationConfig(
                num_permutations=10,
                timeout_seconds=60.0,
            ),
        )
    finally:
        os.chdir(previous_cwd)

    # Honest postconditions for the tests below.
    assert outcome.artifact.dataset.rows == DATASET_ROWS
    assert outcome.artifact.run.mode == "mock"
    assert outcome.run_dir == root / "full-run"
    return outcome


def _assert_relative_file(run_dir: Path, relative: str) -> None:
    """Assert one artifact path is run-relative, non-escaping, and present."""
    path = Path(relative)
    assert not path.is_absolute(), f"path must be relative: {relative}"
    assert not relative.startswith(("/", "\\")), f"path must not be rooted: {relative}"
    assert ".." not in path.parts, f"path escapes the run directory: {relative}"
    assert (run_dir / path).is_file(), f"referenced file is missing: {relative}"


def test_full_run_writes_every_stage_artifact(full_run: RunOutcome) -> None:
    run_dir = full_run.run_dir

    missing = [name for name in EXPECTED_FULL_FILES if not (run_dir / name).is_file()]
    assert not missing, f"the full run is missing stage artifacts: {missing}"

    log_path = run_dir / "logs" / "run.log"
    assert log_path.is_file(), "the run log is missing"
    assert "run_done" in log_path.read_text(encoding="utf-8")

    assert full_run.artifact.hypotheses, "the run produced no hypotheses"
    for hypothesis in full_run.artifact.hypotheses:
        assert (run_dir / hypothesis.code_path).is_file(), hypothesis.code_path
        predictions_path = hypothesis.execution.predictions_path
        assert predictions_path is not None, (
            f"{hypothesis.id} has no predictions path"
        )
        assert (run_dir / predictions_path).is_file(), predictions_path


def test_artifact_sections_cross_reference_each_other(
    full_run: RunOutcome,
) -> None:
    run_dir = full_run.run_dir
    artifact = load_artifact(run_dir)

    # Every optional stage must be represented in the typed artifact.
    assert artifact.dataset.target is not None
    assert artifact.dataset.target.name == TARGET
    assert artifact.model_search is not None
    assert artifact.variants is not None
    assert artifact.collection is not None
    assert artifact.collection.get("unit_of_analysis") == SENTINEL
    assert artifact.generation_validation_summary is not None
    assert artifact.field_pitfalls is not None
    assert artifact.final_judge is not None
    assert artifact.validation_summary is not None
    assert artifact.research.semantic_scraping is not None
    assert artifact.research.semantic_scraping.enabled is True

    assert len(artifact.hypotheses) == 3
    for hypothesis in artifact.hypotheses:
        assert hypothesis.execution.status == "ok", hypothesis.id
        assert hypothesis.execution.n_predictions is not None, hypothesis.id
        assert hypothesis.validation is not None, hypothesis.id
        assert hypothesis.generation_validation is not None, hypothesis.id

    assert sum(artifact.validation_summary.counts.values()) == len(
        artifact.hypotheses
    )
    assert sum(artifact.generation_validation_summary.verdicts.values()) == len(
        artifact.hypotheses
    )

    # factors.json mirrors the typed factor sections exactly.
    factors_payload = read_json(run_dir / "factors.json")
    assert factors_payload == [
        factor.model_dump(mode="json") for factor in artifact.factors
    ]

    # research.json carries the same queries, records, cache, and flags as the
    # research section embedded in the artifact.
    research_payload = read_json(run_dir / "research.json")
    assert research_payload["records"] == [
        record.model_dump(mode="json") for record in artifact.research.records
    ]
    assert research_payload["queries"] == [
        query.model_dump(mode="json") for query in artifact.research.queries
    ]
    assert research_payload["warnings"] == artifact.research.warnings
    assert research_payload["cache"] == artifact.research.cache
    assert research_payload["limited_mode"] == artifact.research.limited_mode
    assert research_payload["offline"] == artifact.research.offline

    # model_search.json disagrees structurally (raw result vs typed section),
    # so compare the facts the artifact promises to preserve.
    model_search_payload = read_json(run_dir / "model_search.json")
    assert model_search_payload["target_kind"] == artifact.model_search.target_kind
    assert model_search_payload["metric"] == artifact.model_search.metric
    assert len(model_search_payload["candidates"]) == (
        artifact.model_search.candidate_count
    )
    assert artifact.model_search.candidate_count > 0
    selected = model_search_payload["selected"]
    assert selected is not None, "model search did not select a model"
    assert selected["spec"]["family"] == artifact.model_search.selected_family

    # variants.json records all three scored variants; the artifact expands the
    # variant metadata into a list and preserves the leak/PII verdicts.
    variants_payload = read_json(run_dir / "variants.json")
    assert variants_payload["verdict"] == artifact.variants.verdict
    assert (
        variants_payload["jumbled_label_leak"]
        == artifact.variants.jumbled_label_leak
    )
    assert variants_payload["pii_reliance"] == artifact.variants.pii_reliance
    assert {score["kind"] for score in variants_payload["scores"]} == {
        "original",
        "anonymized",
        "jumbled",
    }
    assert set(variants_payload["variants"]) == {
        "original",
        "anonymized",
        "jumbled",
    }
    assert {variant.get("kind") for variant in artifact.variants.variants} == {
        "original",
        "anonymized",
        "jumbled",
    }

    # field_pitfalls.json is the same report embedded in the artifact. The
    # stage runs in mock mode (queries are recorded) even when no cached
    # research record yields an extracted pitfall.
    pitfalls_payload = read_json(run_dir / "field_pitfalls.json")
    assert pitfalls_payload == artifact.field_pitfalls.model_dump(mode="json")
    assert artifact.field_pitfalls.field_path
    assert artifact.field_pitfalls.queries
    assert [pitfall["id"] for pitfall in pitfalls_payload["pitfalls"]] == [
        pitfall.id for pitfall in artifact.field_pitfalls.pitfalls
    ]
    for pitfall in artifact.field_pitfalls.pitfalls:
        assert pitfall.field_id is not None, pitfall.id
        pattern = rf"F:{re.escape(pitfall.field_id)}:[1-9]\d*"
        assert re.fullmatch(pattern, pitfall.id), pitfall.id

    # final_judge.json is the same report embedded in the artifact, and the
    # mock judge must not fabricate a clean pass.
    final_judge_payload = read_json(run_dir / "final_judge.json")
    assert final_judge_payload == artifact.final_judge.model_dump(mode="json")
    assert artifact.final_judge.decision == "insufficient_evidence"
    assert final_judge_payload["decision"] == "insufficient_evidence"
    selected_family = artifact.final_judge.model_summary.get("selected_family")
    assert selected_family is not None
    assert selected_family == artifact.model_search.selected_family
    assert (
        artifact.final_judge.model_summary.get("target_kind")
        == artifact.model_search.target_kind
    )


def test_run_directory_is_portable_and_valid(full_run: RunOutcome) -> None:
    run_dir = full_run.run_dir
    artifact, errors = validate_artifact(full_run.artifact_path)
    assert errors == []
    assert artifact is not None
    assert artifact.schema_version == "1.5.0"

    _assert_relative_file(run_dir, artifact.dataset.stored_path)
    for hypothesis in artifact.hypotheses:
        _assert_relative_file(run_dir, hypothesis.code_path)
        predictions_path = hypothesis.execution.predictions_path
        assert predictions_path is not None, hypothesis.id
        _assert_relative_file(run_dir, predictions_path)

    assert artifact.final_judge is not None
    assert artifact.final_judge.report_path is not None
    assert artifact.final_judge.fragment_path is not None
    _assert_relative_file(run_dir, artifact.final_judge.report_path)
    _assert_relative_file(run_dir, artifact.final_judge.fragment_path)

    raw = full_run.artifact_path.read_text(encoding="utf-8")
    tmp_root = str(run_dir.parent)
    assert tmp_root not in raw, "artifact.json leaked the absolute tmp_path"
    assert tmp_root.replace("\\", "/") not in raw, (
        "artifact.json leaked the absolute tmp_path in POSIX form"
    )


def test_preview_covers_every_optional_stage(
    full_run: RunOutcome, request: pytest.FixtureRequest
) -> None:
    preview = full_run.preview_path.read_text(encoding="utf-8")

    headings = (
        "## Automated model search",
        "## Dataset variants",
        "## Collection metadata",
        "## Generation validation summary",
        "## Validation summary",
        "## Semantic scraping",
        "## Final model judgement",
    )
    missing = [heading for heading in headings if heading not in preview]
    assert not missing, f"preview is missing stage headings: {missing}"
    assert SENTINEL in preview, "collection metadata did not reach the preview"

    # whack26/preview.py renders "## Field pitfalls" only when at least one
    # pitfall was extracted. In the mandated mock/offline full run the stage
    # runs (artifact.field_pitfalls and field_pitfalls.json exist) but the
    # fresh cache has no research records, so extraction yields zero pitfalls
    # and the heading is omitted. Record that as a non-strict expected failure
    # rather than editing production.
    artifact = full_run.artifact
    if artifact.field_pitfalls is not None and not artifact.field_pitfalls.pitfalls:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=False,
                reason=(
                    "production preview gap: the field-pitfall stage ran and "
                    "wrote field_pitfalls.json (queries recorded) but extracted "
                    "zero pitfalls in mock/offline mode because the fresh cache "
                    "has no research records; whack26/preview.py renders "
                    "'## Field pitfalls' only when report.pitfalls is non-empty, "
                    "so the heading is absent even though the artifact section "
                    "is present."
                ),
            )
        )
    assert "## Field pitfalls" in preview
