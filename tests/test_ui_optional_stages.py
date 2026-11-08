"""Optional-stage report coverage (plan section G, SA-6).

Covers UI-80 to UI-85. The pipeline handoff basics (UI-70 to UI-79) live in
``tests/test_ui_pipeline.py`` and are deliberately not re-authored here.

The renderer currently exposes a selected-model table but has no
variant-specific, generation-specific, or collection-specific sections. Tests
for those stages therefore assert that the report and fragment remain valid and
mark the missing renderer section xfail rather than inventing output.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from UI_frontEnd import QuantDashboard  # noqa: E402
from ui_fixtures import make_pipeline_settings, make_ui_dataset  # noqa: E402
from whack26.final_judge.types import FinalJudgeReport  # noqa: E402
from whack26.run import run_pipeline  # noqa: E402

pytestmark = [pytest.mark.ui, pytest.mark.integration]

# Stable text contract from the final judge: report decision -> UI label.
DECISION_LABELS = {
    "accepted": "Accepted",
    "accepted_with_concerns": "Accepted with concerns",
    "requires_reaudit": "Re-audit required",
    "rejected": "Rejected",
    "insufficient_evidence": "Insufficient evidence",
}

COLLECTION_SENTINEL = "UI83-ROW-UNIT-SENTINEL"
COLLECTION_METADATA = {
    "design": "experimental",
    "unit_of_analysis": COLLECTION_SENTINEL,
    "treatment": "assigned treatment",
    "sampling": "randomised draw from the study population",
}


@pytest.fixture
def dashboard(qtbot):
    """A real dashboard whose lifetime is owned by pytest-qt."""
    widget = QuantDashboard()
    qtbot.addWidget(widget)
    try:
        yield widget
    finally:
        widget.close()
        widget.deleteLater()


@pytest.fixture(scope="module")
def optional_run(tmp_path_factory):
    """One small mock/offline run with every optional report stage enabled."""
    tmp_path = tmp_path_factory.mktemp("ui_optional_stages")
    dataset = make_ui_dataset(tmp_path / "input.csv")
    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        mock=True,
        offline=True,
        model_search=True,
        variants=True,
        validate_generation=True,
        final_judge=True,
        collection=COLLECTION_METADATA,
        settings=make_pipeline_settings(tmp_path),
        out_dir=tmp_path / "run-optional",
    )
    assert outcome.final_judge is not None
    return outcome


@pytest.fixture(scope="module")
def minimal_run(tmp_path_factory):
    """One small mock/offline run with every optional report stage disabled."""
    tmp_path = tmp_path_factory.mktemp("ui_optional_stages_minimal")
    dataset = make_ui_dataset(tmp_path / "input.csv")
    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        mock=True,
        offline=True,
        model_search=False,
        variants=False,
        validate_generation=False,
        field_pitfalls=False,
        final_judge=True,
        settings=make_pipeline_settings(tmp_path),
        out_dir=tmp_path / "run-minimal",
    )
    assert outcome.final_judge is not None
    return outcome


def _fragment(outcome) -> str:
    return (outcome.run_dir / "final_judge.fragment.html").read_text(
        encoding="utf-8"
    )


def _load_text(dashboard, outcome) -> str:
    workspace = dashboard.current_workspace()
    dashboard.load_final_judge(outcome.run_dir, workspace)
    return workspace.summary_output.toPlainText()


def test_ui80_model_search_shows_selected_model_table(
    dashboard, optional_run
) -> None:
    """UI-80: ``model_search=True`` renders the selected-model table in the UI."""
    report = optional_run.final_judge
    assert report is not None
    assert optional_run.model_search is not None

    family = report.model_summary.get("selected_family")
    assert family, "mock/offline model search should select a model family"

    fragment = _fragment(optional_run)
    assert "<h2>Selected model</h2>" in fragment
    assert "Selected family" in fragment
    assert str(family) in fragment
    assert "Candidates" in fragment

    text = _load_text(dashboard, optional_run)
    assert "Selected model" in text
    assert "Selected family" in text
    assert str(family) in text


def test_ui81_variants_keep_fragment_and_report_valid(
    dashboard, optional_run
) -> None:
    """UI-81: ``variants=True`` does not break fragment rendering or the report."""
    report = optional_run.final_judge
    assert report is not None
    assert optional_run.variants is not None

    payload = (optional_run.run_dir / "final_judge.json").read_text(
        encoding="utf-8"
    )
    round_tripped = FinalJudgeReport.model_validate_json(payload)
    assert round_tripped.decision == report.decision

    fragment = _fragment(optional_run)
    assert "Final model judgement" in fragment
    assert DECISION_LABELS[report.decision] in fragment

    text = _load_text(dashboard, optional_run)
    assert "Final model judgement" in text
    assert DECISION_LABELS[report.decision] in text

    # No variant-specific findings are rendered today. Per the plan, assert the
    # report stays valid and the fragment still renders rather than inventing a
    # variant section.


def test_ui82_validate_generation_renders_evidence(dashboard, optional_run) -> None:
    """UI-82: ``validate_generation=True`` keeps the fragment renderable."""
    report = optional_run.final_judge
    assert report is not None

    fragment = _fragment(optional_run)
    assert "Final model judgement" in fragment

    text = _load_text(dashboard, optional_run)
    assert DECISION_LABELS[report.decision] in text

    if report.missing_evidence:
        assert "missing evidence" in fragment.lower()
        assert "not a pass" in fragment.lower()

    if "generation" not in fragment.lower():
        pytest.xfail(
            "current fragment renderer has no generation-specific section; "
            "generation evidence reaches the judge prompt but is not rendered "
            "(production change required)"
        )
    assert "Generation validation" in fragment
    assert "generation" in text.lower()


def test_ui83_collection_metadata_is_not_dropped(dashboard, optional_run) -> None:
    """UI-83: collection metadata survives the run and does not break the UI."""
    report = optional_run.final_judge
    assert report is not None

    collection = optional_run.artifact.collection
    assert collection is not None
    assert collection.get("unit_of_analysis") == COLLECTION_SENTINEL

    fragment = _fragment(optional_run)
    assert "Final model judgement" in fragment

    text = _load_text(dashboard, optional_run)
    assert DECISION_LABELS[report.decision] in text

    if COLLECTION_SENTINEL not in fragment:
        pytest.xfail(
            "current fragment renderer does not expose collection metadata; "
            "collection context is present in artifact.json but has no UI "
            "section (production change required)"
        )
    assert COLLECTION_SENTINEL in text


def test_ui84_all_optional_stages_off_render_minimal_fragment(
    dashboard, minimal_run
) -> None:
    """UI-84: optional stages off leaves no empty placeholder sections."""
    report = minimal_run.final_judge
    assert report is not None
    assert minimal_run.model_search is None
    assert minimal_run.variants is None
    assert minimal_run.generation_validations == {}

    fragment = _fragment(minimal_run)
    assert "Final model judgement" in fragment
    assert DECISION_LABELS[report.decision] in fragment

    # Sections whose data is absent must be omitted rather than emitted empty.
    for title, values in (
        ("Code-verified flags", report.code_verified_flags),
        ("Vetoes", report.vetoes),
        ("Pending re-audit", report.pending_reaudit),
        ("Concerns", report.concerns),
        ("Advisories", report.advisories),
        ("Next steps", report.reaudit_instructions),
    ):
        if not values:
            assert title not in fragment
    assert '<ul class="flags"></ul>' not in fragment
    assert '<ol class="steps"></ol>' not in fragment
    assert "<table></table>" not in fragment

    text = _load_text(dashboard, minimal_run)
    assert "Final model judgement" in text
    assert DECISION_LABELS[report.decision] in text


def test_ui85_mock_judge_is_not_a_pass_in_the_ui(dashboard, minimal_run) -> None:
    """UI-85: the mock judge is insufficient evidence and never ``Accepted``."""
    report = minimal_run.final_judge
    assert report is not None
    assert report.decision == "insufficient_evidence"

    fragment = _fragment(minimal_run)
    assert "Insufficient evidence" in fragment
    assert "not a pass" in fragment.lower()
    assert "Accepted" not in fragment

    text = _load_text(dashboard, minimal_run)
    assert "Insufficient evidence" in text
    assert "not a pass" in text.lower()
    assert "Accepted" not in text
