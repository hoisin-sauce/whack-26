"""Renderer tests for the final judge report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ui_fixtures import make_report as make_shared_report

from whack26.final_judge.report import (
    render_document,
    render_fragment,
    render_json,
    render_markdown,
    write_reports,
)
from whack26.final_judge.types import FinalJudgeReport, JudgeFinding

# Documented UI-fragment cap for a dense report (1 MB).  The renderer is linear
# in the report size; this guards against a pathological blow-up.
FRAGMENT_SIZE_CAP_BYTES = 1_000_000


def make_report(**overrides: Any) -> FinalJudgeReport:
    """Build the accepted-with-concerns fixture on top of the shared helper."""
    values: dict[str, Any] = {
        "decision": "accepted_with_concerns",
        "model_summary": {
            "target": "y",
            "target_kind": "continuous",
            "metric": "r2",
            "selected_family": "linear",
            "selected_features": ["x"],
            "candidate_count": 4,
            "validation_verdict": "pass",
        },
        "n_critics_evaluated": 7,
        "code_verified_flags": ["Train-CV gap is large (0.15)."],
        "concerns": [
            JudgeFinding(
                critic="overfitting_skeptic",
                severity="medium",
                finding="The gap is <script>alert(1)</script>.",
                evidence="train 0.95 vs cv 0.80",
                suggested_fix="Use repeated CV.",
                falsification_test="Re-run with more folds.",
                confidence="medium",
                status="suspected",
            )
        ],
        "unevaluated_critics": ["subgroup_critic"],
        "missing_evidence": ["subgroup_slices"],
        "note": None,
    }
    values.update(overrides)
    return make_shared_report(**values)


def test_fragment_is_ui_safe_and_escaped() -> None:
    fragment = render_fragment(make_report())

    assert "Final model judgement" in fragment
    assert "Accepted with concerns" in fragment
    assert "<style" not in fragment
    assert "<script" not in fragment
    assert "<details" not in fragment
    assert "&lt;script&gt;" in fragment
    assert "Subgroup critic" in fragment


def test_markdown_and_document_render() -> None:
    report = make_report()

    markdown = render_markdown(report)
    document = render_document(report)

    assert "# Final model judgement" in markdown
    assert "## Selected model" in markdown
    assert "Train-CV gap" in markdown
    assert "<!doctype html>" in document
    assert "<style>" in document


def test_json_and_write_reports(tmp_path: Path) -> None:
    report = make_report()

    payload = render_json(report)
    assert '"decision": "accepted_with_concerns"' in payload

    paths = write_reports(tmp_path, report)
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["html"].is_file()
    assert paths["fragment"].is_file()
    assert "<style>" in paths["html"].read_text(encoding="utf-8")
    assert "<style>" not in paths["fragment"].read_text(encoding="utf-8")


def test_write_reports_honours_a_format_subset(tmp_path: Path) -> None:
    report = make_report()

    paths = write_reports(tmp_path, report, formats=("markdown",))

    assert set(paths) == {"markdown"}
    assert paths["markdown"].is_file()
    assert not (tmp_path / "final_judge.json").exists()
    assert not (tmp_path / "final_judge.html").exists()


def test_write_reports_supports_a_custom_stem(tmp_path: Path) -> None:
    paths = write_reports(
        tmp_path, make_report(), stem="judge", formats=("json", "fragment")
    )

    assert set(paths) == {"json", "fragment"}
    assert (tmp_path / "judge.json").is_file()
    assert (tmp_path / "judge.fragment.html").is_file()


def test_write_reports_rejects_unknown_formats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown report format"):
        write_reports(tmp_path, make_report(), formats=("pdf",))


@pytest.mark.parametrize(
    ("decision", "label", "tone"),
    [
        ("accepted", "Accepted", "ok"),
        ("accepted_with_concerns", "Accepted with concerns", "warn"),
        ("requires_reaudit", "Re-audit required", "alert"),
        ("rejected", "Rejected", "bad"),
        ("insufficient_evidence", "Insufficient evidence", "neutral"),
    ],
)
def test_each_decision_gets_its_banner(decision: str, label: str, tone: str) -> None:
    report = FinalJudgeReport(decision=decision)

    fragment = render_fragment(report)

    assert f'banner {tone}' in fragment
    assert label in fragment


def test_empty_report_omits_empty_sections() -> None:
    fragment = render_fragment(FinalJudgeReport())

    assert "Final model judgement" in fragment
    assert "<h2>Vetoes" not in fragment
    assert "<h2>Concerns" not in fragment
    assert "<h2>Advisories" not in fragment
    assert "<h2>Next steps" not in fragment
    assert "<h2>Code-verified flags" not in fragment


def test_markdown_heading_level_is_configurable() -> None:
    markdown = render_markdown(make_report(), heading_level=2)

    assert markdown.startswith("## Final model judgement")


def test_reaudit_steps_and_special_names_are_rendered() -> None:
    report = FinalJudgeReport(
        decision="requires_reaudit",
        reaudit_instructions=[
            {
                "from": "devils_advocate",
                "fix": "Re-run with blocked CV.",
                "test": "Compare fold scores.",
            }
        ],
    )

    fragment = render_fragment(report)
    markdown = render_markdown(report)

    assert "<h2>Next steps</h2>" in fragment
    assert "Devil&#x27;s advocate" in fragment or "Devil's advocate" in fragment
    assert "Re-run with blocked CV." in fragment
    assert "## Next steps" in markdown
    assert "Devil's advocate" in markdown


def test_counts_and_selected_model_table_are_rendered() -> None:
    """UI-54: audit counts and selected-model rows appear when supplied."""
    report = make_report(
        vetoes=[
            JudgeFinding(
                critic="leakage_hunter",
                severity="high",
                finding="A leaked feature predicts the target.",
                evidence="fold 0 R2 is 0.99",
                confidence="high",
                status="established",
            )
        ],
        pending_reaudit=[
            JudgeFinding(
                critic="overfitting_skeptic",
                severity="high",
                finding="The train-CV gap needs confirmation.",
                evidence="train 0.95 vs cv 0.80",
            )
        ],
    )

    fragment = render_fragment(report)

    assert "<th>critics evaluated / minimum</th><td>7/5</td>" in fragment
    assert "<th>blocking findings</th><td>2</td>" in fragment
    assert "<h2>Selected model</h2>" in fragment
    for label, value in (
        ("Target", "y"),
        ("Target kind", "continuous"),
        ("Metric", "r2"),
        ("Selected family", "linear"),
        ("Features", "x"),
        ("Candidates", "4"),
        ("Validation", "pass"),
    ):
        assert f"<tr><th>{label}</th><td>{value}</td></tr>" in fragment


def test_unevaluated_critics_and_missing_evidence_are_visible() -> None:
    """UI-56: the non-pass wording and missing items are visible."""
    report = make_report(
        decision="insufficient_evidence",
        n_critics_evaluated=1,
        min_required=5,
        unevaluated_critics=["overfitting_skeptic", "subgroup_critic"],
        missing_evidence=["out_of_sample_holdout", "generation_metrics"],
        note="The audit did not complete; this is NOT a pass.",
    )

    fragment = render_fragment(report)

    assert "Insufficient evidence" in fragment
    assert "This is not a pass." in fragment
    assert "Absence of evidence is not a pass." in fragment
    assert "NOT a pass" in fragment
    assert "<h2>Unevaluated critics and missing evidence</h2>" in fragment
    assert "Overfitting skeptic" in fragment
    assert "Subgroup critic" in fragment
    assert "<li>out_of_sample_holdout</li>" in fragment
    assert "<li>generation_metrics</li>" in fragment


def test_fragment_rendering_is_deterministic() -> None:
    """UI-58: the same report renders byte-identical fragments."""
    report = make_report()

    first = render_fragment(report).encode("utf-8")
    second = render_fragment(report).encode("utf-8")

    assert first == second


def test_dense_report_stays_under_fragment_size_cap() -> None:
    """UI-59: a dense report stays below the documented 1 MB cap."""
    long_text = "Evidence with <markup> & quotes, repeated for density. " * 16
    findings = [
        JudgeFinding(
            critic=f"critic_{index}",
            severity="medium",
            finding=f"Finding {index}: {long_text}",
            evidence=long_text,
            suggested_fix=long_text,
            falsification_test=long_text,
        )
        for index in range(100)
    ]
    report = make_report(
        decision="requires_reaudit",
        vetoes=findings[:20],
        pending_reaudit=findings[20:40],
        concerns=findings[40:70],
        advisories=findings[70:],
        code_verified_flags=[
            f"flag {index}: {long_text}" for index in range(40)
        ],
        unevaluated_critics=[f"critic_{index}" for index in range(40)],
        missing_evidence=[f"evidence_{index}" for index in range(40)],
    )

    fragment = render_fragment(report)

    assert "Finding 99" in fragment
    assert len(fragment.encode("utf-8")) < FRAGMENT_SIZE_CAP_BYTES
