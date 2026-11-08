from __future__ import annotations

from pathlib import Path

from artifact_fixtures import make_artifact

from whack26.pitfalls.types import FieldPitfall, FieldPitfallReport
from whack26.preview import _field_pitfalls_section, render_preview


def make_pitfall(identifier: str = "F:fld_leaf:1") -> FieldPitfall:
    return FieldPitfall(
        id=identifier,
        field_id="fld_leaf",
        field_name="Regression",
        field_level=2,
        scope="leaf",
        title="Selection into the sample",
        pitfall="Sampling on the outcome biases the fitted effect.",
        category="selection",
        applies_to_dataset="unknown",
        applicability_reason="The profile cannot establish the sampling rule.",
        check="Review how rows enter the sample.",
        evidence_id="openalex:W1",
        source="openalex",
        url="https://example.test/W1",
        year=2024,
        relevance=0.9,
        confidence=0.7,
    )


def make_report(
    pitfalls: list[FieldPitfall] | None = None,
) -> FieldPitfallReport:
    return FieldPitfallReport(
        generated_at="2026-11-08T10:00:00Z",
        field_path=["Mock Domain", "Regression"],
        pitfalls=[make_pitfall()] if pitfalls is None else pitfalls,
        warnings=["field_pitfalls_mock_records_only"],
        degraded=True,
        offline=True,
        cache={"hits": 0, "misses": 1},
    )


def test_preview_renders_field_pitfalls_section(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact.field_pitfalls = make_report()

    text = render_preview(artifact, base_dir=run_dir)

    assert "## Field pitfalls" in text
    assert "Regression" in text
    assert "level 2" in text
    assert "scope leaf" in text
    assert "selection" in text
    assert "applies: unknown" in text
    assert "Sampling on the outcome biases the fitted effect." in text
    assert "Check: Review how rows enter the sample." in text
    assert "openalex:W1" in text
    assert "[https://example.test/W1](https://example.test/W1)" in text
    assert text.index("## Field pitfalls") < text.index("## Research")


def test_preview_has_no_field_pitfalls_section_when_report_is_none(
    tmp_path: Path,
) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    assert artifact.field_pitfalls is None

    text = render_preview(artifact, base_dir=run_dir)

    assert "## Field pitfalls" not in text
    assert _field_pitfalls_section(artifact) == []


def test_field_pitfalls_section_returns_empty_for_empty_report(
    tmp_path: Path,
) -> None:
    artifact, _, _ = make_artifact(tmp_path)
    artifact.field_pitfalls = make_report(pitfalls=[])

    assert _field_pitfalls_section(artifact) == []


def test_preview_limits_field_pitfalls_to_twelve(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    pitfalls = [
        make_pitfall(f"F:fld_leaf:{index}").model_copy(
            update={"pitfall": f"Caveat number {index}."}
        )
        for index in range(1, 15)
    ]
    artifact.field_pitfalls = make_report(pitfalls=pitfalls)

    text = render_preview(artifact, base_dir=run_dir)

    assert "Caveat number 12." in text
    assert "Caveat number 13." not in text
    assert "Caveat number 14." not in text
    assert "2 more pitfalls" in text
