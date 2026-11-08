from __future__ import annotations

from pathlib import Path

from artifact_fixtures import make_artifact

from whack26.artifact import SemanticScrapingDocumentSection, SemanticScrapingSection
from whack26.preview import preview_from_path, render_preview, write_preview


def test_preview_renders_every_required_section(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)

    text = render_preview(artifact, base_dir=run_dir)

    for heading in (
        "## Dataset",
        "## Field path",
        "## Research",
        "## Factors",
        "## Hypotheses",
        "## Critic summary",
        "## Warnings and degraded modes",
        "## Provenance",
    ):
        assert heading in text
    assert artifact.run.run_id in text
    assert "Mock Domain" in text
    assert "household income" in text
    assert "linear_model hypothesis" in text
    assert "group_comparison hypothesis" in text
    assert "interaction_or_derived hypothesis" in text
    assert "def run(df: pd.DataFrame" in text
    assert "verdict" in text.lower()


def test_preview_shows_critic_issues_and_missing_information(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact.hypotheses[0].critic.missing_information = [  # type: ignore[union-attr]
        "the split protocol"
    ]
    artifact.hypotheses[1].critic.issues[0].severity = "fatal"  # type: ignore[index,union-attr]

    text = render_preview(artifact, base_dir=run_dir)

    assert "A split protocol is not described." in text
    assert "Fix: Split first." in text
    assert "the split protocol" in text
    assert "fatal" in text


def test_preview_limits_research_to_ten_records(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    template = artifact.research.records[0]
    artifact.research.records = [
        template.model_copy(update={"title": f"Study number {index}"})
        for index in range(12)
    ]

    text = render_preview(artifact, base_dir=run_dir)

    assert "Study number 9" in text
    assert "Study number 10" not in text
    assert "2 more records" in text


def test_preview_has_no_absolute_paths_or_prediction_arrays(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)

    text = render_preview(artifact, base_dir=run_dir)

    assert str(run_dir) not in text
    assert "C:\\" not in text
    assert "predictions/hyp_" in text
    assert "[0.0, 0.0, 0.0" not in text


def test_preview_handles_missing_code_file(tmp_path: Path) -> None:
    artifact, run_dir, hypotheses = make_artifact(tmp_path)
    (run_dir / hypotheses[0].code_path).unlink()

    text = render_preview(artifact, base_dir=run_dir)

    assert "(code file unavailable)" in text


def test_write_preview_and_preview_from_path(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    artifact_path.write_text(
        artifact.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    direct = tmp_path / "direct.md"
    write_preview(direct, artifact, base_dir=run_dir)
    generated = preview_from_path(artifact_path)

    assert direct.is_file()
    assert generated == run_dir / "preview.md"
    assert generated.is_file()
    assert direct.read_text(encoding="utf-8") == generated.read_text(encoding="utf-8")


def test_preview_renders_semantic_scraping_section(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact.research.semantic_scraping = SemanticScrapingSection(
        documents=[
            SemanticScrapingDocumentSection(
                id="doc_a",
                kind="paper",
                source="openalex",
                title="A scraped paper",
                url="https://example.test/a",
                summary="A summary.",
                authors=["A. Author"],
                year=2024,
                credibility=0.8,
                credible=True,
                field_path=["Statistics", "Methods"],
                critique_verdict="refine",
                comparisons=2,
            )
        ],
        warnings=["scrape_budget_exhausted"],
        stats={"documents": 1},
    )

    text = render_preview(artifact, base_dir=run_dir)

    assert "## Semantic scraping" in text
    assert "A scraped paper" in text
    assert "`Statistics` > `Methods`" in text
    assert "refine" in text
    assert "scrape_budget_exhausted" in text
