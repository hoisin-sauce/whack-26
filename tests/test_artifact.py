from __future__ import annotations

import json
from pathlib import Path

from artifact_fixtures import make_artifact

from whack26.artifact import (
    SCHEMA_VERSION,
    _semantic_scraping_section,
    artifact_json_schema,
    validate_artifact,
    write_artifact,
)
from whack26.research.scrape.models import (
    Document,
    DocumentComparison,
    DocumentCritique,
    DocumentLink,
    ScrapeResult,
)


def test_build_write_and_validate_artifact(tmp_path: Path) -> None:
    artifact, run_dir, hypotheses = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"

    write_artifact(artifact_path, artifact)
    loaded, errors = validate_artifact(artifact_path)

    assert errors == []
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.dataset.stored_path == "dataset.csv"
    assert loaded.dataset.sha256
    assert len(loaded.hypotheses) == 3
    assert [item.id for item in loaded.hypotheses] == [item.id for item in hypotheses]
    assert loaded.critic_summary.verdicts == {
        "pass": 1,
        "refine": 1,
        "reject": 0,
        "insufficient": 1,
    }
    assert loaded.critic_summary.degraded is False
    assert loaded.warnings == ["demo_warning"]


def test_artifact_contains_no_prediction_arrays(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert "predictions" not in payload["research"]["records"][0]
    for hypothesis in payload["hypotheses"]:
        assert "predictions" not in hypothesis
        assert hypothesis["execution"]["predictions_path"].endswith(".csv")


def test_validate_rejects_missing_handoff_files(tmp_path: Path) -> None:
    artifact, run_dir, hypotheses = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    (run_dir / hypotheses[1].code_path).unlink()
    (run_dir / hypotheses[2].execution.predictions_path).unlink()

    _, errors = validate_artifact(artifact_path)

    assert any("hypothesis code is missing" in error for error in errors)
    assert any("predictions are missing" in error for error in errors)


def test_validate_rejects_escaping_paths(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["dataset"]["stored_path"] = "..\\outside.csv"
    payload["hypotheses"][0]["execution"]["predictions_path"] = "C:\\temp\\pred.csv"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, errors = validate_artifact(artifact_path, check_files=False)

    assert any("escapes the run directory" in error for error in errors)
    assert any("path must be relative" in error for error in errors)


def test_validate_allows_risk_adjusted_language(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["warnings"] = ["risk-adjusted weights are not supported yet"]
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, errors = validate_artifact(artifact_path)

    assert not any("credential" in error for error in errors)


def test_validate_detects_credential_material(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["warnings"] = ["AIza" + "a" * 35]
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _, errors = validate_artifact(artifact_path)

    assert any("credential" in error for error in errors)


def test_validate_rejects_unknown_schema_major(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "2.0.0"
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded, errors = validate_artifact(artifact_path)

    assert loaded is None
    assert any("unknown schema major" in error for error in errors)


def test_validate_accepts_legacy_1_0_0_artifact(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"
    write_artifact(artifact_path, artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0.0"
    payload["research"].pop("semantic_scraping", None)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    loaded, errors = validate_artifact(artifact_path)

    assert errors == []
    assert loaded is not None
    assert loaded.schema_version == "1.0.0"
    assert loaded.research.semantic_scraping is None


def test_artifact_json_schema_is_exportable() -> None:
    schema = artifact_json_schema()

    assert schema["type"] == "object"
    assert "schema_version" in schema["properties"]
    assert set(schema["$defs"]) >= {"RunInfo", "DatasetSection", "HypothesisSection"}
    json.dumps(schema)


def test_semantic_scraping_section_maps_documents() -> None:
    scrape = ScrapeResult(
        documents=[
            Document(
                id="doc_a",
                kind="paper",
                source="openalex",
                title="A paper",
                summary="A summary.",
                summary_embedding=[1.0, 0.0],
                canonical_url="https://example.test/a",
                authors=["A. Author"],
                published_at="2024-01-01",
                credibility_score=0.8,
                credible=True,
                metadata={"field_path": ["Statistics", "Methods"]},
            )
        ],
        links=[
            DocumentLink(
                from_id="doc_a",
                to_id="doc_b",
                relation="references",
            )
        ],
        comparisons=[
            DocumentComparison(
                left_id="doc_a",
                right_id="doc_b",
                kind="method_overlap",
                similarity=0.9,
                verdict="supports",
            )
        ],
        critiques=[
            DocumentCritique(
                document_id="doc_a",
                corpus_version="1.0.0+abc",
                verdict="refine",
                summary="A method review.",
            )
        ],
        warnings=["scrape_budget_exhausted"],
        stats={"documents": 1},
    )

    section = _semantic_scraping_section(scrape)

    assert section is not None
    assert section.enabled is True
    assert section.documents[0].field_path == ["Statistics", "Methods"]
    assert section.documents[0].year == 2024
    assert section.documents[0].critique_verdict == "refine"
    assert section.links[0].relation == "references"
    assert section.comparisons[0].verdict == "supports"
    assert section.critiques[0].corpus_version == "1.0.0+abc"
    assert section.warnings == ["scrape_budget_exhausted"]
    assert section.stats == {"documents": 1}
