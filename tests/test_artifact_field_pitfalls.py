from __future__ import annotations

import json
from pathlib import Path

from artifact_fixtures import make_artifact

from whack26.artifact import (
    SCHEMA_VERSION,
    artifact_json_schema,
    validate_artifact,
    write_artifact,
)
from whack26.pitfalls.types import (
    FieldPitfall,
    FieldPitfallQuery,
    FieldPitfallReport,
)


def make_report() -> FieldPitfallReport:
    return FieldPitfallReport(
        generated_at="2026-11-08T10:00:00Z",
        field_path=["Mock Domain", "Regression"],
        queries=[
            FieldPitfallQuery(
                field_id="fld_leaf",
                field_name="Regression",
                field_level=2,
                source="openalex",
                query="Regression limitations bias",
            )
        ],
        pitfalls=[
            FieldPitfall(
                id="F:fld_leaf:1",
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
        ],
        warnings=["field_pitfalls_mock_records_only"],
        degraded=True,
        offline=True,
        cache={"hits": 0, "misses": 1},
    )


def test_artifact_round_trips_field_pitfall_report(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    report = make_report()
    artifact.field_pitfalls = report
    artifact_path = run_dir / "artifact.json"

    write_artifact(artifact_path, artifact)
    loaded, errors = validate_artifact(artifact_path)

    assert errors == []
    assert loaded is not None
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.field_pitfalls is not None
    assert loaded.field_pitfalls.model_dump(mode="json") == report.model_dump(
        mode="json"
    )
    assert loaded.field_pitfalls.pitfalls[0].evidence_id == "openalex:W1"
    assert loaded.field_pitfalls.queries[0].field_name == "Regression"


def test_artifact_without_field_pitfalls_still_validates(tmp_path: Path) -> None:
    artifact, run_dir, _ = make_artifact(tmp_path)
    artifact_path = run_dir / "artifact.json"

    write_artifact(artifact_path, artifact)
    loaded, errors = validate_artifact(artifact_path)

    assert errors == []
    assert loaded is not None
    assert loaded.field_pitfalls is None


def test_artifact_schema_version_is_1_5_0() -> None:
    assert SCHEMA_VERSION == "1.5.0"


def test_artifact_json_schema_includes_field_pitfalls() -> None:
    schema = artifact_json_schema()

    assert "field_pitfalls" in schema["properties"]
    assert "FieldPitfallReport" in schema["$defs"]
    assert "FieldPitfall" in schema["$defs"]
    json.dumps(schema)


def test_checked_in_schema_parses_and_includes_field_pitfalls() -> None:
    checked_in = (
        Path(__file__).resolve().parent.parent
        / "schemas"
        / "artifact.schema.json"
    )
    schema = json.loads(checked_in.read_text(encoding="utf-8"))

    assert "field_pitfalls" in schema["properties"]
    assert "FieldPitfallReport" in schema["$defs"]
    assert schema["properties"]["field_pitfalls"]["anyOf"] == [
        {"$ref": "#/$defs/FieldPitfallReport"},
        {"type": "null"},
    ]
