from __future__ import annotations

import json
from pathlib import Path

import pytest

from whack26.store import FieldStore
from whack26.validation.collection import (
    CollectionMetadataError,
    check_collection_design,
    collection_to_json,
    has_causal_design,
    load_collection,
    parse_collection_fields,
    parse_collection_json,
    parse_collection_metadata,
    persist_collection,
    redact_collection,
    redact_text,
)
from whack26.validation.generation_models import CollectionMetadata

DIMS = 8
DATASET_ID = "ds_collection0001"


def vec(*values: float) -> list[float]:
    return [*values, *([0.0] * (DIMS - len(values)))]


def make_dataset(store: FieldStore) -> None:
    field = store.create_field(
        parent_id=None,
        name="Test",
        description="A test field.",
        description_embedding=vec(1.0),
    )
    store.commit_path(
        dataset_id=DATASET_ID,
        sha256="a" * 64,
        profile_json='{"rows": 10}',
        summary="A test dataset",
        embedding=vec(1.0),
        path_json=json.dumps([field.id]),
        confidence=0.75,
        refined_after_research=False,
        field_ids=[field.id],
    )


def clean_metadata(**overrides) -> CollectionMetadata:
    values = {
        "design": "experimental",
        "unit_of_analysis": "participant",
        "sampling": "randomized assignment of consenting adults",
        "time_period": "2024",
        "treatment": "treatment arm",
        "identification_strategy": "randomized assignment",
        "known_biases": ["attrition"],
        "measurement_notes": ["outcome measured with a validated scale"],
    }
    values.update(overrides)
    return CollectionMetadata(**values)


def test_collection_schema_defaults() -> None:
    metadata = CollectionMetadata()

    assert metadata.design == "observational"
    assert metadata.known_biases == []
    assert metadata.measurement_notes == []
    assert metadata.ethical_constraints == []
    assert metadata.unit_of_analysis is None


def test_parse_collection_fields_merges_repeated_values() -> None:
    metadata = parse_collection_metadata(
        fields=[
            "design=experimental",
            "unit_of_analysis=participant",
            "known_biases=attrition;measurement error",
            "known_biases=selection",
            "treatment=treatment arm",
        ]
    )

    assert metadata.design == "experimental"
    assert metadata.unit_of_analysis == "participant"
    assert metadata.known_biases == ["attrition", "measurement error", "selection"]
    assert metadata.treatment == "treatment arm"


def test_parse_collection_json_from_path_and_text(tmp_path: Path) -> None:
    payload = {
        "design": "survey",
        "unit_of_analysis": "household",
        "known_biases": ["non-response"],
    }
    path = tmp_path / "collection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    from_path = parse_collection_json(path)
    from_text = parse_collection_json(json.dumps(payload))

    assert from_path == from_text
    assert from_path.design == "survey"
    assert from_path.unit_of_analysis == "household"


def test_parse_collection_metadata_fields_override_json(tmp_path: Path) -> None:
    path = tmp_path / "collection.json"
    path.write_text(
        json.dumps({"design": "observational", "sampling": "convenience"}),
        encoding="utf-8",
    )

    metadata = parse_collection_metadata(
        json_source=path, fields=["design=experimental", "sampling=census"]
    )

    assert metadata.design == "experimental"
    assert metadata.sampling == "census"


def test_parse_collection_fields_rejects_unknown_keys_and_designs() -> None:
    with pytest.raises(CollectionMetadataError, match="unknown collection field"):
        parse_collection_fields(["not_a_field=1"])

    with pytest.raises(CollectionMetadataError, match="unknown design"):
        parse_collection_fields(["design=magic"])

    with pytest.raises(CollectionMetadataError, match="key=value"):
        parse_collection_fields(["design"])


def test_causal_gate_rejects_plain_observational_design() -> None:
    metadata = CollectionMetadata(
        design="observational",
        unit_of_analysis="individual row",
        sampling="simple random sample",
    )

    issues = check_collection_design(
        metadata, claim_type="causal", outcome="income"
    )

    fatal = [issue for issue in issues if issue.severity == "fatal"]
    assert fatal, [issue.evidence for issue in issues]
    assert {issue.evidence for issue in fatal} >= {"claim.design", "design.treatment"}
    assert not has_causal_design(metadata, outcome="income")


def test_causal_gate_accepts_experimental_design() -> None:
    metadata = clean_metadata()

    issues = check_collection_design(
        metadata, claim_type="causal", outcome="score"
    )

    assert [issue for issue in issues if issue.severity == "fatal"] == []
    assert has_causal_design(metadata, outcome="score")


def test_quasi_experimental_requires_running_variable_and_strategy() -> None:
    metadata = CollectionMetadata(
        design="quasi_experimental",
        unit_of_analysis="district",
        sampling="all districts",
        treatment="policy",
        identification_strategy=None,
        running_variable=None,
        instrument=None,
    )

    issues = check_collection_design(
        metadata, claim_type="causal", outcome="employment"
    )
    evidence = {issue.evidence for issue in issues}

    assert "design.identification" in evidence
    assert "design.instrument" in evidence
    assert not has_causal_design(metadata, outcome="employment")


def test_causal_gate_needs_outcome_and_unit() -> None:
    metadata = clean_metadata(unit_of_analysis=None)

    issues = check_collection_design(metadata, claim_type="causal", outcome=None)

    fatal = [issue for issue in issues if issue.severity == "fatal"]
    assert {issue.evidence for issue in fatal} == {"claim.design", "unit.of_analysis"}


def test_missing_collection_for_prediction_is_not_an_issue() -> None:
    assert check_collection_design(None, claim_type="prediction") == []
    assert not has_causal_design(None, outcome="y")


def test_non_causal_metadata_does_not_require_treatment() -> None:
    metadata = CollectionMetadata(
        design="observational",
        unit_of_analysis="county",
        sampling="census",
    )

    issues = check_collection_design(
        metadata, claim_type="prediction", outcome="rate"
    )

    assert [issue for issue in issues if issue.evidence == "design.treatment"] == []


def test_pii_redaction_removes_identifiers_and_secrets() -> None:
    metadata = CollectionMetadata(
        design="observational",
        unit_of_analysis="individual row",
        sampling="survey of jane.doe@example.com",
        data_source="https://example.test/data?api_key=supersecret",
        collection_method="follow-up by +1 (555) 123-4567 from 192.168.1.20",
        treatment="record 123e4567-e89b-12d3-a456-426614174000",
        known_biases=["contact jane.doe@example.com"],
        measurement_notes=["called +44 20 7946 0958"],
    )

    safe = redact_collection(metadata)
    text = json.dumps(safe.model_dump(mode="json"))

    assert "jane.doe@example.com" not in text
    assert "123e4567-e89b-12d3-a456-426614174000" not in text
    assert "555" not in text
    assert "7946" not in text
    assert "192.168.1.20" not in text
    assert "supersecret" not in text
    assert "[redacted]" in text


def test_redact_text_leaves_dates_alone() -> None:
    assert redact_text("2020-2023") == "2020-2023"
    assert redact_text("2024") == "2024"


def test_persist_and_load_collection_round_trip(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        make_dataset(store)
        metadata = clean_metadata()

        persist_collection(store, DATASET_ID, metadata)
        loaded = load_collection(store, DATASET_ID)

        assert loaded is not None
        assert loaded == metadata
        assert store.get_collection(DATASET_ID)["design"] == "experimental"

        persisted = collection_to_json(metadata)
        assert json.loads(persisted)["unit_of_analysis"] == "participant"


def test_persist_redacts_by_default_and_can_disable(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        make_dataset(store)
        metadata = clean_metadata(sampling="contact jane.doe@example.com")

        persist_collection(store, DATASET_ID, metadata)
        stored = store.get_collection(DATASET_ID)
        assert stored is not None
        assert "jane.doe@example.com" not in stored["sampling"]

        persist_collection(store, DATASET_ID, metadata, redact=False)
        assert "jane.doe@example.com" in store.get_collection(DATASET_ID)["sampling"]


def test_load_collection_missing_returns_none(tmp_path: Path) -> None:
    with FieldStore(tmp_path / "fields.db", dims=DIMS) as store:
        make_dataset(store)
        assert load_collection(store, DATASET_ID) is None
