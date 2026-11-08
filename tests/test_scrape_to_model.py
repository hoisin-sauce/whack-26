from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from whack26.config import Settings
from whack26.factors import Factor
from whack26.llm import LLMError, MockLLMClient
from whack26.profile import ColumnProfile, DatasetProfile, TargetChoice
from whack26.research.scrape.models import Document
from whack26.research.to_model import (
    RESEARCH_MODEL_SCHEMA,
    ResearchModelSpec,
    extract_research_models,
    mark_research_model_applied,
)
from whack26.store import FieldStore

EVIDENCE = "Household income predicts house prices."


def make_document(
    identifier: str = "doc_paper",
    *,
    summary: str = f"{EVIDENCE} We fit an ordinary least squares model.",
    abstract: str | None = None,
    kind: str = "paper",
) -> Document:
    metadata: dict[str, Any] = {}
    if abstract is not None:
        metadata["abstract"] = abstract
    return Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source="openalex",
        source_id=identifier,
        title="Income and house prices",
        summary=summary,
        summary_embedding=[1.0, 0.0],
        embedding_model="fake@2",
        metadata=metadata,
        credible=True,
    )


def column(name: str, role: str = "feature", *, dtype: str = "float64") -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype=dtype,
        role=role,  # type: ignore[arg-type]
        missing_pct=0.0,
        cardinality=10,
        sample_values=[],
        numeric=None,
        top_values=None,
        datetime=False,
    )


def make_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_id="ds_aaaaaaaaaaaa",
        source_path="data/housing.csv",
        sha256="a" * 64,
        format="csv",
        rows=1000,
        columns=[
            column("median_income"),
            column("ocean_proximity", dtype="object"),
            column("night_lights"),
            column("median_house_value", role="target"),
            column("row_id", role="id", dtype="int64"),
        ],
        description="Housing prices by block group.",
        target=TargetChoice(
            name="median_house_value", source="user", confidence=1.0
        ),
        time_column=None,
        warnings=[],
    )


class FakeEmbeddings:
    """Small deterministic embeddings for the mapping branches."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            lower = text.lower()
            if "income" in lower or "earning" in lower:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "ocean" in lower:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "house" in lower and "price" not in lower:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "night" in lower or "nocur" in lower or "radiance" in lower:
                vectors.append([0.0, 0.0, 0.0, 1.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 0.0])
        return vectors


def make_model(**overrides: Any) -> dict[str, Any]:
    model: dict[str, Any] = {
        "family": "linear_model",
        "outcome": "house price",
        "predictors": ["household income", "ocean proximity"],
        "target_transform": "none",
        "feature_transforms": [
            {"feature": "household income", "transform": "log1p"}
        ],
        "interactions": [
            {"left": "household income", "right": "ocean proximity"}
        ],
        "assumptions": ["A linear relationship is adequate."],
        "reported_metrics": {"r2": 0.62, "n_observations": 1000},
        "evidence": EVIDENCE,
        "confidence": 0.82,
    }
    model.update(overrides)
    return model


def run_extract(
    llm: MockLLMClient,
    *,
    documents: list[Document] | None = None,
    factors: list[Factor] | None = None,
    store: Any | None = None,
    warnings: list[str] | None = None,
) -> list[ResearchModelSpec]:
    return extract_research_models(
        documents if documents is not None else [make_document()],
        factors or [],
        make_profile(),
        llm,
        FakeEmbeddings(),
        Settings(embed_dims=4, factor_map_cosine=0.72),
        store,
        warnings=warnings,
    )


def test_schema_is_strict_and_declares_every_spec_field() -> None:
    assert RESEARCH_MODEL_SCHEMA["type"] == "object"
    model_schema = RESEARCH_MODEL_SCHEMA["properties"]["models"]["items"]
    assert set(model_schema["required"]) == {
        "family",
        "outcome",
        "predictors",
        "target_transform",
        "feature_transforms",
        "interactions",
        "assumptions",
        "reported_metrics",
        "evidence",
        "confidence",
    }
    assert model_schema["properties"]["family"]["enum"] == [
        "linear_model",
        "group_comparison",
        "interaction_or_derived",
    ]


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        ("", "missing_evidence"),
        ("This sentence does not occur in the document.", "evidence_not_found"),
    ],
)
def test_evidence_is_required_and_must_be_a_span(
    evidence: str, reason: str
) -> None:
    llm = MockLLMClient(
        fixtures={"research_models": {"models": [make_model(evidence=evidence)]}}
    )
    warnings: list[str] = []

    specs = run_extract(llm, warnings=warnings)

    assert specs == []
    assert any(reason in warning for warning in warnings)
    assert llm.calls == ["research_models"]


def test_predictors_map_by_exact_and_synonym_and_spec_is_complete() -> None:
    llm = MockLLMClient(fixtures={"research_models": {"models": [make_model()]}})

    specs = run_extract(llm)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.document_id == "doc_paper"
    assert spec.family == "linear_model"
    assert spec.outcome == "house price"
    assert spec.predictors == ["household income", "ocean proximity"]
    assert spec.mapped_columns == ["median_income", "ocean_proximity"]
    assert spec.target_transform == "none"
    assert spec.feature_transforms == {"median_income": "log1p"}
    assert spec.interactions == [("median_income", "ocean_proximity")]
    assert spec.reported_metrics == {"r2": 0.62, "n_observations": 1000.0}
    assert spec.evidence == EVIDENCE
    assert spec.applied is False
    assert spec.confidence == pytest.approx(0.82 * 0.8)


def test_factor_and_synonym_mapping_branches() -> None:
    factors = [
        Factor(
            name="ocean proximity",
            description="Distance to the coast.",
            evidence_ids=["doc_paper"],
            mapped_column="ocean_proximity",
            in_dataset=True,
            mapping_method="exact",
            mapping_confidence=0.95,
        )
    ]
    fixture = make_model(predictors=["earnings", "ocean proximity"])
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})

    specs = run_extract(llm, factors=factors)

    assert specs[0].mapped_columns == ["median_income", "ocean_proximity"]


def test_embedding_mapping_branch() -> None:
    fixture = make_model(predictors=["nocturnal radiance", "ocean proximity"])
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})

    specs = run_extract(llm)

    assert specs[0].mapped_columns == ["night_lights", "ocean_proximity"]


def test_unmappable_predictor_rejects_the_model() -> None:
    fixture = make_model(predictors=["unobserved social capital index"])
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})
    warnings: list[str] = []

    specs = run_extract(llm, warnings=warnings)

    assert specs == []
    assert any("unmapped_predictor" in warning for warning in warnings)


def test_predictor_mapped_to_target_is_rejected_as_target_leakage() -> None:
    fixture = make_model(
        predictors=["median house value", "ocean proximity"],
    )
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})
    warnings: list[str] = []

    specs = run_extract(llm, warnings=warnings)

    assert specs == []
    assert any("target_leakage" in warning for warning in warnings)


def test_unsupported_family_is_rejected() -> None:
    fixture = make_model(family="random_forest")
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})
    warnings: list[str] = []

    specs = run_extract(llm, warnings=warnings)

    assert specs == []
    assert any("unsupported_family" in warning for warning in warnings)


def test_family_alias_maps_to_the_three_family_contract() -> None:
    fixture = make_model(family="ols", predictors=["household income"])
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})

    specs = run_extract(llm)

    assert [spec.family for spec in specs] == ["linear_model"]


def test_outcome_mismatch_rejects_the_model() -> None:
    fixture = make_model(outcome="student test scores", predictors=["household income"])
    llm = MockLLMClient(fixtures={"research_models": {"models": [fixture]}})
    warnings: list[str] = []

    specs = run_extract(llm, warnings=warnings)

    assert specs == []
    assert any("outcome_mismatch" in warning for warning in warnings)


def test_one_llm_call_per_document() -> None:
    documents = [make_document("doc_a"), make_document("doc_b", kind="web")]
    llm = MockLLMClient(fixtures={"research_models": {"models": [make_model()]}})

    specs = run_extract(llm, documents=documents)

    assert llm.calls == ["research_models", "research_models"]
    assert [spec.document_id for spec in specs] == ["doc_a", "doc_b"]


def test_llm_failure_is_reported_and_does_not_stop_the_batch() -> None:
    class FailingOnce(MockLLMClient):
        def __init__(self) -> None:
            super().__init__({"research_models": {"models": [make_model()]}})
            self.failed = False

        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            if not self.failed:
                self.failed = True
                self.calls.append("research_models")
                raise LLMError("quota", purpose="research_models")
            return super().generate_json(**kwargs)

    warnings: list[str] = []
    specs = run_extract(
        FailingOnce(),
        documents=[make_document("doc_a"), make_document("doc_b")],
        warnings=warnings,
    )

    assert [spec.document_id for spec in specs] == ["doc_b"]
    assert any("research_models_llm_failed" in warning for warning in warnings)


def test_storage_round_trip_and_idempotent_rewrite(tmp_path: Path) -> None:
    llm = MockLLMClient(fixtures={"research_models": {"models": [make_model()]}})
    document = make_document("doc_stored")

    with FieldStore(tmp_path / "fields.db", dims=2) as store:
        store.upsert_document(document)
        specs = run_extract(llm, documents=[document], store=store)
        stored = store.list_research_models(document_id="doc_stored")

        assert len(specs) == 1
        assert len(stored) == 1
        record = stored[0]
        assert record["family"] == "linear_model"
        assert record["evidence"] == EVIDENCE
        assert record["applied"] is False
        assert record["spec"]["mapped_columns"] == [
            "median_income",
            "ocean_proximity",
        ]
        assert record["mapping"]["predictors"] == [
            "household income",
            "ocean proximity",
        ]

        # Re-running extraction upserts the deterministic id instead of
        # creating a duplicate research model.
        run_extract(llm, documents=[document], store=store)
        assert len(store.list_research_models(document_id="doc_stored")) == 1

        updated = mark_research_model_applied(store, specs[0])
        assert updated.applied is True
        assert store.list_research_models(document_id="doc_stored", applied=True)


class FakeStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def put_research_model(self, **kwargs: Any) -> str:
        self.records.append(kwargs)
        return str(kwargs["model_id"])


def test_applied_flag_round_trips() -> None:
    spec = ResearchModelSpec(
        document_id="doc_paper",
        family="linear_model",
        outcome="house price",
        predictors=["household income"],
        evidence=EVIDENCE,
        mapped_columns=["median_income"],
        confidence=0.7,
    )
    store = FakeStore()

    assert spec.applied is False
    updated = mark_research_model_applied(store, spec)

    assert updated.applied is True
    assert spec.applied is False
    assert store.records[-1]["applied"] is True
