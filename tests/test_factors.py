from __future__ import annotations

import copy
from typing import Any

import pytest

from whack26.config import Settings
from whack26.factors import (
    Factor,
    _exact_mapping,
    _singular,
    extract_factors,
    map_factors,
)
from whack26.llm import LLMError
from whack26.profile import ColumnProfile, DatasetProfile, TargetChoice
from whack26.research.base import ResearchRecord


def make_profile() -> DatasetProfile:
    columns = [
        ColumnProfile(
            name="median_income",
            dtype="float64",
            role="feature",
            missing_pct=0.0,
            cardinality=100,
            sample_values=["1.5", "2.0"],
            numeric={"min": 1.0, "max": 9.0, "mean": 3.0, "std": 1.0, "zeros": 0, "negatives": 0},
            top_values=None,
            datetime=False,
        ),
        ColumnProfile(
            name="ocean_proximity",
            dtype="str",
            role="feature",
            missing_pct=0.0,
            cardinality=5,
            sample_values=["INLAND", "NEAR BAY"],
            numeric=None,
            top_values=[("INLAND", 3)],
            datetime=False,
        ),
        ColumnProfile(
            name="median_house_value",
            dtype="float64",
            role="target",
            missing_pct=0.0,
            cardinality=100,
            sample_values=["100.0", "200.0"],
            numeric={"min": 1.0, "max": 9.0, "mean": 3.0, "std": 1.0, "zeros": 0, "negatives": 0},
            top_values=None,
            datetime=False,
        ),
        ColumnProfile(
            name="row_id",
            dtype="int64",
            role="id",
            missing_pct=0.0,
            cardinality=100,
            sample_values=["1", "2"],
            numeric={"min": 1.0, "max": 9.0, "mean": 3.0, "std": 1.0, "zeros": 0, "negatives": 0},
            top_values=None,
            datetime=False,
        ),
    ]
    return DatasetProfile(
        dataset_id="ds_aaaaaaaaaaaa",
        source_path="data/housing.csv",
        sha256="a" * 64,
        format="csv",
        rows=100,
        columns=columns,
        description="Housing prices by block group.",
        target=TargetChoice(name="median_house_value", source="user", confidence=1.0),
        time_column=None,
        warnings=[],
    )


def make_records() -> list[ResearchRecord]:
    return [
        ResearchRecord(
            id="openalex:W1",
            source="openalex",
            source_id="W1",
            title="Household income and house prices",
            abstract="Income predicts prices.",
            url="https://example.test/W1",
            retrieved_at="2026-01-01T00:00:00Z",
            query="housing prices",
            relevance=0.9,
        ),
        ResearchRecord(
            id="zenodo:123",
            source="zenodo",
            source_id="123",
            title="Housing datasets",
            abstract=None,
            url="https://example.test/123",
            retrieved_at="2026-01-01T00:00:00Z",
            query="housing dataset",
            relevance=0.7,
        ),
    ]


class FakeEmbeddings:
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        vectors = []
        for text in texts:
            lower = text.lower()
            if "income" in lower:
                vectors.append([1.0, 0.0])
            elif "ocean" in lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.6, 0.8])
        return vectors


class ScriptedLLM:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, purpose: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"purpose": purpose, **kwargs})
        payload = self.payloads[purpose]
        if isinstance(payload, Exception):
            raise payload
        return copy.deepcopy(payload)


def _factor_payload(*factors: dict[str, Any]) -> dict[str, Any]:
    return {"factors": list(factors)}


def _mapping_payload(*mappings: dict[str, Any]) -> dict[str, Any]:
    return {"mappings": list(mappings)}


def test_extract_factors_drops_unknown_evidence_and_duplicates() -> None:
    records = make_records()
    llm = ScriptedLLM(
        {
            "factors": _factor_payload(
                {
                    "name": "household income",
                    "description": "Income of households in the area.",
                    "direction": "positive",
                    "claim_type": "association",
                    "evidence_ids": ["openalex:W1"],
                    "candidate_columns": ["median_income"],
                    "rationale": "income predicts prices",
                },
                {
                    "name": "invented factor",
                    "description": "No record supports this.",
                    "direction": "unclear",
                    "claim_type": "association",
                    "evidence_ids": ["openalex:missing"],
                    "candidate_columns": [],
                    "rationale": "none",
                },
                {
                    "name": "Household income",
                    "description": "Duplicate factor.",
                    "direction": "positive",
                    "claim_type": "association",
                    "evidence_ids": ["zenodo:123"],
                    "candidate_columns": [],
                    "rationale": "duplicate",
                },
                {
                    "name": "ocean proximity",
                    "description": "Distance to the ocean.",
                    "direction": "unclear",
                    "claim_type": "descriptive",
                    "evidence_ids": ["zenodo:123"],
                    "candidate_columns": ["ocean_proximity"],
                    "rationale": "location matters",
                },
                {
                    "name": "macro conditions",
                    "description": "Broader economic conditions.",
                    "direction": "unclear",
                    "claim_type": "association",
                    "evidence_ids": ["openalex:W1"],
                    "candidate_columns": [],
                    "rationale": "context",
                },
            ),
            "factor_mapping": _mapping_payload(
                {
                    "factor": "macro conditions",
                    "column": "",
                    "confidence": 0.2,
                    "in_dataset": False,
                    "rationale": "not measured",
                }
            ),
        }
    )
    settings = Settings(embed_dims=2, factor_map_cosine=0.72)
    extraction = extract_factors(
        profile=make_profile(),
        summary="Dataset summary",
        path_names=["Housing", "Housing Markets"],
        records=records,
        llm=llm,
        embeddings=FakeEmbeddings(),
        settings=settings,
    )
    by_name = {factor.name: factor for factor in extraction.factors}
    assert set(by_name) == {"household income", "ocean proximity", "macro conditions"}
    assert by_name["household income"].mapped_column == "median_income"
    assert by_name["household income"].mapping_method == "llm_candidate"
    assert by_name["ocean proximity"].mapped_column == "ocean_proximity"
    assert by_name["macro conditions"].in_dataset is False
    assert any("factors_without_evidence" in warning for warning in extraction.warnings)
    assert llm.calls[0]["purpose"] == "factors"
    assert llm.calls[1]["purpose"] == "factor_mapping"


def test_mapping_excludes_target_and_id_columns() -> None:
    factors = [
        Factor(
            name="house value",
            description="Price of the house.",
            evidence_ids=["openalex:W1"],
            candidate_columns=["median_house_value"],
        )
    ]
    llm = ScriptedLLM(
        {
            "factor_mapping": _mapping_payload(
                {
                    "factor": "house value",
                    "column": "median_house_value",
                    "confidence": 0.9,
                    "in_dataset": True,
                    "rationale": "the target",
                }
            )
        }
    )
    profile = make_profile()
    mapped = map_factors(
        factors,
        columns=[c for c in profile.columns if c.role in ("feature", "unknown")],
        llm=llm,
        embeddings=FakeEmbeddings(),
        settings=Settings(embed_dims=2),
    )
    assert mapped[0].in_dataset is False
    assert mapped[0].mapped_column is None


def test_mapping_llm_failure_falls_back_to_embedding() -> None:
    factors = [
        Factor(
            name="social disadvantage",
            description="Income of households.",
            evidence_ids=["openalex:W1"],
        )
    ]
    llm = ScriptedLLM({"factor_mapping": LLMError("quota", purpose="factor_mapping")})
    warnings: list[str] = []
    mapped = map_factors(
        factors,
        columns=[column for column in make_profile().columns if column.role == "feature"],
        llm=llm,
        embeddings=FakeEmbeddings(),
        settings=Settings(embed_dims=2, factor_map_cosine=0.72),
        warnings=warnings,
    )
    assert mapped[0].in_dataset is True
    assert mapped[0].mapped_column == "median_income"
    assert mapped[0].mapping_method == "embedding"
    assert warnings and "factor_mapping_llm_failed" in warnings[0]


def test_off_prompt_verification_cannot_overwrite_exact_mapping() -> None:
    # FACT-01: only unresolved factors are shown in the prompt. A response
    # naming an already exact-mapped factor must not clear or replace it.
    factors = [
        Factor(
            name="household income",
            description="Income of households.",
            evidence_ids=["openalex:W1"],
            candidate_columns=["median_income"],
        ),
        Factor(
            name="macro conditions",
            description="Broader economic conditions.",
            evidence_ids=["openalex:W1"],
        ),
    ]
    llm = ScriptedLLM(
        {
            "factor_mapping": _mapping_payload(
                {
                    "factor": "household income",
                    "column": "",
                    "confidence": 0.1,
                    "in_dataset": False,
                    "rationale": "off prompt",
                },
                {
                    "factor": "macro conditions",
                    "column": "",
                    "confidence": 0.1,
                    "in_dataset": False,
                    "rationale": "not measured",
                },
            )
        }
    )
    mapped = map_factors(
        factors,
        columns=[column for column in make_profile().columns if column.role == "feature"],
        llm=llm,
        embeddings=FakeEmbeddings(),
        settings=Settings(embed_dims=2, factor_map_cosine=0.72),
    )
    by_name = {factor.name: factor for factor in mapped}

    exact = by_name["household income"]
    assert exact.in_dataset is True
    assert exact.mapped_column == "median_income"
    assert exact.mapping_method == "llm_candidate"
    assert exact.mapping_confidence == 0.9
    assert by_name["macro conditions"].in_dataset is False
    assert by_name["macro conditions"].mapped_column is None


@pytest.mark.parametrize(
    ("plural", "expected"),
    [
        ("house", "house"),
        ("houses", "house"),
        ("class", "class"),
        ("classes", "class"),
        ("analysis", "analysis"),
        ("analyses", "analysis"),
        ("bus", "bus"),
        ("buses", "bus"),
        ("category", "category"),
        ("categories", "category"),
        ("box", "box"),
        ("boxes", "box"),
    ],
)
def test_singular_handles_ambiguous_plurals(plural: str, expected: str) -> None:
    # FACT-02: "-ses" is ambiguous and must not collapse "houses" to "hous".
    assert _singular(plural) == expected


def _column_profile(name: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        role="feature",
        missing_pct=0.0,
        cardinality=100,
        sample_values=[],
        numeric=None,
        top_values=None,
        datetime=False,
    )


def test_description_match_beats_short_name_subset() -> None:
    # FACT-03: a specific description match (0.8) outranks a short partial
    # name match (0.6).
    factor = Factor(
        name="house price",
        description="house size adjusted for inflation",
        evidence_ids=["openalex:W1"],
    )
    mapping = _exact_mapping(
        factor,
        [_column_profile("price"), _column_profile("house_size")],
    )

    assert mapping == ("house_size", "exact", 0.8)


def test_exact_name_match_still_beats_description_match() -> None:
    factor = Factor(
        name="median income",
        description="median house value adjusted for inflation",
        evidence_ids=["openalex:W1"],
    )
    mapping = _exact_mapping(
        factor,
        [_column_profile("price"), _column_profile("median_income")],
    )

    assert mapping == ("median_income", "exact", 1.0)


def test_extract_factors_without_records_is_skipped() -> None:
    extraction = extract_factors(
        profile=make_profile(),
        summary="Dataset summary",
        path_names=["Housing"],
        records=[],
        llm=ScriptedLLM({}),
        embeddings=FakeEmbeddings(),
        settings=Settings(embed_dims=2),
    )
    assert extraction.factors == []
    assert extraction.warnings == [
        "factors_skipped: no research records were available"
    ]


def test_extract_factors_reports_llm_failure() -> None:
    extraction = extract_factors(
        profile=make_profile(),
        summary="Dataset summary",
        path_names=["Housing"],
        records=make_records(),
        llm=ScriptedLLM({"factors": LLMError("quota", purpose="factors")}),
        embeddings=FakeEmbeddings(),
        settings=Settings(embed_dims=2),
    )
    assert extraction.factors == []
    assert any("factor_extraction_failed" in warning for warning in extraction.warnings)
