"""Shared test fakes for the parallel implementation packages.

These are deliberately small and dependency-free so every work package can run
unit tests before the schema or another package lands.
"""

from __future__ import annotations

from typing import Any

from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)
from whack26.research.scrape.models import Document
from whack26.validation.generation_models import (
    CollectionMetadata,
    GenerationValidationResult,
    ValidationBundle,
)
from whack26.validation.models import (
    ValidationLayerReport,
    ValidationReport,
)


def make_document(
    identifier: str = "doc_test",
    *,
    kind: str = "paper",
    title: str = "A test document",
    summary: str = "A test summary.",
    embedding: list[float] | None = None,
    credible: bool = True,
) -> Document:
    return Document(
        id=identifier,
        kind=kind,  # type: ignore[arg-type]
        source="openalex",
        title=title,
        summary=summary,
        summary_embedding=embedding or [1.0, 0.0],
        embedding_model="fake@2",
        credible=credible,
        credibility_score=0.8 if credible else 0.2,
    )


def make_model_search_result(*, selected_family: str = "linear") -> ModelSearchResult:
    spec = ModelSpec(family=selected_family)  # type: ignore[arg-type]
    candidate = ModelCandidate(
        spec=spec,
        folds=[
            FoldResult(
                fold=0,
                train_rows=70,
                test_rows=30,
                metrics={"r2": 0.8},
                baseline_metrics={"r2": 0.0},
            )
        ],
        metrics={"r2": 0.8},
        metric_std={"r2": 0.02},
        diagnostics=[Diagnostic(name="residuals", status="ok")],
        selected=True,
    )
    return ModelSearchResult(
        target_kind="continuous",
        metric="r2",
        candidates=[candidate],
        selected=candidate,
        baseline=ModelCandidate(spec=ModelSpec(family="baseline")),
    )


def make_validation_report(
    hypothesis_id: str = "hyp_test",
    *,
    status: str = "passed",
    verdict: str = "pass",
) -> ValidationReport:
    return ValidationReport(
        hypothesis_id=hypothesis_id,
        status=status,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        layers=[
            ValidationLayerReport(
                layer="Layer 1 (DSR)",
                status="passed",
                reason="fake",
            )
        ],
    )


def make_generation_result(
    hypothesis_id: str = "hyp_test",
    *,
    verdict: str = "pass",
) -> GenerationValidationResult:
    return GenerationValidationResult(
        hypothesis_id=hypothesis_id,
        verdict=verdict,  # type: ignore[arg-type]
        summary="Fake generation validation.",
    )


def make_validation_bundle(
    hypothesis_id: str = "hyp_test",
) -> ValidationBundle:
    return ValidationBundle(
        hypothesis_id=hypothesis_id,
        generation=make_generation_result(hypothesis_id),
        statistical=make_validation_report(hypothesis_id),
    )


def make_collection_metadata(**overrides: Any) -> CollectionMetadata:
    values: dict[str, Any] = {
        "design": "observational",
        "unit_of_analysis": "block group",
        "sampling": "census",
    }
    values.update(overrides)
    return CollectionMetadata(**values)
