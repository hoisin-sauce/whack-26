"""New validation models that extend the existing statistical ValidationReport."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .models import ValidationReport

Design = Literal[
    "observational", "experimental", "quasi_experimental", "survey",
    "administrative", "other",
]
GenerationVerdict = Literal["pass", "refine", "reject", "insufficient"]


class CollectionMetadata(BaseModel):
    design: Design = "observational"
    unit_of_analysis: str | None = None
    sampling: str | None = None
    time_period: str | None = None
    data_source: str | None = None
    collection_method: str | None = None
    known_biases: list[str] = Field(default_factory=list)
    measurement_notes: list[str] = Field(default_factory=list)
    treatment: str | None = None
    instrument: str | None = None
    running_variable: str | None = None
    identification_strategy: str | None = None
    ethical_constraints: list[str] = Field(default_factory=list)


class ValidationCheck(BaseModel):
    id: str
    status: Literal["pass", "flag", "fail", "not_applicable"] = "pass"
    severity: Literal["none", "minor", "major", "fatal"] = "none"
    reason: str = ""
    citations: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    chunk: str = "S0-how-to-use"
    severity: Literal["minor", "major", "fatal"] = "minor"
    finding: str
    evidence: str = ""
    why: str = ""
    fix: str = ""


class GenerationValidationResult(BaseModel):
    hypothesis_id: str
    verdict: GenerationVerdict = "insufficient"
    checks: list[ValidationCheck] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""
    model: str = ""
    corpus_version: str = ""


class ValidationBundle(BaseModel):
    hypothesis_id: str
    generation: GenerationValidationResult | None = None
    statistical: ValidationReport | None = None
    backend: str = "in_slice"
    warnings: list[str] = Field(default_factory=list)
