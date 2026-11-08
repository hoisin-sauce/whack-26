"""Typed final-judge report.

The final judge is the run-level audit over the selected model, its
statistical validation, and the method-critic results.  It is advisory: it
never re-ranks candidates and never blocks artifact emission.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Severity = Literal["none", "low", "medium", "high", "critical"]
Confidence = Literal["low", "medium", "high"]
FindingStatus = Literal["established", "suspected"]
DefectClass = Literal["model", "evaluation_design", "other"]
Decision = Literal[
    "accepted",
    "accepted_with_concerns",
    "requires_reaudit",
    "rejected",
    "insufficient_evidence",
]

SEVERITIES: tuple[str, ...] = ("none", "low", "medium", "high", "critical")
CONFIDENCES: tuple[str, ...] = ("low", "medium", "high")
STATUSES: tuple[str, ...] = ("established", "suspected")
DEFECT_CLASSES: tuple[str, ...] = ("model", "evaluation_design", "other")


class JudgeFinding(BaseModel):
    """One critic verdict, normalised by code before aggregation."""

    critic: str
    severity: Severity = "none"
    finding: str = ""
    evidence: str = ""
    suggested_fix: str = ""
    falsification_test: str = ""
    confidence: Confidence = "low"
    status: FindingStatus = "suspected"
    could_not_evaluate: bool = False
    defect_class: DefectClass = "other"

    @model_validator(mode="after")
    def _normalise(self) -> "JudgeFinding":
        # A finding inferred from missing information cannot be high-confidence.
        if self.status == "suspected" and self.confidence == "high":
            self.confidence = "medium"
        return self

    @property
    def substantive(self) -> bool:
        return bool(self.finding.strip()) and bool(self.evidence.strip())


class FinalJudgeReport(BaseModel):
    """Serializable final-judge report embedded in the artifact."""

    schema_version: str = "1.0.0"
    generated_at: str = ""
    decision: Decision = "insufficient_evidence"
    model_summary: dict[str, Any] = Field(default_factory=dict)
    model_description: str = ""
    n_critics_evaluated: int = 0
    min_required: int = 0
    code_verified_flags: list[str] = Field(default_factory=list)
    vetoes: list[JudgeFinding] = Field(default_factory=list)
    veto_reason: str | None = None
    pending_reaudit: list[JudgeFinding] = Field(default_factory=list)
    concerns: list[JudgeFinding] = Field(default_factory=list)
    advisories: list[JudgeFinding] = Field(default_factory=list)
    unverified_flags: list[dict[str, Any]] = Field(default_factory=list)
    unevaluated_critics: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    devils_advocate_summary: dict[str, Any] = Field(default_factory=dict)
    reaudit_instructions: list[dict[str, Any]] = Field(default_factory=list)
    critics_run: list[str] = Field(default_factory=list)
    tokens_used: int | None = None
    warnings: list[str] = Field(default_factory=list)
    report_path: str | None = None
    fragment_path: str | None = None
    note: str | None = None


__all__ = [
    "CONFIDENCES",
    "DEFECT_CLASSES",
    "Decision",
    "FinalJudgeReport",
    "FindingStatus",
    "JudgeFinding",
    "SEVERITIES",
    "STATUSES",
    "Severity",
    "Confidence",
    "DefectClass",
]
