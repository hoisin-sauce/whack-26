"""Typed outputs shared by the deterministic checks and the LLM critic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["fatal", "major", "minor"]
Verdict = Literal["pass", "refine", "reject", "insufficient"]
ClaimType = Literal["description", "prediction", "causal", "explanatory"]
CheckStatus = Literal["pass", "flag", "fail", "not_applicable"]


class Issue(BaseModel):
    chunk: str
    severity: Severity
    finding: str
    evidence: str = ""
    why: str = ""
    fix: str = ""


class Check(BaseModel):
    id: str
    status: CheckStatus
    severity: Literal["fatal", "major", "minor", "none"] = "none"
    reason: str = ""
    citations: list[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    hypothesis_id: str
    claim_type: ClaimType = "description"
    verdict: Verdict = "insufficient"
    issues: list[Issue] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    summary: str = ""
    retrieved_chunks: list[str] = Field(default_factory=list)
    model: str = "unknown"
    corpus_version: str = "unknown"
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
