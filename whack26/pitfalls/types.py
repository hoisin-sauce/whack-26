"""Typed field-scoped pitfalls.

The field-pitfall stage turns field-specific literature into bounded evidence
records.  A record does not assert that the dataset suffers from the pitfall:
``applies_to_dataset`` records whether the profile shows the precondition, and
``unknown`` means the pitfall is background context only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PitfallCategory = Literal[
    "censoring",
    "selection",
    "aggregation",
    "measurement_error",
    "missing_data",
    "spatial_dependence",
    "temporal_dependence",
    "confounding",
    "multiplicity",
    "rare_group",
    "other",
]

Applicability = Literal["yes", "unknown", "no"]
PitfallScope = Literal["leaf", "ancestor"]

PITFALL_CATEGORIES: tuple[str, ...] = (
    "censoring",
    "selection",
    "aggregation",
    "measurement_error",
    "missing_data",
    "spatial_dependence",
    "temporal_dependence",
    "confounding",
    "multiplicity",
    "rare_group",
    "other",
)

APPLICABILITY_VALUES: tuple[str, ...] = ("yes", "unknown", "no")


class FieldPitfallQuery(BaseModel):
    """One field-scoped search query and the field it belongs to."""

    field_id: str | None = None
    field_name: str
    field_level: int
    source: str
    query: str


class FieldPitfall(BaseModel):
    """One cited, field-scoped caveat."""

    id: str
    field_id: str | None = None
    field_name: str
    field_level: int = 1
    scope: PitfallScope = "ancestor"
    title: str
    pitfall: str
    category: PitfallCategory = "other"
    applies_to_dataset: Applicability = "unknown"
    applicability_reason: str = ""
    check: str = ""
    evidence_id: str
    source: str = ""
    url: str = ""
    year: int | None = None
    relevance: float = 0.0
    confidence: float = 0.0

    @property
    def citation(self) -> str:
        return self.id


class FieldPitfallReport(BaseModel):
    """Serializable result of the field-pitfall stage."""

    schema_version: str = "1.0.0"
    generated_at: str = ""
    field_path: list[str] = Field(default_factory=list)
    queries: list[FieldPitfallQuery] = Field(default_factory=list)
    pitfalls: list[FieldPitfall] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False
    offline: bool = False
    cache: dict[str, Any] = Field(default_factory=dict)

    def by_field(self) -> dict[str, list[FieldPitfall]]:
        """Group pitfalls by field id (or field name when no id exists)."""
        grouped: dict[str, list[FieldPitfall]] = {}
        for pitfall in self.pitfalls:
            key = pitfall.field_id or pitfall.field_name
            grouped.setdefault(key, []).append(pitfall)
        return grouped

    def citations(self) -> list[str]:
        return [pitfall.id for pitfall in self.pitfalls]


__all__ = [
    "APPLICABILITY_VALUES",
    "Applicability",
    "FieldPitfall",
    "FieldPitfallQuery",
    "FieldPitfallReport",
    "PITFALL_CATEGORIES",
    "PitfallCategory",
    "PitfallScope",
]
