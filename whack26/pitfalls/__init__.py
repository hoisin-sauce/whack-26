"""Field-scoped pitfall retrieval and extraction."""

from .extract import (
    DatasetSignals,
    applicability,
    build_signals,
    extract_field_pitfalls,
)
from .pipeline import run_field_pitfalls
from .queries import build_field_queries
from .types import (
    APPLICABILITY_VALUES,
    PITFALL_CATEGORIES,
    FieldPitfall,
    FieldPitfallQuery,
    FieldPitfallReport,
)

__all__ = [
    "APPLICABILITY_VALUES",
    "PITFALL_CATEGORIES",
    "DatasetSignals",
    "FieldPitfall",
    "FieldPitfallQuery",
    "FieldPitfallReport",
    "applicability",
    "build_field_queries",
    "build_signals",
    "extract_field_pitfalls",
    "run_field_pitfalls",
]
