"""Typed configuration and reports for statistical model validation.

The validation stage consumes a generated hypothesis after it has been
executed. It is deliberately independent of the method critic: the critic
audits the derivation method, while this module reports what the four-layer
engine in ``stats_engine.py`` found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

ValidationStatus = Literal["passed", "failed", "partial", "skipped", "error"]
ValidationVerdict = Literal["pass", "fail", "inconclusive"]
LayerStatus = Literal["passed", "failed", "skipped", "error"]

DEFAULT_MIN_DSR = 0.95
DEFAULT_MAX_P = 0.05
DEFAULT_TRAIN_SPLIT = 0.70
DEFAULT_NUM_PERMUTATIONS = 500
DEFAULT_SENSITIVITY_RANGE = 0.10
DEFAULT_MAX_PERFORMANCE_DROP = 0.20
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_PERIODS_PER_YEAR = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds and runtime limits for one validation pass.

    These are validation settings, not model hyperparameters. The engine
    perturbs only the numeric entries in the generated module's ``PARAMETERS``
    dict; everything here is copied into the engine's reserved config keys and
    is never perturbed.
    """

    min_dsr: float = DEFAULT_MIN_DSR
    max_p: float = DEFAULT_MAX_P
    train_split_ratio: float = DEFAULT_TRAIN_SPLIT
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS
    sensitivity_range: float = DEFAULT_SENSITIVITY_RANGE
    max_performance_drop: float = DEFAULT_MAX_PERFORMANCE_DROP
    max_decay_tolerance: float | None = None
    periods_per_year: float | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    seed: int = 0
    max_features: int = 10
    max_categories: int = 20
    max_ece: float | None = None
    max_dispersion: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_dsr <= 1.0:
            raise ValueError("min_dsr must be between 0 and 1")
        if not 0.0 < self.max_p <= 1.0:
            raise ValueError("max_p must be between 0 and 1")
        if not 0.0 < self.train_split_ratio < 1.0:
            raise ValueError("train_split_ratio must be between 0 and 1")
        if self.num_permutations < 1:
            raise ValueError("num_permutations must be positive")
        if self.sensitivity_range <= 0.0:
            raise ValueError("sensitivity_range must be positive")
        if self.max_performance_drop < 0.0:
            raise ValueError("max_performance_drop must be non-negative")
        if self.max_ece is not None and not 0.0 < self.max_ece <= 1.0:
            raise ValueError("max_ece must be between 0 and 1")
        if self.max_dispersion is not None and not (
            math.isfinite(self.max_dispersion) and self.max_dispersion >= 0.0
        ):
            raise ValueError("max_dispersion must be non-negative")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2")
        if self.periods_per_year is not None and self.periods_per_year <= 0.0:
            raise ValueError("periods_per_year must be positive when set")

    def engine_params(self, parameters: dict[str, float]) -> dict[str, Any]:
        """Merge real model hyperparameters with reserved engine settings."""
        params: dict[str, Any] = dict(parameters)
        params.update(
            {
                "min_dsr_limit": self.min_dsr,
                "max_p_value": self.max_p,
                "train_split_ratio": self.train_split_ratio,
                "num_permutations": self.num_permutations,
                "sensitivity_range": self.sensitivity_range,
                "max_performance_drop": self.max_performance_drop,
            }
        )
        if self.max_decay_tolerance is not None:
            params["max_decay_tolerance"] = self.max_decay_tolerance
        return params

    def engine_options(self) -> dict[str, Any]:
        """Return keyword options for ``stats_engine.validate_model``.

        These are engine thresholds, not model hyperparameters. The
        calibration and dispersion gates live here rather than in
        :meth:`engine_params` so parameter-sensitivity testing never
        perturbs a validation threshold.
        """
        return {
            "min_dsr_limit": self.min_dsr,
            "max_p_value": self.max_p,
            "train_split_ratio": self.train_split_ratio,
            "num_permutations": self.num_permutations,
            "sensitivity_range": self.sensitivity_range,
            "max_performance_drop": self.max_performance_drop,
            "max_decay_tolerance": self.max_decay_tolerance,
            "max_ece": self.max_ece,
            "max_dispersion": self.max_dispersion,
            "permutation_seed": self.seed,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_dsr": self.min_dsr,
            "max_p": self.max_p,
            "train_split_ratio": self.train_split_ratio,
            "num_permutations": self.num_permutations,
            "sensitivity_range": self.sensitivity_range,
            "max_performance_drop": self.max_performance_drop,
            "max_ece": self.max_ece,
            "max_dispersion": self.max_dispersion,
            "max_decay_tolerance": self.max_decay_tolerance,
            "periods_per_year": self.periods_per_year,
            "timeout_seconds": self.timeout_seconds,
            "seed": self.seed,
            "max_features": self.max_features,
            "max_categories": self.max_categories,
        }


class ValidationLayerReport(BaseModel):
    layer: str
    status: LayerStatus
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """One hypothesis's validation result, embedded in the artifact."""

    hypothesis_id: str
    status: ValidationStatus = "error"
    verdict: ValidationVerdict = "inconclusive"
    target: str | None = None
    rows_used: int = 0
    dropped_rows: int = 0
    feature_columns: list[str] = Field(default_factory=list)
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR
    periods_per_year_source: str = "default"
    in_sample_fallback: bool = False
    has_fit_predict: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    layers: list[ValidationLayerReport] = Field(default_factory=list)
    dsr_score: float | None = None
    p_value: float | None = None
    oos_score: float | None = None
    decay_rate: float | None = None
    tolerance: float | None = None
    pss_message: str | None = None
    runtime_ms: int = 0
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def layer(self, name: str) -> ValidationLayerReport | None:
        for item in self.layers:
            if item.layer == name:
                return item
        return None

    def failed_layers(self) -> list[str]:
        return [item.layer for item in self.layers if item.status == "failed"]

    def skipped_layers(self) -> list[str]:
        return [item.layer for item in self.layers if item.status == "skipped"]


@dataclass
class ValidationRequest:
    """Inputs passed to the isolated validation runner."""

    hypothesis_id: str
    module_path: str
    dataset_path: str
    predictions_path: str | None
    target: str | None
    required_columns: list[str] = field(default_factory=list)
    optional_columns: list[str] = field(default_factory=list)
    time_column: str | None = None
    target_kind: str | None = None
    description: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "module_path": self.module_path,
            "dataset_path": self.dataset_path,
            "predictions_path": self.predictions_path,
            "target": self.target,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "time_column": self.time_column,
            "target_kind": self.target_kind,
            "description": self.description,
        }


def summary_counts(reports: list[ValidationReport]) -> dict[str, int]:
    counts = {
        "passed": 0,
        "failed": 0,
        "partial": 0,
        "skipped": 0,
        "error": 0,
    }
    for report in reports:
        counts[report.status] = counts.get(report.status, 0) + 1
    return counts


def error_report(
    hypothesis_id: str,
    message: str,
    *,
    runtime_ms: int = 0,
) -> ValidationReport:
    return ValidationReport(
        hypothesis_id=hypothesis_id,
        status="error",
        verdict="inconclusive",
        runtime_ms=runtime_ms,
        error=message,
        layers=[
            ValidationLayerReport(
                layer="Validation",
                status="error",
                reason=message,
            )
        ],
    )


def skipped_report(
    hypothesis_id: str,
    reason: str,
    *,
    target: str | None = None,
    warnings: list[str] | None = None,
) -> ValidationReport:
    return ValidationReport(
        hypothesis_id=hypothesis_id,
        status="skipped",
        verdict="inconclusive",
        target=target,
        warnings=list(warnings or []),
        layers=[
            ValidationLayerReport(
                layer="Target contract" if reason.startswith("target_contract") else "Validation",
                status="skipped",
                reason=reason,
            )
        ],
    )
