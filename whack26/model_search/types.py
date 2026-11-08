"""Shared types for automated regression model search."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TargetKind = Literal["continuous", "binary", "multiclass", "count", "time_series"]
FeatureKind = Literal["numeric", "categorical", "datetime", "id", "leakage_risk"]
ModelFamily = Literal[
    "baseline",
    "linear",
    "logistic",
    "multinomial",
    "poisson",
    "negative_binomial",
    "interaction",
    "group_comparison",
    "tree",
    "forest",
    "gradient_boosting",
    "sindy",
]
MetricName = Literal[
    "r2",
    "rmse",
    "mae",
    "accuracy",
    "balanced_accuracy",
    "auc",
    "log_loss",
    "poisson_deviance",
    "aic",
    "bic",
]


class ModelSpec(BaseModel):
    family: ModelFamily
    features: list[str] = Field(default_factory=list)
    target_transform: Literal["none", "log1p", "sqrt"] = "none"
    feature_transforms: dict[str, str] = Field(default_factory=dict)
    interactions: list[tuple[str, str]] = Field(default_factory=list)
    regularization: float | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    target_kind: TargetKind = "continuous"


class FoldResult(BaseModel):
    fold: int
    train_rows: int
    test_rows: int
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)


class Diagnostic(BaseModel):
    name: str
    status: Literal["ok", "warn", "fail", "not_applicable"] = "ok"
    value: float | None = None
    message: str = ""


class ModelCandidate(BaseModel):
    spec: ModelSpec
    folds: list[FoldResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    metric_std: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    coefficients: dict[str, float] = Field(default_factory=dict)
    aic: float | None = None
    bic: float | None = None
    rank: int | None = None
    selected: bool = False
    warnings: list[str] = Field(default_factory=list)


class ModelSearchResult(BaseModel):
    target_kind: TargetKind
    metric: MetricName = "r2"
    candidates: list[ModelCandidate] = Field(default_factory=list)
    selected: ModelCandidate | None = None
    baseline: ModelCandidate | None = None
    folds: int = 3
    seed: int = 0
    backend: str = "core"
    warnings: list[str] = Field(default_factory=list)


class ModelSearchConfig(BaseModel):
    folds: int = 3
    test_fraction: float = 0.30
    final_holdout_fraction: float = 0.20
    final_holdout_min_rows: int = 200
    max_candidates: int = 100
    top_n: int = 5
    min_rows: int = 50
    margin: float = 0.01
    seed: int = 0
    backend: Literal["core", "sklearn", "statsmodels", "full"] = "core"
