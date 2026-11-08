"""Run the automated regression model search end to end."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..profile import DatasetProfile
from .candidates import generate_candidates
from .data import build_dataset_info
from .fit import fit_model
from .metrics import metrics_for_target
from .select import select_models
from .splits import build_splits
from .types import (
    FoldResult,
    ModelCandidate,
    ModelSearchConfig,
    ModelSearchResult,
    ModelSpec,
)


def _aggregate(folds: list[FoldResult]) -> tuple[dict[str, float], dict[str, float]]:
    if not folds:
        return {}, {}
    names = sorted({name for fold in folds for name in fold.metrics})
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in names:
        values = [
            fold.metrics[name]
            for fold in folds
            if name in fold.metrics and np.isfinite(fold.metrics[name])
        ]
        if not values:
            continue
        means[name] = float(np.mean(values))
        stds[name] = float(np.std(values))
    return means, stds


def _fit_and_score(
    spec: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    splits: Any,
    target_kind: str,
) -> tuple[ModelCandidate | None, list[str]]:
    warnings: list[str] = []
    folds: list[FoldResult] = []
    coefficients: dict[str, float] = {}
    aic_value = None
    bic_value = None
    try:
        for split in splits.splits:
            train = frame.iloc[split.train_index]
            test = frame.iloc[split.test_index]
            fitted = fit_model(spec, train, target)
            predictions = fitted.predict(test)
            probabilities = None
            if target_kind in {"binary", "multiclass"}:
                try:
                    probabilities = fitted.predict_proba(test)
                except Exception:  # noqa: BLE001 - probabilities are optional
                    probabilities = None
            metrics = metrics_for_target(
                target_kind,
                test[target],
                predictions,
                probabilities=probabilities,
                labels=getattr(fitted, "classes", None),
                n_params=int(getattr(fitted, "n_params", 0)) or None,
                log_likelihood=getattr(fitted, "log_likelihood", None),
            )
            baseline_spec = ModelSpec(family="baseline", target_kind=target_kind)  # type: ignore[arg-type]
            baseline_model = fit_model(baseline_spec, train, target)
            baseline_predictions = baseline_model.predict(test)
            baseline_probabilities = None
            if target_kind in {"binary", "multiclass"}:
                try:
                    baseline_probabilities = baseline_model.predict_proba(test)
                except Exception:  # noqa: BLE001
                    baseline_probabilities = None
            baseline_metrics = metrics_for_target(
                target_kind,
                test[target],
                baseline_predictions,
                probabilities=baseline_probabilities,
                labels=getattr(baseline_model, "classes", None),
            )
            folds.append(
                FoldResult(
                    fold=split.fold,
                    train_rows=len(train),
                    test_rows=len(test),
                    metrics=metrics,
                    baseline_metrics=baseline_metrics,
                )
            )
            coefficients.update(getattr(fitted, "coefficients", {}) or {})
            aic_value = metrics.get("aic", aic_value)
            bic_value = metrics.get("bic", bic_value)
    except Exception as exc:  # noqa: BLE001 - one candidate must not stop the search
        return None, [f"candidate_failed:{spec.family}:{type(exc).__name__}: {exc}"]

    means, stds = _aggregate(folds)
    return (
        ModelCandidate(
            spec=spec,
            folds=folds,
            metrics=means,
            metric_std=stds,
            coefficients=coefficients,
            aic=aic_value,
            bic=bic_value,
            warnings=warnings,
        ),
        warnings,
    )


def run_model_search(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
) -> ModelSearchResult:
    """Type, split, screen, fit, and rank candidates for the dataset target."""
    settings = settings or Settings()
    target = profile.target.name if profile.target is not None else None
    if target is None or target not in frame.columns:
        return ModelSearchResult(
            target_kind="continuous",
            warnings=["model_search_skipped:no_target"],
        )
    if len(frame) < getattr(settings, "model_search_min_rows", 50):
        return ModelSearchResult(
            target_kind="continuous",
            warnings=[f"model_search_skipped:too_few_rows:{len(frame)}"],
        )

    time_column = profile.time_column
    info = build_dataset_info(
        frame,
        target,
        time_column=time_column,
        time_aware=bool(time_column),
    )
    splits = build_splits(
        frame,
        target,
        target_kind=info.target.kind,
        n_splits=settings.model_search_folds,
        test_fraction=settings.model_search_test_fraction,
        seed=settings.model_search_seed,
        time_column=time_column,
        time_aware=bool(time_column),
        final_holdout_fraction=settings.model_search_final_holdout_fraction,
        final_holdout_min_rows=200,
    )

    from .screen import screen_features

    screened = screen_features(
        frame,
        target,
        info=info,
        max_features=20,
        seed=settings.model_search_seed,
    )
    feature_kinds = {feature.feature: feature.kind for feature in screened.features}
    feature_scores = {
        feature.feature: abs(feature.pearson or feature.effect_size or 0.0)
        for feature in screened.features
    }
    specs = generate_candidates(
        target=target,
        target_kind=info.target.kind,
        features=screened.selected or info.usable_features,
        feature_kinds=feature_kinds,
        feature_scores=feature_scores,
        frame=frame,
        max_features=10,
        max_candidates=settings.model_search_max_candidates,
    )

    candidates: list[ModelCandidate] = []
    warnings: list[str] = list(splits.warnings) + list(screened.warnings)
    for spec in specs:
        candidate, candidate_warnings = _fit_and_score(
            spec, frame, target, splits, info.target.kind
        )
        warnings.extend(candidate_warnings)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return ModelSearchResult(
            target_kind=info.target.kind,
            candidates=[],
            folds=len(splits.splits),
            seed=settings.model_search_seed,
            warnings=warnings + ["model_search_no_candidates"],
        )

    baseline = next(
        (item for item in candidates if item.spec.family == "baseline"), None
    )
    config = ModelSearchConfig(
        folds=len(splits.splits),
        test_fraction=settings.model_search_test_fraction,
        final_holdout_fraction=settings.model_search_final_holdout_fraction,
        max_candidates=settings.model_search_max_candidates,
        top_n=settings.model_search_top_n,
        seed=settings.model_search_seed,
        backend=settings.model_search_backend,  # type: ignore[arg-type]
    )
    return select_models(
        candidates,
        target_kind=info.target.kind,
        baseline=baseline,
        config=config,
        n_rows=len(frame),
        target=target,
        warnings=warnings,
    )
