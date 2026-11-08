"""Run the automated regression model search end to end."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..profile import DatasetProfile
from .candidates import generate_candidates
from .data import build_dataset_info
from .fit import (
    BackendUnavailableError,
    FittedModel,
    fit_model,
    optional_fitter,
    register_fitter,
    unregister_fitter,
)
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


def _sindy_forecast(fitted: Any, last_state: float, steps: int) -> np.ndarray:
    """Simulate ``steps`` values after the training series."""
    count = max(0, int(steps))
    if count == 0:
        return np.empty(0, dtype=float)
    times = np.arange(1, count + 1, dtype=float)
    try:
        values = fitted.predict(initial=np.array([last_state]), times=times)
        forecast = np.asarray(values, dtype=float).reshape(-1)
        if forecast.size >= count:
            return forecast[:count]
    except Exception:  # noqa: BLE001 - fall back to a flat forecast
        pass
    return np.full(count, last_state, dtype=float)


def _register_optional_fitters(backend: str) -> dict[str, Any]:
    """Register sklearn and pysindy adapters and return the previous registry."""
    backend_name = str(backend or "core").strip().lower()
    previous: dict[str, Any] = {}

    def _register(family: str, fitter: Any) -> None:
        previous.setdefault(family, optional_fitter(family))
        register_fitter(family, fitter)
    if backend_name in {"sklearn", "full"}:
        try:
            from .trees import fit_tree, sklearn_available

            if sklearn_available():

                def _tree_fitter(
                    spec: ModelSpec,
                    frame: pd.DataFrame,
                    target: str,
                    *,
                    sample_weight: Any = None,
                ) -> FittedModel:
                    data = frame.dropna(subset=[target])
                    if data.empty:
                        raise BackendUnavailableError(
                            "no rows with a non-missing target"
                        )
                    features = data.drop(columns=[target])
                    values = data[target]
                    fitted = fit_tree(spec, features, values)
                    if fitted is None:
                        raise BackendUnavailableError(
                            "scikit-learn is not available"
                        )
                    classes = (
                        list(pd.unique(values))
                        if spec.target_kind in {"binary", "multiclass"}
                        else None
                    )
                    return FittedModel(
                        spec=spec,
                        target=target,
                        n_params=max(1, len(spec.features) + 1),
                        classes=classes,
                        _predict=lambda frame: fitted.predict(frame),
                        _predict_proba=(
                            lambda frame: fitted.predict_proba(frame)
                            if hasattr(fitted, "predict_proba")
                            else None
                        ),
                    )

                for family in ("tree", "forest", "gradient_boosting"):
                    _register(family, _tree_fitter)
        except Exception:  # noqa: BLE001 - optional backend must not break core
            pass

    if backend_name == "full":
        try:
            from .sindy import fit_sindy, pysindy_available

            if pysindy_available():

                def _sindy_fitter(
                    spec: ModelSpec,
                    frame: pd.DataFrame,
                    target: str,
                    *,
                    sample_weight: Any = None,
                ) -> FittedModel:
                    data = frame.dropna(subset=[target])
                    values = pd.to_numeric(data[target], errors="coerce").dropna()
                    if len(values) < 5:
                        raise BackendUnavailableError(
                            "SINDy needs at least five training rows"
                        )
                    matrix = values.to_numpy(dtype=float).reshape(-1, 1)
                    times = np.arange(len(matrix), dtype=float)
                    fitted = fit_sindy(
                        spec,
                        matrix,
                        times,
                        feature_names=[target],
                    )
                    if fitted is None:
                        raise BackendUnavailableError("pysindy is not available")
                    last_state = float(matrix[-1, 0])
                    return FittedModel(
                        spec=spec,
                        target=target,
                        n_params=max(
                            1, len(fitted.coefficient_table()) or 1
                        ),
                        warnings=["sindy_forecast_from_training_end_state"],
                        _predict=lambda frame: _sindy_forecast(
                            fitted, last_state, len(frame)
                        ),
                    )

                _register("sindy", _sindy_fitter)
        except Exception:  # noqa: BLE001 - optional backend must not break core
            pass
    return previous


def _restore_optional_fitters(previous: dict[str, Any]) -> None:
    for family, fitter in previous.items():
        if fitter is None:
            unregister_fitter(family)
        else:
            register_fitter(family, fitter)


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
            train_predictions = fitted.predict(train)
            train_probabilities = None
            if target_kind in {"binary", "multiclass"}:
                try:
                    train_probabilities = fitted.predict_proba(train)
                except Exception:  # noqa: BLE001 - probabilities are optional
                    train_probabilities = None
            train_metrics = metrics_for_target(
                target_kind,
                train[target],
                train_predictions,
                probabilities=train_probabilities,
                labels=getattr(fitted, "classes", None),
                n_params=int(getattr(fitted, "n_params", 0)) or None,
                log_likelihood=getattr(fitted, "log_likelihood", None),
            )
            train_target = pd.to_numeric(train[target], errors="coerce")
            train_target_min = (
                float(train_target.min()) if train_target.notna().any() else None
            )
            train_target_max = (
                float(train_target.max()) if train_target.notna().any() else None
            )
            try:
                numeric_predictions = np.asarray(predictions, dtype=float)
                prediction_min = (
                    float(np.nanmin(numeric_predictions))
                    if np.isfinite(numeric_predictions).any()
                    else None
                )
                prediction_max = (
                    float(np.nanmax(numeric_predictions))
                    if np.isfinite(numeric_predictions).any()
                    else None
                )
            except (TypeError, ValueError):
                prediction_min = None
                prediction_max = None
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
                    train_metrics=train_metrics,
                    train_target_min=train_target_min,
                    train_target_max=train_target_max,
                    prediction_min=prediction_min,
                    prediction_max=prediction_max,
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
    previous_fitters = _register_optional_fitters(settings.model_search_backend)
    specs = generate_candidates(
        target=target,
        target_kind=info.target.kind,
        features=screened.selected or info.usable_features,
        feature_kinds=feature_kinds,
        feature_scores=feature_scores,
        frame=frame,
        max_features=10,
        max_candidates=settings.model_search_max_candidates,
        backend=settings.model_search_backend,
    )

    candidates: list[ModelCandidate] = []
    warnings: list[str] = list(splits.warnings) + list(screened.warnings)
    try:
        for spec in specs:
            candidate, candidate_warnings = _fit_and_score(
                spec, frame, target, splits, info.target.kind
            )
            warnings.extend(candidate_warnings)
            if candidate is not None:
                candidates.append(candidate)
    finally:
        _restore_optional_fitters(previous_fitters)

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
    result = select_models(
        candidates,
        target_kind=info.target.kind,
        baseline=baseline,
        config=config,
        n_rows=len(frame),
        target=target,
        warnings=warnings,
    )
    result.dataset_info = info.model_dump(mode="json")
    result.screen = screened.model_dump(mode="json")
    return result
