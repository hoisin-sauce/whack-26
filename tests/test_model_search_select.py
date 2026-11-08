"""Tests for model-search ranking and selection."""

from __future__ import annotations

import math

import pytest

from whack26.model_search.select import (
    confidence_from_result,
    metric_direction,
    parameter_count,
    primary_metric,
    rank_candidates,
    select_models,
    selection_confidence,
)
from whack26.model_search.types import (
    FoldResult,
    ModelCandidate,
    ModelSearchConfig,
    ModelSpec,
)


def make_candidate(
    family: str = "linear",
    *,
    values: tuple[float, ...] = (0.8,),
    metric: str = "r2",
    baseline: float = 0.0,
    baseline_values: tuple[float, ...] | None = None,
    std: float | None = None,
    features: tuple[str, ...] = ("x1",),
    interactions: tuple[tuple[str, str], ...] = (),
    warnings: tuple[str, ...] = (),
    aic: float | None = None,
    bic: float | None = None,
    metrics: dict[str, float] | None = None,
    hyperparameters: dict[str, object] | None = None,
    train_rows: int = 100,
    test_rows: int = 30,
) -> ModelCandidate:
    baselines = baseline_values or tuple(baseline for _ in values)
    folds = [
        FoldResult(
            fold=index,
            train_rows=train_rows,
            test_rows=test_rows,
            metrics={metric: value} if math.isfinite(value) else {metric: value},
            baseline_metrics=(
                {metric: baselines[index]} if index < len(baselines) else {}
            ),
        )
        for index, value in enumerate(values)
    ]
    if metrics is not None:
        candidate_metrics = dict(metrics)
    elif values:
        candidate_metrics = {metric: sum(values) / len(values)}
    else:
        candidate_metrics = {}
    if std is None and len(values) > 1:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
    metric_std = {metric: std} if std is not None else {}
    return ModelCandidate(
        spec=ModelSpec(
            family=family,  # type: ignore[arg-type]
            features=list(features),
            interactions=list(interactions),
            hyperparameters=hyperparameters or {},
        ),
        folds=folds,
        metrics=candidate_metrics,
        metric_std=metric_std,
        aic=aic,
        bic=bic,
        warnings=list(warnings),
    )


def test_rank_by_primary_metric_mean() -> None:
    weak = make_candidate(values=(0.55, 0.50))
    strong = make_candidate(values=(0.90, 0.88))

    result = select_models(
        [weak, strong], target_kind="continuous", n_rows=1000
    )

    assert result.selected is strong
    assert strong.selected is True
    assert weak.selected is False
    assert [candidate.rank for candidate in result.candidates] == [1, 2]
    assert result.candidates == [strong, weak]


def test_lower_is_better_metric_direction() -> None:
    better = make_candidate(
        family="poisson",
        values=(0.5, 0.55),
        metric="poisson_deviance",
        baseline=1.2,
    )
    worse = make_candidate(
        family="poisson",
        values=(0.9, 0.95),
        metric="poisson_deviance",
        baseline=1.2,
    )

    result = select_models(
        [worse, better], target_kind="count", n_rows=1000
    )

    assert result.metric == "poisson_deviance"
    assert result.selected is better
    assert metric_direction("poisson_deviance") == -1
    assert metric_direction("r2") == 1


def test_baseline_margin_rejects_small_improvements() -> None:
    too_small = make_candidate(values=(0.005,), baseline=0.0)
    enough = make_candidate(values=(0.05,), baseline=0.0)
    config = ModelSearchConfig(margin=0.01)

    result = select_models(
        [too_small, enough],
        target_kind="continuous",
        n_rows=1000,
        config=config,
    )

    assert result.selected is enough
    assert too_small not in result.candidates
    assert any("baseline margin not met" in warning for warning in result.warnings)


def test_no_candidate_beats_baseline() -> None:
    weak = make_candidate(values=(0.0,), baseline=0.1)

    result = select_models([weak], target_kind="continuous", n_rows=1000)

    assert result.selected is None
    assert result.candidates == []
    assert "no_candidate_beats_baseline" in result.warnings


def test_stability_breaks_metric_ties() -> None:
    unstable = make_candidate(values=(0.80,), std=0.20)
    stable = make_candidate(values=(0.80,), std=0.01)

    ranked = rank_candidates(
        [unstable, stable], target_kind="continuous", n_rows=1000
    )

    assert ranked == [stable, unstable]
    assert stable.rank == 1
    assert unstable.rank == 2


def test_parsimony_breaks_metric_and_stability_ties() -> None:
    complex_candidate = make_candidate(
        values=(0.80,), std=0.01, features=("x1", "x2", "x3", "x4", "x5")
    )
    simple_candidate = make_candidate(
        values=(0.80,), std=0.01, features=("x1", "x2")
    )

    ranked = rank_candidates(
        [complex_candidate, simple_candidate],
        target_kind="continuous",
        n_rows=10_000,
        complexity_penalty=0.0,
    )

    assert ranked[0] is simple_candidate
    assert parameter_count(simple_candidate.spec) == 3
    assert parameter_count(complex_candidate.spec) == 6


def test_aic_and_bic_break_remaining_ties() -> None:
    first = make_candidate(values=(0.80,), std=0.01, features=("x1",), aic=100, bic=110)
    second = make_candidate(values=(0.80,), std=0.01, features=("x1",), aic=90, bic=105)

    ranked = rank_candidates(
        [first, second], target_kind="continuous", n_rows=1000, complexity_penalty=0.0
    )

    assert ranked == [second, first]


def test_top_n_truncates_and_ranks() -> None:
    candidates = [
        make_candidate(values=(value,)) for value in (0.6, 0.7, 0.8, 0.9, 0.5, 0.4, 0.3)
    ]

    result = select_models(
        candidates, target_kind="continuous", n_rows=1000, top_n=3
    )

    assert len(result.candidates) == 3
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert result.candidates[0].metrics["r2"] == pytest.approx(0.9)
    assert result.selected is result.candidates[0]


def test_target_and_leakage_columns_are_never_features() -> None:
    leaky = make_candidate(values=(0.99,), features=("x1", "target"))
    interaction_leak = make_candidate(
        values=(0.98,), features=("x1",), interactions=(("x1", "leak"),)
    )
    clean = make_candidate(values=(0.60,), features=("x1",))

    result = select_models(
        [leaky, interaction_leak, clean],
        target_kind="continuous",
        n_rows=1000,
        target="target",
        leakage_columns=("leak",),
    )

    assert result.selected is clean
    assert leaky not in result.candidates
    assert interaction_leak not in result.candidates
    assert any("target leakage" in warning for warning in result.warnings)


def test_events_per_variable_parameter_cap() -> None:
    too_complex = make_candidate(
        values=(0.99,), features=tuple(f"x{index}" for index in range(20))
    )
    acceptable = make_candidate(values=(0.60,), features=("x1", "x2", "x3"))

    result = select_models(
        [too_complex, acceptable],
        target_kind="continuous",
        n_rows=100,
    )

    assert result.selected is acceptable
    assert too_complex not in result.candidates
    assert any("parameter cap exceeded" in warning for warning in result.warnings)


def test_nonconverged_and_nonfinite_candidates_are_discarded() -> None:
    nonconverged = make_candidate(
        values=(0.95,), warnings=("optimizer did not converge",)
    )
    nonfinite = make_candidate(
        values=(0.94,), metrics={"r2": float("nan")}
    )
    nonfinite.folds[0].metrics["r2"] = float("nan")
    good = make_candidate(values=(0.60,))

    result = select_models(
        [nonconverged, nonfinite, good], target_kind="continuous", n_rows=1000
    )

    assert result.selected is good
    assert nonconverged not in result.candidates
    assert nonfinite not in result.candidates


def test_baseline_candidate_is_extracted_and_not_ranked() -> None:
    baseline = make_candidate(family="baseline", values=(0.0,), features=())
    model = make_candidate(values=(0.4,), baseline=0.0)

    result = select_models([baseline, model], target_kind="continuous", n_rows=1000)

    assert result.baseline is baseline
    assert baseline not in result.candidates
    assert result.selected is model


def test_selection_confidence_levels() -> None:
    high = make_candidate(values=(0.78, 0.80, 0.82), std=0.02, baseline=0.0)
    high_result = select_models([high], target_kind="continuous", n_rows=1000)
    assert confidence_from_result(high_result) == "high"

    medium = make_candidate(values=(0.6, 1.0), std=0.20, baseline=0.0)
    medium_result = select_models([medium], target_kind="continuous", n_rows=1000)
    assert confidence_from_result(medium_result) == "medium"

    low = make_candidate(values=(0.4, 1.2), std=0.40, baseline=0.0)
    low_result = select_models([low], target_kind="continuous", n_rows=1000)
    assert confidence_from_result(low_result) == "low"

    assert (
        selection_confidence(0.8, 0.01, folds=3, improvement=0.5, margin=0.01)
        == "high"
    )
    assert (
        selection_confidence(0.8, 0.30, folds=1, improvement=0.5, margin=0.01)
        == "low"
    )


def test_primary_metric_uses_target_kind_and_available_metrics() -> None:
    binary = make_candidate(
        family="logistic",
        values=(0.85, 0.9),
        metric="auc",
        baseline=0.5,
    )
    assert primary_metric("binary", [binary]) == "auc"
    assert primary_metric("continuous", []) == "r2"
    assert primary_metric("count", []) == "poisson_deviance"

    with pytest.raises(ValueError):
        primary_metric("continuous", [binary], metric="not_a_metric")  # type: ignore[arg-type]


def test_missing_baseline_does_not_reject() -> None:
    candidate = make_candidate(values=(0.9,))
    candidate.folds[0].baseline_metrics = {}

    result = select_models([candidate], target_kind="continuous", n_rows=1000)

    assert result.selected is candidate
    assert "baseline_missing" in result.warnings
