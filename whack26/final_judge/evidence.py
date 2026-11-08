"""Build the final judge's evidence bundle from existing stage outputs.

The adapter deliberately leaves a critic out of the bundle when its evidence is
missing.  The orchestrator then reports that critic as unevaluated instead of
asking the LLM to pass something it cannot see.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
)
from .checks import metric_is_higher_better

_TRAIN_REASON = frozenset({"target", "target_copy", "leakage_name"})
_SPLIT_REASON = frozenset({"group_column"})


@dataclass
class ModelFacts:
    target: str | None = None
    target_kind: str | None = None
    metric: str = "r2"
    higher_is_better: bool = True
    selected_family: str | None = None
    selected_features: list[str] = field(default_factory=list)
    selected_metrics: dict[str, float] = field(default_factory=dict)
    metric_std: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    folds: list[Any] = field(default_factory=list)
    diagnostics: list[Any] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    screen: dict[str, Any] | None = None
    dataset_info: dict[str, Any] | None = None
    candidate_count: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    model_description: str
    evidence: dict[str, dict]
    model_summary: dict[str, Any]
    missing_evidence: list[str]
    computed: dict[str, dict] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    assert mean is not None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance**0.5


def _fold_metric(fold: Any, key: str, metric: str) -> float | None:
    metrics = _get(fold, key, {}) or {}
    if not isinstance(metrics, dict):
        return None
    return _number(metrics.get(metric))


def _fold_values(folds: Sequence[Any], key: str, metric: str) -> list[float]:
    values: list[float] = []
    for fold in folds:
        value = _fold_metric(fold, key, metric)
        if value is not None:
            values.append(value)
    return values


def _fold_attr(fold: Any, name: str) -> float | None:
    return _number(_get(fold, name))


def _diagnostic_name(diagnostic: Any) -> str:
    return str(_get(diagnostic, "name", ""))


def _diagnostic_value(diagnostic: Any) -> float | None:
    return _number(_get(diagnostic, "value"))


def _diagnostic_metadata(diagnostic: Any) -> dict[str, Any]:
    metadata = _get(diagnostic, "metadata", {}) or {}
    return metadata if isinstance(metadata, dict) else {}


def _parse_statistic_message(message: str) -> dict[str, Any]:
    """Fallback for diagnostics that only encode raw values in ``message``."""
    output: dict[str, Any] = {}
    match = re.search(r"Q=([0-9.eE+-]+)", message)
    if match:
        output["statistic"] = float(match.group(1))
    match = re.search(r"lag=(\d+)", message)
    if match:
        output["lags"] = int(match.group(1))
    match = re.search(r"LM=([0-9.eE+-]+)", message)
    if match:
        output["statistic"] = float(match.group(1))
    match = re.search(r"df=(\d+)", message)
    if match:
        output["model_df"] = int(match.group(1))
    return output


def _facts_from_model_search(model_search: ModelSearchResult | None) -> ModelFacts:
    if model_search is None:
        return ModelFacts()
    selected: ModelCandidate | None = model_search.selected
    metric = str(model_search.metric or "r2")
    higher = metric_is_higher_better(metric)
    if selected is None:
        return ModelFacts(
            target_kind=str(model_search.target_kind or ""),
            metric=metric,
            higher_is_better=higher,
            screen=_as_dict(getattr(model_search, "screen", None)),
            dataset_info=_as_dict(getattr(model_search, "dataset_info", None)),
            candidate_count=len(model_search.candidates),
            warnings=list(model_search.warnings),
        )
    folds = list(selected.folds)
    baseline_folds = _fold_values(folds, "baseline_metrics", metric)
    baseline_metrics = dict(model_search.baseline.metrics) if model_search.baseline else {}
    if baseline_folds:
        baseline_metrics[metric] = _mean(baseline_folds) or 0.0
    return ModelFacts(
        target_kind=str(model_search.target_kind or ""),
        metric=metric,
        higher_is_better=higher,
        selected_family=selected.spec.family,
        selected_features=list(selected.spec.features),
        selected_metrics=dict(selected.metrics),
        metric_std=dict(selected.metric_std),
        baseline_metrics=baseline_metrics,
        folds=folds,
        diagnostics=list(selected.diagnostics),
        coefficients=dict(selected.coefficients),
        screen=_as_dict(getattr(model_search, "screen", None)),
        dataset_info=_as_dict(getattr(model_search, "dataset_info", None)),
        candidate_count=len(model_search.candidates),
        warnings=list(model_search.warnings),
    )


def _facts_from_artifact(artifact: Any | None) -> ModelFacts:
    if artifact is None:
        return ModelFacts()
    section = getattr(artifact, "model_search", None)
    dataset = getattr(artifact, "dataset", None)
    target = getattr(getattr(dataset, "target", None), "name", None)
    if section is None:
        return ModelFacts(target=target)
    metric = str(getattr(section, "metric", "r2") or "r2")
    candidates = list(getattr(section, "candidates", []) or [])
    selected_payload = next(
        (item for item in candidates if item.get("selected")), {}
    )
    diagnostics = list(selected_payload.get("diagnostics", []) or [])
    return ModelFacts(
        target=target,
        target_kind=str(getattr(section, "target_kind", "") or ""),
        metric=metric,
        higher_is_better=metric_is_higher_better(metric),
        selected_family=getattr(section, "selected_family", None),
        selected_features=list(selected_payload.get("features", []) or []),
        selected_metrics=dict(getattr(section, "selected_metrics", {}) or {}),
        metric_std=dict(selected_payload.get("metric_std", {}) or {}),
        diagnostics=diagnostics,
        candidate_count=int(getattr(section, "candidate_count", 0) or 0),
        warnings=list(getattr(section, "warnings", []) or []),
    )


def _validation_summary(report: Any) -> dict[str, Any]:
    payload = _as_dict(report)
    if not payload:
        return {}
    layers = payload.get("layers") or []
    failed = [
        str(layer.get("layer"))
        for layer in layers
        if isinstance(layer, dict) and layer.get("status") == "failed"
    ]
    return {
        "hypothesis_id": payload.get("hypothesis_id"),
        "backend": payload.get("backend"),
        "status": payload.get("status"),
        "verdict": payload.get("verdict"),
        "failed_layers": failed,
        "p_value": payload.get("p_value", payload.get("permutation_p_value")),
        "dsr_score": payload.get("dsr_score"),
        "oos_score": payload.get("oos_score"),
        "decay_rate": payload.get("decay_rate"),
        "reason": payload.get("reason"),
        "error": payload.get("error"),
        "warnings": list(payload.get("warnings", []) or []),
    }


def _critic_summary(critic: Any) -> dict[str, Any]:
    payload = _as_dict(critic)
    if not payload:
        return {}
    return {
        "hypothesis_id": payload.get("hypothesis_id"),
        "claim_type": payload.get("claim_type"),
        "verdict": payload.get("verdict"),
        "issues": len(payload.get("issues", []) or []),
        "missing_information": list(payload.get("missing_information", []) or []),
        "summary": payload.get("summary", ""),
        "degraded": bool(payload.get("degraded", False)),
    }


def _generation_summary(result: Any) -> dict[str, Any]:
    payload = _as_dict(result)
    if not payload:
        return {}
    return {
        "hypothesis_id": payload.get("hypothesis_id"),
        "verdict": payload.get("verdict"),
        "issues": len(payload.get("issues", []) or []),
        "missing_information": list(payload.get("missing_information", []) or []),
    }


def _validation_reports_from_artifact(artifact: Any | None) -> list[Any]:
    if artifact is None:
        return []
    reports: list[Any] = []
    for hypothesis in getattr(artifact, "hypotheses", []) or []:
        report = getattr(hypothesis, "validation", None)
        if report is not None:
            reports.append(report)
    return reports


def _critics_from_artifact(artifact: Any | None) -> list[Any]:
    if artifact is None:
        return []
    return [
        hypothesis.critic
        for hypothesis in getattr(artifact, "hypotheses", []) or []
        if getattr(hypothesis, "critic", None) is not None
    ]


def _generation_from_artifact(artifact: Any | None) -> list[Any]:
    if artifact is None:
        return []
    return [
        hypothesis.generation_validation
        for hypothesis in getattr(artifact, "hypotheses", []) or []
        if getattr(hypothesis, "generation_validation", None) is not None
    ]


def _dataset_context(
    profile: Any | None,
    artifact: Any | None,
) -> dict[str, Any]:
    if profile is not None:
        target = getattr(profile, "target", None)
        return {
            "rows": getattr(profile, "rows", None),
            "description": getattr(profile, "description", None),
            "target": getattr(target, "name", None),
            "time_column": getattr(profile, "time_column", None),
        }
    dataset = getattr(artifact, "dataset", None) if artifact is not None else None
    if dataset is None:
        return {}
    target = getattr(dataset, "target", None)
    return {
        "rows": getattr(dataset, "rows", None),
        "description": getattr(dataset, "description", None),
        "target": getattr(target, "name", None),
        "time_column": getattr(dataset, "time_column", None),
    }


def _factor_context(
    factors: Sequence[Any],
    artifact: Any | None,
) -> list[dict[str, Any]]:
    values = list(factors)
    if not values and artifact is not None:
        values = list(getattr(artifact, "factors", []) or [])
    output: list[dict[str, Any]] = []
    for factor in values[:20]:
        output.append(
            {
                "name": _get(factor, "name", ""),
                "direction": _get(factor, "direction", "unclear"),
                "mapped_column": _get(factor, "mapped_column", None),
            }
        )
    return output


def _research_context(artifact: Any | None) -> list[dict[str, Any]]:
    if artifact is None:
        return []
    research = getattr(artifact, "research", None)
    records = list(getattr(research, "records", []) or [])
    return [
        {
            "title": getattr(record, "title", ""),
            "year": getattr(record, "year", None),
            "source": getattr(record, "source", ""),
        }
        for record in records[:8]
    ]


def _screen_values(screen: dict[str, Any] | None) -> list[float]:
    if not screen:
        return []
    values: list[float] = []
    for feature in screen.get("features", []) or []:
        if not isinstance(feature, dict):
            continue
        value = feature.get("adjusted_p", feature.get("p_value"))
        number = _number(value)
        if number is not None:
            values.append(number)
    return values


def _feature_entries(
    facts: ModelFacts,
) -> list[dict[str, Any]]:
    dataset_info = facts.dataset_info or {}
    feature_info = {
        str(item.get("name")): item
        for item in dataset_info.get("features", []) or []
        if isinstance(item, dict)
    }
    excluded = dataset_info.get("excluded", {}) or {}
    coefficients = facts.coefficients
    magnitudes = [abs(value) for value in coefficients.values() if value]
    mean = _mean(magnitudes) if magnitudes else None
    std = _std(magnitudes) if magnitudes else None

    entries: list[dict[str, Any]] = []
    # Audit the features the selected model actually uses.  Excluded target or
    # leakage columns are provenance context, not model inputs, so flagging
    # them here would be a false positive.
    for name in facts.selected_features:
        info = feature_info.get(name, {})
        reason = str(excluded.get(name, info.get("reason", "")) or "")
        coefficient = coefficients.get(name)
        zscore = None
        if coefficient is not None and mean is not None and std and std > 0:
            zscore = round((abs(coefficient) - mean) / std, 3)
        entries.append(
            {
                "name": name,
                "selected": name in facts.selected_features,
                "kind": info.get("kind"),
                "derived_from_target": reason in _TRAIN_REASON,
                # A time index is known at prediction time; other provenance
                # is not explicit in the current model-search contract.
                "known_at_prediction_time": True,
                "used_to_define_split": reason in _SPLIT_REASON,
                "importance_zscore": zscore,
                "reason": reason or None,
            }
        )
    return entries


def build_evidence_bundle(
    *,
    model_search: ModelSearchResult | None = None,
    validation_reports: Sequence[Any] = (),
    profile: Any | None = None,
    artifact: Any | None = None,
    critics: Sequence[Any] = (),
    generation_validations: Sequence[Any] = (),
    variants: Any | None = None,
    collection: Any | None = None,
    factors: Sequence[Any] = (),
) -> EvidenceBundle:
    """Translate the run's typed outputs into the branch's evidence bundle."""

    facts = (
        _facts_from_model_search(model_search)
        if model_search is not None
        else _facts_from_artifact(artifact)
    )
    reports = list(validation_reports) or _validation_reports_from_artifact(artifact)
    critic_values = list(critics) or _critics_from_artifact(artifact)
    generation_values = list(generation_validations) or _generation_from_artifact(
        artifact
    )
    dataset = _dataset_context(profile, artifact)
    if facts.target is None:
        facts.target = dataset.get("target")
    if facts.target_kind is None:
        facts.target_kind = dataset.get("target_kind")

    metric = facts.metric
    higher = facts.higher_is_better
    folds = facts.folds
    model_scores = _fold_values(folds, "metrics", metric)
    baseline_scores = _fold_values(folds, "baseline_metrics", metric)
    train_scores = _fold_values(folds, "train_metrics", metric)
    model_mean = _mean(model_scores)
    baseline_mean = _mean(baseline_scores)
    if model_mean is None:
        model_mean = _number(facts.selected_metrics.get(metric))
    if baseline_mean is None:
        baseline_mean = _number(facts.baseline_metrics.get(metric))
    model_sd = _std(model_scores)
    if model_sd is None:
        model_sd = _number(facts.metric_std.get(metric))

    evidence: dict[str, dict] = {}
    missing: list[str] = []
    if not facts.selected_family and not facts.selected_metrics:
        missing.append("selected_model")

    if folds:
        evidence["overfitting_skeptic"] = {
            "metric": metric,
            "higher_is_better": higher,
            "train_score": _mean(train_scores) if train_scores else None,
            "cv_mean": model_mean,
            "cv_sd": model_sd,
            "n_folds": len(folds),
            "learning_curve_present": False,
        }
    else:
        missing.append("fold_metrics")
    if not train_scores:
        missing.append("train_metrics")

    diagnostics = {
        _diagnostic_name(item): item
        for item in facts.diagnostics
        if _diagnostic_name(item)
    }
    residual: dict[str, Any] = {}
    ljung = diagnostics.get("ljung_box")
    if ljung is not None:
        metadata = _diagnostic_metadata(ljung)
        message = str(_get(ljung, "message", ""))
        parsed = _parse_statistic_message(message)
        residual.update(
            {
                "p_value": _diagnostic_value(ljung),
                "statistic": metadata.get("statistic", parsed.get("statistic")),
                "lags": metadata.get("lags", parsed.get("lags")),
                "model_df": metadata.get("model_df", parsed.get("model_df")),
            }
        )
    for name, key in (
        ("durbin_watson", "durbin_watson"),
        ("autocorrelation_lag1", "autocorrelation_lag1"),
        ("breusch_pagan", "breusch_pagan_p"),
        ("white", "white_p"),
    ):
        diagnostic = diagnostics.get(name)
        if diagnostic is not None:
            residual[key] = _diagnostic_value(diagnostic)
    if any(value is not None for value in residual.values()):
        evidence["residual_analyst"] = residual
    else:
        missing.append("residual_diagnostics")

    feature_entries = _feature_entries(facts)
    if feature_entries:
        evidence["leakage_detective"] = {"features": feature_entries}
    else:
        missing.append("feature_provenance")

    tests: dict[str, Any] = {}
    for name, key in (
        ("adf", "adf_p"),
        ("durbin_watson", "durbin_watson"),
        ("breusch_pagan", "breusch_pagan_p"),
        ("white", "white_p"),
        ("vif", "vif_max"),
        ("max_vif", "vif_max"),
        ("condition_number", "condition_number"),
        ("calibration", "calibration_ece"),
        ("overdispersion", "overdispersion"),
    ):
        diagnostic = diagnostics.get(name)
        value = _diagnostic_value(diagnostic) if diagnostic is not None else None
        if value is not None:
            tests[key] = value
    if tests:
        evidence["assumptions_auditor"] = {"tests": tests}
    else:
        missing.append("diagnostic_tests")

    slices = {}
    for report in reports:
        payload = _as_dict(report)
        candidate = payload.get("slices")
        if isinstance(candidate, dict) and candidate:
            slices = candidate
            break
    if slices:
        evidence["subgroup_critic"] = {
            "metric": metric,
            "higher_is_better": higher,
            "slices": slices,
        }
    else:
        missing.append("subgroup_slices")

    p_values = _screen_values(facts.screen)
    if p_values:
        corrected = any(
            isinstance(feature, dict) and feature.get("adjusted_p") is not None
            for feature in (facts.screen or {}).get("features", []) or []
        )
        evidence["multiple_comparisons_skeptic"] = {
            "n_tests": max(len(p_values), facts.candidate_count, 1),
            "min_p": min(p_values),
            "p_is_corrected": corrected,
            "n_looks": 1,
        }
    else:
        missing.append("screening_p_values")

    if folds:
        train_min = [
            value for value in (_fold_attr(fold, "train_target_min") for fold in folds)
            if value is not None
        ]
        train_max = [
            value for value in (_fold_attr(fold, "train_target_max") for fold in folds)
            if value is not None
        ]
        prediction_min = [
            value for value in (_fold_attr(fold, "prediction_min") for fold in folds)
            if value is not None
        ]
        prediction_max = [
            value for value in (_fold_attr(fold, "prediction_max") for fold in folds)
            if value is not None
        ]
        if train_min and train_max and prediction_min and prediction_max:
            evidence["extrapolation_critic"] = {
                "train": {"min": min(train_min), "max": max(train_max)},
                "pred": {"min": min(prediction_min), "max": max(prediction_max)},
            }
        else:
            missing.append("prediction_range")
    else:
        missing.append("prediction_range")

    if folds and model_mean is not None and baseline_mean is not None:
        evidence["simplicity_advocate"] = {
            "metric": metric,
            "higher_is_better": higher,
            "baseline_score": baseline_mean,
            "model_score": model_mean,
            "model_fold_scores": model_scores,
            "baseline_fold_scores": baseline_scores,
            "n_train": _fold_attr(folds[0], "train_rows"),
            "n_test": _fold_attr(folds[0], "test_rows"),
        }
    else:
        missing.append("paired_fold_scores")

    domain = {
        "dataset_description": dataset.get("description"),
        "target": facts.target,
        "target_kind": facts.target_kind,
        "factor_directions": _factor_context(factors, artifact),
        "coefficients": facts.coefficients,
        "research_titles": _research_context(artifact),
        "collection": _as_dict(collection) or _as_dict(
            getattr(artifact, "collection", None) if artifact is not None else None
        ),
    }
    if any(
        value not in (None, "", [], {})
        for value in domain.values()
    ):
        evidence["domain_sanity_critic"] = domain
    else:
        missing.append("domain_context")

    if not reports:
        missing.append("statistical_validation")

    validation_context = [_validation_summary(report) for report in reports]
    critic_context = [_critic_summary(critic) for critic in critic_values]
    generation_context = [
        _generation_summary(result) for result in generation_values
    ]
    variants_context = _as_dict(variants) or _as_dict(
        getattr(artifact, "variants", None) if artifact is not None else None
    )
    collection_context = _as_dict(collection) or _as_dict(
        getattr(artifact, "collection", None) if artifact is not None else None
    )
    context = {
        "dataset": dataset,
        "selected_model": {
            "family": facts.selected_family,
            "features": facts.selected_features,
            "metric": metric,
            "higher_is_better": higher,
            "metrics": facts.selected_metrics,
            "baseline_metrics": facts.baseline_metrics,
            "candidate_count": facts.candidate_count,
            "warnings": facts.warnings,
        },
        "validation": validation_context,
        "method_critic": critic_context,
        "generation_validation": generation_context,
        "variants": variants_context,
        "collection": collection_context,
        "warnings": facts.warnings,
    }
    model_description = json.dumps(context, indent=2, default=str, ensure_ascii=False)

    model_summary = {
        "target": facts.target,
        "target_kind": facts.target_kind,
        "metric": metric,
        "higher_is_better": higher,
        "selected_family": facts.selected_family,
        "selected_features": facts.selected_features,
        "selected_metrics": facts.selected_metrics,
        "baseline_metrics": facts.baseline_metrics,
        "candidate_count": facts.candidate_count,
        "n_folds": len(folds),
        "validation_status": (
            validation_context[0].get("status") if validation_context else None
        ),
        "validation_verdict": (
            validation_context[0].get("verdict") if validation_context else None
        ),
        "warnings": facts.warnings,
    }

    return EvidenceBundle(
        model_description=model_description,
        evidence=evidence,
        model_summary=model_summary,
        missing_evidence=list(dict.fromkeys(missing)),
    )


__all__ = [
    "EvidenceBundle",
    "ModelFacts",
    "build_evidence_bundle",
]
