"""JSON and Markdown rendering for model-search results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .select import confidence_from_result, metric_direction
from .types import Diagnostic, ModelCandidate, ModelSearchResult


def _as_candidate_list(result: ModelSearchResult) -> list[ModelCandidate]:
    if result.candidates:
        return list(result.candidates)
    if result.selected is not None:
        return [result.selected]
    return []


def _primary_value(candidate: ModelCandidate, metric: str) -> float | None:
    value = candidate.metrics.get(metric)
    if value is None or not isinstance(value, (int, float)):
        return None
    return float(value)


def _baseline_delta(candidate: ModelCandidate, result: ModelSearchResult) -> float | None:
    metric = result.metric
    value = _primary_value(candidate, metric)
    if value is None:
        return None

    baseline_values: list[float] = []
    for fold in candidate.folds:
        baseline = fold.baseline_metrics.get(metric)
        if isinstance(baseline, (int, float)):
            baseline_values.append(float(baseline))
    if baseline_values:
        baseline = sum(baseline_values) / len(baseline_values)
    elif result.baseline is not None:
        baseline = _primary_value(result.baseline, metric)
    else:
        baseline = None
    if baseline is None:
        return None
    return metric_direction(metric) * (value - baseline)


def report_payload(result: ModelSearchResult) -> dict[str, Any]:
    """Build the JSON-serializable report payload."""

    payload = result.model_dump(mode="json")
    payload["selection_confidence"] = confidence_from_result(result)
    payload["primary_metric"] = result.metric
    payload["candidates"] = [
        {
            **candidate.model_dump(mode="json"),
            "baseline_delta": _baseline_delta(candidate, result),
        }
        for candidate in _as_candidate_list(result)
    ]
    if result.selected is not None:
        payload["selected"] = {
            **result.selected.model_dump(mode="json"),
            "baseline_delta": _baseline_delta(result.selected, result),
        }
    return payload


def render_json(result: ModelSearchResult, *, indent: int = 2) -> str:
    """Render ``model_search.json`` as text."""

    return json.dumps(report_payload(result), indent=indent, ensure_ascii=False) + "\n"


def _format_number(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    if value != value or value in {float("inf"), float("-inf")}:
        return "—"
    return f"{value:.{digits}f}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _feature_label(candidate: ModelCandidate) -> str:
    features = candidate.spec.features or []
    if not features:
        return "—"
    label = ", ".join(features)
    if candidate.spec.interactions:
        pairs = ", ".join(f"{left}×{right}" for left, right in candidate.spec.interactions)
        label = f"{label}; {pairs}"
    if candidate.spec.target_transform != "none":
        label = f"{label} (y {candidate.spec.target_transform})"
    return _escape(label)


def _diagnostic_flag(candidate: ModelCandidate) -> str:
    flagged = [
        item for item in candidate.diagnostics if item.status in {"warn", "fail"}
    ]
    if not flagged:
        return "—"
    return _escape(", ".join(f"{item.name}:{item.status}" for item in flagged))


def _render_diagnostics(diagnostics: Iterable[Diagnostic], lines: list[str]) -> None:
    rows = list(diagnostics)
    if not rows:
        return
    lines.append("## Diagnostics")
    lines.append("")
    lines.append("| Diagnostic | Status | Value | Message |")
    lines.append("|---|---|---:|---|")
    for item in rows:
        lines.append(
            "| {name} | {status} | {value} | {message} |".format(
                name=_escape(item.name),
                status=item.status,
                value=_format_number(item.value),
                message=_escape(item.message),
            )
        )
    lines.append("")


def render_markdown(result: ModelSearchResult) -> str:
    """Render a human-readable model-search report."""

    confidence = confidence_from_result(result)
    lines: list[str] = [
        "# Model search",
        "",
        (
            f"Target kind: **{result.target_kind}** · "
            f"Primary metric: **{result.metric}** · "
            f"Backend: **{result.backend}** · "
            f"Folds: {result.folds} · Seed: {result.seed}"
        ),
        "",
        f"Selection confidence: **{confidence}**",
        "",
    ]

    candidates = _as_candidate_list(result)
    selected = result.selected
    lines.append("## Selected model")
    lines.append("")
    if selected is None:
        lines.append(
            "No candidate beat the baseline by the configured margin."
        )
    else:
        value = _primary_value(selected, result.metric)
        delta = _baseline_delta(selected, result)
        lines.append(
            f"- Family: **{selected.spec.family}**"
        )
        lines.append(f"- Features: {_feature_label(selected)}")
        lines.append(
            f"- {result.metric}: {_format_number(value)} "
            f"(baseline Δ {_format_number(delta)})"
        )
        if selected.aic is not None or selected.bic is not None:
            lines.append(
                f"- AIC: {_format_number(selected.aic)} · "
                f"BIC: {_format_number(selected.bic)}"
            )
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    if not candidates:
        lines.append("_No eligible candidates._")
        lines.append("")
    else:
        lines.append(
            "| Rank | Family | Features | {metric} | Std | Baseline Δ | AIC | BIC | Diagnostics | Selected |".format(
                metric=result.metric
            )
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|---|")
        for index, candidate in enumerate(candidates, start=1):
            rank = candidate.rank if candidate.rank is not None else index
            value = _primary_value(candidate, result.metric)
            std = candidate.metric_std.get(result.metric)
            delta = _baseline_delta(candidate, result)
            lines.append(
                "| {rank} | {family} | {features} | {value} | {std} | {delta} | {aic} | {bic} | {flags} | {selected} |".format(
                    rank=rank,
                    family=_escape(candidate.spec.family),
                    features=_feature_label(candidate),
                    value=_format_number(value),
                    std=_format_number(
                        float(std) if isinstance(std, (int, float)) else None
                    ),
                    delta=_format_number(delta),
                    aic=_format_number(candidate.aic),
                    bic=_format_number(candidate.bic),
                    flags=_diagnostic_flag(candidate),
                    selected="**yes**" if candidate.selected else "",
                )
            )
        lines.append("")

    if selected is not None and selected.diagnostics:
        _render_diagnostics(selected.diagnostics, lines)

    lines.append("## Warnings")
    lines.append("")
    if result.warnings:
        for warning in result.warnings:
            lines.append(f"- {_escape(warning)}")
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)


def write_json(result: ModelSearchResult, path: str | Path, *, indent: int = 2) -> Path:
    """Write ``model_search.json`` and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(result, indent=indent), encoding="utf-8")
    return output


def write_markdown(result: ModelSearchResult, path: str | Path) -> Path:
    """Write the Markdown report and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(result), encoding="utf-8")
    return output


def write_report(
    result: ModelSearchResult,
    directory: str | Path,
    *,
    stem: str = "model_search",
) -> tuple[Path, Path]:
    """Write both reports into ``directory`` and return ``(json, markdown)``."""

    base = Path(directory)
    json_path = write_json(result, base / f"{stem}.json")
    markdown_path = write_markdown(result, base / f"{stem}.md")
    return json_path, markdown_path


__all__ = [
    "render_json",
    "render_markdown",
    "report_payload",
    "write_json",
    "write_markdown",
    "write_report",
]
