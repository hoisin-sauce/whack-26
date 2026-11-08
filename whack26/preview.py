"""Markdown preview rendered from `artifact.json`.

The preview reads only the artifact plus the run-relative files it references
(generated code). It never reads the dataset, the field store, or the cache.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .artifact import (
    Artifact,
    CriticSection,
    DatasetColumn,
    HypothesisSection,
    ResearchRecordSection,
)
from .common import redact_secrets
from .research.scrape.models import ScrapeResult

_MAX_RESEARCH_RECORDS = 10
_MAX_COLUMNS = 12
_MAX_MISSING_COLUMNS = 6
_MAX_EVIDENCE_IDS = 4


def _text(value: Any) -> str:
    """Collapse a value to one safe Markdown table/list fragment."""
    return " ".join(str(value).split()).replace("|", "\\|")


def _number(value: Any, digits: int = 4) -> str:
    if value is None or isinstance(value, bool):
        return "n/a" if value is None else str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value)
    if not math.isfinite(number):
        return "n/a"
    text = f"{number:,.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def _seconds(started_at: str, finished_at: str) -> str:
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    seconds = max(0, int((end - start).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _code_text(base_dir: Path, code_path: str) -> str:
    root = base_dir.resolve()
    candidate = (base_dir / code_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return "(code path escapes the run directory)"
    try:
        return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(code file unavailable)"


def _fence(code: str) -> str:
    fence = "```"
    while fence in code:
        fence += "`"
    return fence


def _sanitize(value: str, base_dir: Path) -> str:
    cleaned = redact_secrets(value)
    for variant in {str(base_dir), str(base_dir).replace("\\", "/")}:
        cleaned = cleaned.replace(variant, ".")
    return cleaned


def _bullets(items: Iterable[str], *, empty: str = "None.") -> list[str]:
    rendered = [f"- {_text(item)}" for item in items if _text(item)]
    return rendered or [empty]


def _dataset_section(artifact: Artifact) -> list[str]:
    dataset = artifact.dataset
    lines = ["## Dataset", ""]
    description = (dataset.description or "").strip()
    if description:
        lines.extend([_text(description), ""])
    lines.append(f"- Shape: {dataset.rows:,} rows x {len(dataset.columns)} columns")
    lines.append(f"- Format: {_text(dataset.format)}")
    lines.append(f"- SHA-256: `{dataset.sha256}`")
    if dataset.target is not None:
        lines.append(
            f"- Target: `{dataset.target.name}` "
            f"(source: {_text(dataset.target.source)}, "
            f"confidence: {_number(dataset.target.confidence, 2)})"
        )
    else:
        lines.append("- Target: none (descriptive run)")
    lines.append(
        f"- Time column: `{dataset.time_column}`"
        if dataset.time_column
        else "- Time column: none"
    )

    target = artifact.screen.target_summary
    if target:
        kind = target.get("kind", "none")
        if kind == "numeric" and target.get("stats"):
            stats = target["stats"]
            lines.append(
                "- Target distribution: numeric | "
                f"mean {_number(stats.get('mean'))}, "
                f"std {_number(stats.get('std'))}, "
                f"median {_number(stats.get('median'))}, "
                f"min {_number(stats.get('min'))}, "
                f"max {_number(stats.get('max'))}"
            )
        elif kind == "categorical" and target.get("top_values"):
            top = ", ".join(
                f"{_text(label)} ({count})"
                for label, count in target["top_values"][:8]
            )
            lines.append(f"- Target distribution: categorical | top: {top}")
        lines.append(
            f"- Target missing: {_number(target.get('missing'))} "
            f"({_number(target.get('missing_pct'), 2)}%)"
        )

    missing = sorted(
        artifact.screen.missingness.items(), key=lambda item: -item[1]
    )[:_MAX_MISSING_COLUMNS]
    if missing:
        lines.append(
            "- Missingness: "
            + ", ".join(
                f"`{_text(name)}` {_number(pct, 2)}%" for name, pct in missing
            )
        )
    else:
        lines.append("- Missingness: none")

    if artifact.screen.time_trend:
        trend = artifact.screen.time_trend
        lines.append(
            f"- Time trend: `{_text(trend.get('column'))}` "
            f"({_text(trend.get('period'))}, {_text(trend.get('first'))} "
            f"to {_text(trend.get('last'))})"
        )

    if artifact.screen.top_correlations:
        pairs = "; ".join(
            f"`{_text(pair.a)}` ~ `{_text(pair.b)}`: "
            f"r={_number(pair.pearson)}, rho={_number(pair.spearman)}"
            for pair in artifact.screen.top_correlations[:5]
        )
        lines.append(f"- Top correlations (descriptive): {pairs}")

    lines.extend(["", "### Columns", ""])
    lines.append("| Column | Type | Role | Missing % | Unique |")
    lines.append("|---|---|---|---:|---:|")
    for column in dataset.columns[:_MAX_COLUMNS]:
        lines.append(_column_row(column))
    if len(dataset.columns) > _MAX_COLUMNS:
        lines.append("")
        lines.append(
            f"_{len(dataset.columns) - _MAX_COLUMNS} more columns are recorded in the artifact._"
        )
    lines.append("")
    return lines


def _column_row(column: DatasetColumn) -> str:
    return (
        f"| `{_text(column.name)}` | {_text(column.dtype)} | "
        f"{_text(column.role)} | {_number(column.missing_pct, 2)} | "
        f"{column.cardinality:,} |"
    )


def _field_path_section(artifact: Artifact) -> list[str]:
    classification = artifact.classification
    lines = ["## Field path", ""]
    if not classification.path:
        return lines + ["No field path was recorded.", ""]
    breadcrumb = " \u2192 ".join(
        f"`{_text(item.name)}`" for item in classification.path
    )
    lines.append(f"**{breadcrumb}**")
    lines.append("")
    lines.append(
        f"- Confidence: {_number(classification.confidence, 3)} | "
        f"created fields: {len(classification.created_fields)} | "
        f"refined after research: {'yes' if classification.refined_after_research else 'no'}"
    )
    if classification.refinement_reason:
        lines.append(f"- Refinement: {_text(classification.refinement_reason)}")
    lines.append("")
    for item in classification.path:
        lines.append(f"- **{_text(item.name)}** (level {item.level}): {_text(item.description)}")
    lines.append("")
    return lines


def _record_line(record: ResearchRecordSection) -> str:
    year = record.year if record.year is not None else "n.d."
    link = f"[{_text(record.url)}]({record.url})" if record.url else "no link"
    relevance = f"relevance {_number(record.relevance, 3)}"
    if record.query:
        relevance += f', query "{_text(record.query)}"'
    return (
        f"- **{_text(record.title)}** ({year}, {_text(record.source)}, "
        f"{relevance}) \u2014 {link}"
    )


def _research_section(artifact: Artifact) -> list[str]:
    research = artifact.research
    lines = ["## Research", ""]
    flags = []
    if research.limited_mode:
        flags.append("limited mode")
    if research.offline:
        flags.append("offline")
    if flags:
        lines.append(f"_Run flags: {', '.join(flags)}._")
        lines.append("")
    lines.append(
        f"{len(research.records)} record(s) from {len(research.queries)} quer(ies)."
    )
    if research.queries:
        lines.append("")
        for query in research.queries:
            lines.append(f"- `{_text(query.source)}`: \u201c{_text(query.query)}\u201d")
    lines.append("")
    lines.extend(["### Top records", ""])
    if not research.records:
        lines.append("No research records were retrieved.")
    else:
        for record in research.records[:_MAX_RESEARCH_RECORDS]:
            lines.append(_record_line(record))
        if len(research.records) > _MAX_RESEARCH_RECORDS:
            lines.append("")
            lines.append(
                f"_{len(research.records) - _MAX_RESEARCH_RECORDS} more records are "
                "recorded in the artifact._"
            )
    lines.append("")
    return lines


def _semantic_scraping_section(artifact: Artifact) -> list[str]:
    scraping = artifact.research.semantic_scraping
    if scraping is None:
        return []
    lines = ["## Semantic scraping", ""]
    lines.append(
        f"- Documents: {len(scraping.documents)} | "
        f"links: {len(scraping.links)} | "
        f"comparisons: {len(scraping.comparisons)} | "
        f"critiques: {len(scraping.critiques)}"
    )
    if scraping.stats:
        stats = ", ".join(
            f"{_text(key)}={_number(value)}"
            for key, value in sorted(scraping.stats.items())
        )
        lines.append(f"- Stats: {stats}")
    lines.append("")
    credible = [item for item in scraping.documents if item.credible]
    shown = (credible or scraping.documents)[:_MAX_RESEARCH_RECORDS]
    if not shown:
        lines.append("No scraped documents were recorded.")
    else:
        lines.append("| Document | Kind | Field path | Credibility | Critique | Comparisons |")
        lines.append("|---|---|---|---:|---|---:|")
        for document in shown:
            field_path = " > ".join(
                f"`{_text(item)}`" for item in document.field_path
            ) or "unassigned"
            critique = document.critique_verdict or "none"
            lines.append(
                f"| [{_text(document.title)}]({document.url}) | "
                f"{_text(document.kind)} | {field_path} | "
                f"{_number(document.credibility, 2)} | {_text(critique)} | "
                f"{document.comparisons:,} |"
                if document.url
                else (
                    f"| {_text(document.title)} | {_text(document.kind)} | "
                    f"{field_path} | {_number(document.credibility, 2)} | "
                    f"{_text(critique)} | {document.comparisons:,} |"
                )
            )
        if len(credible or scraping.documents) > _MAX_RESEARCH_RECORDS:
            lines.append("")
            lines.append(
                f"_{len(credible or scraping.documents) - _MAX_RESEARCH_RECORDS} "
                "more documents are recorded in the artifact._"
            )
    if scraping.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {_text(item)}" for item in scraping.warnings)
    lines.append("")
    return lines


def _factors_section(artifact: Artifact) -> list[str]:
    lines = ["## Factors", ""]
    if not artifact.factors:
        return lines + ["No research factors were extracted.", ""]
    lines.append("| Factor | Direction | Claim | Evidence | Mapped column | Confidence |")
    lines.append("|---|---|---|---|---|---:|")
    for factor in artifact.factors:
        evidence = ", ".join(
            f"`{_text(item)}`" for item in factor.evidence_ids[:_MAX_EVIDENCE_IDS]
        )
        if len(factor.evidence_ids) > _MAX_EVIDENCE_IDS:
            evidence += f" (+{len(factor.evidence_ids) - _MAX_EVIDENCE_IDS})"
        mapping = factor.mapping
        column = f"`{_text(mapping.column)}`" if mapping.column else "unmapped"
        confidence = (
            _number(mapping.confidence, 2) if mapping.confidence is not None else "n/a"
        )
        lines.append(
            f"| {_text(factor.name)} | {_text(factor.direction)} | "
            f"{_text(factor.claim_type)} | {evidence or 'none'} | {column} | "
            f"{confidence} |"
        )
    lines.append("")
    for factor in artifact.factors:
        if factor.description:
            lines.append(f"- **{_text(factor.name)}**: {_text(factor.description)}")
    lines.append("")
    return lines


def _critic_heading(critic: CriticSection | None) -> str:
    if critic is None:
        return "not reviewed"
    return f"{critic.verdict} ({len(critic.issues)} issue(s))"


def _hypothesis_section(
    artifact: Artifact, hypothesis: HypothesisSection, base_dir: Path
) -> list[str]:
    factor_names = {
        factor.id: factor.name for factor in artifact.factors
    }
    lines = [f"### {_text(hypothesis.title)}", ""]
    lines.append(f"- ID: `{hypothesis.id}`")
    lines.append(
        f"- Family: {_text(hypothesis.method_family)} | "
        f"claim: {_text(hypothesis.claim_type)} | "
        f"target: `{_text(hypothesis.target)}`"
    )
    lines.append(f"- Minimum rows: {hypothesis.min_rows:,}")
    if hypothesis.required_columns or hypothesis.optional_columns:
        required = ", ".join(f"`{_text(item)}`" for item in hypothesis.required_columns)
        optional = ", ".join(f"`{_text(item)}`" for item in hypothesis.optional_columns)
        lines.append(
            f"- Columns: required {required or 'none'}; optional {optional or 'none'}"
        )
    if hypothesis.assumptions:
        lines.append("- Assumptions:")
        lines.extend(f"  - {_text(item)}" for item in hypothesis.assumptions)
    if hypothesis.factors_used:
        used = ", ".join(
            f"`{_text(identifier)}` ({_text(factor_names.get(identifier, 'unknown'))})"
            for identifier in hypothesis.factors_used
        )
        lines.append(f"- Factors used: {used}")
    if hypothesis.source_ids:
        lines.append(
            "- Sources: "
            + ", ".join(f"`{_text(item)}`" for item in hypothesis.source_ids)
        )
    lines.append("")
    if hypothesis.rationale:
        lines.extend([_text(hypothesis.rationale), ""])

    execution = hypothesis.execution
    execution_line = (
        f"- Execution: {_text(execution.status)} | "
        f"{execution.runtime_ms:,} ms"
    )
    if execution.n_predictions is not None:
        execution_line += f" | {execution.n_predictions:,} predictions"
    lines.append(execution_line)
    if execution.predictions_path:
        lines.append(f"- Predictions: `{_text(execution.predictions_path)}`")
    if execution.error:
        lines.append(f"- Execution error: {_text(_sanitize(execution.error, base_dir))}")
    lines.append(f"- Critic verdict: {_text(_critic_heading(hypothesis.critic))}")
    if hypothesis.critic and hypothesis.critic.summary:
        lines.extend(["", _text(hypothesis.critic.summary), ""])
    code = _sanitize(_code_text(base_dir, hypothesis.code_path), base_dir).rstrip()
    fence = _fence(code)
    lines.extend(["", fence + "python", code, fence, ""])
    return lines


def _hypotheses_section(artifact: Artifact, base_dir: Path) -> list[str]:
    lines = ["## Hypotheses", ""]
    if not artifact.hypotheses:
        return lines + ["No hypotheses were generated.", ""]
    for hypothesis in artifact.hypotheses:
        lines.extend(_hypothesis_section(artifact, hypothesis, base_dir))
    return lines


def _critic_section(artifact: Artifact) -> list[str]:
    summary = artifact.critic_summary
    lines = ["## Critic summary", ""]
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for verdict in ("pass", "refine", "reject", "insufficient"):
        lines.append(f"| {verdict} | {summary.verdicts.get(verdict, 0)} |")
    lines.append("")
    lines.append(f"- Degraded: {'yes' if summary.degraded else 'no'}")
    lines.append("")
    issues = [
        (hypothesis.id, issue)
        for hypothesis in artifact.hypotheses
        if hypothesis.critic is not None
        for issue in hypothesis.critic.issues
        if issue.severity in {"major", "fatal"}
    ]
    if issues:
        lines.append("### Major and fatal issues")
        lines.append("")
        for hypothesis_id, issue in issues:
            lines.append(
                f"- `{hypothesis_id}` **{_text(issue.severity)}** "
                f"({_text(issue.chunk)}): {_text(issue.finding)}"
            )
            if issue.fix:
                lines.append(f"  - Fix: {_text(issue.fix)}")
        lines.append("")
    missing = [
        (hypothesis.id, item)
        for hypothesis in artifact.hypotheses
        if hypothesis.critic is not None
        for item in hypothesis.critic.missing_information
    ]
    if missing:
        lines.append("### Missing information")
        lines.append("")
        for hypothesis_id, item in missing:
            lines.append(f"- `{hypothesis_id}`: {_text(item)}")
        lines.append("")
    return lines


def _warnings_section(artifact: Artifact) -> list[str]:
    lines = ["## Warnings and degraded modes", ""]
    warnings: list[str] = []
    seen: set[str] = set()
    for warning in (
        list(artifact.warnings)
        + list(artifact.research.warnings)
        + [
            warning
            for hypothesis in artifact.hypotheses
            for warning in hypothesis.warnings
        ]
    ):
        text = _text(warning)
        if text and text not in seen:
            seen.add(text)
            warnings.append(text)
    if artifact.critic_summary.degraded:
        warnings.append("The method critic ran in a degraded mode.")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No warnings.")
    lines.append("")
    return lines


def _provenance_section(artifact: Artifact) -> list[str]:
    provenance = artifact.provenance
    lines = ["## Provenance", ""]
    lines.append(f"- Python: {_text(provenance.python)}")
    if provenance.packages:
        packages = ", ".join(
            f"{name} {version}"
            for name, version in sorted(provenance.packages.items())
        )
        lines.append(f"- Packages: {packages}")
    lines.append(f"- Corpus version: {_text(provenance.corpus_version or 'unknown')}")
    if provenance.git_commit:
        lines.append(f"- Git commit: `{provenance.git_commit[:12]}`")
    lines.append("")
    return lines


def render_scrape_preview(scrape: ScrapeResult) -> str:
    """Render a compact standalone Markdown report for a scrape result."""
    credible = [document for document in scrape.documents if document.credible]
    critique_by_document = {
        critique.document_id: critique for critique in scrape.critiques
    }
    comparison_counts: dict[str, int] = {}
    for comparison in scrape.comparisons:
        comparison_counts[comparison.left_id] = (
            comparison_counts.get(comparison.left_id, 0) + 1
        )
        comparison_counts[comparison.right_id] = (
            comparison_counts.get(comparison.right_id, 0) + 1
        )

    lines = [
        "# Semantic scraping",
        "",
        f"- Documents: {len(scrape.documents)} | credible: {len(credible)} | "
        f"links: {len(scrape.links)} | comparisons: {len(scrape.comparisons)} | "
        f"critiques: {len(scrape.critiques)}",
    ]
    if scrape.stats:
        lines.append(
            "- Stats: "
            + ", ".join(
                f"{_text(key)}={_number(value)}"
                for key, value in sorted(scrape.stats.items())
            )
        )
    lines.append("")
    shown = (credible or scrape.documents)[:20]
    if not shown:
        lines.append("No documents were recorded.")
    else:
        lines.append(
            "| Document | Kind | Field path | Credibility | Critique | Comparisons |"
        )
        lines.append("|---|---|---|---:|---|---:|")
        for document in shown:
            field_path = document.metadata.get("field_path", [])
            if not isinstance(field_path, list):
                field_path = []
            path = " > ".join(f"`{_text(item)}`" for item in field_path) or "unassigned"
            critique = critique_by_document.get(document.id)
            verdict = critique.verdict if critique else "none"
            title = (
                f"[{_text(document.title)}]({document.canonical_url})"
                if document.canonical_url
                else _text(document.title)
            )
            lines.append(
                f"| {title} | {_text(document.kind)} | {path} | "
                f"{_number(document.credibility_score, 2)} | {_text(verdict)} | "
                f"{comparison_counts.get(document.id, 0):,} |"
            )
        if len(credible or scrape.documents) > 20:
            lines.append("")
            lines.append(
                f"_{len(credible or scrape.documents) - 20} more documents are "
                "recorded in scrape.json._"
            )
    if scrape.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        lines.extend(f"- {_text(warning)}" for warning in scrape.warnings)
    return "\n".join(lines) + "\n"


def render_preview(artifact: Artifact, *, base_dir: Path) -> str:
    """Render the full Markdown preview. Paths are relative to ``base_dir``."""
    run = artifact.run
    lines: list[str] = [
        f"# Research-to-hypothesis run: `{run.run_id}`",
        "",
        f"- Mode: {_text(run.mode)} | seed: {run.seed} | tool: {_text(run.tool_version)}",
        f"- LLM: {_text(run.models.get('llm', 'unknown'))} | "
        f"embedding: {_text(run.models.get('embedding', 'unknown'))}",
        f"- Started: {_text(run.started_at)} | finished: {_text(run.finished_at)} "
        f"| duration: {_seconds(run.started_at, run.finished_at)}",
        "",
    ]
    lines.extend(_dataset_section(artifact))
    lines.extend(_field_path_section(artifact))
    lines.extend(_research_section(artifact))
    lines.extend(_semantic_scraping_section(artifact))
    lines.extend(_factors_section(artifact))
    lines.extend(_hypotheses_section(artifact, base_dir))
    lines.extend(_critic_section(artifact))
    lines.extend(_warnings_section(artifact))
    lines.extend(_provenance_section(artifact))
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def write_preview(
    path: Path, artifact: Artifact, *, base_dir: Path | None = None
) -> None:
    directory = base_dir if base_dir is not None else path.parent
    path.write_text(
        render_preview(artifact, base_dir=directory),
        encoding="utf-8",
    )


def preview_from_path(artifact_path: Path, *, out: Path | None = None) -> Path:
    """Validate an artifact and write `preview.md` beside it (or to ``out``)."""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact = Artifact.model_validate(payload)
    destination = out or artifact_path.with_name("preview.md")
    write_preview(destination, artifact, base_dir=artifact_path.parent)
    return destination
