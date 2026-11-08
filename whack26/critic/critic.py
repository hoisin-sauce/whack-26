"""Agentic method critic: deterministic checks plus a corpus-grounded LLM review."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..config import Settings
from ..llm import LLMError
from ..profile import DatasetProfile
from ..screen import ScreenSummary, format_screen
from .checks import run_deterministic_checks
from .corpus import Corpus, CorpusChunk, CorpusError, load_corpus
from .models import Check, ClaimType, CriticResult, Issue, Severity, Verdict

CRITIC_SYSTEM_PROMPT = (
    "You are a method critic for statistical modelling. You audit whether the "
    "derivation method can support the claim being made. You do not evaluate "
    "model performance, p-values, effect sizes, accuracy, AUC, RMSE, calibration, "
    "or out-of-sample results; another component owns that. Use only the supplied "
    "corpus chunks and derivation trace. If the information needed to judge a "
    "check is absent, say so in `missing_information` and use `insufficient` "
    "rather than guessing. Every issue must cite a chunk id from the supplied "
    "chunks. Return JSON matching the schema."
)

CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_type": {
            "type": "string",
            "enum": ["description", "prediction", "causal", "explanatory"],
        },
        "verdict": {
            "type": "string",
            "enum": ["pass", "refine", "reject", "insufficient"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk": {"type": "string", "maxLength": 80},
                    "severity": {
                        "type": "string",
                        "enum": ["fatal", "major", "minor"],
                    },
                    "finding": {"type": "string", "maxLength": 600},
                    "evidence": {"type": "string", "maxLength": 600},
                    "why": {"type": "string", "maxLength": 600},
                    "fix": {"type": "string", "maxLength": 600},
                },
                "required": [
                    "chunk",
                    "severity",
                    "finding",
                    "evidence",
                    "why",
                    "fix",
                ],
            },
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string", "maxLength": 240},
        },
        "summary": {"type": "string", "maxLength": 1500},
    },
    "required": [
        "claim_type",
        "verdict",
        "issues",
        "missing_information",
        "summary",
    ],
    "propertyOrdering": [
        "claim_type",
        "verdict",
        "issues",
        "missing_information",
        "summary",
    ],
}

_CLAIM_ALIASES = {
    "predictive": "prediction",
    "descriptive": "description",
    "association": "description",
    "associational": "description",
    "inferential": "explanatory",
}

_BANNED_PERFORMANCE_RE = re.compile(
    r"\b(?:accuracy|auc|roc|rmse|mse|mae|r[- ]?squared|r2|"
    r"p[- ]value|statistical significance|confidence interval|effect size|"
    r"calibration|discrimination|out[- ]of[- ]sample|benchmark|generalization|"
    r"(?:evaluation|performance)\s+metrics?)\b",
    re.IGNORECASE,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_FIELD_PITFALL_CAUTION = (
    "Field-scoped pitfalls are supplementary. `applies_to_dataset=yes` means "
    "the dataset profile shows the precondition; `unknown` is background only. "
    "Do not raise an issue from a field pitfall unless it applies to this "
    "dataset or the code/metadata clearly demonstrates the defect."
)


class LLMIssue(BaseModel):
    chunk: str
    severity: Severity
    finding: str
    evidence: str = ""
    why: str = ""
    fix: str = ""


class LLMCriticPayload(BaseModel):
    claim_type: ClaimType = "description"
    verdict: Verdict = "insufficient"
    issues: list[LLMIssue] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("claim_type", mode="before")
    @classmethod
    def _normalize_claim(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _CLAIM_ALIASES.get(value.strip().lower(), value.strip().lower())
        return value


def _corpus_missing_result(
    hypothesis_id: str,
    *,
    settings: Settings,
    checks: Sequence[Check],
) -> CriticResult:
    return CriticResult(
        hypothesis_id=hypothesis_id,
        verdict="insufficient",
        issues=[
            Issue(
                chunk="S0-how-to-use",
                severity="major",
                finding="The statistical-pitfalls corpus is missing or unreadable.",
                evidence="critic.corpus_missing",
                why="The critic cannot ground its method review without the corpus.",
                fix="Restore docs/corpus/statistical-pitfalls.md and rerun the critic.",
            )
        ],
        missing_information=["critic corpus"],
        checks=list(checks),
        summary=(
            "The method critic could not run because the preloaded corpus is "
            "missing. The hypothesis was executed, but its method is unassessed."
        ),
        retrieved_chunks=[],
        model=_model_name(settings),
        corpus_version="missing",
        degraded=True,
        warnings=["critic.corpus_missing"],
    )


def _model_name(settings: Settings) -> str:
    return settings.llm_model_name


def _field_path_text(field_path: Sequence[Any]) -> str:
    lines: list[str] = []
    for item in field_path:
        name = getattr(item, "name", str(item))
        description = getattr(item, "description", "")
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n".join(lines) or "(none)"


def _factors_text(factors: Sequence[Any]) -> str:
    lines: list[str] = []
    for factor in factors:
        lines.append(
            f"- id={getattr(factor, 'id', None) or '(none)'} | "
            f"name={getattr(factor, 'name', '')} | "
            f"direction={getattr(factor, 'direction', 'unclear')} | "
            f"evidence={', '.join(getattr(factor, 'evidence_ids', []) or [])} | "
            f"mapped_column={getattr(factor, 'mapped_column', None) or '(unmapped)'}"
        )
    return "\n".join(lines) or "(none)"


def _research_text(records: Sequence[Any], limit: int = 12) -> str:
    lines: list[str] = []
    for record in records[:limit]:
        lines.append(
            f"- {getattr(record, 'id', '')} | {getattr(record, 'year', None)} | "
            f"{getattr(record, 'title', '')}"
        )
    return "\n".join(lines) or "(none)"


def _value(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from a Pydantic model, SimpleNamespace, or mapping."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _field_pitfall_chunks(
    field_pitfalls: Sequence[Any], *, limit: int = 6
) -> list[CorpusChunk]:
    """Convert applicable field pitfalls into capped, prompt-ready chunks."""
    applicable: list[Any] = []
    for pitfall in field_pitfalls:
        applies = _value(pitfall, "applies_to_dataset", "unknown")
        if str(applies or "unknown").strip().lower() == "no":
            continue
        applicable.append(pitfall)
    applicable.sort(
        key=lambda item: (
            0 if str(_value(item, "scope", "ancestor")) == "leaf" else 1,
            -_as_int(_value(item, "field_level", 1), 1),
            -_as_float(_value(item, "confidence", 0.0)),
            str(_value(item, "id", "") or ""),
        )
    )
    chunks: list[CorpusChunk] = []
    for pitfall in applicable:
        chunk_id = str(_value(pitfall, "id", "") or "").strip()
        if not chunk_id:
            continue
        field_name = str(_value(pitfall, "field_name", "") or "")
        scope = str(_value(pitfall, "scope", "ancestor") or "ancestor")
        text = "\n".join(
            [
                f"Field: {field_name}",
                f"Field level: {_as_int(_value(pitfall, 'field_level', 1), 1)}",
                f"Scope: {scope}",
                f"Category: {_value(pitfall, 'category', 'other')}",
                "Applies to dataset: "
                f"{_value(pitfall, 'applies_to_dataset', 'unknown')}",
                "Applicability reason: "
                f"{_value(pitfall, 'applicability_reason', '')}",
                f"Pitfall: {_value(pitfall, 'pitfall', '')}",
                f"Check: {_value(pitfall, 'check', '')}",
                f"Evidence ID: {_value(pitfall, 'evidence_id', '')}",
                f"Source: {_value(pitfall, 'source', '')}",
                f"URL: {_value(pitfall, 'url', '')}",
            ]
        )
        chunks.append(
            CorpusChunk(
                id=chunk_id,
                title=f"{field_name} ({scope} field)",
                text=text,
            )
        )
        if len(chunks) >= limit:
            break
    return chunks


def _hypothesis_text(hypothesis: Any) -> str:
    draft = getattr(hypothesis, "draft", hypothesis)
    return (
        f"id: {getattr(hypothesis, 'id', '')}\n"
        f"method_family: {getattr(hypothesis, 'method_family', '')}\n"
        f"title: {getattr(draft, 'title', '')}\n"
        f"claim_type: {getattr(draft, 'claim_type', '')}\n"
        f"target: {getattr(draft, 'target', None)}\n"
        f"required_columns: {getattr(draft, 'required_columns', [])}\n"
        f"optional_columns: {getattr(draft, 'optional_columns', [])}\n"
        f"min_rows: {getattr(draft, 'min_rows', None)}\n"
        f"assumptions: {getattr(draft, 'assumptions', [])}\n"
        f"factors_used: {getattr(draft, 'factors_used', [])}\n"
        f"source_ids: {getattr(draft, 'source_ids', [])}\n"
        f"rationale: {getattr(draft, 'rationale', '')}"
    )


def _chunks_text(chunks: Sequence[Any]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(f"### {chunk.id} — {chunk.title}\n{chunk.text}")
    return "\n\n".join(blocks) or "(no chunks retrieved)"


def _user_prompt(
    *,
    hypothesis: Any,
    profile: DatasetProfile,
    summary: str,
    field_path: Sequence[Any],
    factors: Sequence[Any],
    screen: ScreenSummary,
    research: Sequence[Any],
    chunks: Sequence[Any],
    field_pitfalls_supplied: bool = False,
    feedback: str | None = None,
) -> str:
    draft = getattr(hypothesis, "draft", hypothesis)
    text = (
        "Dataset profile summary:\n"
        + summary
        + "\n\nField path:\n"
        + _field_path_text(field_path)
        + "\n\nFactors:\n"
        + _factors_text(factors)
        + "\n\nResearch records:\n"
        + _research_text(research)
        + "\n\nHypothesis metadata and derivation trace:\n"
        + _hypothesis_text(hypothesis)
        + "\n\nCode:\n```python\n"
        + str(getattr(draft, "code", "") or "")
        + "\n```\n\nCheap descriptive screen (context only):\n"
        + format_screen(screen, max_chars=2500)
        + "\n\nRetrieved corpus chunks:\n"
        + _chunks_text(chunks)
        + (
            "\n\n" + _FIELD_PITFALL_CAUTION
            if field_pitfalls_supplied
            else ""
        )
        + "\n\nReview the derivation, not the performance. Classify the claim and "
        "return issues that cite a chunk id. Use `insufficient` when the supplied "
        "trace lacks the information needed for a check."
    )
    if feedback:
        text += "\n\nYour previous response failed validation:\n" + feedback
    return text


def _sanitize_text(value: str) -> tuple[str, bool]:
    if not _BANNED_PERFORMANCE_RE.search(value):
        return value, False
    sentences = _SENTENCE_RE.split(value)
    kept = [sentence for sentence in sentences if not _BANNED_PERFORMANCE_RE.search(sentence)]
    return " ".join(kept).strip(), True


def _sanitize_issue(issue: Issue) -> tuple[Issue | None, bool]:
    text = " ".join(
        [issue.finding, issue.evidence, issue.why, issue.fix]
    )
    if _BANNED_PERFORMANCE_RE.search(text):
        return None, True
    return issue, False


def _dedupe_issues(issues: Sequence[Issue]) -> list[Issue]:
    seen: set[tuple[str, str]] = set()
    result: list[Issue] = []
    for issue in issues:
        key = (issue.chunk, issue.finding.strip().lower()[:100])
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _merge_verdict(
    *,
    issues: Sequence[Issue],
    missing_information: Sequence[str],
    degraded: bool,
) -> Verdict:
    if any(issue.severity == "fatal" for issue in issues):
        return "reject"
    if degraded or missing_information:
        return "insufficient"
    if any(issue.severity == "major" for issue in issues):
        return "refine"
    return "pass"


def run_critic(
    hypothesis: Any,
    *,
    profile: DatasetProfile,
    summary: str,
    field_path: Sequence[Any],
    factors: Sequence[Any],
    screen: ScreenSummary,
    research: Sequence[Any],
    llm: Any,
    embeddings: Any,
    settings: Settings,
    corpus: Corpus | None = None,
    field_pitfalls: Sequence[Any] = (),
) -> CriticResult:
    """Run deterministic checks and one corpus-grounded LLM review."""
    hypothesis_id = str(getattr(hypothesis, "id", "hyp_unknown"))
    deterministic = run_deterministic_checks(
        hypothesis,
        profile=profile,
        screen=screen,
        factors=factors,
        description=profile.description,
    )

    if corpus is None:
        try:
            corpus = load_corpus(settings.corpus_path)
        except CorpusError as exc:
            deterministic.warnings.append(f"critic.corpus_error: {exc}")
            corpus = None
    if corpus is None:
        return _corpus_missing_result(
            hypothesis_id, settings=settings, checks=deterministic.checks
        )

    draft = getattr(hypothesis, "draft", hypothesis)
    query = " ".join(
        [
            str(getattr(draft, "rationale", "")),
            summary,
            " ".join(str(getattr(factor, "name", "")) for factor in factors),
        ]
    )
    chunks, retrieval_warnings = corpus.retrieve(query, embeddings)
    field_chunks = _field_pitfall_chunks(field_pitfalls)
    if field_chunks:
        chunks = list(chunks) + field_chunks
    retrieved_ids = [chunk.id for chunk in chunks]
    valid_chunk_ids = {chunk.id for chunk in corpus.chunks}
    valid_chunk_ids.update(chunk.id for chunk in field_chunks)

    payload: dict[str, Any] | None = None
    validation_error: str | None = None
    llm_warnings: list[str] = []
    for attempt in (1, 2):
        try:
            payload = llm.generate_json(
                system=CRITIC_SYSTEM_PROMPT,
                user=_user_prompt(
                    hypothesis=hypothesis,
                    profile=profile,
                    summary=summary,
                    field_path=field_path,
                    factors=factors,
                    screen=screen,
                    research=research,
                    chunks=chunks,
                    field_pitfalls_supplied=bool(field_chunks),
                    feedback=validation_error if attempt == 2 else None,
                ),
                schema=CRITIC_SCHEMA,
                max_tokens=4096,
                temperature=0.0,
                purpose="critic",
            )
            if not isinstance(payload, dict):
                raise ValueError("critic returned a non-object response")
            parsed = LLMCriticPayload.model_validate(payload)
            break
        except (LLMError, ValidationError, ValueError, TypeError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            payload = None
            if attempt == 2:
                llm_warnings.append(f"critic.unavailable: {validation_error}")

    if payload is None:
        issue = Issue(
            chunk=retrieved_ids[0] if retrieved_ids else "S0-how-to-use",
            severity="major",
            finding="The method critic could not produce a valid review.",
            evidence="critic.unavailable",
            why=validation_error or "The critic provider failed.",
            fix="Rerun the critic or inspect the method manually.",
        )
        return CriticResult(
            hypothesis_id=hypothesis_id,
            claim_type=str(getattr(draft, "claim_type", "description")),  # type: ignore[arg-type]
            verdict="insufficient",
            issues=list(deterministic.issues) + [issue],
            missing_information=["method critic review"],
            checks=deterministic.checks,
            summary=(
                "The method critic did not run successfully, so the derivation "
                "is unassessed. The hypothesis itself executed normally."
            ),
            retrieved_chunks=retrieved_ids,
            model=_model_name(settings),
            corpus_version=corpus.version,
            degraded=True,
            warnings=deterministic.warnings + retrieval_warnings + llm_warnings,
        )

    llm_issues: list[Issue] = []
    dropped_citations = 0
    for raw_issue in parsed.issues:
        if raw_issue.chunk not in valid_chunk_ids:
            dropped_citations += 1
            continue
        llm_issues.append(
            Issue(
                chunk=raw_issue.chunk,
                severity=raw_issue.severity,
                finding=raw_issue.finding,
                evidence=raw_issue.evidence,
                why=raw_issue.why,
                fix=raw_issue.fix,
            )
        )
    warnings = deterministic.warnings + retrieval_warnings + llm_warnings
    if dropped_citations:
        warnings.append(
            f"critic.issues_without_valid_citation: dropped {dropped_citations}"
        )

    issues = _dedupe_issues(list(deterministic.issues) + llm_issues)

    sanitized: list[Issue] = []
    removed_performance = False
    for issue in issues:
        cleaned, removed = _sanitize_issue(issue)
        removed_performance = removed_performance or removed
        if cleaned is not None:
            sanitized.append(cleaned)
    summary_text, removed_summary = _sanitize_text(parsed.summary)
    removed_performance = removed_performance or removed_summary
    if removed_performance:
        warnings.append("critic.performance_language_removed")
        if not summary_text:
            summary_text = (
                "The critic review contained out-of-scope performance language, "
                "which was removed."
            )

    missing_information: list[str] = []
    removed_missing = 0
    for item in parsed.missing_information:
        text = str(item).strip()
        if not text:
            continue
        if _BANNED_PERFORMANCE_RE.search(text):
            removed_missing += 1
            continue
        missing_information.append(text)
    if removed_missing:
        warnings.append(
            f"critic.performance_missing_information_removed: {removed_missing}"
        )
    verdict = _merge_verdict(
        issues=sanitized,
        missing_information=missing_information,
        degraded=False,
    )
    if parsed.verdict == "insufficient" and verdict != "insufficient":
        warnings.append("critic.llm_insufficient_overridden")
    return CriticResult(
        hypothesis_id=hypothesis_id,
        claim_type=parsed.claim_type,
        verdict=verdict,
        issues=sanitized,
        missing_information=missing_information,
        checks=deterministic.checks,
        summary=summary_text,
        retrieved_chunks=retrieved_ids,
        model=_model_name(settings),
        corpus_version=corpus.version,
        degraded=False,
        warnings=warnings,
    )


def critic_to_artifact(result: CriticResult) -> dict[str, Any]:
    payload = result.model_dump(mode="json")
    payload.pop("degraded", None)
    payload.pop("warnings", None)
    return payload
