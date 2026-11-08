"""Generation-process validation: does the claim match how the data arose?

This validator is deliberately separate from the statistical ``ValidationReport``
(``whack26.validation.models``). It audits the *derivation*: the claim type, the
collection design, the unit of analysis, sampling, bias/measurement notes,
leakage, model-family suitability, missing-data handling, reproducibility, and
the jumbled/PII variant reports. It never judges model performance.

Deterministic checks always run. A corpus-grounded LLM review is optional; when it
runs, deterministic findings win, so an LLM cannot talk the validator out of a
fatal claim/design mismatch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..critic.corpus import Corpus, CorpusError, load_corpus
from ..llm import LLMError
from .collection import (
    check_collection_design,
    format_collection,
    redact_collection,
)
from .generation_models import (
    CollectionMetadata,
    GenerationValidationResult,
    ValidationCheck,
    ValidationIssue,
)

GENERATION_SYSTEM_PROMPT = (
    "You are a generation-process validator for statistical modelling. You audit "
    "whether the hypothesis's claim matches how the dataset was produced: the "
    "collection design, treatment/instrument, unit of analysis, sampling, known "
    "biases, measurement error, time period, target leakage, model-family "
    "suitability, missing-data handling, reproducibility, and any jumbled-label "
    "or PII-reliance report. You do not evaluate model performance, p-values, "
    "effect sizes, accuracy, AUC, RMSE, or out-of-sample results; another "
    "component owns that. Use only the supplied metadata and corpus chunks. If "
    "the information needed for a check is absent, say so in "
    "`missing_information` and use `insufficient` rather than guessing. Every "
    "issue must cite a chunk id from the supplied chunks. Return JSON matching "
    "the schema."
)

GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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
    "required": ["verdict", "issues", "missing_information", "summary"],
}

_BANNED_PERFORMANCE_RE = re.compile(
    r"\b(?:accuracy|auc|roc|rmse|mse|mae|r[- ]?squared|r2|"
    r"p[- ]value|statistical significance|confidence interval|effect size|"
    r"calibration|discrimination|out[- ]of[- ]sample|benchmark|generalization|"
    r"(?:evaluation|performance)\s+metrics?)\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_MODEL_FAMILIES: dict[str, set[str]] = {
    "continuous": {
        "baseline",
        "linear",
        "interaction",
        "group_comparison",
        "tree",
        "forest",
        "gradient_boosting",
        "sindy",
    },
    "binary": {
        "baseline",
        "logistic",
        "interaction",
        "group_comparison",
        "tree",
        "forest",
        "gradient_boosting",
    },
    "multiclass": {
        "baseline",
        "multinomial",
        "tree",
        "forest",
        "gradient_boosting",
    },
    "count": {
        "baseline",
        "poisson",
        "negative_binomial",
        "tree",
        "forest",
        "gradient_boosting",
    },
    "time_series": {
        "baseline",
        "linear",
        "interaction",
        "sindy",
        "tree",
        "forest",
        "gradient_boosting",
    },
}

_MISSING_HANDLING_RE = re.compile(
    r"fillna|dropna|simpleimputer|iterativeimputer|impute|interpolate|"
    r"isna|isnull|notnull",
    re.IGNORECASE,
)
_SEED_RE = re.compile(r"\b(?:np\.random\.seed|random_state|seed)\b")

_FALLBACK_CHUNKS = (
    "S0-how-to-use",
    "S1-assumptions",
    "S2-leakage",
    "S4-causal-predictive",
    "S8-missing-measurement",
)


class _LLMIssue(BaseModel):
    chunk: str = "S0-how-to-use"
    severity: str = "minor"
    finding: str
    evidence: str = ""
    why: str = ""
    fix: str = ""


class _LLMPayload(BaseModel):
    verdict: str = "insufficient"
    issues: list[_LLMIssue] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""


def _get(hypothesis: Any, name: str, default: Any = None) -> Any:
    """Read a field from a wrapper object or its draft metadata."""
    if hasattr(hypothesis, name):
        return getattr(hypothesis, name)
    draft = getattr(hypothesis, "draft", None)
    if draft is not None and hasattr(draft, name):
        return getattr(draft, name)
    return default


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _check(
    identifier: str,
    *,
    status: str,
    severity: str = "none",
    reason: str = "",
    citations: Sequence[str] | None = None,
) -> ValidationCheck:
    return ValidationCheck(
        id=identifier,
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        reason=reason,
        citations=list(citations or []),
    )


def _check_from_issues(
    identifier: str,
    issues: Sequence[ValidationIssue],
    *,
    pass_reason: str,
) -> ValidationCheck:
    if not issues:
        return _check(identifier, status="pass", reason=pass_reason)
    fatal = next((item for item in issues if item.severity == "fatal"), None)
    major = next((item for item in issues if item.severity == "major"), None)
    chosen = fatal or major or issues[0]
    return _check(
        identifier,
        status="fail" if chosen.severity == "fatal" else "flag",
        severity=chosen.severity,
        reason=chosen.finding,
        citations=[issue.chunk for issue in issues],
    )


def _issue(
    *,
    evidence: str,
    severity: str,
    finding: str,
    why: str,
    fix: str,
    chunk: str = "S0-how-to-use",
) -> ValidationIssue:
    return ValidationIssue(
        chunk=chunk,
        severity=severity,  # type: ignore[arg-type]
        finding=finding,
        evidence=evidence,
        why=why,
        fix=fix,
    )


def _dedupe_issues(issues: Sequence[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str]] = set()
    result: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.evidence, issue.finding.strip().lower()[:120])
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _merge_verdict(
    issues: Sequence[ValidationIssue],
    missing_information: Sequence[str],
) -> str:
    if any(issue.severity == "fatal" for issue in issues):
        return "reject"
    if any(issue.severity == "major" for issue in issues):
        return "refine"
    if missing_information:
        return "insufficient"
    return "pass"


def _sanitize_text(value: str) -> tuple[str, bool]:
    if not _BANNED_PERFORMANCE_RE.search(value):
        return value, False
    sentences = _SENTENCE_RE.split(value)
    kept = [
        sentence
        for sentence in sentences
        if not _BANNED_PERFORMANCE_RE.search(sentence)
    ]
    return " ".join(kept).strip(), True


def _sanitize_issue(issue: ValidationIssue) -> ValidationIssue | None:
    text = " ".join([issue.finding, issue.evidence, issue.why, issue.fix])
    if _BANNED_PERFORMANCE_RE.search(text):
        return None
    return issue


def _model_name(settings: Settings | None, llm: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    if settings is not None:
        if settings.llm_provider == "gemini":
            return settings.gemini_model
        return settings.llm_provider
    if llm is not None:
        return type(llm).__name__
    return "deterministic"


def _target_kind(model_search: Any, screen: Any) -> str | None:
    kind = getattr(model_search, "target_kind", None)
    if isinstance(model_search, Mapping):
        kind = model_search.get("target_kind", kind)
    if kind:
        return str(kind)
    if screen is not None:
        screen_kind = getattr(getattr(screen, "target", None), "kind", None)
        return {
            "numeric": "continuous",
            "categorical": "binary",
        }.get(str(screen_kind))
    return None


def _variant_flag(report: Any, *names: str) -> bool:
    for name in names:
        if isinstance(report, Mapping) and report.get(name):
            return True
        if bool(getattr(report, name, False)):
            return True
    return False


def _hypothesis_text(hypothesis: Any) -> str:
    return (
        f"id: {_get(hypothesis, 'id', '')}\n"
        f"method_family: {_get(hypothesis, 'method_family', '')}\n"
        f"title: {_get(hypothesis, 'title', '')}\n"
        f"claim_type: {_get(hypothesis, 'claim_type', '')}\n"
        f"target: {_get(hypothesis, 'target', None)}\n"
        f"required_columns: {_get(hypothesis, 'required_columns', [])}\n"
        f"optional_columns: {_get(hypothesis, 'optional_columns', [])}\n"
        f"assumptions: {_get(hypothesis, 'assumptions', [])}\n"
        f"rationale: {_get(hypothesis, 'rationale', '')}"
    )


def _profile_text(profile: Any) -> str:
    if profile is None:
        return "(no dataset profile supplied)"
    columns = getattr(profile, "columns", []) or []
    column_text = ", ".join(
        f"{getattr(column, 'name', '?')} ({getattr(column, 'role', 'unknown')})"
        for column in columns[:30]
    )
    return (
        f"rows: {getattr(profile, 'rows', 0)}\n"
        f"description: {getattr(profile, 'description', None)}\n"
        f"time_column: {getattr(profile, 'time_column', None)}\n"
        f"columns: {column_text or '(none)'}"
    )


def _chunks_text(chunks: Sequence[Any]) -> str:
    blocks = [
        f"### {chunk.id} — {chunk.title}\n{chunk.text}" for chunk in chunks
    ]
    return "\n\n".join(blocks) or "(no chunks retrieved)"


class GenerationValidator:
    """Run deterministic generation-process checks and an optional LLM review."""

    def __init__(
        self,
        *,
        llm: Any = None,
        embeddings: Any = None,
        settings: Settings | None = None,
        corpus: Corpus | None = None,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.settings = settings
        self._corpus = corpus
        self.model = _model_name(settings, llm, model)

    def _resolve_corpus(self) -> Corpus | None:
        if self._corpus is not None:
            return self._corpus
        if self.settings is None:
            return None
        try:
            self._corpus = load_corpus(self.settings.corpus_path)
        except CorpusError:
            self._corpus = None
        return self._corpus

    def _retrieve_chunks(
        self, corpus: Corpus | None, query: str
    ) -> tuple[list[Any], list[str]]:
        if corpus is None:
            return [], []
        if self.embeddings is not None:
            return corpus.retrieve(query, self.embeddings)
        selected = [
            chunk
            for chunk in (corpus.get(item) for item in _FALLBACK_CHUNKS)
            if chunk is not None
        ]
        return selected, []

    def validate(
        self,
        hypothesis: Any,
        *,
        profile: Any = None,
        collection: CollectionMetadata | Mapping[str, Any] | None = None,
        collection_metadata: CollectionMetadata | Mapping[str, Any] | None = None,
        screen: Any = None,
        model_search: Any = None,
        diagnostics: Any = None,
        research_models: Sequence[Any] = (),
        variant_report: Any = None,
        code_hash: str | None = None,
        data_hash: str | None = None,
        seed: int | None = None,
    ) -> GenerationValidationResult:
        """Validate one hypothesis against the collection and dataset metadata."""
        if collection is None:
            collection = collection_metadata
        if model_search is None:
            model_search = diagnostics
        if collection is not None and not isinstance(collection, CollectionMetadata):
            try:
                collection = CollectionMetadata.model_validate(collection)
            except ValidationError as exc:
                raise ValueError(f"invalid collection metadata: {exc}") from exc

        hypothesis_id = str(_get(hypothesis, "id", "hyp_unknown"))
        code = str(_get(hypothesis, "code", "") or "")
        claim_type = str(_get(hypothesis, "claim_type", "description") or "description")
        family = str(_get(hypothesis, "method_family", "") or "")
        target = _get(hypothesis, "target", None)
        if target is None and profile is not None:
            profile_target = getattr(profile, "target", None)
            target = getattr(profile_target, "name", None)
        target = str(target) if target else None
        required = _as_list(_get(hypothesis, "required_columns", []))
        optional = _as_list(_get(hypothesis, "optional_columns", []))

        checks: list[ValidationCheck] = []
        issues: list[ValidationIssue] = []
        missing_information: list[str] = []

        design_issues = check_collection_design(
            collection, claim_type=claim_type, outcome=target
        )
        issues.extend(design_issues)

        # 1. Claim type vs design -----------------------------------------
        claim_issues = [
            issue
            for issue in design_issues
            if issue.evidence in {"claim.design", "design.identification"}
        ]
        checks.append(
            _check_from_issues(
                "claim.design",
                claim_issues,
                pass_reason=(
                    "The claim type is consistent with the recorded design."
                    if claim_type == "causal"
                    else "The hypothesis does not make a causal claim."
                ),
            )
        )

        # 2. Treatment / instrument ---------------------------------------
        treatment_issues = [
            issue
            for issue in design_issues
            if issue.evidence in {"design.treatment", "design.instrument"}
        ]
        treatment_required = claim_type == "causal" or (
            collection is not None
            and collection.design in {"experimental", "quasi_experimental"}
        )
        if treatment_required:
            checks.append(
                _check_from_issues(
                    "treatment.instrument",
                    treatment_issues,
                    pass_reason="The required treatment or instrument is recorded.",
                )
            )
        else:
            checks.append(
                _check(
                    "treatment.instrument",
                    status="not_applicable",
                    reason="No treatment or instrument is required for this claim.",
                )
            )

        # 3. Unit of analysis ---------------------------------------------
        unit_issues = [
            issue for issue in design_issues if issue.evidence == "unit.of_analysis"
        ]
        checks.append(
            _check_from_issues(
                "unit.of_analysis",
                unit_issues,
                pass_reason="The unit of analysis is recorded.",
            )
        )
        if unit_issues:
            missing_information.append("unit of analysis")

        # 4. Sampling / selection bias ------------------------------------
        sampling = collection.sampling if collection is not None else None
        if sampling:
            checks.append(
                _check(
                    "sampling.selection_bias",
                    status="pass",
                    reason="The sampling method is recorded.",
                )
            )
        else:
            issue = _issue(
                evidence="sampling.selection_bias",
                severity="major",
                finding="The sampling method is not recorded.",
                why=(
                    "Selection into the sample determines which population the "
                    "estimate describes and whether selection bias is plausible."
                ),
                fix="Record sampling in the collection metadata.",
                chunk="S7-sample-size",
            )
            issues.append(issue)
            missing_information.append("sampling method")
            checks.append(
                _check_from_issues(
                    "sampling.selection_bias",
                    [issue],
                    pass_reason="The sampling method is recorded.",
                )
            )

        # 5. Known biases --------------------------------------------------
        known_biases = (
            list(collection.known_biases) if collection is not None else []
        )
        if known_biases:
            checks.append(
                _check(
                    "known_biases",
                    status="pass",
                    reason="Known biases are recorded.",
                )
            )
        else:
            issue = _issue(
                evidence="known_biases",
                severity="minor",
                finding="No known biases are recorded.",
                why=(
                    "Catalogue the known biases so the derivation can say how "
                    "they were addressed or bounded."
                ),
                fix="Record known_biases in the collection metadata.",
                chunk="S8-missing-measurement",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "known_biases",
                    [issue],
                    pass_reason="Known biases are recorded.",
                )
            )

        # 6. Measurement error --------------------------------------------
        measurement_notes = (
            list(collection.measurement_notes) if collection is not None else []
        )
        if measurement_notes:
            checks.append(
                _check(
                    "measurement.error",
                    status="pass",
                    reason="Measurement notes are recorded.",
                )
            )
        else:
            issue = _issue(
                evidence="measurement.error",
                severity="minor",
                finding="No measurement notes are recorded.",
                why=(
                    "The measurement process determines the error structure of "
                    "the outcome and features."
                ),
                fix="Record measurement_notes in the collection metadata.",
                chunk="S8-missing-measurement",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "measurement.error",
                    [issue],
                    pass_reason="Measurement notes are recorded.",
                )
            )

        # 7. Time period / sample consistency ------------------------------
        time_column = getattr(profile, "time_column", None) if profile else None
        if not time_column and screen is not None:
            time_column = getattr(getattr(screen, "time_trend", None), "column", None)
        if not time_column:
            checks.append(
                _check(
                    "time.period",
                    status="not_applicable",
                    reason="The dataset has no time column.",
                )
            )
        elif collection is not None and collection.time_period:
            checks.append(
                _check(
                    "time.period",
                    status="pass",
                    reason="The time period is recorded.",
                )
            )
        else:
            issue = _issue(
                evidence="time.period",
                severity="major",
                finding="The time period is not recorded for a time-dependent dataset.",
                why="The sample window bounds the population and the claim.",
                fix="Record time_period in the collection metadata.",
                chunk="S9-time-series",
            )
            issues.append(issue)
            missing_information.append("time period")
            checks.append(
                _check_from_issues(
                    "time.period",
                    [issue],
                    pass_reason="The time period is recorded.",
                )
            )

        # 8. Target leakage -------------------------------------------------
        metadata_leak = bool(target and target in set(required) | set(optional))
        code_refs: list[int] = []
        if target:
            try:
                from ..hypotheses.validate import target_feature_references

                code_refs = target_feature_references(code, target)
            except Exception:  # noqa: BLE001 - a static check must not crash
                code_refs = []
        if target is None:
            checks.append(
                _check(
                    "target.leakage",
                    status="not_applicable",
                    reason="No target is recorded.",
                )
            )
        elif metadata_leak or code_refs:
            issue = _issue(
                evidence="target.leakage",
                severity="fatal",
                finding=(
                    "The target or a target-derived column is used as a feature"
                    + (
                        " (code line(s) " + ", ".join(map(str, code_refs)) + ")"
                        if code_refs
                        else ""
                    )
                    + "."
                ),
                why="Target leakage makes the derivation circular.",
                fix=(
                    "Remove the target and target-derived columns from the "
                    "feature set."
                ),
                chunk="S2-leakage",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "target.leakage",
                    [issue],
                    pass_reason="The target is not used as a feature.",
                )
            )
        else:
            checks.append(
                _check(
                    "target.leakage",
                    status="pass",
                    reason="The target is not used as a feature.",
                )
            )

        # 9. Model family vs outcome ---------------------------------------
        kind = _target_kind(model_search, screen)
        allowed = _MODEL_FAMILIES.get(kind or "")
        if not kind or not family or allowed is None:
            checks.append(
                _check(
                    "model.family",
                    status="not_applicable",
                    reason="The outcome kind or model family is not available.",
                )
            )
        elif family not in allowed:
            issue = _issue(
                evidence="model.family",
                severity="major",
                finding=(
                    f"Model family {family!r} is not appropriate for a "
                    f"{kind} outcome."
                ),
                why=(
                    "The family's likelihood and link function imply a "
                    "different outcome type."
                ),
                fix="Choose a family that matches the outcome kind.",
                chunk="S1-assumptions",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "model.family",
                    [issue],
                    pass_reason="The model family matches the outcome kind.",
                )
            )
        else:
            checks.append(
                _check(
                    "model.family",
                    status="pass",
                    reason="The model family matches the outcome kind.",
                )
            )

        # 10. Missing-data handling ----------------------------------------
        high_missing: list[str] = []
        if screen is not None:
            high_missing = [
                str(name)
                for name, pct in (getattr(screen, "missingness", {}) or {}).items()
                if float(pct) > 5.0
            ]
        elif profile is not None:
            high_missing = [
                str(getattr(column, "name", "?"))
                for column in getattr(profile, "columns", []) or []
                if float(getattr(column, "missing_pct", 0.0) or 0.0) > 5.0
            ]
        if not high_missing:
            checks.append(
                _check(
                    "missing.data",
                    status="pass",
                    reason="No column exceeds 5% missing values.",
                )
            )
        elif not code:
            checks.append(
                _check(
                    "missing.data",
                    status="not_applicable",
                    reason="No hypothesis code is supplied.",
                )
            )
        elif _MISSING_HANDLING_RE.search(code):
            checks.append(
                _check(
                    "missing.data",
                    status="pass",
                    reason="The code applies an explicit missing-data rule.",
                )
            )
        else:
            issue = _issue(
                evidence="missing.data",
                severity="major",
                finding=(
                    "Columns with more than 5% missing values are not handled "
                    "explicitly."
                ),
                why="Silent complete-case analysis changes the sample.",
                fix="Impute, drop, or document the missing-data mechanism explicitly.",
                chunk="S8-missing-measurement",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "missing.data",
                    [issue],
                    pass_reason="Missing values are handled explicitly.",
                )
            )

        # 11. Reproducibility ----------------------------------------------
        resolved_code_hash = code_hash or _get(hypothesis, "code_sha256", None)
        resolved_data_hash = data_hash or (
            getattr(profile, "sha256", None) if profile is not None else None
        )
        has_seed = seed is not None or bool(_SEED_RE.search(code))
        missing_repro: list[str] = []
        if not resolved_code_hash:
            missing_repro.append("code hash")
        if not resolved_data_hash:
            missing_repro.append("data hash")
        if not has_seed:
            missing_repro.append("random seed")
        if missing_repro:
            issue = _issue(
                evidence="reproducibility",
                severity="major",
                finding="The derivation is not fully reproducible: missing "
                + ", ".join(missing_repro)
                + ".",
                why=(
                    "A reviewer cannot re-run the derivation without these "
                    "identifiers."
                ),
                fix="Record the seed, code hash, and data hash.",
                chunk="S10-reporting",
            )
            issues.append(issue)
            missing_information.extend(missing_repro)
            checks.append(
                _check_from_issues(
                    "reproducibility",
                    [issue],
                    pass_reason="The seed, code hash, and data hash are recorded.",
                )
            )
        else:
            checks.append(
                _check(
                    "reproducibility",
                    status="pass",
                    reason="The seed, code hash, and data hash are recorded.",
                )
            )

        # 12. Jumbled-label report -----------------------------------------
        if variant_report is None:
            checks.append(
                _check(
                    "variant.jumbled",
                    status="not_applicable",
                    reason="No variant report is available.",
                )
            )
        elif _variant_flag(
            variant_report, "jumbled_label_leak", "jumbled_leak", "leak_detected"
        ):
            issue = _issue(
                evidence="variant.jumbled",
                severity="fatal",
                finding="The model exploits the jumbled labels.",
                why=(
                    "Performance above the permutation null on permuted labels "
                    "means the model recovers leakage rather than signal."
                ),
                fix="Remove the leak and re-run the jumbled-variant gate.",
                chunk="S2-leakage",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "variant.jumbled",
                    [issue],
                    pass_reason="The jumbled-variant gate found no leak.",
                )
            )
        else:
            checks.append(
                _check(
                    "variant.jumbled",
                    status="pass",
                    reason="The jumbled-variant gate found no leak.",
                )
            )

        # 13. PII-reliance report ------------------------------------------
        if variant_report is None:
            checks.append(
                _check(
                    "variant.pii",
                    status="not_applicable",
                    reason="No variant report is available.",
                )
            )
        elif _variant_flag(variant_report, "pii_reliance", "pii_leak"):
            issue = _issue(
                evidence="variant.pii",
                severity="fatal",
                finding="The model relies on personally identifying information.",
                why="The anonymized variant shows that the derivation depends on PII.",
                fix="Remove the identifiers and re-fit on the anonymized variant.",
                chunk="S2-leakage",
            )
            issues.append(issue)
            checks.append(
                _check_from_issues(
                    "variant.pii",
                    [issue],
                    pass_reason="The model does not rely on PII.",
                )
            )
        else:
            checks.append(
                _check(
                    "variant.pii",
                    status="pass",
                    reason="The model does not rely on PII.",
                )
            )

        # Optional corpus-grounded LLM review ------------------------------
        summary = ""
        corpus_version = "none"
        if self.llm is not None:
            corpus = self._resolve_corpus()
            corpus_version = corpus.version if corpus is not None else "missing"
            llm_issues, llm_missing, llm_summary = self._llm_review(
                hypothesis_id=hypothesis_id,
                hypothesis=hypothesis,
                profile=profile,
                collection=collection,
                checks=checks,
                corpus=corpus,
            )
            issues.extend(llm_issues)
            missing_information.extend(llm_missing)
            summary, removed = _sanitize_text(llm_summary)
            if removed and not summary:
                summary = (
                    "The LLM review contained out-of-scope performance language, "
                    "which was removed."
                )

        issues = _dedupe_issues(
            [issue for issue in (_sanitize_issue(item) for item in issues) if issue]
        )
        missing_information = list(
            dict.fromkeys(
                item.strip() for item in missing_information if item.strip()
            )
        )
        verdict = _merge_verdict(issues, missing_information)

        if not summary:
            failed = [check for check in checks if check.status == "fail"]
            flagged = [check for check in checks if check.status == "flag"]
            summary = (
                f"Generation validation ran {len(checks)} deterministic checks "
                f"({len(failed)} failed, {len(flagged)} flagged) and found "
                f"{len(issues)} issue(s)."
            )

        return GenerationValidationResult(
            hypothesis_id=hypothesis_id,
            verdict=verdict,  # type: ignore[arg-type]
            checks=checks,
            issues=issues,
            missing_information=missing_information,
            summary=summary,
            model=self.model,
            corpus_version=corpus_version,
        )

    def _llm_review(
        self,
        *,
        hypothesis_id: str,
        hypothesis: Any,
        profile: Any,
        collection: CollectionMetadata | None,
        checks: Sequence[ValidationCheck],
        corpus: Corpus | None,
    ) -> tuple[list[ValidationIssue], list[str], str]:
        safe_collection = (
            redact_collection(collection) if collection is not None else None
        )
        query = " ".join(
            [
                _get(hypothesis, "rationale", "") or "",
                (safe_collection.sampling if safe_collection else "") or "",
                " ".join(safe_collection.known_biases) if safe_collection else "",
            ]
        )
        chunks, retrieval_warnings = self._retrieve_chunks(corpus, query)
        valid_chunk_ids = {chunk.id for chunk in corpus.chunks} if corpus else set()
        check_text = "\n".join(
            f"- {check.id}: {check.status} ({check.severity}) {check.reason}"
            for check in checks
        )
        user = (
            "Dataset profile:\n"
            + _profile_text(profile)
            + "\n\nCollection metadata:\n"
            + format_collection(safe_collection)
            + "\n\nHypothesis metadata:\n"
            + _hypothesis_text(hypothesis)
            + "\n\nDeterministic checks already run:\n"
            + (check_text or "(none)")
            + "\n\nRetrieved corpus chunks:\n"
            + _chunks_text(chunks)
            + "\n\nReview the generation process, not the performance. Return "
            "issues that cite a chunk id. Use `insufficient` when the supplied "
            "metadata lacks the information needed for a check."
        )
        if retrieval_warnings:
            user += "\n\nRetrieval warnings: " + "; ".join(retrieval_warnings)

        payload: _LLMPayload | None = None
        error: str | None = None
        for attempt in (1, 2):
            try:
                raw = self.llm.generate_json(
                    system=GENERATION_SYSTEM_PROMPT,
                    user=user,
                    schema=GENERATION_SCHEMA,
                    max_tokens=4096,
                    temperature=0.0,
                    purpose="generation_validation",
                )
                if not isinstance(raw, Mapping):
                    raise ValueError("generation validator returned a non-object")
                payload = _LLMPayload.model_validate(raw)
                break
            except (LLMError, ValidationError, ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 2:
                    return (
                        [
                            _issue(
                                evidence="generation_validation.unavailable",
                                severity="major",
                                finding="The LLM generation review did not run.",
                                why=error or "The provider failed.",
                                fix=(
                                    "Rerun the generation validator or inspect "
                                    "the metadata."
                                ),
                            )
                        ],
                        ["LLM generation review"],
                        "",
                    )

        if payload is None:
            return [], [], ""

        issues: list[ValidationIssue] = []
        dropped = 0
        for raw_issue in payload.issues:
            if valid_chunk_ids and raw_issue.chunk not in valid_chunk_ids:
                dropped += 1
                continue
            issues.append(
                ValidationIssue(
                    chunk=raw_issue.chunk,
                    severity=raw_issue.severity,  # type: ignore[arg-type]
                    finding=raw_issue.finding,
                    evidence=raw_issue.evidence or "llm.review",
                    why=raw_issue.why,
                    fix=raw_issue.fix,
                )
            )
        missing = [
            str(item).strip()
            for item in payload.missing_information
            if str(item).strip()
            and not _BANNED_PERFORMANCE_RE.search(str(item))
        ]
        return issues, missing, payload.summary


def validate_generation(
    hypothesis: Any,
    *,
    llm: Any = None,
    embeddings: Any = None,
    settings: Settings | None = None,
    corpus: Corpus | None = None,
    **kwargs: Any,
) -> GenerationValidationResult:
    """Convenience wrapper around :class:`GenerationValidator`."""
    validator = GenerationValidator(
        llm=llm,
        embeddings=embeddings,
        settings=settings,
        corpus=corpus,
    )
    return validator.validate(hypothesis, **kwargs)


# Common alternative spelling used by the pipeline plan.
run_generation_validation = validate_generation
