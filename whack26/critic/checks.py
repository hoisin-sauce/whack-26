"""Deterministic method checks that the LLM critic cannot override."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..hypotheses.validate import (
    target_feature_references,
    validate_code,
)
from ..profile import DatasetProfile
from ..screen import ScreenSummary
from .models import Check, Issue

_CAUSAL_TERMS = re.compile(
    r"\b(?:caus(?:e|es|ed|al|ality)|effect of|leads? to|impact|"
    r"driv(?:e|es|er)|reduc(?:e|es|ed|tion)|increas(?:e|es|ed))\b",
    re.IGNORECASE,
)
_DESIGN_TERMS = re.compile(
    r"randomi[sz]ed|random assignment|experiment|trial|"
    r"quasi-experimental|natural experiment|instrumental variable|"
    r"difference[- ]in[- ]differences|regression discontinuity",
    re.IGNORECASE,
)
_MULTIPLICITY_TERMS = re.compile(
    r"p[- ]value|statistically significant|\bsignificance\b|best model|"
    r"stepwise|selected model|multiple testing|data dredging",
    re.IGNORECASE,
)
_CORRECTION_TERMS = re.compile(
    r"bonferroni|holm|benjamini|fdr|false discovery|"
    r"pre[- ]?specified|preregistered|held[- ]out|holdout",
    re.IGNORECASE,
)

_FIXES = {
    "code.static": "Rewrite the module so it passes the AST and import allowlist.",
    "target.leakage": "Remove the target (and target-derived columns) from the feature set.",
    "split.preprocessing": (
        "Fit every scaler, imputer, and encoder inside the training fold or "
        "after the split."
    ),
    "split.temporal": (
        "Use a time-ordered split and never train on rows dated after the "
        "evaluation rows."
    ),
    "missing.complete_case": (
        "Report the missing-data mechanism and use an explicit imputation or "
        "document the complete-case analysis."
    ),
    "sample.epv": "Reduce the parameter count or add data before fitting this model.",
    "sample.small": "Treat the estimate as exploratory or add more rows.",
    "multiplicity.selection": (
        "Pre-specify one model or report a correction for the search."
    ),
    "claim.causal": (
        "Either weaken the claim to association or state an identification "
        "strategy that supports a causal reading."
    ),
    "model.family": "Use a model family that matches the outcome type.",
    "time.series": (
        "Model the time dependence with lags, differencing, or a time-based split."
    ),
    "research.traceability": (
        "Cite a factor with evidence, or state explicitly why no factor applies."
    ),
    "assumptions.specific": "Replace generic statements with testable assumptions.",
    "reproducibility": "Record the seed, code hash, and data hash.",
}


@dataclass
class DeterministicResult:
    checks: list[Check] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _issue(check: Check) -> Issue | None:
    if check.status not in {"flag", "fail"}:
        return None
    chunk = check.citations[0] if check.citations else "S0-how-to-use"
    return Issue(
        chunk=chunk,
        severity=check.severity if check.severity != "none" else "minor",
        finding=check.reason,
        evidence=check.id,
        why="This is a deterministic pre-check of the derivation.",
        fix=_FIXES.get(check.id, "Revise the method so the claim is supported."),
    )


def _check(
    identifier: str,
    *,
    failed: bool,
    severity: str,
    reason: str,
    citations: Sequence[str],
    applicable: bool = True,
) -> Check:
    if not applicable:
        return Check(
            id=identifier,
            status="not_applicable",
            severity="none",
            reason=reason,
            citations=list(citations),
        )
    if failed:
        return Check(
            id=identifier,
            status="fail" if severity == "fatal" else "flag",
            severity=severity,  # type: ignore[arg-type]
            reason=reason,
            citations=list(citations),
        )
    return Check(
        id=identifier,
        status="pass",
        severity="none",
        reason=reason,
        citations=list(citations),
    )


def _call_names(tree: ast.Module) -> list[tuple[int, str, ast.Call]]:
    calls: list[tuple[int, str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append((node.lineno, node.func.id, node))
        elif isinstance(node.func, ast.Attribute):
            calls.append((node.lineno, node.func.attr, node))
    return calls


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return ""


def _get(hypothesis: Any, name: str, default: Any = None) -> Any:
    """Read a field from the wrapper object or its draft metadata."""
    if hasattr(hypothesis, name):
        return getattr(hypothesis, name)
    draft = getattr(hypothesis, "draft", None)
    if draft is not None and hasattr(draft, name):
        return getattr(draft, name)
    return default


def run_deterministic_checks(
    hypothesis: Any,
    *,
    profile: DatasetProfile,
    screen: ScreenSummary,
    factors: Sequence[Any],
    description: str | None,
) -> DeterministicResult:
    """Run every deterministic check from the method-critic spec."""
    result = DeterministicResult()
    code = str(_get(hypothesis, "code", "") or "")
    target = _get(hypothesis, "target", None) or (
        profile.target.name if profile.target else None
    )
    required = [str(item) for item in _get(hypothesis, "required_columns", [])]
    optional = [str(item) for item in _get(hypothesis, "optional_columns", [])]
    rationale = str(_get(hypothesis, "rationale", "") or "")
    title = str(_get(hypothesis, "title", "") or "")
    claim_type = str(_get(hypothesis, "claim_type", "description") or "description")
    execution = getattr(hypothesis, "execution", None)
    notes = str(
        getattr(execution, "notes", "") or _get(hypothesis, "notes", "") or ""
    )
    factors_used = [
        str(item) for item in _get(hypothesis, "factors_used", []) or []
    ]
    source_ids = [str(item) for item in _get(hypothesis, "source_ids", []) or []]
    assumptions = [
        str(item) for item in _get(hypothesis, "assumptions", []) or []
    ]

    # 1. Static contract ---------------------------------------------------
    static = validate_code(code, target=target)
    result.checks.append(
        _check(
            "code.static",
            failed=not static.ok,
            severity="fatal",
            reason=(
                "The module passes the AST and import allowlist."
                if static.ok
                else "Static validation failed: " + "; ".join(static.messages[:4])
            ),
            citations=["S0-how-to-use"],
        )
    )

    # 2. Target leakage ----------------------------------------------------
    target_refs = target_feature_references(code, target) if target else []
    metadata_leak = bool(target and target in set(required) | set(optional))
    result.checks.append(
        _check(
            "target.leakage",
            failed=metadata_leak or bool(target_refs),
            severity="fatal",
            reason=(
                "The target is not used as a feature."
                if not (metadata_leak or target_refs)
                else (
                    "The target appears on the feature side"
                    + (f" at line(s) {target_refs}" if target_refs else "")
                    + "."
                )
            ),
            citations=["S2-leakage"],
            applicable=target is not None,
        )
    )

    # 3. Preprocessing before the split -----------------------------------
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    calls = _call_names(tree)
    split_lines = [line for line, name, _ in calls if name == "train_test_split"]
    split_line = min(split_lines) if split_lines else None
    pre_split_fits: list[int] = []
    if split_line is not None:
        for line, name, node in calls:
            if name not in {"fit", "fit_transform", "fit_resample"}:
                continue
            if line >= split_line:
                continue
            argument_text = " ".join(_unparse(argument) for argument in node.args).lower()
            if "train" in argument_text:
                continue
            pre_split_fits.append(line)
    result.checks.append(
        _check(
            "split.preprocessing",
            failed=bool(pre_split_fits),
            severity="major",
            reason=(
                "No preprocessing is fitted before a split."
                if not pre_split_fits
                else "Preprocessing is fitted before train_test_split at line(s) "
                + ", ".join(str(line) for line in pre_split_fits)
                + "."
            ),
            citations=["S2-leakage"],
            applicable=split_line is not None,
        )
    )

    # 4. Temporal split ----------------------------------------------------
    time_column = profile.time_column
    shift_negative = bool(re.search(r"\.shift\(\s*-\s*\d", code))
    result.checks.append(
        _check(
            "split.temporal",
            failed=shift_negative,
            severity="fatal",
            reason=(
                "No negative shift is used."
                if not shift_negative
                else "The code shifts values from the future into the past."
            ),
            citations=["S9-time-series"],
            applicable=bool(time_column),
        )
    )

    # 5. Silent complete-case analysis ------------------------------------
    uses_dropna = ".dropna(" in code
    high_missing = [name for name, pct in screen.missingness.items() if pct > 5.0]
    mentioned_missing = any(
        token in (notes + " " + rationale).lower()
        for token in ("missing", "impute", "drop", "complete-case", "complete case")
    )
    complete_case = uses_dropna and bool(high_missing) and not mentioned_missing
    result.checks.append(
        _check(
            "missing.complete_case",
            failed=complete_case,
            severity="major",
            reason=(
                "Missing values are handled with an explicit rule."
                if not complete_case
                else "The code drops rows while columns have >5% missing data "
                "and the note does not explain the rule."
            ),
            citations=["S8-missing-measurement"],
            applicable=uses_dropna,
        )
    )

    # 6. Events per variable ----------------------------------------------
    parameters = max(1, len(required) + len(optional))
    events: int | None = None
    if screen.target.kind == "categorical" and screen.target.top_values:
        counts = [count for _, count in screen.target.top_values]
        if len(counts) == 2:
            events = min(counts)
    elif screen.target.kind == "numeric":
        events = profile.rows
    count_model = bool(
        re.search(r"\bpoisson\b|negative\s*binomial|negbin", code, re.IGNORECASE)
    )
    threshold = 5 if count_model else 10
    epv_failed = events is not None and events / parameters < threshold
    result.checks.append(
        _check(
            "sample.epv",
            failed=epv_failed,
            severity="major",
            reason=(
                f"{events} events for {parameters} parameter(s); "
                f"the threshold is {threshold} per parameter."
                if events is not None
                else "No target was available to estimate events per parameter."
            ),
            citations=["S7-sample-size"],
            applicable=events is not None,
        )
    )

    # 7. Small sample ------------------------------------------------------
    result.checks.append(
        _check(
            "sample.small",
            failed=profile.rows < 100 and parameters >= 2,
            severity="minor",
            reason=(
                f"The dataset has {profile.rows} rows and the model has "
                f"{parameters} parameter(s)."
            ),
            citations=["S7-sample-size"],
        )
    )

    # 8. Multiplicity and selection ---------------------------------------
    text = f"{title} {rationale} {code}"
    searches = bool(_MULTIPLICITY_TERMS.search(text))
    corrected = bool(_CORRECTION_TERMS.search(text))
    result.checks.append(
        _check(
            "multiplicity.selection",
            failed=searches and not corrected,
            severity="major",
            reason=(
                "No undisclosed best-model search was detected."
                if not (searches and not corrected)
                else "The derivation claims a best or notable result without a "
                "stated correction or pre-specification."
            ),
            citations=["S3-multiplicity"],
        )
    )

    # 9. Causal language without a design ---------------------------------
    has_design = bool(_DESIGN_TERMS.search(description or ""))
    causal = claim_type == "causal" or bool(
        _CAUSAL_TERMS.search(f"{title} {rationale}")
    )
    result.checks.append(
        _check(
            "claim.causal",
            failed=causal and not has_design,
            severity="major",
            reason=(
                "The claim is observational and uses associative language."
                if not (causal and not has_design)
                else "The claim uses causal language but the data description "
                "does not identify a randomized or quasi-experimental design."
            ),
            citations=["S4-causal-predictive"],
        )
    )

    # 10. Model family -----------------------------------------------------
    lowered = code.lower()
    family_failed = False
    family_reason = "The model family matches the outcome type."
    target_kind = screen.target.kind
    if target_kind == "numeric" and (
        "logistic" in lowered or "sigmoid" in lowered or "logit" in lowered
    ):
        family_failed = True
        family_reason = "A classification link is used for a continuous target."
    elif target_kind == "categorical":
        top = screen.target.top_values or []
        binary = len(top) == 2
        if binary and "poisson" in lowered:
            family_failed = True
            family_reason = "A count model is used for a binary target."
        elif not binary and (
            "lstsq" in lowered
            and "logistic" not in lowered
            and "softmax" not in lowered
        ):
            family_failed = True
            family_reason = "A linear model is used for a multi-class target."
    result.checks.append(
        _check(
            "model.family",
            failed=family_failed,
            severity="fatal",
            reason=family_reason,
            citations=["S6-count-models"] if count_model else ["S1-assumptions"],
        )
    )

    # 11. Time-series dependence ------------------------------------------
    has_time_handling = bool(
        re.search(r"\.shift\(|\.diff\(|seasonal|rolling\(|expanding\(", code)
    ) or ("sort_values" in code and "train_test_split" in code)
    result.checks.append(
        _check(
            "time.series",
            failed=bool(time_column) and not has_time_handling,
            severity="major",
            reason=(
                "No time column is present."
                if not time_column
                else (
                    "Time dependence is modelled or the data are split by time."
                    if has_time_handling
                    else "A time column exists but the code ignores lag, differencing, "
                    "and time-based splitting."
                )
            ),
            citations=["S9-time-series"],
            applicable=bool(time_column),
        )
    )

    # 12. Research traceability -------------------------------------------
    by_id = {str(getattr(factor, "id", "")): factor for factor in factors}
    missing_evidence: list[str] = []
    unknown: list[str] = []
    for identifier in factors_used:
        factor = by_id.get(identifier)
        if factor is None:
            unknown.append(identifier)
        elif not getattr(factor, "evidence_ids", []):
            missing_evidence.append(identifier)
    no_factor_justified = bool(
        re.search(
            r"no (?:applicable )?factor|none of the factors|domain knowledge|"
            r"no mapped factor",
            rationale,
            re.IGNORECASE,
        )
    )
    trace_failed = bool(
        unknown
        or missing_evidence
        or (factors_used and not source_ids)
        or (not factors_used and not no_factor_justified)
    )
    if not trace_failed:
        trace_reason = (
            "Every used factor has an evidence id, or the absence of a factor "
            "is justified explicitly."
        )
    else:
        parts: list[str] = []
        if unknown:
            parts.append("unknown factor ids: " + ", ".join(unknown))
        if missing_evidence:
            parts.append("factors without evidence: " + ", ".join(missing_evidence))
        if factors_used and not source_ids:
            parts.append("no source ids are cited")
        if not factors_used and not no_factor_justified:
            parts.append("no factor is used and no explicit justification is given")
        trace_reason = "Factor/source traceability is incomplete: " + "; ".join(parts)
    result.checks.append(
        _check(
            "research.traceability",
            failed=trace_failed,
            severity="major",
            reason=trace_reason,
            citations=["S10-reporting"],
        )
    )

    # 13. Specific assumptions --------------------------------------------
    generic = (
        not assumptions
        or any(
            len(item.split()) < 3
            or item.strip().lower() in {"all assumptions met", "none", "n/a"}
            for item in assumptions
        )
    )
    result.checks.append(
        _check(
            "assumptions.specific",
            failed=generic,
            severity="minor",
            reason=(
                "At least one specific assumption is stated."
                if not generic
                else "The assumptions are empty, one-word, or generic."
            ),
            citations=["S1-assumptions"],
        )
    )

    # 14. Reproducibility --------------------------------------------------
    code_sha = str(getattr(hypothesis, "code_sha256", "") or "")
    data_sha = str(getattr(profile, "sha256", "") or "")
    uses_randomness = bool(
        re.search(r"np\.random|random\.|numpy\.random", code)
    )
    seeded = "seed" in code and ".seed(" in code
    reproducible_failure = not code_sha or not data_sha or (
        uses_randomness and not seeded
    )
    result.checks.append(
        _check(
            "reproducibility",
            failed=reproducible_failure,
            severity="minor",
            reason=(
                "The code hash, data hash, and random seed are present."
                if not reproducible_failure
                else "The derivation is missing a code hash, data hash, or seed."
            ),
            citations=["S10-reporting"],
        )
    )

    result.issues = [
        issue for check in result.checks if (issue := _issue(check)) is not None
    ]
    return result
