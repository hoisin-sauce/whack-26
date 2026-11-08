"""Bounded model-building agent loop.

The loop proposes one hypothesis per family, runs the existing static AST
validator and isolated runner, feeds model-search diagnostics and validation
results back to the proposer, and records every attempt. It never edits the
one-shot generator; callers inject the proposer and, optionally, the executor,
diagnostic, and validation callbacks.

The loop is deliberately conservative:

* exactly three unique families are required and each family is accepted at
  most once;
* every attempt is counted against the per-family and global budgets;
* identical failures are stopped after a small number of repetitions;
* accepting an executed model is the fallback when no diagnostics are supplied;
* the audit contains no generated source, prompts, secrets, or raw dataset rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..common import redact_secrets, utc_now_iso
from ..config import Settings
from ..ids import hypothesis_id
from ..model_search.types import Diagnostic
from .execute import ExecutionResult, TIMEOUT_SECONDS
from .generate import (
    FAMILIES,
    GeneratedHypothesis,
    HypothesisContext,
    HypothesisDraft,
    HypothesisGenerationError,
    HypothesisSet,
)
from .validate import validate_code, validate_hypothesis

Decision = Literal["accept", "refine", "replace", "discard"]
Phase = Literal["initial", "refine", "replace"]

_MAX_IDENTICAL_FAILURES = 3
_MAX_AUDIT_TEXT = 1500
_MAX_FEEDBACK_CHARS = 5000

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{9,}\b")

_EXECUTION_STATUSES = {
    "ok",
    "applies_false",
    "runtime_error",
    "timeout",
    "invalid_output",
    "runner_error",
}


# ---------------------------------------------------------------------------
# Requests and callback contracts


@dataclass(frozen=True)
class ProposalRequest:
    """Everything the proposer may use for one attempt."""

    family: str
    attempt: int
    total_attempt: int
    phase: Phase
    feedback: "DiagnosticFeedback | None"
    context: HypothesisContext | None
    context_pack: Any | None


@dataclass(frozen=True)
class ExecutionRequest:
    """One draft to execute in the isolated runner."""

    family: str
    attempt: int
    draft: HypothesisDraft
    module_path: Path | None
    dataset_path: Path | None
    seed: int


@dataclass(frozen=True)
class EvaluationRequest:
    """One executed draft to diagnose and validate."""

    family: str
    attempt: int
    draft: HypothesisDraft
    execution: ExecutionResult
    context: HypothesisContext | None
    context_pack: Any | None


GeneratorCallback = Callable[[ProposalRequest], Any]
ExecutorCallback = Callable[[ExecutionRequest], Any]
DiagnosticCallback = Callable[[EvaluationRequest], Any]
ValidationCallback = Callable[[EvaluationRequest], Any]
ContextBuilder = Callable[[], Any]


# ---------------------------------------------------------------------------
# Structured feedback and audit models


class DiagnosticFeedback(BaseModel):
    """Accumulated feedback for the next attempt in one family."""

    family: str
    attempt: int = 1
    phase: Phase = "refine"
    summary: str = ""
    static_errors: list[str] = Field(default_factory=list)
    static_warnings: list[str] = Field(default_factory=list)
    execution_status: str | None = None
    execution_error: str | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    validation_verdict: str | None = None
    validation_issues: list[str] = Field(default_factory=list)
    leak_flags: list[str] = Field(default_factory=list)
    repeated_failures: int = 0
    hints: list[str] = Field(default_factory=list)

    def as_prompt(self, max_chars: int = _MAX_FEEDBACK_CHARS) -> str:
        """Render the feedback as a safe, human-readable prompt block."""
        lines = [
            f"Family: {self.family}",
            f"Attempt: {self.attempt}",
            f"Requested action: {self.phase}",
        ]
        if self.summary:
            lines.append("Summary: " + self.summary)
        if self.static_errors:
            lines.append("Static validation errors:")
            lines.extend(f"- {error}" for error in self.static_errors)
        if self.static_warnings:
            lines.append("Static validation warnings:")
            lines.extend(f"- {warning}" for warning in self.static_warnings)
        if self.execution_status is not None:
            details = f"Execution: status={self.execution_status}"
            if self.execution_error:
                details += f" | error={self.execution_error}"
            lines.append(details)
        if self.diagnostics:
            lines.append("Model-search diagnostics:")
            lines.extend(
                f"- {item.name} ({item.status})"
                + (f": {item.message}" if item.message else "")
                for item in self.diagnostics
            )
        if self.validation_verdict is not None or self.validation_issues:
            lines.append(
                "Generation/statistical validation: verdict="
                + (self.validation_verdict or "unknown")
            )
            lines.extend(f"- {issue}" for issue in self.validation_issues)
        if self.leak_flags:
            lines.append("Leakage flags: " + ", ".join(self.leak_flags))
        if self.repeated_failures > 1:
            lines.append(
                f"The same failure has repeated {self.repeated_failures} times. "
                "Change the model structure instead of patching the same draft."
            )
        if self.hints:
            lines.append("Hints:")
            lines.extend(f"- {hint}" for hint in self.hints)
        return _clip("\n".join(lines), max_chars)


class ModelAttempt(BaseModel):
    """One proposed model, its execution/diagnostics, and the decision."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    family: str
    draft: HypothesisDraft
    execution: ExecutionResult | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    # GenerationValidationResult, StatisticalValidationResult, ValidationReport,
    # a mapping, or a list of those results.
    validation: Any | None = None
    decision: Decision = "discard"
    reason: str = ""
    attempt: int = 0
    phase: Phase = "initial"
    code_sha256: str = ""
    hypothesis_id: str = ""

    def to_audit(self) -> dict[str, Any]:
        """Return a structured, redacted audit record (no source or rows)."""
        execution = None
        if self.execution is not None:
            execution = {
                "status": self.execution.status,
                "runtime_ms": self.execution.runtime_ms,
                "n_predictions": self.execution.n_predictions,
                "error": _clip(self.execution.error or "", 800),
            }
        return {
            "attempt": self.attempt,
            "family": self.family,
            "phase": self.phase,
            "hypothesis_id": self.hypothesis_id,
            "code_sha256": self.code_sha256,
            "title": _clip(self.draft.title, 200),
            "decision": self.decision,
            "reason": _clip(self.reason, _MAX_AUDIT_TEXT),
            "execution": execution,
            "diagnostics": [
                {
                    "name": item.name,
                    "status": item.status,
                    "value": item.value,
                    "message": _clip(item.message, 600),
                }
                for item in self.diagnostics
            ],
            "validation": _validation_audit(self.validation),
        }


@dataclass
class _ValidationSummary:
    verdict: str | None = None
    status: str | None = None
    issues: list[str] = field(default_factory=list)
    severity: str | None = None
    leak: bool = False


@dataclass
class _Decision:
    decision: Decision
    reason: str
    feedback: DiagnosticFeedback | None = None


# ---------------------------------------------------------------------------
# Small helpers


def _scrub_pii(text: str) -> str:
    """Redact common direct-identifier patterns from audit/feedback text.

    The loop never includes raw dataset rows or prompts. This is a second line
    of defence for exception messages that might echo a value.
    """
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _IPV4_RE.sub("[redacted-ip]", text)
    return _LONG_DIGITS_RE.sub("[redacted-id]", text)


def _clip(text: str, limit: int = _MAX_AUDIT_TEXT) -> str:
    cleaned = _scrub_pii(redact_secrets(str(text)))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _diagnostic_detail(diagnostics: Sequence[Diagnostic]) -> str:
    parts = []
    for item in diagnostics[:5]:
        detail = f"{item.name} ({item.status})"
        if item.message:
            detail += f": {item.message}"
        parts.append(detail)
    return "; ".join(parts) or "no diagnostic detail"


def _is_leak_text(text: str) -> bool:
    """Detect explicit jumbled-label or PII-leakage findings.

    The check is intentionally phrase-based. A neutral note such as "no
    jumbled-label leak detected" must not reject a clean model.
    """
    lower = text.lower()
    if any(
        marker in lower
        for marker in ("jumbled_label_leak", "pii_reliance", "target_leak")
    ):
        return True
    if re.search(r"pii[-\s]?reliance", lower):
        return True
    if re.search(r"target[-\s]?leak(?:age)?", lower) and not re.search(
        r"\b(no|not|without)\s+target[-\s]?leak", lower
    ):
        return True
    if "leakage detected" in lower:
        return True
    if "jumbled" not in lower:
        return False
    if "unscrambl" in lower:
        return True
    if (
        ("permutation null" in lower or "permutation-null" in lower)
        and any(word in lower for word in ("above", "beat", "exceed", "pass"))
        and not re.search(r"\b(no|not|without)\s+jumbled", lower)
    ):
        return True
    if re.search(r"jumbled[-\s]*(label[-\s]*)?leak", lower):
        return not re.search(r"\b(no|not|without)\s+jumbled", lower)
    return False


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _collect(value: Any, key: str) -> list[Any]:
    raw = _get(value, key, None)
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, Mapping)):
        return [raw]
    try:
        return list(raw)
    except TypeError:
        return [raw]


def _issue_severity(item: Any) -> str:
    return str(_get(item, "severity", "") or "").strip().lower()


def _issue_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    parts: list[str] = []
    for key in ("severity", "finding", "evidence", "why", "fix"):
        value = _get(item, key, None)
        if value:
            parts.append(f"{key}: {value}")
    if not parts:
        return str(item)
    return "; ".join(parts)


def _max_severity(severities: Sequence[str]) -> str | None:
    order = {"": 0, "none": 0, "minor": 1, "major": 2, "fatal": 3}
    best = max(severities, key=lambda item: order.get(item, 0), default="")
    return best or None


def _infer_severity(text: str) -> str:
    lower = text.lower()
    if "fatal" in lower:
        return "fatal"
    if "major" in lower or "reject" in lower or _is_leak_text(lower):
        return "major"
    if "minor" in lower:
        return "minor"
    return ""


def _normalize_verdict(verdict: Any, status: Any) -> str | None:
    value = str(verdict).strip().lower() if verdict is not None else ""
    state = str(status).strip().lower() if status is not None else ""
    if value in {"pass", "refine", "reject", "insufficient"}:
        return value
    if value in {"failed", "fail"}:
        return "reject"
    if value in {"inconclusive", "unknown", "skipped", "partial", "error"}:
        return "insufficient"
    if not value and state:
        if state == "passed":
            return "pass"
        if state == "failed":
            return "reject"
        if state in {"partial", "skipped", "error"}:
            return "insufficient"
    return value or None


def _summarize_one(value: Any) -> _ValidationSummary:
    issues_raw = list(_collect(value, "issues"))
    for check in _collect(value, "checks"):
        check_status = str(_get(check, "status", "") or "").lower()
        check_severity = str(_get(check, "severity", "") or "").lower()
        if check_status in {"fail", "flag"} or check_severity in {"fatal", "major"}:
            issues_raw.append(
                {
                    "severity": check_severity
                    or ("fatal" if check_status == "fail" else "major"),
                    "finding": "check "
                    + str(_get(check, "id", "") or "(unnamed)")
                    + ": "
                    + str(_get(check, "reason", "") or check_status),
                }
            )
    for layer in _collect(value, "layers"):
        layer_status = str(_get(layer, "status", "") or "").lower()
        if layer_status in {"failed", "error"}:
            issues_raw.append(
                {
                    "severity": "fatal" if layer_status == "failed" else "major",
                    "finding": "layer "
                    + str(_get(layer, "layer", "") or "(unnamed)")
                    + ": "
                    + str(_get(layer, "reason", "") or layer_status),
                }
            )
    for item in _collect(value, "missing_information"):
        issues_raw.append(
            {"severity": "major", "finding": f"missing information: {item}"}
        )
    error = _get(value, "error", None)
    if error:
        issues_raw.append({"severity": "major", "finding": f"error: {error}"})
    reason = _get(value, "reason", None)
    if reason:
        raw_verdict = str(_get(value, "verdict", "") or "").lower()
        reason_severity = (
            "fatal"
            if raw_verdict in {"fail", "failed", "reject"}
            else "major"
            if raw_verdict == "refine"
            else "minor"
        )
        issues_raw.append(
            {"severity": reason_severity, "finding": f"reason: {reason}"}
        )
    for item in _collect(value, "diagnostics"):
        item_status = str(_get(item, "status", "") or "").lower()
        if item_status in {"warn", "fail"}:
            issues_raw.append(
                {
                    "severity": "fatal" if item_status == "fail" else "major",
                    "finding": "validation diagnostic "
                    + str(_get(item, "name", "") or "(unnamed)")
                    + ": "
                    + str(_get(item, "message", "") or ""),
                }
            )
    for warning in _collect(value, "warnings"):
        issues_raw.append({"severity": "minor", "finding": f"warning: {warning}"})

    issues = [_issue_text(item) for item in issues_raw]
    severity = _max_severity([_issue_severity(item) for item in issues_raw])
    verdict = _normalize_verdict(
        _get(value, "verdict", None), _get(value, "status", None)
    )
    status = _get(value, "status", None)
    leak = _is_leak_text(
        " ".join(
            [
                str(_get(value, "verdict", "") or ""),
                str(status or ""),
                str(_get(value, "summary", "") or ""),
                *issues,
            ]
        )
    )
    report = _get(value, "report", None)
    if report is not None and report is not value:
        report_summary = _summarize_one(report)
        issues.extend(report_summary.issues)
        severity = _max_severity(
            [severity or "", report_summary.severity or ""]
        )
        leak = leak or report_summary.leak
        if verdict is None:
            verdict = report_summary.verdict
    return _ValidationSummary(
        verdict=verdict,
        status=str(status) if status is not None else None,
        issues=issues,
        severity=severity,
        leak=leak,
    )


def _summarize_validation(value: Any) -> _ValidationSummary:
    if value is None:
        return _ValidationSummary()
    if isinstance(value, (list, tuple)):
        if not value:
            return _ValidationSummary()
        if all(isinstance(item, str) for item in value):
            issues = [str(item) for item in value]
            return _ValidationSummary(
                issues=issues,
                severity=_max_severity([_infer_severity(item) for item in issues]),
                leak=any(_is_leak_text(item) for item in issues),
            )
        parts = [_summarize_one(item) for item in value if item is not None]
        if not parts:
            return _ValidationSummary()
        verdicts = [part.verdict for part in parts if part.verdict]
        if "reject" in verdicts:
            verdict = "reject"
        elif "refine" in verdicts:
            verdict = "refine"
        elif any(item == "pass" for item in verdicts):
            verdict = "pass"
        elif verdicts:
            verdict = "insufficient"
        else:
            verdict = None
        statuses = [part.status for part in parts if part.status]
        return _ValidationSummary(
            verdict=verdict,
            status=statuses[0] if statuses else None,
            issues=[issue for part in parts for issue in part.issues],
            severity=_max_severity(
                [part.severity or "" for part in parts]
            ),
            leak=any(part.leak for part in parts),
        )
    return _summarize_one(value)


def _validation_audit(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    summary = _summarize_validation(value)
    return {
        "verdict": summary.verdict,
        "status": summary.status,
        "severity": summary.severity,
        "leak": summary.leak,
        "issues": [_clip(issue, 600) for issue in summary.issues[:10]],
    }


def _coerce_diagnostic(value: Any) -> Diagnostic:
    if isinstance(value, Diagnostic):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        name = str(data.get("name") or "diagnostic")
        status = str(data.get("status") or "warn").lower()
        if status not in {"ok", "warn", "fail", "not_applicable"}:
            status = "warn"
        raw_value = data.get("value")
        try:
            numeric = float(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            numeric = None
        return Diagnostic(
            name=name,
            status=status,  # type: ignore[arg-type]
            value=numeric,
            message=str(data.get("message") or ""),
        )
    name = str(getattr(value, "name", None) or "diagnostic")
    status = str(getattr(value, "status", None) or "warn").lower()
    if status not in {"ok", "warn", "fail", "not_applicable"}:
        status = "warn"
    raw_value = getattr(value, "value", None)
    try:
        numeric = float(raw_value) if raw_value is not None else None
    except (TypeError, ValueError):
        numeric = None
    return Diagnostic(
        name=name,
        status=status,  # type: ignore[arg-type]
        value=numeric,
        message=str(getattr(value, "message", "") or ""),
    )


def _model_search_diagnostics(value: Any) -> list[Any] | None:
    """Extract per-candidate diagnostics from a model-search result.

    This lets callers return a ``ModelSearchResult`` or ``ModelCandidate``
    directly from the diagnostic callback instead of flattening it first.
    """
    if isinstance(value, (Diagnostic, Mapping, str, bytes)):
        return None
    candidates = getattr(value, "candidates", None)
    if candidates is not None:
        selected = getattr(value, "selected", None)
        chosen = [selected] if selected is not None else list(candidates)
        collected: list[Any] = []
        for candidate in chosen:
            collected.extend(getattr(candidate, "diagnostics", None) or [])
        return collected
    diagnostics = getattr(value, "diagnostics", None)
    if diagnostics is not None:
        return list(diagnostics)
    return None


def _coerce_diagnostics(value: Any) -> list[Diagnostic]:
    if value is None:
        return []
    extracted = _model_search_diagnostics(value)
    if extracted is not None:
        value = extracted
    if isinstance(value, Diagnostic) or isinstance(value, Mapping):
        items: list[Any] = [value]
    elif isinstance(value, (str, bytes)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    result: list[Diagnostic] = []
    for item in items:
        try:
            result.append(_coerce_diagnostic(item))
        except Exception as exc:  # noqa: BLE001 - keep the loop bounded
            result.append(
                Diagnostic(
                    name="diagnostic_parse_error",
                    status="warn",
                    message=f"{type(exc).__name__}: {item}",
                )
            )
    return result


# ---------------------------------------------------------------------------
# The loop


class ModelAgentLoop:
    """Run a bounded propose -> execute -> diagnose -> decide loop."""

    def __init__(
        self,
        *,
        generator: GeneratorCallback,
        context: HypothesisContext | None = None,
        settings: Settings | None = None,
        run_dir: Path | None = None,
        dataset_path: Path | None = None,
        seed: int = 0,
        executor: ExecutorCallback | None = None,
        diagnostic: DiagnosticCallback | None = None,
        validator: ValidationCallback | None = None,
        context_pack: Any | None = None,
        context_builder: ContextBuilder | None = None,
        families: Sequence[str] = FAMILIES,
        max_identical_failures: int = _MAX_IDENTICAL_FAILURES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(generator):
            raise ValueError("ModelAgentLoop requires a callable generator callback")
        normalized = tuple(str(family) for family in families)
        if len(normalized) != 3 or len(set(normalized)) != 3:
            raise ValueError(
                "ModelAgentLoop requires exactly three unique families; got "
                f"{normalized!r}"
            )
        if set(normalized) != set(FAMILIES):
            raise ValueError(
                f"families must be exactly {FAMILIES}; got {normalized!r}"
            )
        if max_identical_failures < 1:
            raise ValueError("max_identical_failures must be positive")

        self.generator = generator
        self.context = context
        self.settings = settings or Settings()
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.dataset_path = (
            Path(dataset_path) if dataset_path is not None else None
        )
        self.seed = seed
        self.executor = executor
        self.diagnostic = diagnostic
        self.validator = validator
        self.context_pack = context_pack
        self.context_builder = context_builder
        self.families = normalized
        self.max_identical_failures = max_identical_failures
        self._clock = clock

        self._attempts: list[ModelAttempt] = []
        self._accepted: list[GeneratedHypothesis] = []
        self._warnings: list[str] = []
        self._failure_counts: dict[str, Counter[str]] = {}
        self._family_results: dict[str, dict[str, Any]] = {}
        self._total_attempts = 0
        self._last_elapsed = 0.0
        self._stop_reason: str | None = None
        self._status = "idle"
        self._error: str | None = None
        self._started_at: str | None = None
        self._audit_write_failed = False

    # ------------------------------------------------------------------
    # Public state

    @property
    def attempts(self) -> list[ModelAttempt]:
        return list(self._attempts)

    @property
    def accepted(self) -> list[GeneratedHypothesis]:
        return list(self._accepted)

    @property
    def family_results(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._family_results.items()}

    @property
    def audit_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "agent" / "attempts.json"

    # ------------------------------------------------------------------
    # Run

    def run(self) -> HypothesisSet:
        self._reset_run()
        if self.executor is None and self.dataset_path is None:
            raise ValueError(
                "ModelAgentLoop needs dataset_path when no executor callback is "
                "supplied"
            )
        self._started_at = utc_now_iso()
        self._status = "running"
        started = self._clock()

        context_pack = self.context_pack
        if context_pack is None and self.context_builder is not None:
            try:
                context_pack = self.context_builder()
            except Exception as exc:  # noqa: BLE001 - context is advisory
                self._add_warning(
                    f"context_pack_error: {type(exc).__name__}: {exc}"
                )

        try:
            for family in self.families:
                generated, global_stop = self._run_family(
                    family, started, context_pack
                )
                if generated is not None:
                    self._accepted.append(generated)
                if global_stop:
                    break

            accepted_families = [item.method_family for item in self._accepted]
            if (
                len(accepted_families) != len(self.families)
                or set(accepted_families) != set(self.families)
            ):
                raise self._failure_error()

            self._status = "completed"
            self._write_audit()
            return HypothesisSet(
                hypotheses=list(self._accepted),
                warnings=list(self._warnings),
            )
        except HypothesisGenerationError as exc:
            self._status = "failed"
            self._error = str(exc)
            self._write_audit()
            raise
        except Exception as exc:  # noqa: BLE001 - audit unexpected failures too
            self._status = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
            self._write_audit()
            raise

    def _reset_run(self) -> None:
        self._attempts = []
        self._accepted = []
        self._warnings = []
        self._failure_counts = {}
        self._family_results = {}
        self._total_attempts = 0
        self._last_elapsed = 0.0
        self._stop_reason = None
        self._status = "idle"
        self._error = None
        self._started_at = None
        self._audit_write_failed = False

    # ------------------------------------------------------------------
    # Per-family loop

    def _run_family(
        self,
        family: str,
        started: float,
        context_pack: Any | None,
    ) -> tuple[GeneratedHypothesis | None, bool]:
        info: dict[str, Any] = {
            "accepted": False,
            "attempts": 0,
            "stop_reason": None,
        }
        self._family_results[family] = info
        feedback: DiagnosticFeedback | None = None

        while True:
            budget = self._budget_stop(family, int(info["attempts"]), started)
            if budget is not None:
                reason, is_global = budget
                info["stop_reason"] = reason
                if is_global:
                    self._stop_reason = reason
                return None, is_global

            attempt_no = int(info["attempts"]) + 1
            info["attempts"] = attempt_no
            self._total_attempts += 1
            phase: Phase = (
                "initial"
                if attempt_no == 1
                else (feedback.phase if feedback is not None else "refine")
            )

            draft, proposal_error, proposal_warnings = self._propose(
                ProposalRequest(
                    family=family,
                    attempt=attempt_no,
                    total_attempt=self._total_attempts,
                    phase=phase,
                    feedback=feedback,
                    context=self.context,
                    context_pack=context_pack,
                )
            )
            static_errors: list[str] = []
            static_warnings: list[str] = list(proposal_warnings)
            if proposal_error:
                static_errors.append(proposal_error)
            else:
                static_errors, more_warnings = self._static_checks(draft)
                static_warnings.extend(more_warnings)

            hyp_id, code_sha = self._identify(draft)
            execution: ExecutionResult | None = None
            if not static_errors:
                module_path = self._write_module(draft, hyp_id)
                execution = self._execute(
                    ExecutionRequest(
                        family=family,
                        attempt=attempt_no,
                        draft=draft,
                        module_path=module_path,
                        dataset_path=self.dataset_path,
                        seed=self.seed,
                    )
                )

            diagnostics: list[Diagnostic] = []
            validation: Any | None = None
            if execution is not None:
                evaluation = EvaluationRequest(
                    family=family,
                    attempt=attempt_no,
                    draft=draft,
                    execution=execution,
                    context=self.context,
                    context_pack=context_pack,
                )
                diagnostics = self._diagnose(evaluation)
                validation = self._validate(evaluation)

            decision = self._evaluate(
                family=family,
                attempt_no=attempt_no,
                phase=phase,
                draft=draft,
                static_errors=static_errors,
                static_warnings=static_warnings,
                execution=execution,
                diagnostics=diagnostics,
                validation=validation,
            )

            attempt = ModelAttempt(
                family=family,
                draft=draft,
                execution=execution,
                diagnostics=diagnostics,
                validation=validation,
                decision=decision.decision,
                reason=decision.reason,
                attempt=attempt_no,
                phase=phase,
                code_sha256=code_sha,
                hypothesis_id=hyp_id,
            )
            self._attempts.append(attempt)
            for warning in static_warnings:
                self._add_warning(warning)
            self._write_audit()

            if decision.decision == "accept":
                info["accepted"] = True
                info["stop_reason"] = None
                generated = GeneratedHypothesis(
                    id=hyp_id,
                    method_family=cast(Any, family),
                    draft=draft,
                    code_sha256=code_sha,
                    code_path=f"hypotheses/{hyp_id}.py",
                    execution=cast(ExecutionResult, execution),
                    generation_history=self._generation_history(family),
                    validation_warnings=list(static_warnings),
                )
                return generated, False
            if decision.decision == "discard":
                info["stop_reason"] = decision.reason
                return None, False

            feedback = decision.feedback

    # ------------------------------------------------------------------
    # Budgets and failures

    def _budget_stop(
        self, family: str, family_attempts: int, started: float
    ) -> tuple[str, bool] | None:
        elapsed = max(0.0, self._clock() - started)
        self._last_elapsed = elapsed
        if elapsed >= self.settings.agent_max_seconds:
            return (
                f"time budget exhausted after {elapsed:.1f}s "
                f"(limit {self.settings.agent_max_seconds}s)",
                True,
            )
        if self._total_attempts >= self.settings.agent_max_total_attempts:
            return (
                f"total attempt budget exhausted after {self._total_attempts} "
                f"attempt(s) (limit {self.settings.agent_max_total_attempts})",
                True,
            )
        if family_attempts >= self.settings.agent_max_attempts_per_family:
            return (
                f"per-family attempt budget exhausted for {family!r} after "
                f"{family_attempts} attempt(s) "
                f"(limit {self.settings.agent_max_attempts_per_family})",
                False,
            )
        return None

    def _record_failure(
        self, family: str, draft: HypothesisDraft, category: str, detail: str
    ) -> int:
        signature = (
            hashlib.sha256(draft.code.encode("utf-8")).hexdigest()
            + "|"
            + category
            + "|"
            + detail
        )
        counter = self._failure_counts.setdefault(family, Counter())
        counter[signature] += 1
        return counter[signature]

    # ------------------------------------------------------------------
    # Proposal, static validation, execution, diagnostics, validation

    def _propose(
        self, request: ProposalRequest
    ) -> tuple[HypothesisDraft, str | None, list[str]]:
        try:
            proposal = self.generator(request)
        except Exception as exc:  # noqa: BLE001 - recorded as a proposal failure
            return (
                self._placeholder_draft(request.family),
                f"proposal_error: {type(exc).__name__}: {exc}",
                [],
            )
        if proposal is None:
            return (
                self._placeholder_draft(request.family),
                "proposal_error: generator returned None",
                [],
            )
        mismatch = self._proposal_family_mismatch(proposal, request.family)
        if mismatch is not None:
            return self._placeholder_draft(request.family), mismatch, []
        try:
            return self._to_draft(proposal), None, []
        except (ValidationError, TypeError, ValueError) as exc:
            return (
                self._placeholder_draft(request.family),
                f"proposal_error: invalid hypothesis draft: {exc}",
                [],
            )

    def _proposal_family_mismatch(
        self, proposal: Any, expected: str
    ) -> str | None:
        if isinstance(proposal, Mapping):
            actual = proposal.get("method_family") or proposal.get("family")
        else:
            actual = getattr(proposal, "method_family", None) or getattr(
                proposal, "family", None
            )
        if actual is None:
            return None
        if str(actual) != expected:
            return (
                "proposal_error: generator returned family "
                f"{actual!r} for requested family {expected!r}"
            )
        return None

    def _to_draft(self, proposal: Any) -> HypothesisDraft:
        if isinstance(proposal, HypothesisDraft):
            return proposal
        if isinstance(proposal, GeneratedHypothesis):
            return proposal.draft
        if isinstance(proposal, Mapping):
            data = dict(proposal)
            data.pop("method_family", None)
            data.pop("family", None)
            return HypothesisDraft.model_validate(data)
        nested = getattr(proposal, "draft", None)
        if isinstance(nested, HypothesisDraft):
            return nested
        raise TypeError(
            "generator must return a HypothesisDraft, GeneratedHypothesis, "
            f"or mapping; got {type(proposal).__name__}"
        )

    def _placeholder_draft(self, family: str) -> HypothesisDraft:
        return HypothesisDraft(
            title="proposal failed",
            rationale=(
                "The generator callback did not return a valid hypothesis draft "
                f"for family {family}."
            ),
            assumptions=["The next attempt will return a complete draft."],
            code="",
        )

    def _static_checks(
        self, draft: HypothesisDraft
    ) -> tuple[list[str], list[str]]:
        if self.context is None:
            static = validate_code(draft.code, target=draft.target)
            return list(static.messages), []
        draft.target = self.context.target
        validation = validate_hypothesis(
            draft,
            target=self.context.target,
            dataset_columns=self.context.dataset_columns,
            known_factor_ids=self.context.factor_ids,
            known_source_ids=self.context.source_ids,
        )
        return list(validation.errors), list(validation.warnings)

    def _identify(self, draft: HypothesisDraft) -> tuple[str, str]:
        code_sha = hashlib.sha256(draft.code.encode("utf-8")).hexdigest()
        return hypothesis_id(draft.code, draft.factors_used), code_sha

    def _write_module(
        self, draft: HypothesisDraft, hyp_id: str
    ) -> Path | None:
        if self.run_dir is None:
            if self.executor is not None:
                return None
            self.run_dir = Path(tempfile.mkdtemp(prefix="whack26-agent-"))
        module_dir = self.run_dir / "hypotheses"
        module_dir.mkdir(parents=True, exist_ok=True)
        module_path = module_dir / f"{hyp_id}.py"
        module_path.write_text(draft.code, encoding="utf-8")
        return module_path

    def _execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.executor is not None:
            try:
                value = self.executor(request)
            except Exception as exc:  # noqa: BLE001 - feedback, not a crash
                return ExecutionResult(
                    status="runner_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            return self._coerce_execution(value)
        if request.module_path is None or self.dataset_path is None:
            return ExecutionResult(
                status="runner_error",
                error=(
                    "default executor needs a module path and a dataset path; "
                    "supply an executor callback or dataset_path"
                ),
            )
        try:
            # Lazy import keeps the one-shot path independent of the loop.
            from .execute import run_module

            remaining = (
                self.settings.agent_max_seconds - self._last_elapsed
            )
            timeout = max(1.0, min(TIMEOUT_SECONDS, remaining))
            return run_module(
                request.module_path,
                self.dataset_path,
                self.seed,
                run_dir=cast(Path, self.run_dir),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - reported through execution
            return ExecutionResult(
                status="runner_error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _coerce_execution(self, value: Any) -> ExecutionResult:
        if isinstance(value, ExecutionResult):
            return value
        if isinstance(value, GeneratedHypothesis):
            return value.execution
        if isinstance(value, Mapping):
            data = dict(value)
            status = str(data.get("status") or "runner_error")
            data["status"] = (
                status if status in _EXECUTION_STATUSES else "runner_error"
            )
            try:
                return ExecutionResult(**data)
            except TypeError as exc:
                return ExecutionResult(
                    status="runner_error",
                    error=f"executor returned an invalid result mapping: {exc}",
                )
        return ExecutionResult(
            status="runner_error",
            error=(
                "executor must return an ExecutionResult or mapping; got "
                f"{type(value).__name__}"
            ),
        )

    def _diagnose(self, request: EvaluationRequest) -> list[Diagnostic]:
        if self.diagnostic is None:
            return []
        try:
            value = self.diagnostic(request)
        except Exception as exc:  # noqa: BLE001 - one warning, then bound
            return [
                Diagnostic(
                    name="diagnostic_error",
                    status="warn",
                    message=f"{type(exc).__name__}: {exc}",
                )
            ]
        return _coerce_diagnostics(value)

    def _validate(self, request: EvaluationRequest) -> Any | None:
        if self.validator is None:
            return None
        try:
            value = self.validator(request)
        except Exception as exc:  # noqa: BLE001 - request a refinement
            return {
                "verdict": "refine",
                "status": "error",
                "issues": [
                    {
                        "severity": "major",
                        "finding": f"validation_error: {type(exc).__name__}: {exc}",
                    }
                ],
                "summary": "validator raised an exception",
            }
        if isinstance(value, tuple):
            return list(value)
        return value

    # ------------------------------------------------------------------
    # Decision

    def _evaluate(
        self,
        *,
        family: str,
        attempt_no: int,
        phase: Phase,
        draft: HypothesisDraft,
        static_errors: list[str],
        static_warnings: list[str],
        execution: ExecutionResult | None,
        diagnostics: list[Diagnostic],
        validation: Any | None,
    ) -> _Decision:
        common = {
            "family": family,
            "attempt_no": attempt_no,
            "phase": phase,
            "draft": draft,
            "static_errors": static_errors,
            "static_warnings": static_warnings,
            "execution": execution,
            "diagnostics": diagnostics,
            "validation": validation,
        }

        if static_errors:
            detail = "; ".join(static_errors)
            repeated = self._record_failure(
                family, draft, "static", detail
            )
            if repeated >= self.max_identical_failures:
                return _Decision(
                    "discard",
                    f"static validation failed {repeated} times for the same "
                    f"draft; discard this family: {detail}",
                )
            if repeated >= 2:
                return _Decision(
                    "replace",
                    "static validation failed again for the same draft; "
                    f"replace the model for this family: {detail}",
                    self._feedback(
                        **common,
                        decision="replace",
                        reason="static validation failed again",
                        repeated=repeated,
                        hints=["Fix every static violation and return the "
                               "complete module."],
                    ),
                )
            return _Decision(
                "refine",
                f"static validation failed; refine the same family: {detail}",
                self._feedback(
                    **common,
                    decision="refine",
                    reason="static validation failed",
                    repeated=repeated,
                    hints=["Fix the static validation violations and return the "
                           "complete corrected module."],
                ),
            )

        if execution is None or not execution.ok:
            status = execution.status if execution is not None else "missing"
            error = (
                execution.error
                if execution is not None and execution.error
                else "execution did not return a result"
            )
            detail = f"{status}: {error}"
            repeated = self._record_failure(
                family, draft, "execution", detail
            )
            if repeated >= self.max_identical_failures:
                return _Decision(
                    "discard",
                    f"execution failed {repeated} times for the same draft; "
                    f"discard this family: {detail}",
                )
            if repeated >= 2:
                return _Decision(
                    "replace",
                    "execution failed again for the same draft; replace the "
                    f"model for this family: {detail}",
                    self._feedback(
                        **common,
                        decision="replace",
                        reason="execution failed again",
                        repeated=repeated,
                        hints=["Fix the runtime/contract failure and return the "
                               "complete module."],
                    ),
                )
            return _Decision(
                "refine",
                f"execution failed; refine the same family: {detail}",
                self._feedback(
                    **common,
                    decision="refine",
                    reason="execution failed",
                    repeated=repeated,
                    hints=["Fix the runtime/contract failure and return the "
                           "complete corrected module."],
                ),
            )

        summary = _summarize_validation(validation)
        leak_flags = self._leak_flags(diagnostics, summary)
        if leak_flags:
            return _Decision(
                "discard",
                "leakage detected ("
                + ", ".join(leak_flags)
                + "); reject the hypothesis and discard this family",
            )

        if summary.verdict == "reject" or summary.severity == "fatal":
            detail = (
                summary.issues[0]
                if summary.issues
                else (summary.verdict or "validation rejected the hypothesis")
            )
            return _Decision(
                "discard",
                f"validation rejected the hypothesis; discard this family: "
                f"{detail}",
            )

        failing = [item for item in diagnostics if item.status == "fail"]
        if failing:
            detail = _diagnostic_detail(failing)
            repeated = self._record_failure(
                family, draft, "diagnostics_fail", detail
            )
            if repeated >= self.max_identical_failures:
                return _Decision(
                    "discard",
                    "model-search diagnostics failed "
                    f"{repeated} times for the same draft; discard this "
                    f"family: {detail}",
                )
            return _Decision(
                "replace",
                "model-search diagnostics failed; replace the model for this "
                f"family: {detail}",
                self._feedback(
                    **common,
                    decision="replace",
                    reason="model-search diagnostics failed",
                    repeated=repeated,
                    hints=["Address every failed diagnostic in the model "
                           "specification and return a fresh model."],
                ),
            )

        warning_diagnostics = [
            item for item in diagnostics if item.status == "warn"
        ]
        needs_refinement = (
            bool(warning_diagnostics)
            or summary.verdict == "refine"
            or summary.severity == "major"
        )
        if needs_refinement:
            parts: list[str] = []
            if warning_diagnostics:
                parts.append(_diagnostic_detail(warning_diagnostics))
            if summary.verdict == "refine":
                parts.append("generation validation requested refinement")
            if summary.severity == "major":
                first_issue = summary.issues[0] if summary.issues else ""
                parts.append(first_issue or "a major validation issue")
            detail = "; ".join(part for part in parts if part) or (
                "diagnostics require refinement"
            )
            repeated = self._record_failure(
                family, draft, "diagnostics_refine", detail
            )
            if repeated >= self.max_identical_failures:
                return _Decision(
                    "discard",
                    "the same refinement feedback repeated "
                    f"{repeated} times for the same draft; discard this "
                    f"family: {detail}",
                )
            if repeated >= 2:
                return _Decision(
                    "replace",
                    "the same refinement feedback repeated; replace the model "
                    f"for this family: {detail}",
                    self._feedback(
                        **common,
                        decision="replace",
                        reason="the same refinement feedback repeated",
                        repeated=repeated,
                        hints=["Throw out the current model structure and "
                               "propose a fresh, simpler model for this family."],
                    ),
                )
            return _Decision(
                "refine",
                f"diagnostics require refinement; refine the same family: "
                f"{detail}",
                self._feedback(
                    **common,
                    decision="refine",
                    reason="diagnostics require refinement",
                    repeated=repeated,
                    hints=["Adjust transforms, features, interactions, or "
                           "assumptions based on the feedback above."],
                ),
            )

        if not diagnostics and validation is None:
            return _Decision(
                "accept",
                "execution succeeded and no diagnostics were supplied; accept",
            )
        return _Decision(
            "accept",
            "all diagnostics and validation checks passed; accept",
        )

    def _leak_flags(
        self, diagnostics: Sequence[Diagnostic], summary: _ValidationSummary
    ) -> list[str]:
        flags: list[str] = []
        for item in diagnostics:
            if item.status not in {"warn", "fail"}:
                continue
            if _is_leak_text(f"{item.name} {item.message}"):
                flags.append(item.name or "diagnostic_leak")
        if summary.leak:
            flags.append("validation_leak")
        return list(dict.fromkeys(flags))

    def _feedback(
        self,
        *,
        family: str,
        attempt_no: int,
        phase: Phase,
        draft: HypothesisDraft,
        static_errors: list[str],
        static_warnings: list[str],
        execution: ExecutionResult | None,
        diagnostics: list[Diagnostic],
        validation: Any | None,
        decision: Decision,
        reason: str,
        repeated: int,
        hints: list[str],
    ) -> DiagnosticFeedback:
        summary = _summarize_validation(validation)
        next_phase: Phase = "replace" if decision == "replace" else "refine"
        return DiagnosticFeedback(
            family=family,
            attempt=attempt_no,
            phase=next_phase,
            summary=reason,
            static_errors=list(static_errors),
            static_warnings=list(static_warnings),
            execution_status=(
                execution.status if execution is not None else None
            ),
            execution_error=(
                execution.error if execution is not None else None
            ),
            diagnostics=list(diagnostics),
            validation_verdict=summary.verdict,
            validation_issues=list(summary.issues),
            leak_flags=self._leak_flags(diagnostics, summary),
            repeated_failures=repeated,
            hints=list(hints),
        )

    # ------------------------------------------------------------------
    # Audit and warnings

    def _generation_history(self, family: str) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for attempt in self._attempts:
            if attempt.family != family:
                continue
            history.append(
                {
                    "attempt": attempt.attempt,
                    "status": (
                        "ok" if attempt.decision == "accept" else attempt.decision
                    ),
                    "decision": attempt.decision,
                    "error": (
                        None
                        if attempt.decision == "accept"
                        else _clip(attempt.reason, 1000)
                    ),
                }
            )
        return history

    def _add_warning(self, warning: str) -> None:
        cleaned = _clip(warning, 1000)
        if cleaned and cleaned not in self._warnings:
            self._warnings.append(cleaned)

    def _write_audit(self) -> None:
        path = self.audit_path
        if path is None or self._audit_write_failed:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._audit_payload(), indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - audit must not break the run
            self._audit_write_failed = True
            self._add_warning(
                f"agent_audit_write_failed: {type(exc).__name__}: {exc}"
            )

    def _audit_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "status": self._status,
            "families": list(self.families),
            "settings": {
                "agent_max_attempts_per_family": (
                    self.settings.agent_max_attempts_per_family
                ),
                "agent_max_total_attempts": (
                    self.settings.agent_max_total_attempts
                ),
                "agent_max_seconds": self.settings.agent_max_seconds,
                "max_identical_failures": self.max_identical_failures,
            },
            "started_at": self._started_at,
            "updated_at": utc_now_iso(),
            "elapsed_seconds": round(self._last_elapsed, 3),
            "stop_reason": (
                _clip(self._stop_reason, 1000) if self._stop_reason else None
            ),
            "family_results": {
                family: {
                    "accepted": bool(result.get("accepted")),
                    "attempts": int(result.get("attempts", 0)),
                    "stop_reason": (
                        _clip(result["stop_reason"], 1000)
                        if result.get("stop_reason")
                        else None
                    ),
                }
                for family, result in self._family_results.items()
            },
            "accepted": [
                {
                    "family": item.method_family,
                    "hypothesis_id": item.id,
                    "code_sha256": item.code_sha256,
                }
                for item in self._accepted
            ],
            "attempts": [attempt.to_audit() for attempt in self._attempts],
            "warnings": [_clip(warning, 1000) for warning in self._warnings],
            "error": _clip(self._error, 3000) if self._error else None,
        }

    def _failure_error(self) -> HypothesisGenerationError:
        accepted_families = [item.method_family for item in self._accepted]
        missing = [
            family
            for family in self.families
            if family not in accepted_families
        ]
        details: list[str] = []
        for family in missing:
            info = self._family_results.get(family, {})
            reason = info.get("stop_reason") or "not attempted"
            attempts = int(info.get("attempts", 0))
            details.append(f"{family}: {reason} ({attempts} attempt(s))")
        stop = self._stop_reason
        if stop and stop not in " ".join(details):
            details.insert(0, f"stop: {stop}")
        message = (
            f"agent loop produced {len(accepted_families)} of "
            f"{len(self.families)} required families; missing: "
            + ", ".join(missing)
            + ". "
            + "; ".join(details)
        )
        path = self.audit_path
        if path is not None:
            message += f". See {path}"
        return HypothesisGenerationError(_clip(message, 3000))
