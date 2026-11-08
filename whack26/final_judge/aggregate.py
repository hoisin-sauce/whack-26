"""Decision aggregation for the final judge.

The rules preserve the branch's core principle: a pass without evidence is not
a pass.  A plain ``accepted`` requires zero concerns, zero code-verified flags,
zero missing evidence, and every critic evaluated.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .prompts import CRITICS
from .types import JudgeFinding

SEV_W = {"critical": 10, "high": 7, "medium": 4, "low": 1, "none": 0}
CONF_W = {"high": 3, "medium": 2, "low": 1}


def _normalise(finding: JudgeFinding) -> JudgeFinding:
    """could_not_evaluate => severity 'none', confidence 'low', unless the
    critic still produced a substantive finding and evidence."""
    if finding.could_not_evaluate:
        if finding.substantive:
            finding.could_not_evaluate = False
        else:
            finding.severity = "none"
            finding.confidence = "low"
    elif not finding.finding.strip() and not finding.evidence.strip():
        finding.could_not_evaluate = True
        finding.severity = "none"
        finding.confidence = "low"
    return finding


def aggregate(
    verdicts: Sequence[JudgeFinding],
    devil: JudgeFinding,
    flags: Sequence[str],
    *,
    min_evaluated_fraction: float = 0.5,
    missing_evidence: Sequence[str] = (),
) -> dict[str, Any]:
    raw_all = list(verdicts) + [devil]
    pre = {
        finding.critic: (finding.severity, finding.confidence)
        for finding in raw_all
    }
    all_findings = [_normalise(finding.model_copy(deep=True)) for finding in raw_all]
    critics = all_findings[:-1]
    devil_finding = all_findings[-1]

    critics_evaluated = [
        finding for finding in critics if not finding.could_not_evaluate
    ]
    evaluated = critics_evaluated + (
        [] if devil_finding.could_not_evaluate else [devil_finding]
    )

    unverified = [
        {
            "critic": finding.critic,
            "self_reported_severity": pre[finding.critic][0],
            "self_reported_confidence": pre[finding.critic][1],
            "note": "Critic could not evaluate; severity is unverified.",
        }
        for finding in all_findings
        if finding.could_not_evaluate
        and pre[finding.critic][0] in ("high", "critical")
    ]

    high = [
        finding for finding in evaluated if finding.severity in ("high", "critical")
    ]
    strict = [
        finding for finding in high if finding.confidence == "high"
    ]  # high confidence implies established after normalisation
    by_class: dict[str, list[JudgeFinding]] = {}
    for finding in high:
        by_class.setdefault(finding.defect_class, []).append(finding)
    corroborated = [
        finding for findings in by_class.values() if len(findings) >= 2
        for finding in findings
    ]
    veto_map = {finding.critic: finding for finding in strict + corroborated}
    vetoes = list(veto_map.values())

    # High-severity but not strong enough to veto. A re-audit is demanded when
    # the finding is established, or when it is an evaluation-design flaw.
    pending = [
        finding
        for finding in high
        if finding.critic not in veto_map
        and finding.confidence != "low"
        and (
            finding.status == "established"
            or finding.defect_class == "evaluation_design"
        )
    ]
    flagged = set(veto_map) | {finding.critic for finding in pending}

    concerns = [
        finding
        for finding in evaluated
        if finding.critic not in flagged
        and finding.severity in ("medium", "high", "critical")
        and finding.confidence in ("medium", "high")
    ]
    concern_names = {finding.critic for finding in concerns}

    advisories = sorted(
        [
            finding
            for finding in evaluated
            if finding.critic not in flagged | concern_names
            and finding.severity != "none"
        ],
        key=lambda finding: (
            SEV_W[finding.severity] * CONF_W[finding.confidence]
        ),
        reverse=True,
    )
    unevaluated = [
        finding.critic for finding in all_findings if finding.could_not_evaluate
    ]

    min_required = max(
        1, math.ceil(len(CRITICS) * max(0.0, min(1.0, min_evaluated_fraction)))
    )
    if len(critics_evaluated) < min_required:
        decision = "insufficient_evidence"
    elif vetoes:
        decisive = [
            finding
            for finding in vetoes
            if finding.status == "established"
            and finding.defect_class != "evaluation_design"
        ]
        decision = "rejected" if decisive else "requires_reaudit"
    elif pending:
        decision = "requires_reaudit"
    elif concerns or flags or unevaluated or missing_evidence:
        decision = "accepted_with_concerns"
    else:
        decision = "accepted"

    result: dict[str, Any] = {
        "decision": decision,
        "n_critics_evaluated": len(critics_evaluated),
        "min_required": min_required,
        "code_verified_flags": list(flags),
        "vetoes": [finding.model_dump(mode="json") for finding in vetoes],
        "veto_reason": (
            "strict" if strict else "corroborated"
        ) if vetoes else None,
        "pending_reaudit": [
            finding.model_dump(mode="json") for finding in pending
        ],
        "concerns": [finding.model_dump(mode="json") for finding in concerns],
        "advisories": [
            finding.model_dump(mode="json") for finding in advisories[:5]
        ],
        "unverified_flags": unverified,
        "unevaluated_critics": unevaluated,
        "missing_evidence": list(dict.fromkeys(str(item) for item in missing_evidence)),
        "devils_advocate_summary": {
            "severity": devil_finding.severity,
            "confidence": devil_finding.confidence,
            "status": devil_finding.status,
            "defect_class": devil_finding.defect_class,
            "evaluated": not devil_finding.could_not_evaluate,
        },
    }
    if decision in ("requires_reaudit", "rejected"):
        result["reaudit_instructions"] = [
            {
                "from": finding.critic,
                "fix": finding.suggested_fix,
                "test": finding.falsification_test,
            }
            for finding in vetoes + pending
        ]
    if decision == "insufficient_evidence":
        result["note"] = (
            f"Only {len(critics_evaluated)} critic(s) produced a usable verdict "
            f"(minimum {min_required}). The audit did not complete; this is NOT a pass."
        )
    elif decision == "accepted_with_concerns" and unevaluated:
        result["note"] = (
            "Some critics had no evidence. Absence of evidence is not a pass: "
            "supply it to reach a plain 'accepted'."
        )
    return result


__all__ = ["aggregate"]
