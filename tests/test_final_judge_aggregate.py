"""Decision-rule tests for the final judge."""

from __future__ import annotations

from whack26.final_judge.aggregate import aggregate
from whack26.final_judge.prompts import CRITICS
from whack26.final_judge.types import JudgeFinding


def finding(
    critic: str,
    *,
    severity: str = "none",
    confidence: str = "low",
    status: str = "suspected",
    defect_class: str = "model",
    could_not_evaluate: bool = False,
    text: str = "",
) -> JudgeFinding:
    return JudgeFinding(
        critic=critic,
        severity=severity,
        confidence=confidence,
        status=status,
        defect_class=defect_class,
        could_not_evaluate=could_not_evaluate,
        finding=text,
        evidence=text,
    )


def all_clear() -> list[JudgeFinding]:
    return [
        finding(
            spec.name,
            severity="none",
            confidence="low",
            text="No defect found.",
        )
        for spec in CRITICS
    ]


def test_all_clear_is_accepted() -> None:
    result = aggregate(
        all_clear(),
        finding("devils_advocate", text="No novel alternative."),
        [],
    )

    assert result["decision"] == "accepted"
    assert result["n_critics_evaluated"] == len(CRITICS)


def test_established_high_confidence_model_defect_is_rejected() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="high",
        status="established",
        text="A real defect.",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "rejected"
    assert result["veto_reason"] == "strict"
    assert result["vetoes"][0]["critic"] == verdicts[0].critic


def test_suspected_high_severity_is_a_concern_not_a_veto() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="high",
        status="suspected",
        text="A suspected defect.",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "accepted_with_concerns"
    assert result["vetoes"] == []
    assert result["concerns"][0]["confidence"] == "medium"


def test_evaluation_design_finding_requires_reaudit() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="high",
        status="established",
        defect_class="evaluation_design",
        text="The split leaks.",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "requires_reaudit"
    assert result["reaudit_instructions"]


def test_too_few_evaluated_critics_is_insufficient_evidence() -> None:
    verdicts = [
        finding(
            spec.name,
            could_not_evaluate=True,
            text="",
        )
        for spec in CRITICS
    ]

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "insufficient_evidence"
    assert "NOT a pass" in result["note"]


def test_missing_evidence_prevents_plain_acceptance() -> None:
    result = aggregate(
        all_clear(),
        finding("devils_advocate"),
        [],
        missing_evidence=["statistical_validation"],
    )

    assert result["decision"] == "accepted_with_concerns"
    assert result["missing_evidence"] == ["statistical_validation"]


def test_two_high_severity_findings_in_the_same_class_corroborate_a_veto() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="medium",
        status="suspected",
        text="First sign of the same flaw.",
    )
    verdicts[1] = finding(
        verdicts[1].critic,
        severity="high",
        confidence="medium",
        status="suspected",
        text="Second sign of the same flaw.",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "requires_reaudit"
    assert result["veto_reason"] == "corroborated"
    assert {item["critic"] for item in result["vetoes"]} == {
        verdicts[0].critic,
        verdicts[1].critic,
    }


def test_unverified_high_severity_is_reported_as_unverified() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="high",
        could_not_evaluate=True,
        text="",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "accepted_with_concerns"
    assert result["unverified_flags"][0]["critic"] == verdicts[0].critic
    assert result["unverified_flags"][0]["self_reported_severity"] == "high"


def test_non_substantive_findings_are_treated_as_unevaluated() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(
        verdicts[0].critic,
        severity="high",
        confidence="high",
        text="",
    )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert result["decision"] == "accepted_with_concerns"
    assert verdicts[0].critic in result["unevaluated_critics"]
    assert result["vetoes"] == []


def test_advisories_are_sorted_and_capped() -> None:
    verdicts = all_clear()
    severities = ["low", "medium", "low", "medium", "low", "medium", "low"]
    for index, severity in enumerate(severities):
        verdicts[index] = finding(
            verdicts[index].critic,
            severity=severity,
            confidence="low",
            text="A lower-priority observation.",
        )

    result = aggregate(verdicts, finding("devils_advocate"), [])

    assert len(result["advisories"]) == 5
    weights = [
        {"low": 1, "medium": 4, "high": 7, "critical": 10}[item["severity"]]
        for item in result["advisories"]
    ]
    assert weights == sorted(weights, reverse=True)


def test_min_evaluated_fraction_controls_insufficient_evidence() -> None:
    verdicts = all_clear()
    verdicts[0] = finding(verdicts[0].critic, could_not_evaluate=True, text="")

    lenient = aggregate(
        verdicts,
        finding("devils_advocate"),
        [],
        min_evaluated_fraction=0.5,
    )
    strict = aggregate(
        verdicts,
        finding("devils_advocate"),
        [],
        min_evaluated_fraction=1.0,
    )

    assert lenient["decision"] == "accepted_with_concerns"
    assert strict["decision"] == "insufficient_evidence"


def test_unevaluated_devils_advocate_is_noted() -> None:
    result = aggregate(
        all_clear(),
        finding("devils_advocate", could_not_evaluate=True, text=""),
        [],
    )

    assert result["decision"] == "accepted_with_concerns"
    assert "devils_advocate" in result["unevaluated_critics"]
    assert result["devils_advocate_summary"]["evaluated"] is False
