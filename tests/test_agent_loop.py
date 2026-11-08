from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from whack26.config import Settings
from whack26.hypotheses.agent_loop import (
    DiagnosticFeedback,
    ModelAgentLoop,
    ModelAttempt,
)
from whack26.hypotheses.execute import ExecutionResult
from whack26.hypotheses.generate import (
    FAMILIES,
    HypothesisDraft,
    HypothesisGenerationError,
)
from whack26.model_search.types import Diagnostic
from whack26.validation.generation_models import (
    GenerationValidationResult,
    ValidationIssue,
)

VALID_CODE = """import numpy as np
import pandas as pd

PARAMETERS = {"ridge": 1e-3}

def applies(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and len(df) > 0

def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    return 0.0

def _predict(model, df: pd.DataFrame):
    return np.zeros(len(df))

def run(df: pd.DataFrame, seed: int = 0) -> dict:
    model = _fit(df, seed, PARAMETERS)
    return {"predictions": _predict(model, df), "target": None}
"""


def make_draft(
    title: str = "A baseline model",
    *,
    code: str | None = None,
):
    return HypothesisDraft(
        title=title,
        rationale="No factor applies; this is a simple baseline model.",
        assumptions=["The input frame is tabular."],
        required_columns=[],
        optional_columns=[],
        min_rows=1,
        claim_type="prediction",
        factors_used=[],
        source_ids=[],
        code=VALID_CODE if code is None else code,
    )


def ok_execution() -> ExecutionResult:
    return ExecutionResult(
        status="ok",
        applies=True,
        runtime_ms=1,
        n_predictions=4,
        target=None,
    )


def family_code(family: str, attempt: int) -> str:
    return VALID_CODE + f"\n# family: {family}\n# attempt: {attempt}\n"


def test_accepts_three_unique_families_and_writes_audit(tmp_path: Path) -> None:
    pack = {"summary": "injected context pack"}
    seen: list[tuple[str, str, int, Any]] = []

    def generator(request):
        seen.append(
            (request.family, request.phase, request.attempt, request.context_pack)
        )
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        context_pack=pack,
    )

    result = loop.run()

    assert [item.method_family for item in result.hypotheses] == list(FAMILIES)
    assert len({item.method_family for item in result.hypotheses}) == 3
    assert [entry[0] for entry in seen] == list(FAMILIES)
    assert all(entry[1] == "initial" for entry in seen)
    assert all(entry[3] is pack for entry in seen)
    assert len(loop.attempts) == 3
    assert all(attempt.decision == "accept" for attempt in loop.attempts)

    assert loop.audit_path is not None
    audit = json.loads(loop.audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "completed"
    assert audit["families"] == list(FAMILIES)
    assert len(audit["attempts"]) == 3
    assert audit["accepted"][0]["family"] == FAMILIES[0]
    assert audit["attempts"][0]["decision"] == "accept"
    assert "code" not in audit["attempts"][0]
    assert "rationale" not in audit["attempts"][0]
    assert audit["settings"]["agent_max_attempts_per_family"] == 6


def test_refine_then_accept_on_warning_diagnostic(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        if request.attempt == 1:
            return [
                Diagnostic(
                    name="heteroskedasticity",
                    status="warn",
                    message="residual variance is not constant",
                )
            ]
        return []

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
    )

    result = loop.run()

    assert len(result.hypotheses) == 3
    assert [attempt.decision for attempt in loop.attempts] == [
        "refine",
        "accept",
    ] * 3
    assert loop.attempts[1].phase == "refine"
    assert "heteroskedasticity" in loop.attempts[0].reason


def test_replace_on_failed_diagnostic(tmp_path: Path) -> None:
    phases: list[str] = []

    def generator(request):
        phases.append(request.phase)
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        if request.attempt == 1:
            return [
                Diagnostic(
                    name="nonlinearity",
                    status="fail",
                    message="a linear form misses a curved relationship",
                )
            ]
        return []

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
    )

    loop.run()

    assert [attempt.decision for attempt in loop.attempts] == [
        "replace",
        "accept",
    ] * 3
    assert phases[1] == "replace"


def test_jumbled_label_leak_discards_the_family(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        return [
            Diagnostic(
                name="jumbled_label_leak",
                status="fail",
                message="the jumbled-label model beat the permutation null",
            )
        ]

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
    )

    with pytest.raises(HypothesisGenerationError, match="missing"):
        loop.run()

    assert len(loop.attempts) == 3
    assert all(attempt.decision == "discard" for attempt in loop.attempts)
    assert "leakage" in loop.attempts[0].reason
    assert loop.family_results[FAMILIES[0]]["stop_reason"]

    assert loop.audit_path is not None
    audit = json.loads(loop.audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["stop_reason"] is None or "leak" in str(audit["stop_reason"])


def test_neutral_jumbled_note_does_not_discard(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def validator(request):
        return GenerationValidationResult(
            hypothesis_id="hyp_test",
            verdict="pass",
            checks=[],
            issues=[],
            summary="No jumbled-label leak detected; the clean model passes.",
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        validator=validator,
    )

    result = loop.run()

    assert len(result.hypotheses) == 3
    assert all(attempt.decision == "accept" for attempt in loop.attempts)


def test_model_search_result_diagnostics_are_consumed(tmp_path: Path) -> None:
    from tests.contracts.fakes import make_model_search_result

    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        if request.attempt != 1:
            return []
        result = make_model_search_result()
        assert result.selected is not None
        result.selected.diagnostics = [
            Diagnostic(
                name="nonlinearity",
                status="fail",
                message="the selected candidate misses curvature",
            )
        ]
        return result

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
    )

    loop.run()

    assert [attempt.decision for attempt in loop.attempts] == [
        "replace",
        "accept",
    ] * 3
    assert loop.attempts[0].diagnostics[0].name == "nonlinearity"


def test_validation_refine_then_accept(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def validator(request):
        if request.attempt == 1:
            return GenerationValidationResult(
                hypothesis_id="hyp_test",
                verdict="refine",
                issues=[
                    ValidationIssue(
                        severity="major",
                        finding="the unit of analysis is not explicit",
                        fix="state the row unit in the rationale",
                    )
                ],
                summary="A refinement is required.",
            )
        return GenerationValidationResult(
            hypothesis_id="hyp_test",
            verdict="pass",
            summary="The claim and design agree.",
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        validator=validator,
    )

    loop.run()

    assert [attempt.decision for attempt in loop.attempts] == [
        "refine",
        "accept",
    ] * 3
    assert loop.attempts[0].validation is not None


def test_validation_reject_discards_the_family(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def validator(request):
        return GenerationValidationResult(
            hypothesis_id="hyp_test",
            verdict="reject",
            issues=[
                ValidationIssue(
                    severity="fatal",
                    finding="a causal claim has no identification strategy",
                )
            ],
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        validator=validator,
    )

    with pytest.raises(HypothesisGenerationError):
        loop.run()

    assert len(loop.attempts) == 3
    assert all(attempt.decision == "discard" for attempt in loop.attempts)


def test_per_family_budget_exhaustion(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        return [
            Diagnostic(
                name="multicollinearity",
                status="warn",
                message="two predictors are highly correlated",
            )
        ]

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
        settings=Settings(
            agent_max_attempts_per_family=2,
            agent_max_total_attempts=18,
            agent_max_seconds=900,
        ),
    )

    with pytest.raises(HypothesisGenerationError, match="per-family"):
        loop.run()

    assert len(loop.attempts) == 6
    for family in FAMILIES:
        assert loop.family_results[family]["attempts"] == 2
        assert "per-family" in loop.family_results[family]["stop_reason"]


def test_total_attempt_budget_exhaustion(tmp_path: Path) -> None:
    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    def diagnostic(request):
        return [
            Diagnostic(
                name="overdispersion",
                status="warn",
                message="the variance exceeds the mean",
            )
        ]

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
        settings=Settings(
            agent_max_attempts_per_family=6,
            agent_max_total_attempts=2,
            agent_max_seconds=900,
        ),
    )

    with pytest.raises(HypothesisGenerationError, match="total attempt budget"):
        loop.run()

    assert len(loop.attempts) == 2


def test_time_budget_exhaustion(tmp_path: Path) -> None:
    ticks = iter([0.0, 0.1, 2.0])

    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        settings=Settings(
            agent_max_attempts_per_family=6,
            agent_max_total_attempts=18,
            agent_max_seconds=1,
        ),
        clock=lambda: next(ticks),
    )

    with pytest.raises(HypothesisGenerationError, match="time budget"):
        loop.run()

    assert len(loop.attempts) == 1
    assert len(loop.accepted) == 1


def test_no_infinite_loop_on_repeated_identical_failure(tmp_path: Path) -> None:
    same_draft = make_draft("the same model", code=VALID_CODE)

    def generator(request):
        return same_draft

    def diagnostic(request):
        return [
            Diagnostic(
                name="autocorrelation",
                status="warn",
                message="errors are serially correlated",
            )
        ]

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
        diagnostic=diagnostic,
        settings=Settings(
            agent_max_attempts_per_family=100,
            agent_max_total_attempts=100,
            agent_max_seconds=900,
        ),
    )

    with pytest.raises(HypothesisGenerationError, match="attempts.json"):
        loop.run()

    assert len(loop.attempts) == 9
    assert [attempt.decision for attempt in loop.attempts] == [
        "refine",
        "replace",
        "discard",
    ] * 3


def test_static_ast_contract_is_repaired_before_execution(tmp_path: Path) -> None:
    def generator(request):
        if request.attempt == 1:
            return make_draft(
                "bad import",
                code="import sklearn\n" + VALID_CODE,
            )
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ok_execution(),
    )

    loop.run()

    assert loop.attempts[0].decision == "refine"
    assert loop.attempts[0].execution is None
    assert "outside the allowlist" in loop.attempts[0].reason
    assert [attempt.decision for attempt in loop.attempts] == [
        "refine",
        "accept",
    ] * 3


def test_audit_redacts_secrets(tmp_path: Path) -> None:
    secret_text = (
        "request failed for user@example.com from 192.168.10.20: "
        "?api_key=supersecret and Authorization: Bearer topsecret-token; "
        "account 123456789012"
    )

    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=tmp_path,
        executor=lambda request: ExecutionResult(
            status="runtime_error",
            error=secret_text,
        ),
        settings=Settings(
            agent_max_attempts_per_family=1,
            agent_max_total_attempts=18,
            agent_max_seconds=900,
        ),
    )

    with pytest.raises(HypothesisGenerationError):
        loop.run()

    assert loop.audit_path is not None
    text = loop.audit_path.read_text(encoding="utf-8")
    assert "supersecret" not in text
    assert "topsecret-token" not in text
    assert "user@example.com" not in text
    assert "192.168.10.20" not in text
    assert "123456789012" not in text
    assert "[redacted]" in text
    assert "[redacted-email]" in text


def test_default_executor_uses_the_isolated_runner(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def generator(request):
        return make_draft(
            f"{request.family} attempt {request.attempt}",
            code=family_code(request.family, request.attempt),
        )

    loop = ModelAgentLoop(
        generator=generator,
        run_dir=run_dir,
        dataset_path=dataset_path,
        seed=0,
    )

    result = loop.run()

    assert len(result.hypotheses) == 3
    for hypothesis in result.hypotheses:
        assert hypothesis.execution.status == "ok"
        assert hypothesis.execution.n_predictions == 4
        assert (run_dir / hypothesis.code_path).is_file()
        assert hypothesis.execution.predictions_path is not None
        assert (run_dir / hypothesis.execution.predictions_path).is_file()


def test_constructor_requires_exactly_three_unique_families() -> None:
    with pytest.raises(ValueError, match="exactly three unique families"):
        ModelAgentLoop(
            generator=lambda request: make_draft(),
            executor=lambda request: ok_execution(),
            families=("linear_model", "linear_model", "group_comparison"),
        )

    with pytest.raises(ValueError, match="families must be exactly"):
        ModelAgentLoop(
            generator=lambda request: make_draft(),
            executor=lambda request: ok_execution(),
            families=("linear_model", "group_comparison", "research_derived"),
        )


def test_attempt_and_feedback_models_are_structured() -> None:
    attempt = ModelAttempt(
        family="linear_model",
        draft=make_draft("structured"),
        execution=ok_execution(),
        diagnostics=[
            Diagnostic(name="residuals", status="warn", message="fat tails")
        ],
        decision="refine",
        reason="needs a robust transform",
        attempt=2,
        phase="refine",
    )
    assert attempt.to_audit()["diagnostics"][0]["name"] == "residuals"

    feedback = DiagnosticFeedback(
        family="linear_model",
        attempt=2,
        phase="replace",
        summary="the same diagnostic repeated",
        static_errors=["bad import"],
        diagnostics=[
            Diagnostic(name="nonlinearity", status="fail", message="curved")
        ],
        validation_verdict="refine",
        validation_issues=["unit of analysis unclear"],
        repeated_failures=2,
        hints=["throw out the structure"],
    )
    prompt = feedback.as_prompt()
    assert "Requested action: replace" in prompt
    assert "bad import" in prompt
    assert "nonlinearity" in prompt
    assert "unit of analysis unclear" in prompt
    assert "repeated 2 times" in prompt
