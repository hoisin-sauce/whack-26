"""Orchestration tests for the final judge."""

from __future__ import annotations

from typing import Any

from whack26.config import Settings
from whack26.final_judge.pipeline import devils_advocate, run_critic, run_final_judge
from whack26.final_judge.prompts import CRITICS
from whack26.final_judge.types import JudgeFinding
from whack26.llm import LLMError
from whack26.model_search.types import (
    Diagnostic,
    FoldResult,
    ModelCandidate,
    ModelSearchResult,
    ModelSpec,
)
from whack26.validation.models import ValidationLayerReport, ValidationReport


class CleanLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        self.calls.append(purpose)
        return {
            "severity": "none",
            "finding": "No defect found.",
            "evidence": "fake evidence",
            "suggested_fix": "",
            "falsification_test": "",
            "confidence": "low",
            "status": "suspected",
            "could_not_evaluate": False,
            "defect_class": "other",
        }


class BrokenLLM:
    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        raise LLMError("provider unavailable", purpose=kwargs.get("purpose"))


def make_model_search() -> ModelSearchResult:
    folds = [
        FoldResult(
            fold=index,
            train_rows=80,
            test_rows=20,
            metrics={"r2": 0.80 + index * 0.01},
            baseline_metrics={"r2": 0.10},
            train_metrics={"r2": 0.82},
            train_target_min=0.0,
            train_target_max=10.0,
            prediction_min=0.0,
            prediction_max=10.0,
        )
        for index in range(3)
    ]
    candidate = ModelCandidate(
        spec=ModelSpec(family="linear", features=["x"]),
        folds=folds,
        metrics={"r2": 0.81},
        metric_std={"r2": 0.01},
        diagnostics=[
            Diagnostic(name="ljung_box", status="ok", value=0.41, metadata={"statistic": 9.2, "lags": 10}),
            Diagnostic(name="durbin_watson", status="ok", value=2.0),
            Diagnostic(name="breusch_pagan", status="ok", value=0.2),
        ],
        coefficients={"x": 1.0},
        selected=True,
    )
    return ModelSearchResult(
        target_kind="continuous",
        metric="r2",
        candidates=[candidate],
        selected=candidate,
        dataset_info={
            "time_column": None,
            "features": [{"name": "x", "kind": "numeric", "reason": ""}],
            "excluded": {},
        },
        screen={"features": [{"feature": "x", "p_value": 0.01, "adjusted_p": 0.02}]},
    )


def make_validation() -> ValidationReport:
    return ValidationReport(
        hypothesis_id="hyp_test",
        status="passed",
        verdict="pass",
        layers=[ValidationLayerReport(layer="Layer 1", status="passed")],
    )


def test_clean_critics_produce_a_report() -> None:
    llm = CleanLLM()

    report = run_final_judge(
        model_search=make_model_search(),
        validation_reports=[make_validation()],
        llm=llm,
        settings=None,
        max_workers=1,
    )

    assert report.decision == "accepted_with_concerns"
    assert report.n_critics_evaluated >= report.min_required
    assert report.critics_run
    # Eight critics have evidence; subgroup_critic is unevaluated, plus the
    # devil's advocate.
    assert len(llm.calls) == 9
    assert "subgroup_slices" in report.missing_evidence
    assert report.model_summary["selected_family"] == "linear"


def test_broken_provider_is_reported_as_insufficient_not_a_pass() -> None:
    report = run_final_judge(
        model_search=make_model_search(),
        validation_reports=[make_validation()],
        llm=BrokenLLM(),
        settings=None,
        max_workers=1,
    )

    assert report.decision == "insufficient_evidence"
    assert "subgroup_critic" in report.unevaluated_critics
    assert "devils_advocate" in report.unevaluated_critics
    assert report.critics_run == []


class RecordingLLM(CleanLLM):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(dict(kwargs))
        return super().generate_json(**kwargs)


class InvalidThenValidLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(kwargs["user"])
        if len(self.prompts) == 1:
            return {"severity": "catastrophic"}
        return {
            "severity": "none",
            "finding": "No defect found.",
            "evidence": "fake evidence",
            "suggested_fix": "",
            "falsification_test": "",
            "confidence": "low",
            "status": "suspected",
            "could_not_evaluate": False,
            "defect_class": "other",
        }


def test_critic_retries_an_invalid_response_with_feedback() -> None:
    llm = InvalidThenValidLLM()

    finding = run_critic(
        CRITICS[0],
        model_description="A linear model.",
        evidence={
            "overfitting_skeptic": {
                "metric": "r2",
                "train_score": 0.9,
                "cv_mean": 0.8,
                "cv_sd": 0.02,
            }
        },
        llm=llm,
        model=None,
        max_tokens=256,
    )

    assert finding.severity == "none"
    assert len(llm.prompts) == 2
    assert "previous response was rejected" in llm.prompts[1]
    assert "catastrophic" in llm.prompts[1]


def test_devils_advocate_receives_verdicts_and_flags() -> None:
    llm = RecordingLLM()

    finding = devils_advocate(
        model_description="A linear model.",
        verdicts=[JudgeFinding(critic="critic_a", severity="none")],
        flags=["Train-CV gap is large."],
        llm=llm,
        model="da-model",
        max_tokens=512,
    )

    assert finding.critic == "devils_advocate"
    assert llm.requests[0]["purpose"] == "final_judge_devils_advocate"
    assert llm.requests[0]["model"] == "da-model"
    assert "critic_a" in llm.requests[0]["user"]
    assert "Train-CV gap is large." in llm.requests[0]["user"]


def test_settings_control_models_tokens_and_parallelism() -> None:
    llm = RecordingLLM()
    settings = Settings(
        final_judge_model="judge-model",
        final_judge_da_model="da-model",
        final_judge_max_workers=2,
        final_judge_max_tokens=256,
        final_judge_min_evaluated_fraction=0.5,
    )

    report = run_final_judge(
        model_search=make_model_search(),
        validation_reports=[make_validation()],
        llm=llm,
        settings=settings,
    )

    assert report.n_critics_evaluated >= report.min_required
    assert len(llm.requests) == 9
    critique_requests = [
        request
        for request in llm.requests
        if request["purpose"].startswith("final_judge_")
        and request["purpose"] != "final_judge_devils_advocate"
    ]
    assert all(request["model"] == "judge-model" for request in critique_requests)
    assert all(request["max_tokens"] == 256 for request in llm.requests)
    da_request = next(
        request
        for request in llm.requests
        if request["purpose"] == "final_judge_devils_advocate"
    )
    assert da_request["model"] == "da-model"
