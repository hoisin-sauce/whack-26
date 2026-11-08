from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from whack26.io import read_dataset
from whack26.llm import MockLLMClient
from whack26.profile import build_profile
from whack26.screen import build_screen
from whack26.validation.collection import check_collection_design
from whack26.validation.generation import GenerationValidator, validate_generation
from whack26.validation.generation_models import (
    CollectionMetadata,
    GenerationValidationResult,
)

CLEAN_CODE = '''import numpy as np
import pandas as pd


def applies(df: pd.DataFrame) -> bool:
    return {"x", "y"}.issubset(set(df.columns))


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    np.random.seed(seed)
    df = df.copy()
    x = df["x"].fillna(df["x"].mean()).to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"predictions": design @ coefficients, "target": "y"}
'''

TARGET_LEAK_CODE = CLEAN_CODE.replace(
    'design = np.column_stack([np.ones(len(x)), x])',
    'design = np.column_stack([np.ones(len(x)), x, df["y"]])',
)


def make_context(tmp_path: Path):
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 120)
    frame = pd.DataFrame({"x": x, "y": 2.0 * x + rng.normal(0.0, 0.1, 120)})
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    loaded = read_dataset(dataset_path, target="y")
    profile = build_profile(
        loaded,
        target="y",
        description="An observational cross-sectional sample of units.",
    )
    screen = build_screen(profile, loaded.frame)
    return profile, screen


def make_hypothesis(code: str = CLEAN_CODE, **overrides: Any):
    values = {
        "id": "hyp_generation0001",
        "method_family": "linear",
        "title": "A linear baseline",
        "claim_type": "prediction",
        "target": "y",
        "required_columns": ["x"],
        "optional_columns": [],
        "min_rows": 100,
        "assumptions": ["The conditional mean is approximately linear in x."],
        "factors_used": [],
        "source_ids": [],
        "rationale": "This baseline is justified from domain knowledge.",
        "code": code,
        "code_sha256": "a" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def clean_metadata(**overrides: Any) -> CollectionMetadata:
    values = {
        "design": "observational",
        "unit_of_analysis": "individual row",
        "sampling": "simple random sample of adults",
        "time_period": "2024",
        "known_biases": ["self-report bias"],
        "measurement_notes": ["outcome measured with a validated scale"],
    }
    values.update(overrides)
    return CollectionMetadata(**values)


def causal_metadata(**overrides: Any) -> CollectionMetadata:
    values = {
        "design": "experimental",
        "unit_of_analysis": "participant",
        "sampling": "randomized assignment of consenting adults",
        "time_period": "2024",
        "treatment": "treatment arm",
        "identification_strategy": "randomized assignment",
        "known_biases": ["attrition"],
        "measurement_notes": ["outcome measured with a validated scale"],
    }
    values.update(overrides)
    return CollectionMetadata(**values)


def model_search() -> SimpleNamespace:
    return SimpleNamespace(target_kind="continuous")


def run(
    tmp_path: Path,
    *,
    hypothesis: Any | None = None,
    collection: CollectionMetadata | None = None,
    llm: Any = None,
    **kwargs: Any,
) -> GenerationValidationResult:
    profile, screen = make_context(tmp_path)
    validator = GenerationValidator(llm=llm)
    return validator.validate(
        hypothesis or make_hypothesis(),
        profile=profile,
        collection=clean_metadata() if collection is None else collection,
        screen=screen,
        model_search=model_search(),
        **kwargs,
    )


def test_generation_result_schema(tmp_path: Path) -> None:
    result = run(tmp_path)

    assert isinstance(result, GenerationValidationResult)
    assert result.hypothesis_id == "hyp_generation0001"
    assert result.verdict in {"pass", "refine", "reject", "insufficient"}
    assert result.checks
    assert result.summary
    assert result.model
    assert result.corpus_version == "none"


def test_clean_hypothesis_passes(tmp_path: Path) -> None:
    result = run(tmp_path)

    assert result.verdict == "pass", [
        (issue.evidence, issue.finding) for issue in result.issues
    ]
    assert result.issues == []
    assert result.missing_information == []
    statuses = {check.id: check.status for check in result.checks}
    assert statuses["claim.design"] == "pass"
    assert statuses["unit.of_analysis"] == "pass"
    assert statuses["sampling.selection_bias"] == "pass"
    assert statuses["target.leakage"] == "pass"
    assert statuses["model.family"] == "pass"
    assert statuses["reproducibility"] == "pass"


def test_observational_causal_claim_is_rejected(tmp_path: Path) -> None:
    metadata = CollectionMetadata(
        design="observational",
        unit_of_analysis="individual row",
        sampling="convenience sample",
        known_biases=["selection"],
        measurement_notes=["single measurement"],
    )

    result = run(
        tmp_path,
        hypothesis=make_hypothesis(claim_type="causal"),
        collection=metadata,
    )

    assert result.verdict == "reject"
    claim = next(check for check in result.checks if check.id == "claim.design")
    assert claim.status == "fail"
    assert claim.severity == "fatal"
    assert any(issue.evidence == "claim.design" for issue in result.issues)


def test_experimental_causal_claim_is_accepted(tmp_path: Path) -> None:
    result = run(
        tmp_path,
        hypothesis=make_hypothesis(claim_type="causal"),
        collection=causal_metadata(),
    )

    assert result.verdict == "pass", [
        (issue.evidence, issue.finding) for issue in result.issues
    ]
    claim = next(check for check in result.checks if check.id == "claim.design")
    treatment = next(
        check for check in result.checks if check.id == "treatment.instrument"
    )
    assert claim.status == "pass"
    assert treatment.status == "pass"


def test_causal_claim_missing_treatment_is_rejected(tmp_path: Path) -> None:
    metadata = causal_metadata(
        treatment=None, identification_strategy="randomized assignment"
    )

    result = run(
        tmp_path,
        hypothesis=make_hypothesis(claim_type="causal"),
        collection=metadata,
    )

    assert result.verdict == "reject"
    treatment = next(
        check for check in result.checks if check.id == "treatment.instrument"
    )
    assert treatment.status == "fail"
    assert treatment.severity == "fatal"


def test_known_bias_is_acknowledged(tmp_path: Path) -> None:
    acknowledged = run(tmp_path)
    unacknowledged = run(
        tmp_path, collection=clean_metadata(known_biases=[])
    )

    acknowledged_check = next(
        check for check in acknowledged.checks if check.id == "known_biases"
    )
    missing_check = next(
        check for check in unacknowledged.checks if check.id == "known_biases"
    )
    assert acknowledged_check.status == "pass"
    assert missing_check.status == "flag"
    assert missing_check.severity == "minor"


def test_target_leakage_is_fatal(tmp_path: Path) -> None:
    result = run(tmp_path, hypothesis=make_hypothesis(TARGET_LEAK_CODE))

    assert result.verdict == "reject"
    leakage = next(check for check in result.checks if check.id == "target.leakage")
    assert leakage.status == "fail"
    assert leakage.severity == "fatal"


def test_missing_sampling_and_unit_refine(tmp_path: Path) -> None:
    metadata = clean_metadata(sampling=None, unit_of_analysis=None)

    result = run(tmp_path, collection=metadata)

    assert result.verdict == "refine"
    statuses = {check.id: check.status for check in result.checks}
    assert statuses["sampling.selection_bias"] == "flag"
    assert statuses["unit.of_analysis"] == "flag"
    assert "sampling method" in result.missing_information
    assert "unit of analysis" in result.missing_information


def test_jumbled_leak_and_pii_reliance_are_rejected(tmp_path: Path) -> None:
    jumbled = run(
        tmp_path,
        variant_report=SimpleNamespace(jumbled_label_leak=True),
    )
    pii = run(
        tmp_path,
        variant_report=SimpleNamespace(pii_reliance=True),
    )

    assert jumbled.verdict == "reject"
    assert pii.verdict == "reject"
    assert any(issue.evidence == "variant.jumbled" for issue in jumbled.issues)
    assert any(issue.evidence == "variant.pii" for issue in pii.issues)


def test_deterministic_checks_override_llm_pass(tmp_path: Path) -> None:
    llm = MockLLMClient(
        fixtures={
            "generation_validation": {
                "verdict": "pass",
                "issues": [],
                "missing_information": [],
                "summary": "Everything is consistent.",
            }
        }
    )
    metadata = CollectionMetadata(
        design="observational",
        unit_of_analysis="individual row",
        sampling="convenience sample",
    )

    result = run(
        tmp_path,
        hypothesis=make_hypothesis(claim_type="causal"),
        collection=metadata,
        llm=llm,
    )

    assert result.verdict == "reject"
    assert result.model == "MockLLMClient"
    assert llm.calls == ["generation_validation"]


def test_llm_major_issue_refines_clean_hypothesis(tmp_path: Path) -> None:
    llm = MockLLMClient(
        fixtures={
            "generation_validation": {
                "verdict": "pass",
                "issues": [
                    {
                        "chunk": "S4-causal-predictive",
                        "severity": "major",
                        "finding": "The sample selection is not justified.",
                        "evidence": "S7-sample-size",
                        "why": "Selection could bias the estimate.",
                        "fix": "Describe the sampling frame.",
                    }
                ],
                "missing_information": [],
                "summary": "The derivation is plausible.",
            }
        }
    )

    result = run(tmp_path, llm=llm)

    assert result.verdict == "refine"
    assert any(
        issue.finding == "The sample selection is not justified."
        for issue in result.issues
    )


def test_llm_prompt_is_pii_redacted(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    class RecordingLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            captured["user"] = kwargs["user"]
            return {
                "verdict": "pass",
                "issues": [],
                "missing_information": [],
                "summary": "Consistent.",
            }

    result = run(
        tmp_path,
        collection=clean_metadata(sampling="survey of jane.doe@example.com"),
        llm=RecordingLLM(),
    )

    assert result.verdict == "pass"
    assert "jane.doe@example.com" not in captured["user"]
    assert "[redacted]" in captured["user"]


def test_validate_generation_wrapper(tmp_path: Path) -> None:
    profile, screen = make_context(tmp_path)

    result = validate_generation(
        make_hypothesis(),
        profile=profile,
        collection=clean_metadata(),
        screen=screen,
        model_search=model_search(),
    )

    assert result.verdict == "pass"
    assert check_collection_design(
        clean_metadata(), claim_type="prediction", outcome="y"
    ) == []
