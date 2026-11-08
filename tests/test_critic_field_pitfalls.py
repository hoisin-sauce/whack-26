from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from whack26.config import Settings
from whack26.critic.critic import run_critic
from whack26.embeddings import MockEmbeddingClient
from whack26.io import read_dataset
from whack26.llm import MockLLMClient
from whack26.pitfalls.types import FieldPitfall
from whack26.profile import build_profile, build_summary
from whack26.screen import build_screen

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
    return {
        "predictions": design @ coefficients,
        "target": "y",
        "notes": "Missing x values were mean-imputed.",
    }
'''

CAUTION = (
    "Field-scoped pitfalls are supplementary. `applies_to_dataset=yes` means "
    "the dataset profile shows the precondition; `unknown` is background only. "
    "Do not raise an issue from a field pitfall unless it applies to this "
    "dataset or the code/metadata clearly demonstrates the defect."
)


def make_context(tmp_path: Path):
    rng = np.random.default_rng(3)
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
    summary = build_summary(profile, loaded.frame)
    screen = build_screen(profile, loaded.frame)
    return profile, summary, screen


def make_hypothesis() -> SimpleNamespace:
    return SimpleNamespace(
        id="hyp_aaaaaaaaaaaa",
        method_family="linear_model",
        title="A linear baseline",
        claim_type="prediction",
        target="y",
        required_columns=["x"],
        optional_columns=[],
        min_rows=100,
        assumptions=["The conditional mean is approximately linear in x."],
        factors_used=[],
        source_ids=[],
        rationale="No factor applies; this baseline is justified from domain knowledge.",
        code=CLEAN_CODE,
        code_sha256="a" * 64,
        execution=SimpleNamespace(notes="Missing x values were mean-imputed."),
    )


def settings() -> Settings:
    return Settings.from_env({}, mock=True)


def make_pitfall(**overrides: Any) -> FieldPitfall:
    values: dict[str, Any] = {
        "id": "F:fld_leaf:1",
        "field_id": "fld_leaf",
        "field_name": "Leaf field",
        "field_level": 1,
        "scope": "leaf",
        "title": "Leaf pitfall",
        "pitfall": "Censoring at the detection limit biases the mean.",
        "category": "censoring",
        "applies_to_dataset": "yes",
        "applicability_reason": "The profile shows a detection limit.",
        "check": "Confirm values are not clipped at the limit.",
        "evidence_id": "rec:1",
        "source": "Field methods review",
        "url": "https://example.test/censoring",
        "year": 2024,
        "relevance": 0.9,
        "confidence": 0.8,
    }
    values.update(overrides)
    return FieldPitfall(**values)


class CapturingLLM(MockLLMClient):
    def __init__(self, fixtures: dict[str, Any] | None = None) -> None:
        super().__init__(fixtures)
        self.user_prompts: list[str] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.user_prompts.append(str(kwargs.get("user", "")))
        return super().generate_json(**kwargs)


def run(
    profile: Any,
    summary: str,
    screen: Any,
    *,
    llm: Any,
    field_pitfalls: Any = (),
):
    return run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=llm,
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
        field_pitfalls=field_pitfalls,
    )


def test_critic_keeps_field_pitfall_citation_and_cautions_prompt(
    tmp_path: Path,
) -> None:
    profile, summary, screen = make_context(tmp_path)
    pitfall = make_pitfall()
    payload = {
        "claim_type": "prediction",
        "verdict": "refine",
        "issues": [
            {
                "chunk": pitfall.id,
                "severity": "major",
                "finding": "The derivation ignores detection-limit censoring.",
                "evidence": "code",
                "why": "The mean includes clipped values.",
                "fix": "Report or model the censoring.",
            }
        ],
        "missing_information": [],
        "summary": "One field-scoped concern.",
    }
    llm = CapturingLLM(fixtures={"critic": payload})

    result = run(profile, summary, screen, llm=llm, field_pitfalls=[pitfall])

    assert [issue.chunk for issue in result.issues] == [pitfall.id]
    assert pitfall.id in result.retrieved_chunks
    assert "S0-how-to-use" in result.retrieved_chunks
    prompt = llm.user_prompts[-1]
    assert pitfall.pitfall in prompt
    assert "Leaf field (leaf field)" in prompt
    assert CAUTION in prompt


def test_critic_drops_invalid_field_pitfall_citation(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    pitfall = make_pitfall()
    payload = {
        "claim_type": "prediction",
        "verdict": "refine",
        "issues": [
            {
                "chunk": "F:missing:99",
                "severity": "major",
                "finding": "A made-up field-pitfall citation.",
                "evidence": "x",
                "why": "x",
                "fix": "x",
            }
        ],
        "missing_information": [],
        "summary": "One issue.",
    }
    llm = CapturingLLM(fixtures={"critic": payload})

    result = run(profile, summary, screen, llm=llm, field_pitfalls=[pitfall])

    assert result.issues == []
    assert pitfall.id in result.retrieved_chunks
    assert "F:missing:99" not in result.retrieved_chunks
    assert any(
        "critic.issues_without_valid_citation" in warning
        for warning in result.warnings
    )


def test_critic_omits_field_pitfalls_marked_not_applicable(
    tmp_path: Path,
) -> None:
    profile, summary, screen = make_context(tmp_path)
    excluded = {
        "id": "F:fld_leaf:2",
        "field_name": "Excluded field",
        "field_level": 1,
        "scope": "leaf",
        "pitfall": "EXCLUDED_PITFALL_TEXT",
        "applies_to_dataset": "no",
    }
    included = SimpleNamespace(
        id="F:fld_leaf:3",
        field_name="Included field",
        field_level=2,
        scope="ancestor",
        pitfall="INCLUDED_PITFALL_TEXT",
        category="confounding",
        applies_to_dataset="unknown",
        applicability_reason="The profile is incomplete.",
        check="Check the confounder listing.",
        evidence_id="rec:3",
        source="Another review",
        url="https://example.test/confounding",
        confidence=0.4,
    )
    llm = CapturingLLM()

    result = run(
        profile, summary, screen, llm=llm, field_pitfalls=[excluded, included]
    )

    prompt = llm.user_prompts[-1]
    assert "EXCLUDED_PITFALL_TEXT" not in prompt
    assert excluded["id"] not in result.retrieved_chunks
    assert "INCLUDED_PITFALL_TEXT" in prompt
    assert included.id in result.retrieved_chunks
    assert CAUTION in prompt


def test_field_pitfalls_are_sorted_and_capped_at_six(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    pitfalls = [
        make_pitfall(
            id="F:anc:1",
            field_name="Ancestor high",
            scope="ancestor",
            field_level=9,
            confidence=0.99,
        ),
        make_pitfall(
            id="F:leaf:1",
            field_name="Leaf low",
            scope="leaf",
            field_level=1,
            confidence=0.2,
        ),
        make_pitfall(
            id="F:leaf:3",
            field_name="Leaf tie b",
            scope="leaf",
            field_level=2,
            confidence=0.9,
        ),
        make_pitfall(
            id="F:leaf:0",
            field_name="Leaf tie a",
            scope="leaf",
            field_level=2,
            confidence=0.9,
        ),
        make_pitfall(
            id="F:leaf:2",
            field_name="Leaf mid",
            scope="leaf",
            field_level=2,
            confidence=0.5,
        ),
        make_pitfall(
            id="F:anc:0",
            field_name="Ancestor higher",
            scope="ancestor",
            field_level=10,
            confidence=0.1,
        ),
        make_pitfall(
            id="F:anc:2",
            field_name="Ancestor low",
            scope="ancestor",
            field_level=1,
            confidence=0.0,
        ),
        make_pitfall(
            id="F:leaf:9",
            field_name="Leaf excluded",
            scope="leaf",
            field_level=5,
            confidence=1.0,
            applies_to_dataset="no",
        ),
    ]
    llm = CapturingLLM()

    run(profile, summary, screen, llm=llm, field_pitfalls=pitfalls)

    prompt = llm.user_prompts[-1]
    supplied = re.findall(r"### (F:[^\s]+) —", prompt)
    assert supplied == [
        "F:leaf:0",
        "F:leaf:3",
        "F:leaf:2",
        "F:leaf:1",
        "F:anc:0",
        "F:anc:1",
    ]
