from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from whack26.config import Settings
from whack26.critic.checks import run_deterministic_checks
from whack26.critic.corpus import Corpus, load_corpus
from whack26.critic.critic import run_critic
from whack26.embeddings import EmbeddingError, MockEmbeddingClient
from whack26.io import read_dataset
from whack26.llm import LLMError, MockLLMClient
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

SCALER_LEAK_CODE = CLEAN_CODE.replace(
    '    design = np.column_stack([np.ones(len(x)), x])',
    '    scaler = np.array([0.0])\n'
    '    scaler.fit = lambda values: values\n'
    '    scaler.fit(x)\n'
    '    train_test_split(x, y)\n'
    '    design = np.column_stack([np.ones(len(x)), x])',
)

TARGET_LEAK_CODE = CLEAN_CODE.replace(
    'design = np.column_stack([np.ones(len(x)), x])',
    'design = np.column_stack([np.ones(len(x)), x, df["y"]])',
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


def make_hypothesis(code: str = CLEAN_CODE, **overrides: Any):
    values = {
        "id": "hyp_aaaaaaaaaaaa",
        "method_family": "linear_model",
        "title": "A linear baseline",
        "claim_type": "prediction",
        "target": "y",
        "required_columns": ["x"],
        "optional_columns": [],
        "min_rows": 100,
        "assumptions": ["The conditional mean is approximately linear in x."],
        "factors_used": [],
        "source_ids": [],
        "rationale": "No factor applies; this baseline is justified from domain knowledge.",
        "code": code,
        "code_sha256": "a" * 64,
        "execution": SimpleNamespace(notes="Missing x values were mean-imputed."),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def settings(corpus_path: Path | None = None) -> Settings:
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(corpus_path=corpus_path) if corpus_path else base


def test_corpus_parses_version_and_retrieves_five_chunks() -> None:
    corpus = load_corpus(Settings.from_env({}, mock=True).corpus_path)
    assert corpus is not None
    assert corpus.version.startswith("1.0.1+")
    assert corpus.get("S0-how-to-use") is not None
    assert corpus.get("S2-leakage") is not None
    assert corpus.get("S10-reporting") is not None
    assert len(corpus.ranked_chunks()) == 10
    assert corpus.get("S11") is None

    chunks, warnings = corpus.retrieve(
        "target leakage and preprocessing", MockEmbeddingClient(16)
    )
    assert warnings == []
    assert chunks[0].id == "S0-how-to-use"
    assert len(chunks) == 5
    assert len({chunk.id for chunk in chunks}) == 5


def test_corpus_retrieval_falls_back_when_embeddings_fail() -> None:
    corpus = load_corpus(Settings.from_env({}, mock=True).corpus_path)
    assert corpus is not None

    class FailingEmbeddings:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingError("offline")

    chunks, warnings = corpus.retrieve("anything", FailingEmbeddings())
    ids = [chunk.id for chunk in chunks]
    assert ids[0] == "S0-how-to-use"
    assert {"S1-assumptions", "S2-leakage", "S5-diagnostics"} <= set(ids)
    assert any("retrieval_failed" in warning for warning in warnings)


def test_missing_corpus_returns_none(tmp_path: Path) -> None:
    assert load_corpus(tmp_path / "missing.md") is None


def test_deterministic_checks_pass_clean_code(tmp_path: Path) -> None:
    profile, _, screen = make_context(tmp_path)
    result = run_deterministic_checks(
        make_hypothesis(), profile=profile, screen=screen, factors=[], description=profile.description
    )
    failed = [check for check in result.checks if check.status in {"flag", "fail"}]
    assert failed == [], [check.reason for check in failed]
    assert result.issues == []


def test_deterministic_checks_flag_scaler_before_split(tmp_path: Path) -> None:
    profile, _, screen = make_context(tmp_path)
    result = run_deterministic_checks(
        make_hypothesis(SCALER_LEAK_CODE),
        profile=profile,
        screen=screen,
        factors=[],
        description=profile.description,
    )
    check = next(item for item in result.checks if item.id == "split.preprocessing")
    assert check.status == "flag"
    assert check.severity == "major"


def test_deterministic_checks_fail_target_leakage(tmp_path: Path) -> None:
    profile, _, screen = make_context(tmp_path)
    result = run_deterministic_checks(
        make_hypothesis(TARGET_LEAK_CODE),
        profile=profile,
        screen=screen,
        factors=[],
        description=profile.description,
    )
    check = next(item for item in result.checks if item.id == "target.leakage")
    assert check.status == "fail"
    assert check.severity == "fatal"


def test_deterministic_checks_mark_time_series_not_applicable_without_time_column(
    tmp_path: Path,
) -> None:
    profile, _, screen = make_context(tmp_path)
    result = run_deterministic_checks(
        make_hypothesis(),
        profile=profile,
        screen=screen,
        factors=[],
        description=profile.description,
    )
    check = next(item for item in result.checks if item.id == "time.series")
    assert check.status == "not_applicable"
    assert check.reason == "No time column is present."


def test_critic_passes_clean_hypothesis(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    assert result.verdict == "pass", [issue.finding for issue in result.issues]
    assert result.corpus_version.startswith("1.0.1+")
    assert result.retrieved_chunks[0] == "S0-how-to-use"
    assert result.degraded is False


def test_critic_fails_scaler_and_target_leaks(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    common = dict(
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    scaler = run_critic(make_hypothesis(SCALER_LEAK_CODE), **common)
    target = run_critic(make_hypothesis(TARGET_LEAK_CODE), **common)

    assert scaler.verdict in {"refine", "reject", "insufficient"}
    assert any(
        check.id == "split.preprocessing" and check.status == "flag"
        for check in scaler.checks
    )
    assert target.verdict == "reject"
    assert any(
        check.id == "target.leakage" and check.status == "fail"
        for check in target.checks
    )


def test_critic_strips_performance_language(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    payload = {
        "claim_type": "prediction",
        "verdict": "pass",
        "issues": [
            {
                "chunk": "S2-leakage",
                "severity": "major",
                "finding": "The reported RMSE suggests leakage.",
                "evidence": "code",
                "why": "accuracy is inflated",
                "fix": "resplit",
            }
        ],
        "missing_information": [],
        "summary": "The model has excellent accuracy. The split is clean.",
    }
    llm = MockLLMClient(fixtures={"critic": payload})
    result = run_critic(
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
    )
    text = result.summary + " " + " ".join(issue.finding + issue.why for issue in result.issues)
    assert "accuracy" not in text.lower()
    assert "rmse" not in text.lower()
    assert any("performance_language_removed" in warning for warning in result.warnings)


def test_critic_removes_performance_language_from_missing_information(
    tmp_path: Path,
) -> None:
    profile, summary, screen = make_context(tmp_path)
    payload = {
        "claim_type": "prediction",
        "verdict": "insufficient",
        "issues": [],
        "missing_information": [
            "split protocol",
            "out-of-sample performance metrics",
            "evaluation metrics on a test set",
            "model generalization performance",
        ],
        "summary": "The split protocol is not described.",
    }
    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(fixtures={"critic": payload}),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    assert result.missing_information == ["split protocol"]
    assert any(
        "performance_missing_information_removed" in warning
        for warning in result.warnings
    )
    assert result.verdict == "insufficient"


def test_critic_overrides_unexplained_llm_insufficient(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    payload = {
        "claim_type": "prediction",
        "verdict": "insufficient",
        "issues": [],
        "missing_information": ["model generalization performance"],
        "summary": "The split protocol is not described.",
    }
    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(fixtures={"critic": payload}),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    assert result.verdict == "pass"
    assert any(
        "llm_insufficient_overridden" in warning for warning in result.warnings
    )


def test_critic_drops_issue_with_invalid_citation(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    payload = {
        "claim_type": "prediction",
        "verdict": "refine",
        "issues": [
            {
                "chunk": "S99-unknown",
                "severity": "major",
                "finding": "A made-up citation.",
                "evidence": "x",
                "why": "x",
                "fix": "x",
            }
        ],
        "missing_information": [],
        "summary": "One issue.",
    }
    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[SimpleNamespace(name="Test", description="A domain.")],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(fixtures={"critic": payload}),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    assert result.issues == []
    assert any("citation" in warning for warning in result.warnings)
    assert result.verdict == "pass"


def test_critic_missing_corpus_is_insufficient(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)
    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[],
        factors=[],
        screen=screen,
        research=[],
        llm=MockLLMClient(),
        embeddings=MockEmbeddingClient(16),
        settings=settings(tmp_path / "missing.md"),
    )
    assert result.verdict == "insufficient"
    assert result.degraded is True
    assert any(
        issue.evidence == "critic.corpus_missing" for issue in result.issues
    )


def test_critic_llm_failure_is_degraded(tmp_path: Path) -> None:
    profile, summary, screen = make_context(tmp_path)

    class FailingLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            raise LLMError("provider down", purpose="critic")

    result = run_critic(
        make_hypothesis(),
        profile=profile,
        summary=summary,
        field_path=[],
        factors=[],
        screen=screen,
        research=[],
        llm=FailingLLM(),
        embeddings=MockEmbeddingClient(16),
        settings=settings(),
    )
    assert result.verdict == "insufficient"
    assert result.degraded is True
    assert any(issue.evidence == "critic.unavailable" for issue in result.issues)
