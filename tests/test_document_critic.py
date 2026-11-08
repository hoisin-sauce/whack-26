from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from whack26.config import Settings
from whack26.critic.corpus import Corpus, load_corpus
from whack26.critic.document_critic import run_document_critic
from whack26.llm import LLMError, MockLLMClient
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

DIMS = 4
SETTINGS = Settings.from_env({}, mock=True)


def vec(*values: float) -> list[float]:
    assert len(values) <= DIMS
    return [*values, *([0.0] * (DIMS - len(values)))]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    loaded = load_corpus(Settings.from_env({}, mock=True).corpus_path)
    assert loaded is not None
    return loaded


@pytest.fixture
def store() -> Iterator[FieldStore]:
    with FieldStore(":memory:", dims=DIMS) as opened:
        yield opened


def make_document(**overrides: Any) -> Document:
    values: dict[str, Any] = {
        "id": "doc_paper_1",
        "kind": "paper",
        "source": "openalex",
        "title": "A cohort study of regional outcomes",
        "summary": (
            "The authors follow a regional cohort for five years and fit a "
            "regression of outcomes on baseline covariates."
        ),
        "summary_embedding": vec(1.0),
        "embedding_model": "mock",
        "metadata": {
            "claims": ["Baseline covariates are associated with the outcome."],
            "methods": ["Prospective cohort", "Linear regression"],
            "population": "Adults in one region",
            "time_period": "2015-2020",
        },
    }
    values.update(overrides)
    return Document(**values)


def critique_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "claim_type": "explanatory",
        "verdict": "refine",
        "issues": [
            {
                "chunk": "S2-leakage",
                "severity": "major",
                "finding": "The preprocessing order relative to the split is unclear.",
                "evidence": "No split protocol is described.",
                "why": "Preprocessing before splitting can leak information.",
                "fix": "Describe the split and fit preprocessing inside it.",
            }
        ],
        "missing_information": [],
        "summary": "The split protocol needs to be reported.",
    }
    payload.update(overrides)
    return payload


def test_critique_is_stored_with_corpus_version(
    store: FieldStore, corpus: Corpus
) -> None:
    document = store.upsert_document(make_document())
    llm = MockLLMClient(fixtures={"critic": critique_fixture()})

    result = run_document_critic(
        document,
        corpus=corpus,
        llm=llm,
        settings=SETTINGS,
        store=store,
    )

    assert result.document_id == document.id
    assert result.corpus_version == corpus.version
    assert result.verdict == "refine"
    assert result.issues[0]["chunk"] == "S2-leakage"
    assert llm.calls == ["critic"]
    assert store.get_critique(document.id, corpus.version) == result


def test_performance_language_is_removed(store: FieldStore, corpus: Corpus) -> None:
    document = store.upsert_document(make_document())
    payload = critique_fixture(
        verdict="refine",
        issues=[
            {
                "chunk": "S2-leakage",
                "severity": "major",
                "finding": "The reported RMSE suggests leakage.",
                "evidence": "accuracy is inflated",
                "why": "the benchmark is contaminated",
                "fix": "resplit",
            }
        ],
        missing_information=[
            "out-of-sample performance metrics",
            "split protocol",
        ],
        summary="The model has excellent accuracy. The split protocol is unclear.",
    )
    llm = MockLLMClient(fixtures={"critic": payload})

    result = run_document_critic(
        document,
        corpus=corpus,
        llm=llm,
        settings=SETTINGS,
        store=store,
    )

    text = (
        result.summary
        + " "
        + str(result.issues)
        + " "
        + str(result.missing_information)
    )
    lowered = text.lower()
    assert "accuracy" not in lowered
    assert "rmse" not in lowered
    assert "out-of-sample" not in lowered
    assert "benchmark" not in lowered
    assert result.summary == "The split protocol is unclear."
    assert result.missing_information == ["split protocol"]
    assert result.issues == []


def test_invalid_json_returns_insufficient_issue(
    store: FieldStore, corpus: Corpus
) -> None:
    document = store.upsert_document(make_document())
    llm = MockLLMClient(
        fixtures={"critic": {"verdict": "not-a-verdict", "issues": "not-a-list"}}
    )

    result = run_document_critic(
        document,
        corpus=corpus,
        llm=llm,
        settings=SETTINGS,
        store=store,
    )

    assert result.verdict == "insufficient"
    assert result.issues
    assert "critic.unavailable" in str(result.issues[0])
    assert store.get_critique(document.id, corpus.version) == result


def test_llm_failure_returns_insufficient_issue(
    store: FieldStore, corpus: Corpus
) -> None:
    document = store.upsert_document(make_document())

    class FailingLLM:
        def generate_json(self, **kwargs: Any) -> dict[str, Any]:
            raise LLMError("provider down", purpose="critic")

    result = run_document_critic(
        document,
        corpus=corpus,
        llm=FailingLLM(),
        settings=SETTINGS,
        store=store,
    )

    assert result.verdict == "insufficient"
    assert "critic.unavailable" in str(result.issues[0])
    assert store.get_critique(document.id, corpus.version) == result


def test_missing_corpus_is_insufficient_without_llm_call(
    store: FieldStore, tmp_path: Path
) -> None:
    document = store.upsert_document(make_document())
    llm = MockLLMClient()
    settings = SETTINGS.with_overrides(corpus_path=tmp_path / "missing.md")

    result = run_document_critic(
        document,
        corpus=None,
        llm=llm,
        settings=settings,
        store=store,
    )

    assert result.verdict == "insufficient"
    assert result.corpus_version == "missing"
    assert "critic.corpus_missing" in str(result.issues)
    assert llm.calls == []


def test_prompt_does_not_expect_python_code(store: FieldStore, corpus: Corpus) -> None:
    document = store.upsert_document(make_document())

    class CapturingLLM:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload
            self.calls: list[str | None] = []
            self.last_user = ""
            self.last_system = ""

        def generate_json(
            self,
            *,
            system: str,
            user: str,
            schema: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            self.calls.append(kwargs.get("purpose"))
            self.last_system = system
            self.last_user = user
            return copy.deepcopy(self.payload)

    llm = CapturingLLM(
        critique_fixture(verdict="pass", issues=[], summary="The method is described.")
    )
    research = [
        SimpleNamespace(
            title="Pre-registration and reporting standards",
            summary="Pre-specify analyses to limit selective reporting.",
        )
    ]

    result = run_document_critic(
        document,
        corpus=corpus,
        llm=llm,
        settings=SETTINGS,
        research=research,
        store=store,
    )

    assert result.verdict == "pass"
    assert llm.calls == ["critic"]
    assert "```python" not in llm.last_user.lower()
    assert "No Python code" in llm.last_user
    assert "Do not assess performance" in llm.last_user
    assert document.title in llm.last_user
    assert "Pre-registration and reporting standards" in llm.last_user
    assert "S0-how-to-use" in llm.last_user
    assert "not runnable Python code" in llm.last_system
