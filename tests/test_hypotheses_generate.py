from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from whack26.embeddings import MockEmbeddingClient
from whack26.hypotheses.generate import (
    FAMILIES,
    HypothesisContext,
    HypothesisGenerator,
    HypothesisGenerationError,
    _base_prompt,
)
from whack26.io import read_dataset
from whack26.llm import MockLLMClient, _mock_hypothesis_fixture
from whack26.profile import build_profile, build_summary
from whack26.screen import build_screen


def make_context(tmp_path: Path) -> tuple[HypothesisContext, Path]:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "z": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "y": [2.0, 4.5, 5.5, 8.5, 9.0, 12.5, 13.0, 16.5],
        }
    )
    dataset_path = tmp_path / "dataset.csv"
    frame.to_csv(dataset_path, index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dataset.csv").write_text(
        dataset_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    loaded = read_dataset(dataset_path, target="y")
    profile = build_profile(loaded, target="y")
    context = HypothesisContext(
        profile=profile,
        summary=build_summary(profile, loaded.frame),
        path_names=["Mock", "Domain"],
        factors=[],
        records=[],
        screen=build_screen(profile, loaded.frame),
    )
    return context, run_dir / "dataset.csv"


class QueueLLM:
    """Return queued payloads per purpose, then fall back to the mock fixture."""

    def __init__(self, queues: dict[str, list[dict[str, Any]]]) -> None:
        self.queues = {key: list(value) for key, value in queues.items()}
        self.calls: list[str] = []

    def generate_json(self, *, purpose: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(purpose)
        queue = self.queues.get(purpose)
        if queue:
            return copy.deepcopy(queue.pop(0))
        return copy.deepcopy(_mock_hypothesis_fixture(purpose))


def generate(context: HypothesisContext, dataset: Path, llm: Any):
    generator = HypothesisGenerator(
        context=context,
        llm=llm,
        embeddings=MockEmbeddingClient(16),
        run_dir=context.run_dir if hasattr(context, "run_dir") else dataset.parent,
        dataset_path=dataset,
    )
    return generator.generate()


def test_prompt_excludes_target_from_feature_column_lists(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)

    prompt = _base_prompt(context, "linear_model")

    assert "[TARGET - never a feature]" in prompt
    assert "`required_columns` and `optional_columns` list feature columns only" in prompt
    assert "Never put the target column (y) in either list" in prompt


def test_prompt_for_descriptive_run_does_not_name_a_target_column(
    tmp_path: Path,
) -> None:
    context, _ = make_context(tmp_path)
    context.profile = context.profile.model_copy(update={"target": None})

    prompt = _base_prompt(context, "group_comparison")

    assert "No target column is defined" in prompt
    assert "Never put the target column" not in prompt


def test_mock_generation_produces_three_families(tmp_path: Path) -> None:
    context, dataset = make_context(tmp_path)

    result = generate(context, dataset, MockLLMClient())

    assert [item.method_family for item in result.hypotheses] == list(FAMILIES)
    assert len({item.id for item in result.hypotheses}) == 3
    for item in result.hypotheses:
        assert item.execution.status == "ok"
        assert item.execution.n_predictions == 8
        assert item.generation_history[-1]["status"] == "ok"
        assert (dataset.parent / item.code_path).is_file()


def test_static_failure_is_repaired_once(tmp_path: Path) -> None:
    context, dataset = make_context(tmp_path)
    bad = _mock_hypothesis_fixture("hypothesis_linear_model")
    bad["code"] = "import sklearn\n" + bad["code"]
    llm = QueueLLM({"hypothesis_linear_model": [bad]})

    result = generate(context, dataset, llm)

    linear = result.hypotheses[0]
    assert linear.method_family == "linear_model"
    statuses = [entry["status"] for entry in linear.generation_history]
    assert statuses[0] == "static_invalid"
    assert statuses[-1] == "ok"
    assert llm.calls.count("hypothesis_linear_model") == 2


def test_applies_false_is_repaired(tmp_path: Path) -> None:
    context, dataset = make_context(tmp_path)
    bad = _mock_hypothesis_fixture("hypothesis_group_comparison")
    bad["code"] = bad["code"].replace(
        "return isinstance(df, pd.DataFrame) and len(df) > 0", "return False"
    )
    llm = QueueLLM({"hypothesis_group_comparison": [bad]})

    result = generate(context, dataset, llm)

    group = result.hypotheses[1]
    statuses = [entry["status"] for entry in group.generation_history]
    assert "applies_false" in statuses
    assert statuses[-1] == "ok"


def test_duplicate_code_is_regenerated(tmp_path: Path) -> None:
    context, dataset = make_context(tmp_path)
    linear = _mock_hypothesis_fixture("hypothesis_linear_model")
    duplicate = copy.deepcopy(linear)
    llm = QueueLLM({"hypothesis_group_comparison": [duplicate, duplicate]})

    # The second queue entry is the replacement; QueueLLM falls back to the
    # family fixture after both duplicate attempts.
    result = generate(context, dataset, llm)

    group = result.hypotheses[1]
    statuses = [entry["status"] for entry in group.generation_history]
    assert "duplicate" in statuses
    assert statuses[-1] == "ok"
    assert group.draft.code != linear["code"]


def test_generation_fails_when_a_family_never_succeeds(tmp_path: Path) -> None:
    context, dataset = make_context(tmp_path)
    always_bad = _mock_hypothesis_fixture("hypothesis_linear_model")
    always_bad["code"] = "import os\n" + always_bad["code"]
    llm = QueueLLM(
        {"hypothesis_linear_model": [copy.deepcopy(always_bad) for _ in range(12)]}
    )
    generator = HypothesisGenerator(
        context=context,
        llm=llm,
        embeddings=MockEmbeddingClient(16),
        run_dir=dataset.parent,
        dataset_path=dataset,
    )

    try:
        generator.generate()
    except HypothesisGenerationError as exc:
        message = str(exc)
        assert "linear_model" in message
        assert "static_invalid" in message
    else:
        raise AssertionError("generation should fail when one family is invalid")

    history_path = dataset.parent / "logs" / "generation.json"
    assert history_path.is_file()
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["linear_model"][-1]["status"] == "static_invalid"
