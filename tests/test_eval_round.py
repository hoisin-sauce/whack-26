"""Tests for the per-round eval driver."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from eval import field_classification as eval_module
from eval.field_classification import EvalDataset, LoadedInput, run_round
from whack26.config import Settings


def make_input(tmp_path: Path) -> LoadedInput:
    import pandas as pd

    from whack26.io import read_dataset
    from whack26.profile import build_profile

    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]}).to_csv(
        path, index=False
    )
    loaded = read_dataset(path, target="y")
    profile = build_profile(loaded, target="y")
    entry = EvalDataset(
        slug="demo",
        name="Demo",
        domain="demo",
        url="https://example.test/demo.csv",
        filename="demo.csv",
        description="A demo dataset.",
        target="y",
        acceptable_roots=("demo",),
        forbidden_labels=(),
        good_path="Demo > Data",
    )
    return LoadedInput(entry=entry, profile=profile, frame=loaded.frame)


class FakeClassifier:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def classify(self, profile, frame):
        return SimpleNamespace(
            names=["Demo", "Data"],
            ids=["fld_demo", "fld_data"],
            confidence=0.87,
            created_fields=["fld_demo", "fld_data"],
            warnings=["classification_warning"],
        )


class FailingClassifier:
    def __init__(self, **kwargs) -> None:
        pass

    def classify(self, profile, frame):
        raise ValueError("classifier exploded")


def test_run_round_records_paths_and_stats(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(eval_module, "Classifier", FakeClassifier)
    settings = Settings(embed_dims=4, max_depth=4)

    result = run_round(
        name="cold",
        inputs=[make_input(tmp_path)],
        settings=settings,
        llm=object(),
        embeddings=object(),
        store_path=tmp_path / "fields.db",
        fresh=True,
    )

    assert result.name == "cold"
    assert result.elapsed_s >= 0
    assert len(result.records) == 1
    record = result.records[0]
    assert record.slug == "demo"
    assert record.path_names == ["Demo", "Data"]
    assert record.path_ids == ["fld_demo", "fld_data"]
    assert record.confidence == 0.87
    assert record.created_fields == ["fld_demo", "fld_data"]
    assert record.warnings == ["classification_warning"]
    assert record.error is None
    assert record.depth == 2
    assert record.root_name == "Demo"
    assert "documents" in result.stats


def test_run_round_records_classifier_failures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(eval_module, "Classifier", FailingClassifier)

    result = run_round(
        name="cold",
        inputs=[make_input(tmp_path)],
        settings=Settings(embed_dims=4),
        llm=object(),
        embeddings=object(),
        store_path=tmp_path / "fields.db",
        fresh=True,
    )

    record = result.records[0]
    assert record.error == "ValueError: classifier exploded"
    assert record.path_names == []
    assert record.depth == 0
