"""Tests for the shared integration fixtures.

These tests only exercise helpers in ``integration_fixtures``; one test runs a
real mock/offline pipeline to prove the fixture contract works end to end.
There are no UI imports and no network calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from integration_fixtures import (
    EXPECTED_FULL_FILES,
    TARGET,
    load_artifact,
    make_collection_metadata,
    make_dataset,
    make_research_record,
    make_settings,
    read_json,
    run_mock_pipeline,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["continuous", "binary", "multiclass", "count"])
def test_make_dataset_has_the_expected_target_shape(
    tmp_path: Path, kind: str
) -> None:
    dataset = make_dataset(tmp_path / f"{kind}.csv", kind=kind, rows=60, seed=3)

    assert dataset.is_file()
    frame = pd.read_csv(dataset)
    assert len(frame) == 60
    assert {"x", "z", "group", TARGET}.issubset(frame.columns)
    assert frame["x"].nunique() > 1
    assert frame["z"].nunique() > 1
    assert frame["group"].nunique() >= 2

    target = frame[TARGET]
    assert target.nunique() > 1, "target must not be constant"

    if kind == "continuous":
        assert pd.api.types.is_numeric_dtype(target)
        assert not pd.api.types.is_bool_dtype(target)
        assert float(target.var()) > 0.0
    elif kind == "binary":
        assert pd.api.types.is_integer_dtype(target)
        assert set(target.unique().tolist()) == {0, 1}
    elif kind == "multiclass":
        assert pd.api.types.is_integer_dtype(target)
        assert sorted(target.unique().tolist()) == [0, 1, 2]
    else:  # count
        assert pd.api.types.is_integer_dtype(target)
        assert bool((target >= 0).all())
        assert int(target.nunique()) > 1
        assert float(target.var()) > 0.0


def test_make_settings_isolates_paths_and_applies_overrides(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    assert settings.mock is True
    assert settings.llm_provider == "mock"
    assert settings.embed_provider == "mock"
    assert settings.offline is True
    assert settings.cache_dir == tmp_path / "cache"
    assert settings.runs_dir == tmp_path / "runs"
    assert settings.data_dir == tmp_path / "data"
    for directory in (
        settings.cache_dir,
        settings.runs_dir,
        settings.data_dir,
    ):
        assert directory.parent == tmp_path
    assert settings.deepseek_api_key is None
    assert settings.gemini_api_key is None

    overridden = make_settings(
        tmp_path,
        top_k=7,
        scrape_enabled=True,
        max_depth=2,
        model_search_top_n=3,
    )
    assert overridden.top_k == 7
    assert overridden.scrape_enabled is True
    assert overridden.max_depth == 2
    assert overridden.model_search_top_n == 3
    assert overridden.mock is True
    assert overridden.offline is True
    assert overridden.cache_dir == tmp_path / "cache"
    assert overridden.runs_dir == tmp_path / "runs"
    assert overridden.data_dir == tmp_path / "data"


def test_run_mock_pipeline_writes_a_loadable_artifact(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "input.csv", rows=40, seed=0)

    outcome = run_mock_pipeline(tmp_path, dataset)

    assert outcome.run_dir == tmp_path / "run"
    assert (outcome.run_dir / "artifact.json").is_file()
    assert outcome.preview_path == outcome.run_dir / "preview.md"
    assert outcome.preview_path.is_file()

    artifact = load_artifact(outcome.run_dir)
    assert artifact.dataset.rows == 40
    assert artifact.dataset.target is not None
    assert artifact.dataset.target.name == TARGET
    assert len(artifact.hypotheses) == 3
    assert artifact.run.mode == "mock"

    assert read_json(outcome.artifact_path)["dataset"]["rows"] == 40


def test_research_record_and_collection_metadata_defaults() -> None:
    record = make_research_record()
    assert record.id == "openalex:W1"
    assert record.source == "openalex"
    assert record.source_id == "W1"
    assert record.title
    assert record.abstract
    assert record.url.startswith("https://")
    assert record.license
    assert record.retrieved_at
    assert record.query
    assert 0.0 <= record.relevance <= 1.0

    customized = make_research_record(
        "arxiv:2401.00001",
        title="A custom title",
        relevance=0.42,
    )
    assert customized.source == "arxiv"
    assert customized.source_id == "2401.00001"
    assert customized.title == "A custom title"
    assert customized.relevance == pytest.approx(0.42)

    metadata = make_collection_metadata()
    assert metadata == {
        "design": "observational",
        "unit_of_analysis": "row",
        "sampling": "synthetic",
    }
    assert make_collection_metadata(design="experiment")["design"] == "experiment"
    assert make_collection_metadata() == {
        "design": "observational",
        "unit_of_analysis": "row",
        "sampling": "synthetic",
    }


def test_read_json_and_expected_full_files(tmp_path: Path) -> None:
    payload = {"ok": True, "values": [1, 2, 3]}
    path = tmp_path / "payload.json"
    path.write_text('{"ok": true, "values": [1, 2, 3]}', encoding="utf-8")
    assert read_json(path) == payload

    assert len(EXPECTED_FULL_FILES) == 17
    assert set(EXPECTED_FULL_FILES) == {
        "artifact.json",
        "preview.md",
        "dataset.csv",
        "research.json",
        "factors.json",
        "validation.json",
        "generation_validation.json",
        "model_search.json",
        "variants.json",
        "research_models.json",
        "field_pitfalls.json",
        "scrape.json",
        "scrape.md",
        "final_judge.json",
        "final_judge.md",
        "final_judge.html",
        "final_judge.fragment.html",
    }
