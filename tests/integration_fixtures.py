"""Shared fixtures for the component-integration test slices.

This module is deliberately free of UI imports, network access, and live
provider credentials. Every helper builds its data under the caller's
``tmp_path`` so integration slices can run in parallel without sharing state.

Public API is imported by other integration test modules; keep the signatures
stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from whack26.artifact import Artifact, validate_artifact
from whack26.config import Settings
from whack26.research.base import ResearchRecord
from whack26.run import RunOutcome, run_pipeline

TARGET = "y"

DATASET_KINDS: tuple[str, ...] = ("continuous", "binary", "multiclass", "count")
MIN_DATASET_ROWS = 20

EXPECTED_FULL_FILES: tuple[str, ...] = (
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
)


def make_dataset(
    path: Path,
    *,
    kind: str = "continuous",
    rows: int = 80,
    seed: int = 0,
) -> Path:
    """Write a synthetic dataset and return its path.

    Every dataset has ``x`` (numeric), ``z`` (numeric), ``group``
    (categorical), and ``y`` (target). The target shape depends on ``kind``:

    * ``continuous``: ``y = 2*x - z`` plus small Gaussian noise.
    * ``binary``: a logistic outcome with both classes present.
    * ``multiclass``: three integer classes, all present.
    * ``count``: non-negative Poisson counts with real variance.
    """
    if kind not in DATASET_KINDS:
        raise ValueError(
            f"kind must be one of {DATASET_KINDS}, got {kind!r}"
        )
    if rows < MIN_DATASET_ROWS:
        raise ValueError(
            f"rows must be at least {MIN_DATASET_ROWS} for a usable target, "
            f"got {rows}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 1.0, rows)
    z = rng.normal(0.0, 1.0, rows)
    group = np.asarray(["a", "b", "c"], dtype=object)[np.arange(rows) % 3]

    if kind == "continuous":
        y: np.ndarray = 2.0 * x - z + rng.normal(0.0, 0.2, rows)
    elif kind == "binary":
        logits = 0.9 * x - 0.7 * z
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(rows) < probabilities).astype(int)
        # Guarantee both classes even for unlucky seeds.
        y[0], y[1] = 0, 1
    elif kind == "multiclass":
        class_scores = np.column_stack(
            [
                0.6 * x - 0.4 * z,
                -0.5 * x + 0.8 * z,
                0.2 * x + 0.9 * z + 0.5,
            ]
        )
        y = class_scores.argmax(axis=1).astype(int)
        # Guarantee all three classes even for unlucky seeds.
        y[:3] = [0, 1, 2]
    else:  # count
        rate = np.exp(0.35 * x - 0.25 * z)
        y = rng.poisson(rate).astype(int)
        # Poisson draws are almost surely non-constant, but guarantee it.
        if np.unique(y).size < 2:
            y[0], y[1] = 0, 1

    frame = pd.DataFrame({"x": x, "z": z, "group": group, TARGET: y})

    # Honest postconditions: every downstream slice assumes these hold.
    assert set(("x", "z", "group", TARGET)).issubset(frame.columns)
    assert len(frame) >= MIN_DATASET_ROWS
    target = frame[TARGET]
    assert target.nunique() > 1, f"{kind} target is constant"
    if kind == "continuous":
        assert pd.api.types.is_numeric_dtype(target)
        assert float(target.var()) > 0.0
    elif kind == "binary":
        assert set(target.unique().tolist()) == {0, 1}
    elif kind == "multiclass":
        assert sorted(target.unique().tolist()) == [0, 1, 2]
    else:
        assert pd.api.types.is_integer_dtype(target)
        assert bool((target >= 0).all())
        assert int(target.nunique()) > 1

    frame.to_csv(destination, index=False)
    return destination


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Return mock/offline settings whose cache, runs, and data live in tmp_path."""
    values: dict[str, Any] = {
        "cache_dir": tmp_path / "cache",
        "runs_dir": tmp_path / "runs",
        "data_dir": tmp_path / "data",
        "offline": True,
    }
    values.update(overrides)
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(**values)


def run_mock_pipeline(
    tmp_path: Path,
    dataset: Path,
    *,
    target: str = TARGET,
    settings: Settings | None = None,
    out_name: str = "run",
    **kwargs: Any,
) -> RunOutcome:
    """Run the real pipeline in mock/offline mode and return its outcome."""
    outcome = run_pipeline(
        input_path=Path(dataset),
        target=target,
        mock=True,
        offline=True,
        settings=settings or make_settings(tmp_path),
        out_dir=tmp_path / out_name,
        **kwargs,
    )
    assert outcome.artifact_path.is_file(), (
        f"run did not produce an artifact: {outcome.artifact_path}"
    )
    return outcome


def load_artifact(run_dir: Path) -> Artifact:
    """Validate ``run_dir/artifact.json`` and return the parsed artifact."""
    artifact, errors = validate_artifact(Path(run_dir) / "artifact.json")
    assert not errors, f"artifact validation errors: {errors}"
    assert artifact is not None, "artifact could not be parsed"
    return artifact


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_research_record(
    identifier: str = "openalex:W1", **overrides: Any
) -> ResearchRecord:
    """Build a realistic research record reusable across research slices."""
    source, separator, source_id = identifier.partition(":")
    if not separator or not source_id:
        source = "openalex"
        source_id = identifier

    values: dict[str, Any] = {
        "id": identifier,
        "source": source,
        "source_id": source_id,
        "title": (
            "Household income, housing supply, and neighbourhood change: "
            "a panel study"
        ),
        "abstract": (
            "We study how household income and local housing supply relate to "
            "prices across 120 metropolitan areas using an observational panel "
            "with area and year fixed effects."
        ),
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "year": 2024,
        "doi": f"10.1000/{source_id.lower()}",
        "url": f"https://example.test/{source_id}",
        "license": "CC-BY-4.0",
        "retrieved_at": "2026-11-08T10:00:00Z",
        "query": "housing prices income supply",
        "relevance": 0.9,
    }
    values.update(overrides)

    unknown = sorted(set(values) - set(ResearchRecord.model_fields))
    if unknown:
        raise TypeError(
            "unknown ResearchRecord field(s): " + ", ".join(unknown)
        )
    return ResearchRecord(**values)


def make_collection_metadata(**overrides: Any) -> dict[str, Any]:
    """Return advisory collection metadata with synthetic-study defaults."""
    values: dict[str, Any] = {
        "design": "observational",
        "unit_of_analysis": "row",
        "sampling": "synthetic",
    }
    values.update(overrides)
    return values


__all__ = [
    "DATASET_KINDS",
    "EXPECTED_FULL_FILES",
    "MIN_DATASET_ROWS",
    "TARGET",
    "load_artifact",
    "make_collection_metadata",
    "make_dataset",
    "make_research_record",
    "make_settings",
    "read_json",
    "run_mock_pipeline",
]
