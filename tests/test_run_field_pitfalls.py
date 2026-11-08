from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

import whack26.run as run_module
from whack26.config import Settings
from whack26.pitfalls import FieldPitfall, FieldPitfallReport
from whack26.run import run_pipeline


def make_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    x = rng.normal(5.0, 1.0, 16)
    frame = pd.DataFrame(
        {
            "x": x,
            "group": ["a", "b"] * 8,
            "y": 2.0 * x + rng.normal(0.0, 0.2, 16),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def make_settings(tmp_path: Path) -> Settings:
    base = Settings.from_env({}, mock=True)
    return base.with_overrides(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
        offline=True,
    )


def make_report() -> FieldPitfallReport:
    return FieldPitfallReport(
        generated_at="2026-11-08T10:00:00Z",
        field_path=["root", "leaf"],
        pitfalls=[
            FieldPitfall(
                id="pitfall_1",
                field_id="field_leaf",
                field_name="leaf",
                field_level=2,
                title="Selection into the sample",
                pitfall="Sampling on the outcome biases the fitted effect.",
                category="selection",
                applies_to_dataset="unknown",
                evidence_id="openalex:W1",
                source="openalex",
                url="https://example.test/W1",
                year=2024,
            )
        ],
        warnings=["field_pitfalls_mock_records_only: external searches disabled"],
        degraded=True,
        offline=True,
        cache={"hits": 0, "misses": 0},
    )


def observe_optional_kwarg(monkeypatch, name: str) -> list[object]:
    """Wrap a downstream builder, tolerating a not-yet-landed parameter.

    The artifact/context/critic streams add ``field_pitfalls`` independently;
    record what this stage passes and only strip the keyword when the callee
    does not accept it yet.
    """
    original = getattr(run_module, name)
    accepted = "field_pitfalls" in inspect.signature(original).parameters
    seen: list[object] = []

    def wrapper(*args, **kwargs):
        seen.append(kwargs.get("field_pitfalls"))
        if not accepted:
            kwargs.pop("field_pitfalls", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(run_module, name, wrapper)
    return seen


def test_run_without_field_pitfalls_skips_stage(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"
    artifact_values = observe_optional_kwarg(monkeypatch, "build_artifact")
    context_values = observe_optional_kwarg(monkeypatch, "build_context_pack")
    critic_values = observe_optional_kwarg(monkeypatch, "run_critic")

    def unexpected(**_kwargs):
        raise AssertionError("field pitfalls ran while disabled")

    monkeypatch.setattr(run_module, "run_field_pitfalls", unexpected)

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
        log=lambda _: None,
    )

    assert outcome.field_pitfalls is None
    assert not (run_dir / "field_pitfalls.json").exists()
    assert artifact_values and all(value is None for value in artifact_values)
    assert context_values == [()]
    assert critic_values and all(value == () for value in critic_values)


def test_run_with_field_pitfalls_writes_report_and_wires_it(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"
    settings = make_settings(tmp_path)
    report = make_report()
    calls: list[dict] = []

    def fake_run_field_pitfalls(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(run_module, "run_field_pitfalls", fake_run_field_pitfalls)
    artifact_values = observe_optional_kwarg(monkeypatch, "build_artifact")
    context_values = observe_optional_kwarg(monkeypatch, "build_context_pack")
    critic_values = observe_optional_kwarg(monkeypatch, "run_critic")

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        field_pitfalls=True,
        settings=settings,
        out_dir=run_dir,
        log=lambda _: None,
    )

    assert calls, "run_field_pitfalls was not called"
    call = calls[0]
    assert call["field_path"]
    assert isinstance(call["existing_records"], list)
    assert isinstance(call["summary"], str) and call["summary"]
    assert call["cache_path"] == settings.research_db
    assert call["offline"] is True
    assert call["refresh"] is False
    assert call["frame"] is not None

    payload = json.loads(
        (run_dir / "field_pitfalls.json").read_text(encoding="utf-8")
    )
    assert payload == report.model_dump(mode="json")
    assert outcome.field_pitfalls == report
    assert artifact_values and all(value == report for value in artifact_values)
    assert context_values == [report.pitfalls]
    assert critic_values and all(value == report.pitfalls for value in critic_values)
    assert any(
        warning.startswith("field_pitfalls: ")
        for warning in outcome.artifact.warnings
    )


def test_field_pitfall_failure_is_visible_but_does_not_fail_run(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "input.csv"
    make_dataset(dataset)
    run_dir = tmp_path / "run"

    def broken(**_kwargs):
        raise RuntimeError("search exploded")

    monkeypatch.setattr(run_module, "run_field_pitfalls", broken)
    observe_optional_kwarg(monkeypatch, "build_artifact")
    observe_optional_kwarg(monkeypatch, "build_context_pack")
    observe_optional_kwarg(monkeypatch, "run_critic")

    outcome = run_pipeline(
        input_path=dataset,
        target="y",
        seed=0,
        mock=True,
        offline=True,
        field_pitfalls=True,
        settings=make_settings(tmp_path),
        out_dir=run_dir,
        log=lambda _: None,
    )

    assert outcome.field_pitfalls is None
    assert not (run_dir / "field_pitfalls.json").exists()
    assert any(
        warning.startswith("field_pitfalls_failed: RuntimeError:")
        for warning in outcome.artifact.warnings
    )
