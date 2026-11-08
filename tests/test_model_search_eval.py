"""Tests for the golden-dataset model-search evaluation harness."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import eval.model_search as model_search_eval
from eval.model_search import (
    DATASET_SPECS,
    FIXTURE_DIR,
    SPECS_BY_SLUG,
    build_golden_dataset,
    build_golden_datasets,
    evaluate_all,
    evaluate_dataset,
    generate_dataset,
    infer_target_kind,
    load_golden_fixture,
    write_golden_fixtures,
)

SLUGS = [spec.slug for spec in DATASET_SPECS]


def assert_frames_close(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    assert list(expected.columns) == list(actual.columns)
    assert len(expected) == len(actual)
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
            np.testing.assert_allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"column {column}",
            )
        else:
            assert left.astype(str).tolist() == right.astype(str).tolist()


def ols_r2(frame: pd.DataFrame, features: list[str], target: str) -> float:
    y = frame[target].to_numpy(dtype=float)
    columns = [np.ones(len(frame))]
    for name in features:
        columns.append(frame[name].to_numpy(dtype=float))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - predictions) ** 2)) / total


def interactive_r2(frame: pd.DataFrame, target: str) -> float:
    y = frame[target].to_numpy(dtype=float)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["x1"].to_numpy(dtype=float),
            frame["x2"].to_numpy(dtype=float),
            frame["x1"].to_numpy(dtype=float) * frame["x2"].to_numpy(dtype=float),
        ]
    )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - predictions) ** 2)) / total


@pytest.fixture(scope="module")
def local_evaluation():
    return evaluate_all(force_local=True)


def test_generators_are_deterministic_and_complete() -> None:
    for spec in DATASET_SPECS:
        first = generate_dataset(spec)
        second = generate_dataset(spec)
        assert_frames_close(first, second)
        assert len(first) == spec.rows
        assert spec.target in first.columns
        inferred = infer_target_kind(
            first,
            spec.target,
            time_column=spec.time_column,
        )
        assert inferred == spec.target_kind, (spec.slug, inferred)


def test_build_golden_datasets_returns_every_spec() -> None:
    datasets = build_golden_datasets()
    assert set(datasets) == set(SLUGS)
    for slug in SLUGS:
        golden = build_golden_dataset(slug)
        assert golden.spec.slug == slug
        assert golden.frame.equals(datasets[slug].frame)


def test_fixtures_match_generators() -> None:
    for spec in DATASET_SPECS:
        fixture = load_golden_fixture(spec.slug)
        generated = generate_dataset(spec)
        assert_frames_close(generated, fixture)


def test_fixture_manifest_matches_specs() -> None:
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    entries = {entry["slug"]: entry for entry in manifest["datasets"]}
    assert set(entries) == set(SLUGS)
    for spec in DATASET_SPECS:
        entry = entries[spec.slug]
        assert entry["target"] == spec.target
        assert entry["target_kind"] == spec.target_kind
        assert entry["rows"] == spec.rows
        assert tuple(entry["expected_families"]) == spec.expected_families
        assert tuple(entry["leakage_columns"]) == spec.leakage_columns
        assert entry["fixture_file"] == spec.fixture_file


def test_write_golden_fixtures_round_trips(tmp_path) -> None:
    destination = write_golden_fixtures(tmp_path)
    assert (destination / "manifest.json").is_file()
    for spec in DATASET_SPECS:
        generated = generate_dataset(spec)
        loaded = load_golden_fixture(spec.slug, tmp_path)
        assert_frames_close(generated, loaded)


def test_linear_dataset_has_known_signs_and_a_leaked_column() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["linear"])
    target = frame["y"].to_numpy(dtype=float)
    assert np.corrcoef(frame["x1"], target)[0, 1] > 0.5
    assert np.corrcoef(frame["x2"], target)[0, 1] < -0.4
    assert np.corrcoef(frame["y_leak"], target)[0, 1] > 0.999
    assert frame["record_id"].nunique() == len(frame)


def test_interaction_dataset_needs_the_product_term() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["interaction"])
    main_effects = ols_r2(frame, ["x1", "x2", "x3"], "y")
    with_interaction = interactive_r2(frame, "y")
    assert with_interaction > main_effects + 0.1
    assert with_interaction > 0.75


def test_binary_logistic_dataset_is_balanced_and_directional() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["binary_logistic"])
    values = frame["y_binary"]
    assert set(values.unique()) == {0, 1}
    assert values.mean() > 0.3
    assert values.mean() < 0.7
    target = values.to_numpy(dtype=float)
    assert np.corrcoef(frame["x1"], target)[0, 1] > 0.2
    assert np.corrcoef(frame["x2"], target)[0, 1] < -0.15
    assert np.corrcoef(frame["outcome_copy"], target)[0, 1] > 0.999


def test_multiclass_dataset_has_three_balanced_classes() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["multiclass"])
    values = frame["y_class"]
    assert set(values.unique()) == {0, 1, 2}
    counts = values.value_counts(normalize=True)
    assert counts.min() > 0.15
    assert counts.max() < 0.6
    # x1 separates class 1 from the rest, x2 separates class 2.
    assert frame.loc[frame["y_class"] == 1, "x1"].mean() > frame.loc[
        frame["y_class"] != 1, "x1"
    ].mean()
    assert frame.loc[frame["y_class"] == 2, "x2"].mean() < frame.loc[
        frame["y_class"] != 2, "x2"
    ].mean()


def test_count_dataset_is_overdispersed_and_directional() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["count_overdispersed"])
    values = frame["y_count"]
    assert (values >= 0).all()
    assert values.dtype.kind in {"i", "u"}
    ratio = float(values.var(ddof=1) / values.mean())
    assert ratio > 1.5
    target = values.to_numpy(dtype=float)
    assert np.corrcoef(frame["x1"], target)[0, 1] > 0.25
    assert np.corrcoef(frame["x2"], target)[0, 1] < -0.2


def test_time_series_has_trend_and_a_twelve_period_cycle() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["time_series"])
    time = frame["time"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    assert np.corrcoef(time, y)[0, 1] > 0.3

    coefficients = np.polyfit(time, y, 1)
    detrended = y - np.polyval(coefficients, time)
    period = 12
    seasonal = np.column_stack(
        [
            np.ones(len(time)),
            np.sin(2.0 * np.pi * time / period),
            np.cos(2.0 * np.pi * time / period),
        ]
    )
    beta, *_ = np.linalg.lstsq(seasonal, detrended, rcond=None)
    predicted = seasonal @ beta
    total = float(np.sum((detrended - detrended.mean()) ** 2))
    r2 = 1.0 - float(np.sum((detrended - predicted) ** 2)) / total
    assert r2 > 0.8


def test_group_means_dataset_separates_groups() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["group_means"])
    means = frame.groupby("group")["y"].mean()
    assert means["gamma"] > means["beta"] > means["alpha"] > means["delta"]
    group_means = frame["group"].map(means)
    between = float(((group_means - frame["y"].mean()) ** 2).sum())
    within = float(((frame["y"] - group_means) ** 2).sum())
    assert between > within


def test_missing_and_collinear_dataset_has_both_problems() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["missing_collinear"])
    assert frame["x2"].isna().sum() >= 5
    assert frame["x3"].isna().sum() >= 5
    mask = frame["x1"].notna() & frame["x2"].notna()
    correlation = np.corrcoef(
        frame.loc[mask, "x1"].to_numpy(dtype=float),
        frame.loc[mask, "x2"].to_numpy(dtype=float),
    )[0, 1]
    assert correlation > 0.8


def test_small_n_high_cardinality_dataset_is_small_and_sparse() -> None:
    frame = generate_dataset(SPECS_BY_SLUG["small_n_high_cardinality"])
    assert len(frame) < 60
    assert frame["site"].nunique() >= 20
    assert frame.groupby("site").size().max() <= 2
    target = frame["y"].to_numpy(dtype=float)
    assert np.corrcoef(frame["x1"], target)[0, 1] > 0.2


def test_local_runner_recovers_every_known_family(local_evaluation) -> None:
    assert local_evaluation.top_family_recovery == 1.0
    assert local_evaluation.baseline_improvement_rate == 1.0
    assert local_evaluation.leakage_clean
    for item in local_evaluation.datasets:
        assert item.recovery, (
            item.slug,
            item.top_two_families,
            item.family_ranking,
        )
        assert item.baseline_improved, (item.slug, item.improvement)
        assert item.leakage_ok, (item.slug, item.leakage_violations)
        assert item.candidate_count >= 1
        assert item.fold_count == 3
        assert all(
            candidate.score_std >= 0.0 for candidate in item.candidates
        )


def test_local_runner_excludes_leakage_and_identifier_columns(
    local_evaluation,
) -> None:
    linear = next(item for item in local_evaluation.datasets if item.slug == "linear")
    binary = next(
        item for item in local_evaluation.datasets if item.slug == "binary_logistic"
    )
    assert "y_leak" in linear.excluded_features
    assert "record_id" in linear.excluded_features
    assert "outcome_copy" in binary.excluded_features
    for item in (linear, binary):
        for candidate in item.candidates:
            assert item.target not in candidate.features
            assert "y_leak" not in candidate.features
            assert "outcome_copy" not in candidate.features


def test_local_runner_holdout_policy(local_evaluation) -> None:
    binary = next(
        item for item in local_evaluation.datasets if item.slug == "binary_logistic"
    )
    assert binary.holdout_rows == 40
    assert binary.holdout_score is not None

    small = next(
        item
        for item in local_evaluation.datasets
        if item.slug == "small_n_high_cardinality"
    )
    assert small.holdout_rows == 0
    assert small.holdout_score is None


def test_evaluation_summary_is_json_serializable(local_evaluation) -> None:
    payload = local_evaluation.to_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["top_family_recovery"] == 1.0
    assert decoded["baseline_improvement_rate"] == 1.0
    assert decoded["leakage_clean"] is True
    assert len(decoded["datasets"]) == len(DATASET_SPECS)
    assert all(gate["status"] == "pass" for gate in decoded["gates"])


def test_optional_backend_and_module_availability_are_reported(
    local_evaluation,
) -> None:
    assert set(local_evaluation.backend_availability) == {
        "sklearn",
        "statsmodels",
        "pysindy",
    }
    for status in local_evaluation.backend_availability.values():
        assert isinstance(status["available"], bool)
        assert "version" in status
    assert "pipeline" in local_evaluation.module_availability
    assert all(isinstance(value, bool) for value in local_evaluation.module_availability.values())
    assert local_evaluation.force_local is True


def test_runner_falls_back_to_the_local_baseline_when_modules_are_absent(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_search_eval, "import_optional_module", lambda name: None)
    evaluation = evaluate_dataset(SPECS_BY_SLUG["linear"], force_local=False)
    assert evaluation.backend == "local"
    assert evaluation.recovery is True
    assert evaluation.baseline_improved is True
    assert evaluation.leakage_ok is True
    assert any("local fallback" in warning for warning in evaluation.warnings)


def test_runner_uses_a_module_pipeline_when_one_is_available(monkeypatch) -> None:
    linear_spec = SimpleNamespace(family="linear", features=["x1", "x2"], interactions=[])
    baseline_spec = SimpleNamespace(family="baseline", features=[], interactions=[])
    linear_candidate = SimpleNamespace(
        spec=linear_spec,
        metrics={"r2": 0.9},
        metric_std={"r2": 0.01},
        aic=10.0,
        folds=[
            SimpleNamespace(
                metrics={"r2": 0.9},
                baseline_metrics={"r2": 0.0},
            )
        ],
    )
    baseline_candidate = SimpleNamespace(
        spec=baseline_spec,
        metrics={"r2": 0.0},
        metric_std={"r2": 0.0},
        aic=None,
        folds=[],
    )
    fake_result = SimpleNamespace(
        candidates=[linear_candidate, baseline_candidate],
        metric="r2",
        baseline=baseline_candidate,
        backend="fake",
    )
    fake_pipeline = SimpleNamespace(run_model_search=lambda *args, **kwargs: fake_result)

    def fake_import(name: str):
        if name == "whack26.model_search.pipeline":
            return fake_pipeline
        return None

    monkeypatch.setattr(model_search_eval, "import_optional_module", fake_import)
    evaluation = evaluate_dataset(SPECS_BY_SLUG["linear"], force_local=False)
    assert evaluation.backend == "modules:fake"
    assert evaluation.recovery is True
    assert evaluation.baseline_improved is True
    assert evaluation.leakage_ok is True
    assert evaluation.candidates[0].family == "linear"


def test_target_kind_inference_handles_every_golden_dataset() -> None:
    expected = {spec.slug: spec.target_kind for spec in DATASET_SPECS}
    for spec in DATASET_SPECS:
        frame = generate_dataset(spec)
        assert (
            infer_target_kind(frame, spec.target, time_column=spec.time_column)
            == expected[spec.slug]
        )
