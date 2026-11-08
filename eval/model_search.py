"""Golden-dataset evaluation for automated regression model search.

The harness builds nine deterministic synthetic datasets with known structure
(linear, interaction, logistic, multinomial, over-dispersed count, trend +
seasonality, group means, missing/collinear, and small-n/high-cardinality),
runs the model-search candidates over repeated holdouts, and reports:

- top-family recovery (the known family or a better alternative in the top two);
- improvement over the mean/majority baseline;
- target-leakage checks (no target-derived column may enter a candidate);
- optional backend availability (scikit-learn, statsmodels, pysindy).

The model-search package is imported lazily.  If ``whack26.model_search.pipeline``
is absent or unusable, the harness composes the data, candidate, split, and fit
modules when they are available.  If that composition is also unavailable, it
falls back to a compact numpy implementation that covers the same core families.
The fallback is intentionally small: it is an evaluation probe, not a
replacement for the production search.

No network access is required.

Usage::

    python eval/model_search.py --local
    python eval/model_search.py --local --json-output runs/eval/model_search.json
    python eval/model_search.py --write-fixtures
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whack26.model_search.types import ModelSearchConfig, TargetKind  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "model_search"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "eval" / "model-search-results.md"

LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "log_loss",
    "poisson_deviance",
}

# ---------------------------------------------------------------------------
# Golden dataset specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one known synthetic relationship."""

    slug: str
    name: str
    target: str
    target_kind: TargetKind
    expected_families: tuple[str, ...]
    seed: int
    notes: str = ""
    expected_features: tuple[str, ...] = ()
    expected_signs: tuple[tuple[str, int], ...] = ()
    expected_interaction: tuple[str, str] | None = None
    leakage_columns: tuple[str, ...] = ()
    time_column: str | None = None
    group_column: str | None = None
    required_feature_tokens: tuple[str, ...] = ()
    seasonality_period: int | None = None
    overdispersed: bool = False
    missing_columns: tuple[str, ...] = ()
    collinear_pair: tuple[str, str] | None = None
    high_cardinality_column: str | None = None
    rows: int = 0

    @property
    def fixture_file(self) -> str:
        return f"{self.slug}.csv"


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        slug="linear",
        name="Linear relationship with known signs",
        target="y",
        target_kind="continuous",
        expected_families=("linear",),
        expected_features=("x1", "x2"),
        expected_signs=(("x1", 1), ("x2", -1)),
        leakage_columns=("y_leak",),
        seed=1001,
        rows=160,
        notes="y = 3 + 2*x1 - 1.5*x2 + 0.4*x3 + noise; y_leak is target-derived.",
    ),
    DatasetSpec(
        slug="interaction",
        name="Interaction relationship",
        target="y",
        target_kind="continuous",
        expected_families=("interaction",),
        expected_features=("x1", "x2"),
        expected_interaction=("x1", "x2"),
        seed=1002,
        rows=160,
        notes="y = 1 + 0.6*x1 + 0.6*x2 + 1.5*x1*x2 + noise.",
    ),
    DatasetSpec(
        slug="binary_logistic",
        name="Binary logistic relationship",
        target="y_binary",
        target_kind="binary",
        expected_families=("logistic",),
        expected_features=("x1", "x2"),
        expected_signs=(("x1", 1), ("x2", -1)),
        leakage_columns=("outcome_copy",),
        seed=1003,
        rows=200,
        notes="logit(p) = -0.3 + 1.4*x1 - 1.0*x2 + 0.3*x3.",
    ),
    DatasetSpec(
        slug="multiclass",
        name="Multiclass relationship",
        target="y_class",
        target_kind="multiclass",
        expected_families=("multinomial",),
        expected_features=("x1", "x2"),
        seed=1004,
        rows=180,
        notes="Three-class softmax driven by x1 and x2.",
    ),
    DatasetSpec(
        slug="count_overdispersed",
        name="Over-dispersed count relationship",
        target="y_count",
        target_kind="count",
        expected_families=("negative_binomial", "poisson"),
        expected_features=("x1", "x2"),
        expected_signs=(("x1", 1), ("x2", -1)),
        seed=1005,
        rows=180,
        overdispersed=True,
        notes="Negative-binomial counts with mean exp(0.5 + 0.8*x1 - 0.6*x2).",
    ),
    DatasetSpec(
        slug="time_series",
        name="Time-series trend and seasonality",
        target="y",
        target_kind="time_series",
        expected_families=("linear", "sindy"),
        expected_features=("time",),
        time_column="time",
        required_feature_tokens=("time", "sin", "cos"),
        seasonality_period=12,
        seed=1006,
        rows=144,
        notes="y = 10 + 0.05*t + 2*sin(2*pi*t/12) + 0.8*cos(2*pi*t/12) + noise.",
    ),
    DatasetSpec(
        slug="group_means",
        name="Categorical group means",
        target="y",
        target_kind="continuous",
        expected_families=("group_comparison", "linear"),
        expected_features=("group",),
        group_column="group",
        seed=1007,
        rows=120,
        notes="Four groups with different means plus a weak numeric covariate.",
    ),
    DatasetSpec(
        slug="missing_collinear",
        name="Missing values and collinear features",
        target="y",
        target_kind="continuous",
        expected_families=("linear",),
        expected_features=("x1", "x3"),
        collinear_pair=("x1", "x2"),
        missing_columns=("x2", "x3"),
        seed=1008,
        rows=120,
        notes="x2 is nearly collinear with x1; x2 and x3 contain missing values.",
    ),
    DatasetSpec(
        slug="small_n_high_cardinality",
        name="Small n with a high-cardinality category",
        target="y",
        target_kind="continuous",
        expected_families=("linear", "group_comparison"),
        expected_features=("x1",),
        high_cardinality_column="site",
        seed=1009,
        rows=48,
        notes="24 sites with two observations each; only x1 generalises.",
    ),
)

SPECS_BY_SLUG = {spec.slug: spec for spec in DATASET_SPECS}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _record_ids(n: int) -> list[str]:
    return [f"rec_{index:04d}" for index in range(n)]


def generate_linear(seed: int = 1001) -> pd.DataFrame:
    """Continuous target with known coefficient signs and a leaked column."""

    rng = np.random.default_rng(seed)
    n = 160
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + 0.4 * x3 + noise
    y_leak = y + rng.normal(0.0, 0.01, n)
    return pd.DataFrame(
        {
            "record_id": _record_ids(n),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y": y,
            "y_leak": y_leak,
        }
    )


def generate_interaction(seed: int = 1002) -> pd.DataFrame:
    """Continuous target whose signal is dominated by an interaction term."""

    rng = np.random.default_rng(seed)
    n = 160
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    y = 1.0 + 0.6 * x1 + 0.6 * x2 + 1.5 * x1 * x2 + 0.2 * x3 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def generate_binary_logistic(seed: int = 1003) -> pd.DataFrame:
    """Binary target drawn from a known logistic model."""

    rng = np.random.default_rng(seed)
    n = 200
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    logit = -0.3 + 1.4 * x1 - 1.0 * x2 + 0.3 * x3
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, probability).astype(int)
    outcome_copy = y + rng.normal(0.0, 0.01, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y_binary": y,
            "outcome_copy": outcome_copy,
        }
    )


def generate_multiclass(seed: int = 1004) -> pd.DataFrame:
    """Three-class target from a softmax model."""

    rng = np.random.default_rng(seed)
    n = 180
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    scores = np.column_stack(
        [
            np.zeros(n),
            1.2 * x1 - 0.5 * x3,
            -1.0 * x2 + 0.4 * x3,
        ]
    )
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y = np.array(
        [rng.choice(3, p=row) for row in probabilities],
        dtype=int,
    )
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y_class": y})


def generate_count_overdispersed(seed: int = 1005) -> pd.DataFrame:
    """Negative-binomial counts with variance well above the mean."""

    rng = np.random.default_rng(seed)
    n = 180
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    mu = np.exp(0.5 + 0.8 * x1 - 0.6 * x2 + 0.1 * x3)
    dispersion = 0.5
    # numpy's negative_binomial(n, p) has mean n*(1-p)/p.
    probability = 1.0 / (1.0 + dispersion * mu)
    y = rng.negative_binomial(1.0 / dispersion, probability).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y_count": y})


def generate_time_series(seed: int = 1006) -> pd.DataFrame:
    """Monthly series with a linear trend and a 12-period cycle."""

    rng = np.random.default_rng(seed)
    n = 144
    time = np.arange(n, dtype=float)
    noise = rng.normal(0.0, 0.35, n)
    y = (
        10.0
        + 0.05 * time
        + 2.0 * np.sin(2.0 * np.pi * time / 12.0)
        + 0.8 * np.cos(2.0 * np.pi * time / 12.0)
        + noise
    )
    return pd.DataFrame({"time": time, "y": y})


def generate_group_means(seed: int = 1007) -> pd.DataFrame:
    """Four categorical groups with different means."""

    rng = np.random.default_rng(seed)
    n = 120
    groups = np.array(["alpha", "beta", "gamma", "delta"])
    group = rng.choice(groups, size=n)
    effects = {"alpha": 0.0, "beta": 2.0, "gamma": 4.0, "delta": -1.0}
    x1 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.6, n)
    y = np.array([1.0 + effects[value] for value in group]) + 0.3 * x1 + noise
    return pd.DataFrame({"group": group, "x1": x1, "y": y})


def generate_missing_collinear(seed: int = 1008) -> pd.DataFrame:
    """Collinear features with values missing at random."""

    rng = np.random.default_rng(seed)
    n = 120
    x1 = rng.normal(0.0, 1.0, n)
    x2 = 0.99 * x1 + rng.normal(0.0, 0.05, n)
    x3 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    y = 1.0 + 1.5 * x1 - 0.8 * x3 + 0.3 * x2 + noise
    frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    frame.loc[rng.random(n) < 0.15, "x2"] = np.nan
    frame.loc[rng.random(n) < 0.10, "x3"] = np.nan
    return frame


def generate_small_n_high_cardinality(seed: int = 1009) -> pd.DataFrame:
    """Small sample with a high-cardinality categorical column."""

    rng = np.random.default_rng(seed)
    n = 48
    sites = np.array([f"site_{index:02d}" for index in range(24)])
    site = np.repeat(sites, 2)
    rng.shuffle(site)
    site_effect = {name: rng.normal(0.0, 1.2) for name in sites}
    x1 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.45, n)
    y = np.array([site_effect[name] for name in site]) + 1.2 * x1 + noise
    return pd.DataFrame({"record_id": _record_ids(n), "site": site, "x1": x1, "y": y})


GENERATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "linear": generate_linear,
    "interaction": generate_interaction,
    "binary_logistic": generate_binary_logistic,
    "multiclass": generate_multiclass,
    "count_overdispersed": generate_count_overdispersed,
    "time_series": generate_time_series,
    "group_means": generate_group_means,
    "missing_collinear": generate_missing_collinear,
    "small_n_high_cardinality": generate_small_n_high_cardinality,
}


@dataclass(frozen=True)
class GoldenDataset:
    spec: DatasetSpec
    frame: pd.DataFrame


def generate_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """Generate one golden dataset from its specification."""

    generator = GENERATORS[spec.slug]
    return generator(spec.seed)


def build_golden_dataset(slug: str) -> GoldenDataset:
    spec = SPECS_BY_SLUG[slug]
    return GoldenDataset(spec=spec, frame=generate_dataset(spec))


def build_golden_datasets() -> dict[str, GoldenDataset]:
    return {spec.slug: build_golden_dataset(spec.slug) for spec in DATASET_SPECS}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spec_payload(spec: DatasetSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in fields(spec):
        value = getattr(spec, item.name)
        if isinstance(value, tuple):
            value = [list(part) if isinstance(part, tuple) else part for part in value]
        payload[item.name] = value
    payload["fixture_file"] = spec.fixture_file
    return payload


def write_golden_fixtures(directory: Path | None = None) -> Path:
    """Write every golden dataset and its manifest; returns the directory."""

    destination = Path(directory or FIXTURE_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_by": "eval/model_search.py",
        "datasets": [],
    }
    for spec in DATASET_SPECS:
        frame = generate_dataset(spec)
        frame.to_csv(
            destination / spec.fixture_file,
            index=False,
            float_format="%.12g",
            lineterminator="\n",
        )
        manifest["datasets"].append(_spec_payload(spec))
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_golden_fixture(slug: str, directory: Path | None = None) -> pd.DataFrame:
    spec = SPECS_BY_SLUG[slug]
    path = Path(directory or FIXTURE_DIR) / spec.fixture_file
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Target and feature typing
# ---------------------------------------------------------------------------


def infer_target_kind(
    frame: pd.DataFrame,
    target: str,
    *,
    time_column: str | None = None,
) -> TargetKind:
    """Small local mirror of the production target-typing rules."""

    if target not in frame.columns:
        raise KeyError(f"target {target!r} is not a column")
    if time_column and time_column in frame.columns:
        return "time_series"
    values = frame[target].dropna()
    if values.empty:
        return "continuous"
    unique = values.nunique()
    if unique == 2:
        return "binary"
    is_integer = pd.api.types.is_integer_dtype(values)
    is_categorical = not pd.api.types.is_numeric_dtype(values)
    name = target.lower()
    count_like = any(token in name for token in ("count", "num", "n_"))
    # A count-like name is authoritative even when the integer values happen to
    # have a small cardinality; otherwise "y_count" would look multiclass.
    if is_integer and bool((values >= 0).all()) and count_like:
        return "count"
    if 3 <= unique <= 20 and (is_categorical or is_integer):
        return "multiclass"
    if is_integer and bool((values >= 0).all()):
        mean = float(values.mean())
        variance = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        if mean > 0 and 0.5 <= variance / mean <= 2.0:
            return "count"
    return "continuous"


def _looks_like_identifier(series: pd.Series, name: str, n_rows: int) -> bool:
    lowered = name.lower()
    if lowered in {"id", "index", "record_id", "row_id", "uuid"}:
        return True
    if lowered.endswith("_id") and not pd.api.types.is_numeric_dtype(series):
        return True
    if not pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) >= n_rows:
        return True
    return False


def _correlation(series: pd.Series, target: np.ndarray) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(target)
    if mask.sum() < 5:
        return 0.0
    left = values[mask]
    right = target[mask]
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _target_derived_columns(
    frame: pd.DataFrame,
    target: str,
    numeric_columns: Sequence[str],
    explicit: Sequence[str],
) -> set[str]:
    """Return columns that must never be used as features."""

    leaked = {name for name in explicit if name in frame.columns}
    leaked.add(target)
    target_values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    for name in frame.columns:
        if name == target:
            continue
        lowered = name.lower()
        if "leak" in lowered or lowered in {"target_copy", "y_copy", "outcome_copy"}:
            leaked.add(name)
            continue
        if name not in numeric_columns:
            continue
        correlation = _correlation(frame[name], target_values)
        if abs(correlation) >= 0.999:
            leaked.add(name)
    return leaked


def _split_feature_columns(
    frame: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[list[str], list[str], set[str], list[str]]:
    """Return numeric, categorical, excluded columns and warnings."""

    target = spec.target
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: set[str] = set()
    warnings: list[str] = []

    for name in frame.columns:
        if name == target:
            continue
        series = frame[name]
        if _looks_like_identifier(series, name, len(frame)):
            excluded.add(name)
            continue
        if spec.time_column and name == spec.time_column:
            excluded.add(name)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            excluded.add(name)
            continue
        if pd.api.types.is_bool_dtype(series):
            categorical.append(name)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(name)
        else:
            categorical.append(name)

    leaked = _target_derived_columns(frame, target, numeric, spec.leakage_columns)
    for name in leaked:
        excluded.add(name)
        if name in numeric:
            numeric.remove(name)
        if name in categorical:
            categorical.remove(name)

    for name in spec.leakage_columns:
        if name in frame.columns:
            warnings.append(f"excluded target-derived column {name!r}")
    return numeric, categorical, excluded, warnings


# ---------------------------------------------------------------------------
# Splits and metrics
# ---------------------------------------------------------------------------


def _stratified_indices(
    y: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    test: list[int] = []
    for value in np.unique(y):
        indices = np.flatnonzero(y == value)
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * test_fraction)))
        cut = min(cut, len(indices) - 1) if len(indices) > 1 else 1
        test.extend(indices[:cut].tolist())
        train.extend(indices[cut:].tolist())
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int)


def _make_folds(
    n_rows: int,
    target_kind: TargetKind,
    y: np.ndarray,
    *,
    folds: int,
    test_fraction: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if target_kind == "time_series":
        segments = folds + 1
        boundaries = np.linspace(0, n_rows, segments + 1, dtype=int)
        result = []
        for index in range(folds):
            train = np.arange(0, boundaries[index + 1], dtype=int)
            test = np.arange(boundaries[index + 1], boundaries[index + 2], dtype=int)
            if len(train) and len(test):
                result.append((train, test))
        return result

    result = []
    classification = target_kind in {"binary", "multiclass"}
    for fold in range(folds):
        rng = np.random.default_rng(seed + 101 * fold)
        if classification:
            train, test = _stratified_indices(y, rng, test_fraction)
        else:
            order = rng.permutation(n_rows)
            cut = max(1, int(round(n_rows * test_fraction)))
            test = np.sort(order[:cut])
            train = np.sort(order[cut:])
        if len(train) and len(test):
            result.append((train, test))
    return result


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1e-12:
        return 0.0
    value = 1.0 - residual / total
    return value if math.isfinite(value) else -1.0


def _poisson_deviance(y_true: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    terms = np.zeros_like(y_true, dtype=float)
    positive = y_true > 0
    terms[positive] = y_true[positive] * np.log(y_true[positive] / mu[positive])
    return float(np.mean(2.0 * (terms - (y_true - mu))))


def _score_predictions(
    target_kind: TargetKind,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float]:
    """Return ``(score, raw_metric)`` where a higher score is better."""

    if target_kind in {"continuous", "time_series"}:
        return _r2(y_true, predictions), _r2(y_true, predictions)
    if target_kind in {"binary", "multiclass"}:
        accuracy = float(np.mean(np.asarray(predictions) == y_true))
        return accuracy, accuracy
    if target_kind == "count":
        deviance = _poisson_deviance(y_true, predictions)
        return -deviance, deviance
    return _r2(y_true, predictions), _r2(y_true, predictions)


def _primary_metric_name(target_kind: TargetKind) -> str:
    return {
        "continuous": "r2",
        "time_series": "r2",
        "binary": "accuracy",
        "multiclass": "accuracy",
        "count": "poisson_deviance",
    }[target_kind]


# ---------------------------------------------------------------------------
# Core numeric fallback
# ---------------------------------------------------------------------------


def _top_numeric_features(
    frame: pd.DataFrame,
    target: np.ndarray,
    columns: Sequence[str],
    k: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for name in columns:
        correlation = _correlation(frame[name], target)
        scored.append((abs(correlation), name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[: max(1, min(k, len(scored)))]]


def _impute_standardise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    for name in names:
        train_values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        test_values = pd.to_numeric(test[name], errors="coerce").to_numpy(dtype=float)
        median = float(np.nanmedian(train_values)) if np.isfinite(train_values).any() else 0.0
        train_values = np.where(np.isfinite(train_values), train_values, median)
        test_values = np.where(np.isfinite(test_values), test_values, median)
        mean = float(train_values.mean())
        std = float(train_values.std())
        if std < 1e-12:
            std = 1.0
        train_columns.append((train_values - mean) / std)
        test_columns.append((test_values - mean) / std)
    if not train_columns:
        return np.empty((len(train), 0)), np.empty((len(test), 0))
    return np.column_stack(train_columns), np.column_stack(test_columns)


def _fit_least_squares(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    if X.shape[1] == 0:
        return np.array([float(np.mean(y))])
    if alpha > 0.0:
        penalty = alpha * np.eye(X_design.shape[1])
        penalty[0, 0] = 0.0
        try:
            return np.linalg.solve(X_design.T @ X_design + penalty, X_design.T @ y)
        except np.linalg.LinAlgError:
            pass
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def _predict_least_squares(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    iterations: int = 40,
) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X_design.shape[1])
    beta[0] = math.log(max(float(y.mean()), 1e-6) / max(1.0 - float(y.mean()), 1e-6))
    penalty = alpha * np.eye(X_design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iterations):
        probability = _sigmoid(X_design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = X_design.T @ (y - probability) - penalty @ beta
        hessian = X_design.T @ (X_design * weights[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        beta += step
        if float(np.max(np.abs(step))) < 1e-8:
            break
        if not np.isfinite(beta).all():
            return np.zeros(X_design.shape[1])
    return beta


def _predict_logistic(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([np.ones(len(X)), X]) @ beta)


def _fit_poisson(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X_design.shape[1])
    beta[0] = math.log(max(float(y.mean()), 1e-6))
    penalty = alpha * np.eye(X_design.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(50):
        eta = np.clip(X_design @ beta, -20.0, 20.0)
        mu = np.exp(eta)
        weight = np.clip(mu, 1e-8, None)
        working = eta + (y - mu) / weight
        left = X_design.T @ (X_design * weight[:, None]) + penalty
        right = X_design.T @ (weight * working)
        try:
            updated = np.linalg.solve(left, right)
        except np.linalg.LinAlgError:
            break
        if float(np.max(np.abs(updated - beta))) < 1e-9:
            beta = updated
            break
        beta = updated
        if not np.isfinite(beta).all():
            break
    return beta


def _predict_poisson(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.column_stack([np.ones(len(X)), X]) @ beta, -20.0, 20.0))


def _gammaln(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import gammaln  # type: ignore

        return gammaln(values)
    except Exception:  # pragma: no cover - scipy is a core dependency
        return np.array([math.lgamma(float(value)) for value in values])


def _poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    return float(np.sum(y * np.log(mu) - mu - _gammaln(y + 1.0)))


def _negative_binomial_loglik(
    y: np.ndarray,
    mu: np.ndarray,
    dispersion: float,
) -> float:
    mu = np.clip(mu, 1e-9, None)
    dispersion = max(float(dispersion), 1e-6)
    size = 1.0 / dispersion
    probability = 1.0 / (1.0 + dispersion * mu)
    return float(
        np.sum(
            _gammaln(y + size)
            - _gammaln(size)
            - _gammaln(y + 1.0)
            + size * np.log(np.clip(probability, 1e-12, 1.0))
            + y * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
        )
    )


def _estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    estimate = float(np.mean(((y - mu) ** 2 - mu) / (mu**2)))
    return float(np.clip(estimate, 1e-4, 10.0))


def _fit_linear_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    interactions: Sequence[tuple[str, str]],
    *,
    alpha: float,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str], ...], int, float | None]:
    X_train, X_test = _impute_standardise(train, test, features)
    names = list(features)
    if interactions:
        train_parts = [X_train]
        test_parts = [X_test]
        for left, right in interactions:
            if left not in features or right not in features:
                continue
            left_index = list(features).index(left)
            right_index = list(features).index(right)
            train_parts.append(
                (X_train[:, left_index] * X_train[:, right_index])[:, None]
            )
            test_parts.append((X_test[:, left_index] * X_test[:, right_index])[:, None])
            names.append(f"{left}*{right}")
        X_train = np.column_stack(train_parts)
        X_test = np.column_stack(test_parts)
    beta = _fit_least_squares(X_train, y_train, alpha=alpha)
    predictions = _predict_least_squares(beta, X_test)
    residual = y_train - _predict_least_squares(beta, X_train)
    sse = float(np.sum(residual**2))
    n = len(y_train)
    param_count = int(X_train.shape[1]) + 1
    aic = None
    if n > 1 and sse > 1e-12:
        aic = n * math.log(sse / n) + 2.0 * param_count
    return predictions, tuple(names), tuple(interactions), param_count, aic


def _fit_classification_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    interactions: Sequence[tuple[str, str]],
    *,
    multiclass: bool,
    alpha: float = 1.0,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str], ...], int, float | None]:
    X_train, X_test = _impute_standardise(train, test, features)
    names = list(features)
    if interactions:
        train_parts = [X_train]
        test_parts = [X_test]
        for left, right in interactions:
            if left not in features or right not in features:
                continue
            left_index = list(features).index(left)
            right_index = list(features).index(right)
            train_parts.append(
                (X_train[:, left_index] * X_train[:, right_index])[:, None]
            )
            test_parts.append((X_test[:, left_index] * X_test[:, right_index])[:, None])
            names.append(f"{left}*{right}")
        X_train = np.column_stack(train_parts)
        X_test = np.column_stack(test_parts)

    classes = np.unique(y_train)
    if len(classes) < 2:
        predictions = np.full(len(test), classes[0])
        return predictions, tuple(names), tuple(interactions), 1, None

    if not multiclass or len(classes) == 2:
        binary = (y_train == classes[1]).astype(float)
        beta = _fit_logistic(X_train, binary, alpha=alpha)
        probability = _predict_logistic(beta, X_test)
        predictions = np.where(probability >= 0.5, classes[1], classes[0])
        return predictions, tuple(names), tuple(interactions), len(beta), None

    scores: list[np.ndarray] = []
    for value in classes:
        binary = (y_train == value).astype(float)
        if binary.sum() == 0:
            scores.append(np.full(len(test), -np.inf))
            continue
        beta = _fit_logistic(X_train, binary, alpha=alpha)
        scores.append(_predict_logistic(beta, X_test))
    stacked = np.column_stack(scores)
    predictions = classes[np.argmax(stacked, axis=1)]
    param_count = int(len(classes) * (X_train.shape[1] + 1))
    return predictions, tuple(names), tuple(interactions), param_count, None


def _fit_count_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    features: Sequence[str],
    *,
    negative_binomial: bool,
) -> tuple[np.ndarray, tuple[str, ...], int, float | None]:
    X_train, X_test = _impute_standardise(train, test, features)
    beta = _fit_poisson(X_train, y_train)
    mu_train = _predict_poisson(beta, X_train)
    predictions = _predict_poisson(beta, X_test)
    if negative_binomial:
        dispersion = _estimate_dispersion(y_train, mu_train)
        loglik = _negative_binomial_loglik(y_train, mu_train, dispersion)
        param_count = int(X_train.shape[1]) + 2
    else:
        loglik = _poisson_loglik(y_train, mu_train)
        param_count = int(X_train.shape[1]) + 1
    aic = -2.0 * loglik + 2.0 * param_count
    return predictions, tuple(features), param_count, aic


def _fit_group_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    column: str,
    target_kind: TargetKind,
    *,
    shrinkage: float = 1.0,
) -> tuple[np.ndarray, tuple[str, ...], int, float | None]:
    train_groups = train[column].astype(str).to_numpy()
    test_groups = test[column].astype(str).to_numpy()
    values = np.unique(train_groups)
    param_count = max(1, len(values))

    if target_kind in {"binary", "multiclass"}:
        classes = np.unique(y_train)
        global_counts = np.array([(y_train == value).sum() for value in classes], dtype=float)
        global_probability = (global_counts + 0.5) / (global_counts.sum() + 0.5 * len(classes))
        probabilities: dict[str, np.ndarray] = {}
        for value in values:
            mask = train_groups == value
            counts = np.array([(y_train[mask] == cls).sum() for cls in classes], dtype=float)
            probabilities[value] = (counts + 0.5) / (counts.sum() + 0.5 * len(classes))
        predictions = np.empty(len(test), dtype=y_train.dtype)
        for index, value in enumerate(test_groups):
            probability = probabilities.get(value, global_probability)
            predictions[index] = classes[int(np.argmax(probability))]
        return predictions, (column,), param_count, None

    global_mean = float(np.mean(y_train))
    means: dict[str, float] = {}
    for value in values:
        mask = train_groups == value
        count = float(mask.sum())
        group_mean = float(np.mean(y_train[mask])) if count else global_mean
        means[value] = (count * group_mean + shrinkage * global_mean) / (count + shrinkage)
    predictions = np.array(
        [means.get(value, global_mean) for value in test_groups],
        dtype=float,
    )
    return predictions, (column,), param_count, None


def _fit_group_bin_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    column: str,
    *,
    bins: int = 4,
) -> tuple[np.ndarray, tuple[str, ...], int, float | None]:
    train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
    test_values = pd.to_numeric(test[column], errors="coerce").to_numpy(dtype=float)
    median = float(np.nanmedian(train_values)) if np.isfinite(train_values).any() else 0.0
    train_values = np.where(np.isfinite(train_values), train_values, median)
    test_values = np.where(np.isfinite(test_values), test_values, median)
    edges = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return np.full(len(test), float(np.mean(y_train))), (column,), 1, None
    train_bins = np.clip(np.digitize(train_values, edges[1:-1]), 0, len(edges) - 2)
    test_bins = np.clip(np.digitize(test_values, edges[1:-1]), 0, len(edges) - 2)
    global_mean = float(np.mean(y_train))
    means = {
        value: float(np.mean(y_train[train_bins == value]))
        for value in np.unique(train_bins)
    }
    predictions = np.array([means.get(value, global_mean) for value in test_bins])
    return predictions, (column,), len(means), None


def _fit_time_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_column: str,
    y_train: np.ndarray,
    period: int | None,
    *,
    sindy: bool,
    alpha: float = 1e-4,
) -> tuple[np.ndarray, tuple[str, ...], int, float | None]:
    def design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        time = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=float)
        known = np.isfinite(time)
        if not known.all():
            fill = float(np.nanmedian(time)) if known.any() else 0.0
            time = np.where(known, time, fill)
        parts = [time]
        names = ["time"]
        if sindy:
            parts.extend([time**2, time**3])
            names.extend(["time_sq", "time_cube"])
        if period:
            parts.extend(
                [
                    np.sin(2.0 * np.pi * time / period),
                    np.cos(2.0 * np.pi * time / period),
                ]
            )
            names.extend([f"sin_{period}", f"cos_{period}"])
        return np.column_stack(parts), names

    X_train, names = design(train)
    X_test, _ = design(test)
    beta = _fit_least_squares(X_train, y_train, alpha=alpha)
    predictions = _predict_least_squares(beta, X_test)
    residual = y_train - _predict_least_squares(beta, X_train)
    sse = float(np.sum(residual**2))
    n = len(y_train)
    param_count = int(X_train.shape[1]) + 1
    aic = None
    if n > 1 and sse > 1e-12:
        aic = n * math.log(sse / n) + 2.0 * param_count
    return predictions, tuple(names), param_count, aic


# ---------------------------------------------------------------------------
# Candidate templates and scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CandidateTemplate:
    family: str
    strategy: str
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def key(self) -> str:
        rendered = ",".join(f"{name}={value}" for name, value in self.params)
        return f"{self.strategy}({rendered})"


@dataclass
class _FoldScore:
    score: float
    raw_metric: float
    features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    param_count: int
    aic: float | None


@dataclass
class ScoredCandidate:
    family: str
    strategy: str
    features: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    mean_score: float
    score_std: float
    raw_metric: str
    raw_metric_mean: float
    fold_scores: list[float]
    aic: float | None
    param_count: int
    template_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "strategy": self.strategy,
            "features": list(self.features),
            "interactions": [list(pair) for pair in self.interactions],
            "mean_score": round(self.mean_score, 6),
            "score_std": round(self.score_std, 6),
            "metric": self.raw_metric,
            "metric_mean": round(self.raw_metric_mean, 6),
            "fold_scores": [round(value, 6) for value in self.fold_scores],
            "aic": None if self.aic is None else round(self.aic, 6),
            "param_count": self.param_count,
            "template": self.template_key,
        }


def _candidate_templates(
    spec: DatasetSpec,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[_CandidateTemplate]:
    kind = spec.target_kind
    templates: list[_CandidateTemplate] = [
        _CandidateTemplate("baseline", "baseline"),
    ]
    if kind == "continuous":
        for k in (1, 3, 5):
            templates.append(_CandidateTemplate("linear", "linear", (("k", k),)))
        templates.append(
            _CandidateTemplate("linear", "ridge", (("k", 5), ("alpha", 1.0)))
        )
        if len(numeric_columns) >= 2:
            templates.append(
                _CandidateTemplate("interaction", "interaction", (("rank", 0),))
            )
            templates.append(
                _CandidateTemplate("interaction", "interaction", (("rank", 1),))
            )
        for column in categorical_columns:
            templates.append(
                _CandidateTemplate("group_comparison", "group", (("column", column),))
            )
        if numeric_columns:
            templates.append(
                _CandidateTemplate("group_comparison", "group_bin", (("rank", 0),))
            )
    elif kind == "binary":
        for k in (1, 3):
            templates.append(_CandidateTemplate("logistic", "logistic", (("k", k),)))
        if len(numeric_columns) >= 2:
            templates.append(
                _CandidateTemplate("logistic", "logistic_interaction", (("rank", 0),))
            )
        for column in categorical_columns:
            templates.append(
                _CandidateTemplate("group_comparison", "group", (("column", column),))
            )
    elif kind == "multiclass":
        for k in (1, 3):
            templates.append(
                _CandidateTemplate("multinomial", "multinomial", (("k", k),))
            )
        for column in categorical_columns:
            templates.append(
                _CandidateTemplate("group_comparison", "group", (("column", column),))
            )
    elif kind == "count":
        for k in (1, 3):
            templates.append(_CandidateTemplate("poisson", "poisson", (("k", k),)))
            templates.append(
                _CandidateTemplate(
                    "negative_binomial",
                    "negative_binomial",
                    (("k", k),),
                )
            )
        for column in categorical_columns:
            templates.append(
                _CandidateTemplate("group_comparison", "group", (("column", column),))
            )
    elif kind == "time_series":
        templates.append(_CandidateTemplate("linear", "time_trend", (("period", 0),)))
        for period in (4, 7, 12, 24):
            templates.append(
                _CandidateTemplate("linear", "time", (("period", period),))
            )
        templates.append(
            _CandidateTemplate("sindy", "sindy", (("period", 12),))
        )
    return templates


def _fit_template(
    template: _CandidateTemplate,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    spec: DatasetSpec,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[np.ndarray, _FoldScore]:
    kind = spec.target_kind
    strategy = template.strategy
    params = dict(template.params)

    if strategy == "baseline":
        if kind in {"binary", "multiclass"}:
            classes, counts = np.unique(y_train, return_counts=True)
            predictions = np.full(len(test), classes[int(np.argmax(counts))])
        else:
            predictions = np.full(len(test), float(np.mean(y_train)))
        return predictions, _FoldScore(
            score=0.0,
            raw_metric=0.0,
            features=(),
            interactions=(),
            param_count=1,
            aic=None,
        )

    if strategy in {"linear", "ridge", "interaction"}:
        k = int(params.get("k", 3))
        features = _top_numeric_features(train, y_train, numeric_columns, k)
        interactions: tuple[tuple[str, str], ...] = ()
        if strategy == "interaction":
            rank = int(params.get("rank", 0))
            candidates = _top_numeric_features(train, y_train, numeric_columns, 2)
            if rank > 0:
                candidates = _top_numeric_features(train, y_train, numeric_columns, rank + 2)
                candidates = candidates[rank : rank + 2]
            if len(candidates) >= 2:
                features = candidates[:2]
                interactions = (tuple(candidates[:2]),)  # type: ignore[arg-type]
        alpha = float(params.get("alpha", 1e-6))
        predictions, names, pairs, param_count, aic = _fit_linear_candidate(
            train,
            test,
            y_train,
            features,
            interactions,
            alpha=alpha,
        )
        return predictions, _FoldScore(0.0, 0.0, names, pairs, param_count, aic)

    if strategy in {"logistic", "logistic_interaction"}:
        k = int(params.get("k", 3))
        features = _top_numeric_features(train, y_train, numeric_columns, k)
        interactions: tuple[tuple[str, str], ...] = ()
        if strategy == "logistic_interaction":
            candidates = _top_numeric_features(train, y_train, numeric_columns, 2)
            if len(candidates) >= 2:
                features = candidates[:2]
                interactions = (tuple(candidates[:2]),)  # type: ignore[arg-type]
        predictions, names, pairs, param_count, aic = _fit_classification_candidate(
            train,
            test,
            y_train,
            features,
            interactions,
            multiclass=False,
        )
        return predictions, _FoldScore(0.0, 0.0, names, pairs, param_count, aic)

    if strategy == "multinomial":
        k = int(params.get("k", 3))
        features = _top_numeric_features(train, y_train, numeric_columns, k)
        predictions, names, pairs, param_count, aic = _fit_classification_candidate(
            train,
            test,
            y_train,
            features,
            (),
            multiclass=True,
        )
        return predictions, _FoldScore(0.0, 0.0, names, pairs, param_count, aic)

    if strategy in {"poisson", "negative_binomial"}:
        k = int(params.get("k", 3))
        features = _top_numeric_features(train, y_train, numeric_columns, k)
        predictions, names, param_count, aic = _fit_count_candidate(
            train,
            test,
            y_train,
            features,
            negative_binomial=strategy == "negative_binomial",
        )
        return predictions, _FoldScore(0.0, 0.0, names, (), param_count, aic)

    if strategy == "group":
        column = str(params["column"])
        predictions, names, param_count, aic = _fit_group_candidate(
            train,
            test,
            y_train,
            column,
            kind,
        )
        return predictions, _FoldScore(0.0, 0.0, names, (), param_count, aic)

    if strategy == "group_bin":
        column = _top_numeric_features(train, y_train, numeric_columns, 1)[0]
        predictions, names, param_count, aic = _fit_group_bin_candidate(
            train,
            test,
            y_train,
            column,
        )
        return predictions, _FoldScore(0.0, 0.0, names, (), param_count, aic)

    if strategy in {"time", "time_trend", "sindy"}:
        if not spec.time_column:
            raise ValueError("time-series candidate without a time column")
        period = int(params.get("period", 0)) or None
        predictions, names, param_count, aic = _fit_time_candidate(
            train,
            test,
            spec.time_column,
            y_train,
            period,
            sindy=strategy == "sindy",
        )
        return predictions, _FoldScore(0.0, 0.0, names, (), param_count, aic)

    raise ValueError(f"unknown candidate strategy {strategy!r}")


def _rank_candidates(candidates: Iterable[ScoredCandidate]) -> list[ScoredCandidate]:
    def key(candidate: ScoredCandidate) -> tuple[float, float, float, int, str]:
        aic = float("inf") if candidate.aic is None else candidate.aic
        return (
            -candidate.mean_score,
            candidate.score_std,
            aic,
            candidate.param_count,
            candidate.family,
        )

    return sorted(candidates, key=key)


# ---------------------------------------------------------------------------
# Optional backends and lazy module integration
# ---------------------------------------------------------------------------


def import_optional_module(name: str) -> Any | None:
    """Import a module, returning ``None`` when it is absent or broken."""

    try:
        return importlib.import_module(name)
    except Exception:
        return None


def optional_backend_availability() -> dict[str, dict[str, Any]]:
    """Report optional third-party backends without importing them."""

    distributions = {
        "sklearn": "scikit-learn",
        "statsmodels": "statsmodels",
        "pysindy": "pysindy",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, distribution in distributions.items():
        available = False
        version: str | None = None
        try:
            available = importlib.util.find_spec(name) is not None
        except Exception:
            available = False
        if available:
            try:
                version = importlib.metadata.version(distribution)
            except Exception:
                version = None
        result[name] = {"available": available, "version": version}
    return result


MODULE_NAMES: tuple[str, ...] = (
    "data",
    "splits",
    "screen",
    "candidates",
    "fit",
    "metrics",
    "select",
    "diagnostics",
    "trees",
    "sindy",
    "report",
    "pipeline",
)


def model_search_module_availability() -> dict[str, bool]:
    availability: dict[str, bool] = {}
    for name in MODULE_NAMES:
        availability[name] = (
            import_optional_module(f"whack26.model_search.{name}") is not None
        )
    return availability


def _module_result_to_candidates(
    result: Any,
    spec: DatasetSpec,
    forbidden_features: set[str] | None = None,
) -> tuple[list[ScoredCandidate], str, float | None, list[str]] | None:
    """Convert a contract ``ModelSearchResult`` into local candidate records.

    Candidates that use the target or a target-derived column are dropped and
    reported instead of being silently ranked.
    """

    candidates = getattr(result, "candidates", None)
    if not candidates:
        return None
    forbidden = {str(name) for name in (forbidden_features or set())}
    metric = str(getattr(result, "metric", _primary_metric_name(spec.target_kind)))
    lower_is_better = metric in LOWER_IS_BETTER
    scored: list[ScoredCandidate] = []
    baseline_score: float | None = None
    violations: list[str] = []
    for candidate in candidates:
        candidate_spec = getattr(candidate, "spec", None)
        if candidate_spec is None:
            continue
        values = getattr(candidate, "metrics", {}) or {}
        value = values.get(metric)
        if value is None and values:
            metric = next(iter(values))
            value = values[metric]
        if value is None or not math.isfinite(float(value)):
            continue
        raw_features = tuple(getattr(candidate_spec, "features", []) or [])
        offending = [
            name
            for name in raw_features
            if name in forbidden or name.split("*")[0] in forbidden
        ]
        if offending:
            family = str(getattr(candidate_spec, "family", "unknown"))
            violations.append(f"{family}:{','.join(offending)}")
            continue
        mean_score = -float(value) if lower_is_better else float(value)
        standard_deviations = getattr(candidate, "metric_std", {}) or {}
        std = float(standard_deviations.get(metric, 0.0) or 0.0)
        features = raw_features
        raw_interactions = getattr(candidate_spec, "interactions", []) or []
        interactions = tuple(tuple(pair) for pair in raw_interactions)
        family = str(getattr(candidate_spec, "family", "baseline"))
        param_count = len(features) + len(interactions) + 1
        item = ScoredCandidate(
            family=family,
            strategy="module",
            features=features,
            interactions=interactions,  # type: ignore[arg-type]
            mean_score=mean_score,
            score_std=std,
            raw_metric=metric,
            raw_metric_mean=float(value),
            fold_scores=[
                float(fold.metrics.get(metric, value))
                for fold in getattr(candidate, "folds", [])
            ],
            aic=getattr(candidate, "aic", None),
            param_count=param_count,
            template_key="module",
        )
        scored.append(item)
        if family == "baseline":
            baseline_score = mean_score
    if not scored:
        return None

    if baseline_score is None:
        baseline = getattr(result, "baseline", None)
        if baseline is not None:
            values = getattr(baseline, "metrics", {}) or {}
            value = values.get(metric)
            if value is not None and math.isfinite(float(value)):
                baseline_score = -float(value) if lower_is_better else float(value)
        if baseline_score is None:
            for candidate in candidates:
                folds = getattr(candidate, "folds", []) or []
                if folds:
                    values = getattr(folds[0], "baseline_metrics", {}) or {}
                    value = values.get(metric)
                    if value is not None and math.isfinite(float(value)):
                        baseline_score = -float(value) if lower_is_better else float(value)
                        break
    return scored, metric, baseline_score, violations


@dataclass
class _ModuleRun:
    candidates: list[ScoredCandidate]
    metric: str
    baseline_score: float | None
    backend: str
    holdout_score: float | None = None
    holdout_rows: int = 0
    leakage_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _run_module_pipeline(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    config: ModelSearchConfig,
    forbidden_features: set[str] | None = None,
) -> _ModuleRun | None:
    pipeline = import_optional_module("whack26.model_search.pipeline")
    if pipeline is None:
        return None
    runner = getattr(pipeline, "run_model_search", None)
    if not callable(runner):
        return None
    result: Any | None = None
    attempts = (
        lambda: runner(frame, target=spec.target, config=config),
        lambda: runner(frame, target=spec.target),
        lambda: runner(frame, spec.target),
        lambda: runner(frame),
    )
    for attempt in attempts:
        try:
            result = attempt()
            break
        except TypeError:
            continue
        except Exception:
            return None
    if result is None:
        return None
    converted = _module_result_to_candidates(result, spec, forbidden_features)
    if converted is None:
        return None
    scored, metric, baseline_score, violations = converted
    if not any(candidate.family != "baseline" for candidate in scored):
        return None
    backend = str(getattr(result, "backend", "modules"))
    return _ModuleRun(
        candidates=scored,
        metric=metric,
        baseline_score=baseline_score,
        backend=f"modules:{backend}",
        leakage_violations=violations,
    )


def _family_matches(expected: str, actual: str) -> bool:
    """Whether ``actual`` is the expected family or a strictly richer one."""

    if expected == actual:
        return True
    # An interaction model is a linear model with product terms, and a
    # negative-binomial model is a richer Poisson; both are documented
    # "strictly better alternatives" when the simpler family is expected.
    return (expected, actual) in {
        ("linear", "interaction"),
        ("logistic", "interaction"),
        ("poisson", "negative_binomial"),
        ("group_comparison", "linear"),
    }


def _engineer_time_features(
    frame: pd.DataFrame,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, list[str]]:
    """Add the trend and seasonal terms the core fitter expects for series."""

    if not spec.time_column or spec.time_column not in frame.columns:
        return frame, []
    engineered = frame.copy()
    time = pd.to_numeric(engineered[spec.time_column], errors="coerce").to_numpy(
        dtype=float
    )
    known = np.isfinite(time)
    if not known.all():
        fill = float(np.nanmedian(time)) if known.any() else 0.0
        time = np.where(known, time, fill)
    engineered["time_trend"] = time
    names = ["time_trend"]
    period = spec.seasonality_period or 12
    engineered[f"sin_{period}"] = np.sin(2.0 * np.pi * time / period)
    engineered[f"cos_{period}"] = np.cos(2.0 * np.pi * time / period)
    names.extend([f"sin_{period}", f"cos_{period}"])
    return engineered, names


def _run_composed_modules(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    config: ModelSearchConfig,
) -> _ModuleRun | None:
    """Best-effort composition of data/candidates/splits/fit when no pipeline exists."""

    modules = {
        name: import_optional_module(f"whack26.model_search.{name}")
        for name in ("data", "candidates", "splits", "fit")
    }
    if any(module is None for module in modules.values()):
        return None
    data_module = modules["data"]
    candidate_module = modules["candidates"]
    split_module = modules["splits"]
    fit_module = modules["fit"]

    numeric_columns, _categorical_columns, excluded, _warnings = (
        _split_feature_columns(frame, spec)
    )
    leaked = _target_derived_columns(
        frame,
        spec.target,
        numeric_columns,
        spec.leakage_columns,
    )
    fit_frame = frame
    engineered: list[str] = []
    if spec.target_kind == "time_series":
        fit_frame, engineered = _engineer_time_features(frame, spec)

    try:
        info = data_module.build_dataset_info(
            fit_frame,
            spec.target,
            time_column=spec.time_column,
            group_column=spec.group_column,
            time_aware=bool(spec.time_column),
        )
    except Exception:
        return None

    usable = [
        name
        for name in info.usable_features
        if name not in leaked and name not in excluded
    ]
    for name in engineered:
        if name not in usable:
            usable.append(name)
    feature_kinds = {
        feature.name: feature.kind
        for feature in info.features
        if feature.usable and feature.name in usable
    }
    for name in engineered:
        feature_kinds[name] = "numeric"
    if spec.time_column and spec.time_column in feature_kinds:
        feature_kinds[spec.time_column] = "numeric"

    try:
        specs = candidate_module.generate_candidates(
            target=spec.target,
            target_kind=spec.target_kind,
            features=usable,
            feature_kinds=feature_kinds,
            frame=fit_frame,
            max_features=10,
            max_candidates=config.max_candidates,
        )
    except Exception:
        return None
    if not specs:
        return None

    violations: list[str] = []
    accepted_specs = []
    for model_spec in specs:
        features = list(getattr(model_spec, "features", []) or [])
        offending = [name for name in features if name in leaked]
        if offending:
            violations.append(
                f"{getattr(model_spec, 'family', 'unknown')}:{','.join(offending)}"
            )
            continue
        accepted_specs.append(model_spec)
    if not accepted_specs:
        return None

    try:
        plan = split_module.build_splits(
            fit_frame,
            spec.target,
            target_kind=spec.target_kind,
            n_splits=config.folds,
            test_fraction=config.test_fraction,
            seed=config.seed,
            group_column=spec.group_column,
            time_column=spec.time_column,
            time_aware=bool(spec.time_column),
            final_holdout_fraction=config.final_holdout_fraction,
            final_holdout_min_rows=config.final_holdout_min_rows,
        )
    except Exception:
        return None
    if not plan.splits:
        return None

    metric = _primary_metric_name(spec.target_kind)
    collected: list[ScoredCandidate] = []
    fitted_pairs: list[tuple[ScoredCandidate, Any]] = []
    for model_spec in accepted_specs:
        fold_scores: list[float] = []
        raw_values: list[float] = []
        first_fit: Any | None = None
        failed = False
        for split in plan.splits:
            train = fit_frame.iloc[split.train_index]
            test = fit_frame.iloc[split.test_index]
            try:
                fitted = fit_module.fit_model(model_spec, train, spec.target)
                predictions = fitted.predict(test)
                score, raw = _score_predictions(
                    spec.target_kind,
                    test[spec.target].to_numpy(),
                    predictions,
                )
                if not math.isfinite(score):
                    failed = True
                    break
            except Exception:
                failed = True
                break
            fold_scores.append(score)
            raw_values.append(raw)
            first_fit = fitted
        if failed or len(fold_scores) != len(plan.splits):
            continue
        features = tuple(getattr(model_spec, "features", []) or [])
        raw_interactions = getattr(model_spec, "interactions", []) or []
        interactions = tuple(tuple(pair) for pair in raw_interactions)
        parameter_count = len(features) + len(interactions) + 1
        log_likelihood = getattr(first_fit, "log_likelihood", None)
        aic = (
            None
            if log_likelihood is None
            else -2.0 * float(log_likelihood) + 2.0 * parameter_count
        )
        candidate = ScoredCandidate(
            family=str(getattr(model_spec, "family", "baseline")),
            strategy="module",
            features=features,
            interactions=interactions,  # type: ignore[arg-type]
            mean_score=float(np.mean(fold_scores)),
            score_std=float(np.std(fold_scores, ddof=0)),
            raw_metric=metric,
            raw_metric_mean=float(np.mean(raw_values)),
            fold_scores=fold_scores,
            aic=aic,
            param_count=parameter_count,
            template_key="module",
        )
        collected.append(candidate)
        if first_fit is not None:
            fitted_pairs.append((candidate, model_spec))

    if not collected or not any(
        candidate.family != "baseline" for candidate in collected
    ):
        return None
    ranked = _rank_candidates(collected)
    baseline_score = next(
        (
            candidate.mean_score
            for candidate in ranked
            if candidate.family == "baseline"
        ),
        None,
    )

    holdout_score: float | None = None
    holdout_rows = 0
    if plan.final_holdout is not None:
        holdout_rows = int(plan.final_holdout.test_rows)
        best = next(
            (candidate for candidate in ranked if candidate.family != "baseline"),
            None,
        )
        if best is not None:
            best_spec = next(
                (
                    model_spec
                    for candidate, model_spec in fitted_pairs
                    if candidate is best
                ),
                None,
            )
            if best_spec is not None:
                train = fit_frame.iloc[plan.final_holdout.train_index]
                test = fit_frame.iloc[plan.final_holdout.test_index]
                if len(train) == 0:
                    train = fit_frame.drop(index=plan.final_holdout.test_index)
                try:
                    fitted = fit_module.fit_model(best_spec, train, spec.target)
                    predictions = fitted.predict(test)
                    holdout_score, _ = _score_predictions(
                        spec.target_kind,
                        test[spec.target].to_numpy(),
                        predictions,
                    )
                except Exception:
                    holdout_score = None

    return _ModuleRun(
        candidates=collected,
        metric=metric,
        baseline_score=baseline_score,
        backend="modules:composed",
        holdout_score=holdout_score,
        holdout_rows=holdout_rows,
        leakage_violations=violations,
        warnings=list(getattr(plan, "warnings", []) or []),
    )


# ---------------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------------


@dataclass
class DatasetEvaluation:
    slug: str
    name: str
    target: str
    target_kind: TargetKind
    metric: str
    backend: str
    fold_count: int
    candidate_count: int
    family_ranking: list[str]
    top_two_families: list[str]
    recovered_family: str | None
    recovery: bool
    baseline_score: float
    best_score: float
    improvement: float
    baseline_improved: bool
    leakage_ok: bool
    leakage_violations: list[str]
    excluded_features: list[str]
    candidates: list[ScoredCandidate] = field(default_factory=list)
    holdout_score: float | None = None
    holdout_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "name": self.name,
            "target": self.target,
            "target_kind": self.target_kind,
            "metric": self.metric,
            "backend": self.backend,
            "folds": self.fold_count,
            "candidate_count": self.candidate_count,
            "family_ranking": self.family_ranking,
            "top_two_families": self.top_two_families,
            "recovered_family": self.recovered_family,
            "recovery": self.recovery,
            "baseline_score": round(self.baseline_score, 6),
            "best_score": round(self.best_score, 6),
            "improvement": round(self.improvement, 6),
            "baseline_improved": self.baseline_improved,
            "leakage_ok": self.leakage_ok,
            "leakage_violations": self.leakage_violations,
            "excluded_features": self.excluded_features,
            "holdout_score": (
                None if self.holdout_score is None else round(self.holdout_score, 6)
            ),
            "holdout_rows": self.holdout_rows,
            "warnings": self.warnings,
            "errors": self.errors,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def _check_recovery(
    spec: DatasetSpec,
    ranked: list[ScoredCandidate],
) -> tuple[list[str], list[str], str | None, bool]:
    non_baseline = [candidate for candidate in ranked if candidate.family != "baseline"]
    family_ranking: list[str] = []
    for candidate in non_baseline:
        if candidate.family not in family_ranking:
            family_ranking.append(candidate.family)
    top_two = non_baseline[:2]
    top_two_families = [candidate.family for candidate in top_two]
    recovered = next(
        (
            candidate.family
            for candidate in top_two
            if any(
                _family_matches(expected, candidate.family)
                for expected in spec.expected_families
            )
        ),
        None,
    )
    if recovered is None:
        return family_ranking, top_two_families, None, False

    combined_features = {
        feature for candidate in top_two for feature in candidate.features
    }
    if spec.required_feature_tokens:
        for token in spec.required_feature_tokens:
            if not any(token in feature for feature in combined_features):
                return family_ranking, top_two_families, recovered, False
    if spec.expected_interaction:
        expected = tuple(sorted(spec.expected_interaction))
        for candidate in top_two:
            for pair in candidate.interactions:
                if tuple(sorted(pair)) == expected:
                    return family_ranking, top_two_families, recovered, True
        return family_ranking, top_two_families, recovered, False
    if spec.group_column and spec.expected_families[0] == "group_comparison":
        if any(candidate.family == "group_comparison" for candidate in top_two):
            return family_ranking, top_two_families, recovered, True
        # A linear candidate that is clearly better is accepted as the
        # documented "strictly better alternative".
        if recovered == "linear":
            return family_ranking, top_two_families, recovered, True
        return family_ranking, top_two_families, recovered, False
    return family_ranking, top_two_families, recovered, True


def _holdout_split(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    config: ModelSearchConfig,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    n_rows = len(frame)
    if n_rows < config.final_holdout_min_rows:
        return frame, None
    fraction = config.final_holdout_fraction
    if fraction <= 0.0:
        return frame, None
    holdout_size = max(1, int(round(n_rows * fraction)))
    holdout_size = min(holdout_size, n_rows - 1)
    if spec.target_kind == "time_series":
        holdout_index = np.arange(n_rows - holdout_size, n_rows)
    else:
        rng = np.random.default_rng(seed + 7919)
        y = frame[spec.target].to_numpy()
        if spec.target_kind in {"binary", "multiclass"}:
            _, holdout_index = _stratified_indices(y, rng, fraction)
        else:
            holdout_index = np.sort(rng.permutation(n_rows)[:holdout_size])
    mask = np.zeros(n_rows, dtype=bool)
    mask[holdout_index] = True
    return frame.loc[~mask].reset_index(drop=True), frame.loc[mask].reset_index(drop=True)


def evaluate_dataset(
    spec: DatasetSpec,
    frame: pd.DataFrame | None = None,
    *,
    config: ModelSearchConfig | None = None,
    force_local: bool = False,
) -> DatasetEvaluation:
    """Run candidate search on one golden dataset and score recovery."""

    search_config = config or ModelSearchConfig()
    dataset = generate_dataset(spec) if frame is None else frame.copy()
    if spec.target not in dataset.columns:
        raise KeyError(f"{spec.slug}: target {spec.target!r} is missing")

    warnings: list[str] = []
    errors: list[str] = []
    numeric_columns, categorical_columns, excluded, split_warnings = (
        _split_feature_columns(dataset, spec)
    )
    warnings.extend(split_warnings)
    leaked_columns = _target_derived_columns(
        dataset,
        spec.target,
        numeric_columns,
        spec.leakage_columns,
    )

    metric = _primary_metric_name(spec.target_kind)
    scored: list[ScoredCandidate] | None = None
    all_candidates: list[ScoredCandidate] | None = None
    baseline_score: float | None = None
    backend = "local"
    holdout_score: float | None = None
    holdout_rows = 0
    module_leakage: list[str] = []

    if not force_local:
        module_result = _run_module_pipeline(
            dataset,
            spec,
            search_config,
            leaked_columns,
        )
        if module_result is None:
            module_result = _run_composed_modules(dataset, spec, search_config)
        if module_result is not None and module_result.baseline_score is not None:
            scored = module_result.candidates
            all_candidates = list(scored)
            metric = module_result.metric
            baseline_score = module_result.baseline_score
            backend = module_result.backend
            holdout_score = module_result.holdout_score
            holdout_rows = module_result.holdout_rows
            module_leakage = list(module_result.leakage_violations)
            warnings.extend(module_result.warnings)
        else:
            warnings.append(
                "model-search modules unavailable or incomplete; using local fallback"
            )

    if scored is None:
        pool, holdout = _holdout_split(dataset, spec, search_config, seed=search_config.seed)
        if holdout is not None:
            holdout_rows = len(holdout)
        y_pool = pd.to_numeric(pool[spec.target], errors="coerce").to_numpy(dtype=float)
        folds = _make_folds(
            len(pool),
            spec.target_kind,
            y_pool,
            folds=search_config.folds,
            test_fraction=search_config.test_fraction,
            seed=search_config.seed,
        )
        if len(folds) < search_config.folds:
            warnings.append(
                f"only {len(folds)} usable folds for {len(pool)} rows"
            )
        templates = _candidate_templates(spec, numeric_columns, categorical_columns)
        fold_records: dict[int, list[_FoldScore]] = defaultdict(list)
        occurrences: dict[int, int] = defaultdict(int)
        for train_index, test_index in folds:
            train = pool.iloc[train_index]
            test = pool.iloc[test_index]
            y_train = pd.to_numeric(
                train[spec.target], errors="coerce"
            ).to_numpy(dtype=float)
            y_test = pd.to_numeric(
                test[spec.target], errors="coerce"
            ).to_numpy(dtype=float)
            if spec.target_kind in {"binary", "multiclass"}:
                y_train = train[spec.target].to_numpy()
                y_test = test[spec.target].to_numpy()
            for index, template in enumerate(templates):
                try:
                    predictions, fold = _fit_template(
                        template,
                        train,
                        test,
                        y_train,
                        spec,
                        numeric_columns,
                        categorical_columns,
                    )
                except Exception as exc:  # keep the remaining candidates running
                    occurrences[index] += 1
                    if occurrences[index] == 1:
                        errors.append(f"{template.key}: {type(exc).__name__}: {exc}")
                    continue
                score, raw_metric = _score_predictions(
                    spec.target_kind,
                    y_test,
                    predictions,
                )
                if not math.isfinite(score):
                    occurrences[index] += 1
                    continue
                fold.score = score
                fold.raw_metric = raw_metric
                fold_records[index].append(fold)

        candidates: list[ScoredCandidate] = []
        for index, records in fold_records.items():
            if len(records) < len(folds):
                continue
            template = templates[index]
            scores = [record.score for record in records]
            raw_values = [record.raw_metric for record in records]
            first = records[0]
            candidates.append(
                ScoredCandidate(
                    family=template.family,
                    strategy=template.strategy,
                    features=first.features,
                    interactions=first.interactions,
                    mean_score=float(np.mean(scores)),
                    score_std=float(np.std(scores, ddof=0)),
                    raw_metric=metric,
                    raw_metric_mean=float(np.mean(raw_values)),
                    fold_scores=scores,
                    aic=first.aic,
                    param_count=first.param_count,
                    template_key=template.key,
                )
            )
        if not candidates:
            raise RuntimeError(f"{spec.slug}: no candidate produced finite predictions")

        ranked = _rank_candidates(candidates)
        all_candidates = list(ranked)
        top_n = max(1, search_config.top_n)
        top = ranked[:top_n]
        scored = top
        baseline_candidate = next(
            (candidate for candidate in ranked if candidate.family == "baseline"),
            None,
        )
        if baseline_candidate is None:
            baseline_score = 0.0
        else:
            baseline_score = baseline_candidate.mean_score

        non_baseline = [candidate for candidate in ranked if candidate.family != "baseline"]
        if not non_baseline:
            raise RuntimeError(f"{spec.slug}: no non-baseline candidate survived")

        best = non_baseline[0]
        if holdout is not None and holdout_rows:
            best_template = next(
                (
                    template
                    for template in templates
                    if template.key == best.template_key
                ),
                None,
            )
            if best_template is not None:
                y_pool = pd.to_numeric(
                    pool[spec.target], errors="coerce"
                ).to_numpy(dtype=float)
                y_holdout = pd.to_numeric(
                    holdout[spec.target], errors="coerce"
                ).to_numpy(dtype=float)
                if spec.target_kind in {"binary", "multiclass"}:
                    y_pool = pool[spec.target].to_numpy()
                    y_holdout = holdout[spec.target].to_numpy()
                try:
                    predictions, _ = _fit_template(
                        best_template,
                        pool,
                        holdout,
                        y_pool,
                        spec,
                        numeric_columns,
                        categorical_columns,
                    )
                    holdout_score, _ = _score_predictions(
                        spec.target_kind,
                        y_holdout,
                        predictions,
                    )
                except Exception as exc:
                    errors.append(f"holdout: {type(exc).__name__}: {exc}")

    assert scored is not None
    assert baseline_score is not None
    ranked = _rank_candidates(scored)
    family_ranking, top_two_families, recovered_family, recovery = _check_recovery(
        spec,
        ranked,
    )
    best_score = max((candidate.mean_score for candidate in scored), default=0.0)
    improvement = best_score - baseline_score
    baseline_improved = improvement > search_config.margin

    leaked = leaked_columns
    violations: list[str] = []
    for candidate in all_candidates or scored:
        for feature in candidate.features:
            parts = [part for part in feature.split("*") if part]
            if any(part in leaked for part in parts):
                violations.append(f"{candidate.family}:{feature}")
        for pair in candidate.interactions:
            for value in pair:
                if value in leaked:
                    violations.append(f"{candidate.family}:interaction:{value}")
    violations = sorted(set(violations + module_leakage))
    excluded_features = sorted(excluded)

    if not recovery:
        warnings.append(
            "expected family "
            f"{list(spec.expected_families)} not in top two {top_two_families}"
        )
    if not baseline_improved:
        warnings.append(
            f"best improvement {improvement:.4f} does not exceed margin "
            f"{search_config.margin:.4f}"
        )
    if violations:
        warnings.append("target-derived columns entered candidate features")

    return DatasetEvaluation(
        slug=spec.slug,
        name=spec.name,
        target=spec.target,
        target_kind=spec.target_kind,
        metric=metric,
        backend=backend,
        fold_count=search_config.folds,
        candidate_count=len(all_candidates or scored),
        family_ranking=family_ranking,
        top_two_families=top_two_families,
        recovered_family=recovered_family,
        recovery=recovery,
        baseline_score=baseline_score,
        best_score=best_score,
        improvement=improvement,
        baseline_improved=baseline_improved,
        leakage_ok=not violations,
        leakage_violations=violations,
        excluded_features=excluded_features,
        candidates=scored,
        holdout_score=holdout_score,
        holdout_rows=holdout_rows,
        warnings=warnings,
        errors=errors,
    )


@dataclass
class ModelSearchEvaluation:
    datasets: list[DatasetEvaluation]
    backend_availability: dict[str, dict[str, Any]]
    module_availability: dict[str, bool]
    force_local: bool
    seed: int
    folds: int
    margin: float

    @property
    def top_family_recovery(self) -> float:
        if not self.datasets:
            return 0.0
        return sum(1 for item in self.datasets if item.recovery) / len(self.datasets)

    @property
    def baseline_improvement_rate(self) -> float:
        if not self.datasets:
            return 0.0
        return sum(1 for item in self.datasets if item.baseline_improved) / len(
            self.datasets
        )

    @property
    def leakage_clean(self) -> bool:
        return all(item.leakage_ok for item in self.datasets)

    def gates(self) -> list[dict[str, Any]]:
        failed = [item.slug for item in self.datasets if not item.recovery]
        baseline_failed = [
            item.slug for item in self.datasets if not item.baseline_improved
        ]
        leakage_failed = [item.slug for item in self.datasets if not item.leakage_ok]
        return [
            {
                "name": "Top-family recovery",
                "status": "pass" if not failed else "fail",
                "detail": (
                    f"{self.top_family_recovery:.0%} of datasets have the known "
                    "family or a better alternative in the top two"
                    + (f"; failed: {', '.join(failed)}" if failed else "")
                ),
            },
            {
                "name": "Baseline improvement",
                "status": "pass" if not baseline_failed else "fail",
                "detail": (
                    f"{self.baseline_improvement_rate:.0%} of datasets beat the "
                    "baseline beyond the configured margin"
                    + (
                        f"; failed: {', '.join(baseline_failed)}"
                        if baseline_failed
                        else ""
                    )
                ),
            },
            {
                "name": "Target leakage",
                "status": "pass" if not leakage_failed else "fail",
                "detail": (
                    "no candidate uses a target or target-derived column"
                    if not leakage_failed
                    else f"violations: {', '.join(leakage_failed)}"
                ),
            },
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "mode": "local" if self.force_local else "auto",
            "seed": self.seed,
            "folds": self.folds,
            "margin": self.margin,
            "top_family_recovery": round(self.top_family_recovery, 6),
            "baseline_improvement_rate": round(self.baseline_improvement_rate, 6),
            "leakage_clean": self.leakage_clean,
            "backend_availability": self.backend_availability,
            "module_availability": self.module_availability,
            "gates": self.gates(),
            "datasets": [item.to_dict() for item in self.datasets],
        }


def evaluate_all(
    *,
    force_local: bool = False,
    config: ModelSearchConfig | None = None,
    specs: Sequence[DatasetSpec] | None = None,
) -> ModelSearchEvaluation:
    search_config = config or ModelSearchConfig()
    selected = list(specs or DATASET_SPECS)
    evaluations = [
        evaluate_dataset(
            spec,
            config=search_config,
            force_local=force_local,
        )
        for spec in selected
    ]
    return ModelSearchEvaluation(
        datasets=evaluations,
        backend_availability=optional_backend_availability(),
        module_availability=model_search_module_availability(),
        force_local=force_local,
        seed=search_config.seed,
        folds=search_config.folds,
        margin=search_config.margin,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(evaluation: ModelSearchEvaluation) -> str:
    lines = [
        "# Automated model-search evaluation",
        "",
        f"Mode: {evaluation.to_dict()['mode']}",
        f"Folds: {evaluation.folds}; seed: {evaluation.seed}",
        f"Top-family recovery: **{evaluation.top_family_recovery:.0%}**",
        f"Baseline improvement: **{evaluation.baseline_improvement_rate:.0%}**",
        f"Leakage clean: **{'yes' if evaluation.leakage_clean else 'no'}**",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for gate in evaluation.gates():
        lines.append(
            f"| {_escape(gate['name'])} | {gate['status']} | {_escape(gate['detail'])} |"
        )
    lines.extend(
        [
            "",
            "## Datasets",
            "",
            "| Dataset | Target | Kind | Metric | Top families | Improvement | Leakage | Holdout |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in evaluation.datasets:
        holdout = "-" if item.holdout_score is None else f"{item.holdout_score:.4f}"
        lines.append(
            f"| {item.slug} | {item.target} | {item.target_kind} | {item.metric} | "
            f"{_escape(', '.join(item.top_two_families))} | "
            f"{item.improvement:+.4f} | "
            f"{'ok' if item.leakage_ok else 'FAIL'} | {holdout} |"
        )
    lines.extend(
        [
            "",
            "## Optional backends",
            "",
            "| Backend | Available | Version |",
            "|---|---|---|",
        ]
    )
    for name, status in evaluation.backend_availability.items():
        lines.append(
            f"| {name} | {'yes' if status['available'] else 'no'} | "
            f"{status.get('version') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Model-search modules",
            "",
            "| Module | Available |",
            "|---|---|",
        ]
    )
    for name, available in evaluation.module_availability.items():
        lines.append(f"| {name}.py | {'yes' if available else 'no'} |")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the golden-dataset model-search evaluation."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="force the local numpy fallback instead of the installed pipeline",
    )
    parser.add_argument(
        "--datasets",
        help="comma-separated dataset slugs (default: all)",
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-fixtures",
        action="store_true",
        help="regenerate tests/fixtures/model_search and exit",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="exit 0 even when a blocking gate fails",
    )
    return parser


def _parse_specs(value: str | None) -> list[DatasetSpec]:
    if not value:
        return list(DATASET_SPECS)
    slugs = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [slug for slug in slugs if slug not in SPECS_BY_SLUG]
    if unknown:
        raise ValueError(
            f"unknown dataset slug(s) {unknown}; choose from {sorted(SPECS_BY_SLUG)}"
        )
    return [SPECS_BY_SLUG[slug] for slug in slugs]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_fixtures:
        destination = write_golden_fixtures()
        print(f"wrote fixtures to {destination}")
        return 0
    try:
        specs = _parse_specs(args.datasets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    config = ModelSearchConfig(
        folds=args.folds,
        seed=args.seed,
        margin=args.margin,
    )
    evaluation = evaluate_all(force_local=args.local, config=config, specs=specs)
    payload = evaluation.to_dict()
    json_path = args.json_output or args.output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(evaluation), encoding="utf-8")
    status = "pass" if all(gate["status"] == "pass" for gate in payload["gates"]) else "fail"
    print(
        f"status={status} recovery={evaluation.top_family_recovery:.0%} "
        f"improvement={evaluation.baseline_improvement_rate:.0%} "
        f"results={args.output}"
    )
    if status == "fail" and not args.allow_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
