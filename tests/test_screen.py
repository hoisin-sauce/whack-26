from __future__ import annotations

import numpy as np
import pandas as pd

from whack26.profile import ColumnProfile, DatasetProfile, TargetChoice
from whack26.screen import build_screen, format_screen


def make_profile(
    frame: pd.DataFrame,
    *,
    target: str | None,
    time_column: str | None = None,
) -> DatasetProfile:
    columns = []
    for name in frame.columns:
        series = frame[name]
        numeric = (
            {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "zeros": int((series == 0).sum()),
                "negatives": int((series < 0).sum()),
            }
            if pd.api.types.is_numeric_dtype(series)
            else None
        )
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                role=(
                    "target"
                    if name == target
                    else "time"
                    if name == time_column
                    else "feature"
                ),
                missing_pct=round(float(series.isna().sum()) / len(frame) * 100, 2),
                cardinality=int(series.nunique(dropna=True)),
                sample_values=[str(value) for value in series.dropna().head(3)],
                numeric=numeric,
                top_values=None,
                datetime=bool(pd.api.types.is_datetime64_any_dtype(series.dtype)),
            )
        )
    return DatasetProfile(
        dataset_id="ds_aaaaaaaaaaaa",
        source_path="data/sample.csv",
        sha256="a" * 64,
        format="csv",
        rows=len(frame),
        columns=columns,
        description="Synthetic data.",
        target=(
            TargetChoice(name=target, source="user", confidence=1.0)
            if target
            else None
        ),
        time_column=time_column,
        warnings=[],
    )


def test_screen_numeric_target_correlations_groups_and_missingness() -> None:
    rng = np.random.default_rng(7)
    income = rng.normal(5.0, 1.0, 80)
    frame = pd.DataFrame(
        {
            "income": income,
            "price": income * 10 + rng.normal(0.0, 1.0, 80),
            "region": ["north"] * 40 + ["south"] * 40,
            "optional": [None] * 10 + list(rng.normal(0.0, 1.0, 70)),
        }
    )
    profile = make_profile(frame, target="price")

    screen = build_screen(profile, frame)

    assert screen.target.kind == "numeric"
    assert screen.target.name == "price"
    assert screen.target.stats is not None
    assert screen.target.stats["count"] == 80
    assert screen.missingness == {"optional": 12.5}
    assert "income" in screen.numeric_columns
    assert "price" in screen.numeric_columns
    pair = next(
        item
        for item in screen.top_correlations
        if {item.a, item.b} == {"income", "price"}
    )
    assert pair.pearson is not None and pair.pearson > 0.9
    regions = next(item for item in screen.group_means if item.column == "region")
    assert regions.kind == "mean"
    assert {group.label for group in regions.groups} == {"north", "south"}
    assert all(group.n == 40 for group in regions.groups)


def test_screen_binary_target_reports_rates() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a"] * 30 + ["b"] * 30,
            "label": [1] * 24 + [0] * 6 + [1] * 6 + [0] * 24,
        }
    )
    profile = make_profile(frame, target="label")

    screen = build_screen(profile, frame)

    assert screen.target.kind == "categorical"
    assert screen.target.top_values is not None
    assert dict(screen.target.top_values) == {"1": 30, "0": 30}
    group = next(item for item in screen.group_means if item.column == "group")
    assert group.kind == "rate"
    values = {item.label: item.value for item in group.groups}
    assert values["a"] == 0.8
    assert values["b"] == 0.2


def test_screen_multiclass_target_skips_group_means() -> None:
    frame = pd.DataFrame(
        {
            "group": ["a", "b", "c"] * 20,
            "species": ["x", "y", "z"] * 20,
        }
    )
    profile = make_profile(frame, target="species")

    screen = build_screen(profile, frame)

    assert screen.target.kind == "categorical"
    assert screen.group_means == []
    assert any("categorical target" in warning for warning in screen.warnings)


def test_screen_time_trend_uses_months() -> None:
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": list(range(24)),
            "region": ["north", "south"] * 12,
        }
    )
    profile = make_profile(frame, target="value", time_column="date")

    screen = build_screen(profile, frame)

    assert screen.time_trend is not None
    assert screen.time_trend.period == "month"
    assert screen.time_trend.first.startswith("2020-01")
    assert len(screen.time_trend.points) == 24
    assert screen.time_trend.points[0].mean == 0.0


def test_screen_time_trend_ordered_values_without_datetime() -> None:
    frame = pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021] * 5,
            "value": [1.0, 2.0, 3.0, 4.0] * 5,
        }
    )
    profile = make_profile(frame, target="value", time_column="year")

    screen = build_screen(profile, frame)

    assert screen.time_trend is not None
    assert screen.time_trend.period == "ordered"
    assert screen.time_trend.first == "2018"
    assert screen.time_trend.last == "2021"


def test_format_screen_is_bounded_and_mentions_target() -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(100, dtype=float),
            "y": np.arange(100, dtype=float) * 2,
        }
    )
    profile = make_profile(frame, target="y")
    screen = build_screen(profile, frame)

    text = format_screen(screen, max_chars=200)

    assert len(text) <= 200
    assert "Target distribution" in text
    assert "correlations" in text.lower()
