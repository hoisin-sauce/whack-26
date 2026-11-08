from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from whack26.io import LoadedDataset, read_dataset
from whack26.profile import (
    DatasetProfile,
    _name_looks_sensitive,
    build_profile,
    build_summary,
)


def _demo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5, 6],
            "order_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ]
            ),
            "price": [100, 200, 300, 400, 500, 600],
            "amount": [0.0, -5.0, 10.0, None, 20.0, 30.0],
            "outcome_flag": [0, 1, 0, 1, 0, 1],
            "region": ["north", "south", "north", "east", "west", "south"],
            "customer_email": [f"user{index}@example.com" for index in range(6)],
            "customer_name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
            "notes": ["short note"] * 6,
        }
    )


def _load_demo(tmp_path: Path) -> LoadedDataset:
    path = tmp_path / "data.parquet"
    _demo_frame().to_parquet(path, index=False)
    return read_dataset(path, target="price")


def test_source_site_is_profiled_and_summarized(tmp_path: Path) -> None:
    loaded = _load_demo(tmp_path)
    profile = build_profile(
        loaded,
        description="Demo data.",
        source_site="https://example.com/datasets/demo",
        target="price",
    )
    assert profile.source_site == "https://example.com/datasets/demo"
    summary = build_summary(profile, loaded.frame)
    assert "Source site: https://example.com/datasets/demo" in summary


def _missing_file_dataset(
    tmp_path: Path, frame: pd.DataFrame, name: str = "missing.csv"
) -> LoadedDataset:
    return LoadedDataset(
        frame=frame,
        format="csv",
        source_path=tmp_path / name,
        warnings=[],
    )


class _FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        purpose: str = "generic",
    ) -> object:
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "purpose": purpose}
        )
        return self.response


class _BrokenLLM:
    def generate_json(self, **_: Any) -> object:
        raise RuntimeError("provider unavailable")


def _columns_by_name(profile: DatasetProfile) -> dict[str, Any]:
    return {column.name: column for column in profile.columns}


def test_profile_id_hash_roles_and_stats(tmp_path: Path) -> None:
    loaded = _load_demo(tmp_path)
    profile = build_profile(loaded, description="Tiny demo.", target="price")

    expected_sha = hashlib.sha256((tmp_path / "data.parquet").read_bytes()).hexdigest()
    assert profile.dataset_id == "ds_" + expected_sha[:12]
    assert profile.sha256 == expected_sha
    assert profile.source_path == str(tmp_path / "data.parquet")
    assert profile.format == "parquet"
    assert profile.rows == 6
    assert profile.description == "Tiny demo."
    assert len(profile.columns) == 9

    assert profile.target is not None
    assert profile.target.name == "price"
    assert profile.target.source == "user"
    assert profile.target.confidence == 1.0
    assert profile.time_column == "order_date"

    columns = _columns_by_name(profile)
    assert columns["price"].role == "target"
    assert columns["row_id"].role == "id"
    assert columns["order_date"].role == "time"
    assert columns["outcome_flag"].role == "leakage_risk"
    assert columns["amount"].role == "feature"
    assert columns["region"].role == "feature"

    assert columns["order_date"].datetime is True
    assert columns["price"].datetime is False

    amount = columns["amount"]
    assert amount.missing_pct == 16.67
    assert amount.cardinality == 5
    assert amount.numeric is not None
    assert amount.numeric["min"] == -5.0
    assert amount.numeric["max"] == 30.0
    assert amount.numeric["zeros"] == 1
    assert amount.numeric["negatives"] == 1
    assert amount.numeric["mean"] == pytest.approx(11.0)
    assert amount.numeric["std"] == pytest.approx(14.3178, rel=1e-4)

    region = columns["region"]
    assert region.numeric is None
    assert region.top_values is not None
    assert ("north", 2) in region.top_values

    assert columns["price"].numeric is not None
    assert columns["price"].numeric["zeros"] == 0
    assert columns["price"].numeric["negatives"] == 0

    for column in profile.columns:
        assert len(column.sample_values) <= 5
        assert all(len(value) <= 80 for value in column.sample_values)


def test_explicit_id_or_time_target_is_rejected(tmp_path: Path) -> None:
    # PROF-02: roles are mutually exclusive and id/time take precedence over
    # target, so an explicit target may not be an identifier or timestamp.
    loaded = _load_demo(tmp_path)

    with pytest.raises(ValueError, match="id column"):
        build_profile(loaded, target="row_id")
    with pytest.raises(ValueError, match="time column"):
        build_profile(loaded, target="order_date")


def test_missing_file_hashes_deterministic_csv(tmp_path: Path) -> None:
    frame = pd.DataFrame({"price": [10.0, 20.0], "size": [1.0, 2.0]})
    loaded = LoadedDataset(
        frame=frame,
        format="csv",
        source_path=tmp_path / "missing.csv",
        warnings=["existing warning"],
    )

    profile = build_profile(loaded, target="price")

    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    expected_sha = hashlib.sha256(csv_bytes).hexdigest()
    assert profile.sha256 == expected_sha
    assert profile.dataset_id == "ds_" + expected_sha[:12]
    assert profile.source_path == str(tmp_path / "missing.csv")
    assert profile.warnings[0] == "existing warning"


def test_build_profile_rejects_missing_target(tmp_path: Path) -> None:
    loaded = _load_demo(tmp_path)

    with pytest.raises(ValueError, match="available columns"):
        build_profile(loaded, target="not_a_column")


def test_target_is_inferred_from_name_and_shape(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "row_id": list(range(1, 11)),
            "feature_a": [index * 0.5 for index in range(10)],
            "feature_b": [index * 2.0 for index in range(10)],
            "label": [0, 1] * 5,
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    profile = build_profile(read_dataset(path))

    assert profile.target is not None
    assert profile.target.name == "label"
    assert profile.target.source == "inferred"
    assert 0.0 <= profile.target.confidence <= 1.0
    assert _columns_by_name(profile)["label"].role == "target"


def test_continuous_target_like_column_is_inferred(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "price": [float(index * 10) for index in range(10)],
            "size": [float(index) for index in range(10)],
        }
    )
    path = tmp_path / "continuous.csv"
    frame.to_csv(path, index=False)

    profile = build_profile(read_dataset(path))

    assert profile.target is not None
    assert profile.target.name == "price"
    assert profile.target.source == "inferred"


def test_no_plausible_target_is_none_and_warned(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"feature_a": [1.5, 2.5, 3.5], "feature_b": [4.5, 5.5, 6.5]}
    )
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)

    profile = build_profile(read_dataset(path))

    assert profile.target is None
    assert any("no plausible target" in warning for warning in profile.warnings)
    assert "Target: none" in build_summary(profile, frame)


def test_llm_target_is_used_when_it_names_a_column(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [index * 0.5 for index in range(10)],
            "feature_b": [index * 2.0 for index in range(10)],
            "label": [0, 1] * 5,
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    llm = _FakeLLM({"target": "feature_b", "confidence": 0.72, "rationale": "test"})

    profile = build_profile(read_dataset(path), llm=llm)

    assert profile.target is not None
    assert profile.target.name == "feature_b"
    assert profile.target.source == "inferred"
    assert profile.target.confidence == 0.72
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["purpose"] == "target"
    assert {"target", "confidence", "rationale"} <= set(call["schema"]["required"])


def test_invalid_llm_target_falls_back_to_heuristic(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [index * 0.5 for index in range(10)],
            "label": [0, 1] * 5,
        }
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    llm = _FakeLLM({"target": "not_a_column", "confidence": 0.9, "rationale": "bad"})

    profile = build_profile(read_dataset(path), llm=llm)

    assert profile.target is not None
    assert profile.target.name == "label"
    assert any("not a column" in warning for warning in profile.warnings)


def test_llm_failure_falls_back_to_heuristic(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"feature_a": [index * 0.5 for index in range(10)], "label": [0, 1] * 5}
    )
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)

    profile = build_profile(read_dataset(path), llm=_BrokenLLM())

    assert profile.target is not None
    assert profile.target.name == "label"
    assert any("LLM failed" in warning for warning in profile.warnings)


@pytest.mark.parametrize(
    ("column", "expected_role"),
    [
        ("row_id", "id"),
        ("order_date", "time"),
        ("id_target", "id"),
        ("date_target", "time"),
        ("outcome_target", "leakage_risk"),
        ("outcome_next_week", "leakage_risk"),
    ],
)
def test_llm_id_time_and_leakage_targets_are_rejected(
    tmp_path: Path, column: str, expected_role: str
) -> None:
    # PROF-01/PROF-02: the LLM's target is validated with the same role rules
    # as the profile. An id/time/leakage choice is rejected and the heuristic
    # target is used instead.
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "row_id": list(range(1, 11)),
            "order_date": dates,
            "id_target": list(range(101, 111)),
            "date_target": dates,
            "outcome_target": [0, 1] * 5,
            "outcome_next_week": [1, 0] * 5,
            "label": [0, 1] * 5,
        }
    )
    path = tmp_path / "targets.csv"
    frame.to_csv(path, index=False)
    llm = _FakeLLM({"target": column, "confidence": 0.95, "rationale": "bad"})

    profile = build_profile(read_dataset(path), llm=llm)

    assert profile.target is not None
    assert profile.target.name == "label"
    assert any("rejecting it" in warning for warning in profile.warnings)
    assert _columns_by_name(profile)[column].role == expected_role


def test_id_detection_requires_a_strong_signal(tmp_path: Path) -> None:
    # PROF-03: dtype plus near-uniqueness is not enough. A near-unique integer
    # measure and a near-unique short string category stay features, while a
    # named id and a UUID column are ids.
    rng = np.random.default_rng(11)
    prices = [int(value) for value in rng.integers(100, 10_000, size=100)]
    prices[-1] = prices[0]  # one repeat: near-unique but not exact
    categories = [f"cat_{index:03d}" for index in range(99)] + ["cat_000"]
    session_tokens = [str(uuid.uuid4()) for _ in range(100)]
    frame = pd.DataFrame(
        {
            "price": prices,
            "category": categories,
            "session_token": session_tokens,
            "row_id": list(range(100)),
            "label": [0, 1] * 50,
        }
    )
    path = tmp_path / "ids.csv"
    frame.to_csv(path, index=False)

    profile = build_profile(read_dataset(path))
    columns = _columns_by_name(profile)

    assert columns["price"].role == "feature"
    assert columns["category"].role == "feature"
    assert columns["session_token"].role == "id"
    assert columns["row_id"].role == "id"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("grid", False),
        ("valid", False),
        ("solid", False),
        ("user_id", True),
        ("customer_id", True),
        ("userid", True),
        ("primary_key", True),
    ],
)
def test_sensitive_name_matching_uses_whole_identifier_tokens(
    name: str, expected: bool
) -> None:
    # PROF-04: ordinary words ending in "id" are not identifiers.
    assert _name_looks_sensitive(name) is expected


def test_sensitive_columns_keep_no_raw_sample_values(tmp_path: Path) -> None:
    loaded = _load_demo(tmp_path)
    profile = build_profile(loaded, target="price")
    by_name = {column.name: column for column in profile.columns}

    assert by_name["customer_email"].sample_values == []
    assert by_name["customer_email"].top_values is None
    assert by_name["customer_name"].sample_values == []
    assert by_name["customer_name"].top_values is None
    assert by_name["region"].sample_values


def test_summary_content_and_sample_row_omission(tmp_path: Path) -> None:
    loaded = _load_demo(tmp_path)
    profile = build_profile(loaded, description="Tiny demo.", target="price")

    summary = build_summary(profile, loaded.frame)

    assert summary.startswith("Dataset: data.parquet")
    assert "Description: Tiny demo." in summary
    assert "Rows: 6; Columns: 9" in summary
    assert "Target: price (int64, 6 unique, 0.0%)" in summary
    assert "Time column: order_date" in summary
    assert "Columns:" in summary

    amount_line = next(
        line for line in summary.splitlines() if line.startswith("- amount ")
    )
    assert "missing 16.67%" in amount_line
    assert "numeric min=-5" in amount_line
    assert "zeros=1" in amount_line
    assert "negatives=1" in amount_line

    _, sample = summary.split("Sample rows (first 3, values truncated):", 1)
    sample_lines = [line for line in sample.strip().splitlines() if line]
    assert len(sample_lines) == 4  # header plus the first three rows
    assert "customer_email" not in sample
    assert "customer_name" not in sample
    assert "row_id" not in sample
    assert "notes" not in sample
    assert "price" in sample
    assert "region" in sample
    assert "user0@example.com" not in summary
    assert "Alice" not in summary


def test_sample_values_collapse_whitespace_and_truncate(tmp_path: Path) -> None:
    long_text = "  alpha\n\tbeta  " + "x" * 120
    frame = pd.DataFrame(
        {"long_label": [long_text, "short"], "value": [1.0, 2.0]}
    )
    loaded = _missing_file_dataset(tmp_path, frame, name="labels.csv")

    profile = build_profile(loaded)
    column = _columns_by_name(profile)["long_label"]

    assert len(column.sample_values[0]) == 80
    assert column.sample_values[0].startswith("alpha beta ")
    assert "\n" not in column.sample_values[0]
    assert "\t" not in column.sample_values[0]
    assert "short" in column.sample_values


def test_summary_is_capped_at_6000_characters(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            f"numeric_feature_column_{index:03d}": [
                float(index),
                float(index + 1),
                float(index + 2),
            ]
            for index in range(120)
        }
    )
    loaded = _missing_file_dataset(tmp_path, frame, name="wide.csv")

    profile = build_profile(loaded)
    summary = build_summary(profile, frame)

    assert len(summary) <= 6000
    assert summary.endswith("... (truncated)")


def test_summary_omits_ticket_and_cabin_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "travel.csv"
    pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
            "group": ["a", "b", "a"],
            "ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
            "cabin": ["C85", "C123", ""],
        }
    ).to_csv(path, index=False)
    loaded = read_dataset(path, target="value")
    profile = build_profile(loaded, target="value")
    summary = build_summary(profile, loaded.frame)

    _, sample = summary.split("Sample rows (first 3, values truncated):", 1)
    assert "ticket" not in sample
    assert "cabin" not in sample
    assert "A/5 21171" not in summary
    assert "C85" not in summary


def test_sample_rows_fall_back_to_safe_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "email": ["alice@example.com", "bob@example.com"],
            "full_name": ["Alice", "Bob"],
            "value": [1.0, 2.0],
        }
    )
    loaded = _missing_file_dataset(tmp_path, frame, name="people.csv")

    profile = build_profile(loaded)
    summary = build_summary(profile, frame)

    _, sample = summary.split("Sample rows (first 3, values truncated):", 1)
    assert "email" not in sample
    assert "full_name" not in sample
    assert "value" in sample
    assert "alice@example.com" not in summary
    assert "Alice" not in summary


def test_summary_has_no_sample_columns_when_all_are_sensitive(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "email": ["alice@example.com", "bob@example.com"],
            "full_name": ["Alice", "Bob"],
        }
    )
    loaded = _missing_file_dataset(tmp_path, frame, name="people.csv")

    profile = build_profile(loaded)
    summary = build_summary(profile, frame)

    _, sample = summary.split("Sample rows (first 3, values truncated):", 1)
    assert sample.strip() == "(no columns)"
    assert "alice@example.com" not in summary
    assert "Alice" not in summary
