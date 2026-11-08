from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from whack26.hypotheses.variants import (
    aggregate_variant_reports,
    anonymize_frame,
    build_variants,
    evaluate_hypothesis_variants,
    evaluate_variants,
    jumble_target,
    redact_text,
)
from whack26.io import read_dataset
from whack26.profile import build_profile


def make_profile(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    target: str | None = "y",
    name: str = "data.csv",
):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    loaded = read_dataset(path, target=target)
    return build_profile(loaded, target=target), loaded.frame


def pii_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(rows, dtype="float64"),
            "y": np.arange(rows, dtype="int64") * 2,
            "full_name": [f"Person {index}" for index in range(rows)],
            "email": [f"person{index}@example.com" for index in range(rows)],
            "phone": [f"+1 555 010 {index:04d}" for index in range(rows)],
            "address": [f"{index} Main Street" for index in range(rows)],
            "account": [f"ACCT-{index:08d}" for index in range(rows)],
            "ip_address": [f"10.0.0.{index}" for index in range(rows)],
            "code": [f"C{index:05d}" for index in range(rows)],
            "meta": [
                f"note {index} mail person{index}@example.com"
                if index % 4 == 0
                else f"note {index} with no contact details"
                for index in range(rows)
            ],
        }
    )


def test_evaluate_hypothesis_variants_detects_target_copying(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "x": np.arange(30, dtype="float64"),
            "y": np.arange(30, dtype="float64") * 2.0,
        }
    )
    profile, frame = make_profile(tmp_path, frame)
    variants, frames = build_variants(profile, frame, seed=7)
    module = tmp_path / "leaky.py"
    module.write_text(
        """
import pandas as pd

PARAMETERS = {"ridge": 0.0}


def applies(df: pd.DataFrame) -> bool:
    return "y" in df.columns


def _fit(df, seed=0, params=None):
    return {}


def _predict(model, df):
    return df["y"]


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    return {"predictions": _predict({}, df), "target": "y"}
""",
        encoding="utf-8",
    )

    report = evaluate_hypothesis_variants(
        hypothesis_id="hyp_leaky",
        module_path=module,
        frames=frames,
        variants=variants,
        target_column="y",
        target_kind="continuous",
        run_dir=tmp_path / "run",
    )

    assert report.hypothesis_id == "hyp_leaky"
    assert report.jumbled_label_leak is True
    assert report.verdict == "reject"
    assert any("jumbled_label_leak" in reason for reason in report.reasons)

    aggregate = aggregate_variant_reports(variants, [report])
    assert aggregate.jumbled_label_leak is True
    assert aggregate.verdict == "reject"
    assert any("hyp_leaky" in warning for warning in aggregate.warnings)


def test_build_variants_is_deterministic(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())

    first, first_frames = build_variants(profile, frame, seed=11)
    second, second_frames = build_variants(profile, frame, seed=11)

    assert first.model_dump() == second.model_dump()
    for kind in ("original", "anonymized", "jumbled"):
        assert first_frames[kind].equals(second_frames[kind])
        assert first_frames[kind].dtypes.equals(second_frames[kind].dtypes)

    other, other_frames = build_variants(profile, frame, seed=12)
    assert other.jumbled.frame_sha256 != first.jumbled.frame_sha256
    assert not other_frames["jumbled"].equals(first_frames["jumbled"])


def test_build_variants_does_not_mutate_inputs(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    frame_before = frame.copy(deep=True)
    profile_before = profile.model_dump()

    build_variants(profile, frame, seed=3)

    assert frame.equals(frame_before)
    assert profile.model_dump() == profile_before


def test_anonymize_drops_direct_identifiers_and_keeps_schema(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    anonymized, dropped, notes = anonymize_frame(
        profile, frame, return_details=True
    )

    for column in (
        "full_name",
        "email",
        "phone",
        "address",
        "account",
        "ip_address",
    ):
        assert column in dropped
        assert column not in anonymized.columns

    # A non-sensitive high-cardinality identifier is hashed, not silently kept.
    assert "code" in anonymized.columns
    assert anonymized["code"].str.startswith("id_").all()
    assert any("hashed" in note for note in notes)

    # Remaining columns keep their name, dtype, and index.
    assert str(anonymized["x"].dtype) == str(frame["x"].dtype)
    assert str(anonymized["y"].dtype) == str(frame["y"].dtype)
    assert anonymized.index.equals(frame.index)
    assert "y" in anonymized.columns
    # PII inside a kept free-text column is redacted.
    assert anonymized["meta"].str.contains("example.com").sum() == 0
    assert anonymized["meta"].str.contains(r"\[redacted\]").sum() > 0


def test_helpers_return_primary_object_by_default(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())

    anonymized = anonymize_frame(profile, frame)
    jumbled = jumble_target(profile, frame, seed=1)

    assert isinstance(anonymized, pd.DataFrame)
    assert "email" not in anonymized.columns
    assert isinstance(jumbled, pd.Series)
    assert jumbled.dtype == frame["y"].dtype


def test_redact_text_removes_pii_and_secrets() -> None:
    text = (
        "contact a@b.test or +1 555 010 9999 from 10.1.2.3; "
        "Authorization: Bearer supersecret"
    )
    cleaned = redact_text(text)
    assert "a@b.test" not in cleaned
    assert "10.1.2.3" not in cleaned
    assert "supersecret" not in cleaned
    assert "[redacted]" in cleaned


def test_jumbled_preserves_distribution_and_dtype(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())

    variants, frames = build_variants(profile, frame, seed=5)
    jumbled = frames["jumbled"]

    assert jumbled["y"].dtype == frame["y"].dtype
    assert sorted(jumbled["y"].tolist()) == sorted(frame["y"].tolist())
    assert not jumbled["y"].equals(frame["y"])
    assert variants.jumbled.target_column == "y"
    assert "method=iid_permutation" in variants.jumbled.notes


def test_time_series_uses_block_shuffle(tmp_path: Path) -> None:
    rows = 80
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="D").astype(str),
            "y": np.arange(rows, dtype="float64"),
            "x": np.linspace(0.0, 1.0, rows),
        }
    )
    profile, frame = make_profile(tmp_path, frame)

    assert profile.time_column == "date"
    result = jumble_target(
        profile, frame, seed=7, block_size=10, return_details=True
    )

    assert result.method == "block_permutation"
    assert result.block_size == 10
    assert result.series.dtype == frame["y"].dtype
    assert sorted(result.series.tolist()) == sorted(frame["y"].tolist())
    assert not result.series.equals(frame["y"])

    original_blocks = [
        frame["y"].iloc[start : start + 10].tolist()
        for start in range(0, rows, 10)
    ]
    shuffled_blocks = [
        result.series.iloc[start : start + 10].tolist()
        for start in range(0, rows, 10)
    ]
    assert all(block in original_blocks for block in shuffled_blocks)
    assert shuffled_blocks != original_blocks


def test_grouped_shuffle_permutes_within_groups(tmp_path: Path) -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "group": ["a", "b"] * (rows // 2),
            "y": np.arange(rows, dtype="float64") * 3.0,
            "x": np.arange(rows, dtype="float64"),
        }
    )
    profile, frame = make_profile(tmp_path, frame)

    result = jumble_target(
        profile, frame, seed=9, group_column="group", return_details=True
    )

    assert result.method == "group_permutation"
    assert result.group_column == "group"
    for label in ("a", "b"):
        original = sorted(frame.loc[frame["group"] == label, "y"].tolist())
        shuffled = sorted(result.series.loc[frame["group"] == label].tolist())
        assert shuffled == original
    assert not result.series.equals(frame["y"])


def test_jumbled_label_leak_is_caught(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    variants, _ = build_variants(profile, frame, seed=1)

    report = evaluate_variants(
        variants,
        metric="r2",
        original=0.8,
        anonymized=0.78,
        jumbled=0.35,
        permutation_p=0.01,
    )

    assert report.jumbled_label_leak is True
    assert report.verdict == "reject"
    assert any("jumbled_label_leak" in reason for reason in report.reasons)


def test_clean_model_is_not_rejected(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    variants, _ = build_variants(profile, frame, seed=1)

    report = evaluate_variants(
        variants,
        metric="r2",
        original=0.8,
        anonymized=0.78,
        jumbled=0.0,
        permutation_p=0.9,
    )

    assert report.jumbled_label_leak is False
    assert report.pii_reliance is False
    assert report.verdict == "pass"


def test_pii_reliance_is_flagged(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    variants, _ = build_variants(profile, frame, seed=1)

    report = evaluate_variants(
        variants,
        metric="r2",
        original=0.8,
        anonymized=0.2,
        jumbled=0.0,
        permutation_p=0.9,
        pii_columns_used=["email"],
    )

    assert report.pii_reliance is True
    assert report.verdict == "reject"
    assert any("pii_reliance" in reason for reason in report.reasons)


def test_variant_report_is_json_serializable(tmp_path: Path) -> None:
    profile, frame = make_profile(tmp_path, pii_frame())
    variants, _ = build_variants(profile, frame, seed=2)
    report = evaluate_variants(
        variants,
        metric="r2",
        original=0.5,
        jumbled=0.0,
        permutation_p=0.5,
    )

    payload = json.loads(report.model_dump_json())
    assert payload["variants"]["original"]["kind"] == "original"
    assert payload["verdict"] == "pass"

    destination = tmp_path / "runs" / "run_1" / "variants.json"
    report.write(destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["verdict"] == "pass"
