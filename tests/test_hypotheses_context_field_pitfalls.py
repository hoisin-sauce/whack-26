"""Field-pitfall section of the hypothesis context pack."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from whack26.hypotheses.context import build_context_pack, render_context_pack
from whack26.io import read_dataset
from whack26.pitfalls.types import FieldPitfall
from whack26.profile import build_profile


def make_profile(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "y": [2.0, 4.5, 5.5, 8.5, 9.0, 12.5, 13.0, 16.5],
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    loaded = read_dataset(path, target="y")
    return build_profile(loaded, target="y")


def make_pitfall(**overrides):
    values = {
        "id": "F:fld_leaf:1",
        "field_id": "fld_leaf",
        "field_name": "Leaf",
        "field_level": 1,
        "scope": "leaf",
        "title": "A field caveat",
        "pitfall": "Values may be censored at the detection limit.",
        "category": "censoring",
        "applies_to_dataset": "unknown",
        "evidence_id": "ev_1",
    }
    values.update(overrides)
    return FieldPitfall(**values)


def test_field_pitfall_section_renders_bounded_record(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    pitfall = make_pitfall(
        applies_to_dataset="yes",
        applicability_reason="The target has a spike at the boundary.",
        check="Plot the target histogram near the boundary.",
        source="openalex",
        url="https://example.org/paper",
        confidence=0.9,
    )

    pack = build_context_pack(
        profile=profile,
        field_path=["Leaf"],
        field_pitfalls=[pitfall],
    )
    rendered = render_context_pack(pack)

    assert "## Field pitfalls" in rendered
    assert rendered.index("## Field pitfalls") < rendered.index("## Factors")
    assert "[F:fld_leaf:1]" in rendered
    assert "Leaf" in rendered
    assert "scope=leaf" in rendered
    assert "category=censoring" in rendered
    assert "applies=yes" in rendered
    assert "Values may be censored at the detection limit." in rendered
    assert "Plot the target histogram near the boundary." in rendered
    assert "evidence: ev_1" in rendered
    assert "https://example.org/paper" in rendered

    record = pack.field_pitfalls[0]
    assert record["field_id"] == "fld_leaf"
    assert record["applies_to_dataset"] == "yes"
    assert record["evidence_id"] == "ev_1"


def test_leaf_pitfalls_sort_before_ancestors(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    pitfalls = [
        make_pitfall(
            id="F:root:1",
            field_name="Root",
            field_level=2,
            scope="ancestor",
            confidence=0.95,
        ),
        make_pitfall(
            id="F:leaf_a:1",
            field_name="Leaf A",
            field_level=1,
            scope="leaf",
            confidence=0.10,
        ),
        make_pitfall(
            id="F:leaf_b:1",
            field_name="Leaf B",
            field_level=3,
            scope="leaf",
            confidence=0.80,
        ),
    ]

    pack = build_context_pack(
        profile=profile,
        field_path=["Root", "Leaf A", "Leaf B"],
        field_pitfalls=pitfalls,
    )

    assert [record["id"] for record in pack.field_pitfalls] == [
        "F:leaf_b:1",
        "F:leaf_a:1",
        "F:root:1",
    ]
    rendered = render_context_pack(pack)
    assert rendered.index("F:leaf_b:1") < rendered.index("F:root:1")


def test_field_pitfalls_are_capped_at_eight(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    pitfalls = [
        make_pitfall(
            id=f"F:fld_{index:02d}:1",
            field_id=f"fld_{index:02d}",
            field_name=f"Field {index}",
        )
        for index in range(12)
    ]

    pack = build_context_pack(
        profile=profile,
        field_path=["Field"],
        field_pitfalls=pitfalls,
    )

    assert len(pack.field_pitfalls) == 8
    assert [record["id"] for record in pack.field_pitfalls] == [
        f"F:fld_{index:02d}:1" for index in range(8)
    ]


def test_field_pitfalls_accept_mappings_and_objects(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    mapping = {
        "id": "F:fld_map:1",
        "field_id": "fld_map",
        "field_name": "Mapped",
        "field_level": 2,
        "scope": "ancestor",
        "category": "selection",
        "pitfall": "Sampling only reaches surviving units.",
        "applies_to_dataset": "unknown",
        "check": "Compare included and excluded units.",
        "evidence_id": "ev_map",
        "confidence": 0.4,
    }
    namespace = SimpleNamespace(
        id="F:fld_obj:1",
        field_id="fld_obj",
        field_name="Object",
        field_level=1,
        scope="leaf",
        category="missing_data",
        pitfall="Missing values may not be missing at random.",
        applies_to_dataset="yes",
        applicability_reason="The profile shows missing values.",
        check="Compare missingness by group.",
        evidence_id="ev_obj",
        source="arxiv",
        url="https://example.org/object",
        confidence=0.7,
    )

    pack = build_context_pack(
        profile=profile,
        field_path=["Object", "Mapped"],
        field_pitfalls=[mapping, namespace],
    )

    assert [record["id"] for record in pack.field_pitfalls] == [
        "F:fld_obj:1",
        "F:fld_map:1",
    ]
    assert pack.field_pitfalls[0]["field_name"] == "Object"
    assert pack.field_pitfalls[0]["url"] == "https://example.org/object"
    assert pack.field_pitfalls[1]["category"] == "selection"
    assert pack.field_pitfalls[1]["check"] == "Compare included and excluded units."


def test_field_pitfall_text_is_redacted(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    pitfall = make_pitfall(
        pitfall=(
            "Contact analyst@example.com before using this measure; "
            "the export used api_key=abc123."
        ),
        applicability_reason="Ask data.owner@example.com to confirm.",
        check="Verify with reviewer@example.com.",
    )

    pack = build_context_pack(
        profile=profile,
        field_path=["Leaf"],
        field_pitfalls=[pitfall],
    )

    payload = pack.model_dump_json()
    rendered = render_context_pack(pack)
    assert "analyst@example.com" not in payload
    assert "data.owner@example.com" not in payload
    assert "reviewer@example.com" not in payload
    assert "abc123" not in payload
    assert "[redacted]" in rendered


def test_field_pitfalls_still_respect_max_chars(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)
    pitfall = make_pitfall(
        pitfall="boundary concentration " * 200,
        applicability_reason="reason " * 200,
        check="check the boundary " * 200,
    )

    pack = build_context_pack(
        profile=profile,
        field_path=["Leaf"],
        field_pitfalls=[pitfall] * 8,
        max_chars=800,
    )

    rendered = render_context_pack(pack, max_chars=800)
    assert len(rendered) <= 800
    assert pack.token_estimate <= 200


def test_no_field_pitfalls_renders_no_section(tmp_path: Path) -> None:
    profile = make_profile(tmp_path)

    pack = build_context_pack(profile=profile, field_path=["Leaf"])

    assert pack.field_pitfalls == []
    assert "## Field pitfalls" not in render_context_pack(pack)
