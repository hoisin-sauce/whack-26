from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from eval.field_classification import (
    DATASETS,
    HUMAN_MEAN_THRESHOLD,
    HUMAN_MIN_ACCEPTABLE_SCORE,
    HUMAN_NAMING_MEAN_THRESHOLD,
    HUMAN_PASS_CRITERIA,
    EvalDataset,
    DatasetRun,
    HumanScore,
    RoundResult,
    _duplicate_roots,
    _parse_datasets,
    evaluate_automated,
    evaluate_human,
    evaluate_naming,
    load_scores,
    overall_status,
    write_scores_template,
)

SLUGS = [item.slug for item in DATASETS]


def make_record(
    slug: str,
    *,
    root: str = "Science",
    path: list[str] | None = None,
    error: str | None = None,
    elapsed_s: float = 1.0,
) -> DatasetRun:
    names = path or [root, "Subfield"]
    return DatasetRun(
        slug=slug,
        path_names=names,
        path_ids=[f"fld_{slug}_{index}" for index in range(len(names))],
        confidence=0.9,
        created_fields=[],
        warnings=[],
        elapsed_s=elapsed_s,
        error=error,
    )


def make_round(
    name: str,
    *,
    roots: dict[str, str] | None = None,
    paths: dict[str, list[str]] | None = None,
    elapsed_s: float = 1.0,
    orphans: int = 0,
    error_slug: str | None = None,
) -> RoundResult:
    records = []
    unique_roots: dict[str, list[float]] = {}
    for index, slug in enumerate(SLUGS):
        root = (roots or {}).get(slug, f"Domain {index}")
        path = (paths or {}).get(slug, [root, f"Subfield {index}"])
        angle = 2.0 * math.pi * index / len(SLUGS)
        embedding = [math.cos(angle), math.sin(angle)]
        unique_roots.setdefault(root, embedding)
        records.append(
            make_record(
                slug,
                root=root,
                path=path,
                error=f"RuntimeError: boom" if slug == error_slug else None,
            )
        )
    return RoundResult(
        name=name,
        elapsed_s=elapsed_s,
        records=records,
        stats={"orphan_fields": orphans},
        roots=list(unique_roots.items()),
    )


def make_entry(
    slug: str = "custom", *, forbidden: tuple[str, ...] = ()
) -> EvalDataset:
    return EvalDataset(
        slug=slug,
        name="Custom Dataset",
        domain="test",
        url="",
        filename="custom.csv",
        description=None,
        target=None,
        acceptable_roots=("science",),
        forbidden_labels=forbidden,
        good_path="Science > Subfield",
    )


def make_naming_round(slug: str, path: list[str]) -> RoundResult:
    return RoundResult(
        name="Cold",
        elapsed_s=1.0,
        records=[make_record(slug, path=path)],
        stats={},
        roots=[],
    )


def test_all_automated_gates_pass() -> None:
    cold = make_round("Cold")
    warm = make_round("Warm")
    shuffled = make_round("Shuffled cold")
    repeat = make_round("Repeat cold")
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=repeat,
        expected=len(SLUGS),
    )
    assert {gate.status for gate in gates} == {"pass"}
    assert overall_status(gates) == "pass"


def test_exact_and_near_duplicate_roots_are_counted() -> None:
    roots = [
        ("Science", [1.0, 0.0]),
        ("science", [1.0, 0.0]),
    ]
    exact, near = _duplicate_roots(roots)
    assert exact == 1
    assert near == 1


def test_duplicate_root_gates_fail() -> None:
    cold = make_round("Cold", roots={slug: "Science" for slug in SLUGS})
    warm = make_round("Warm", roots={slug: "Science" for slug in SLUGS})
    shuffled = make_round("Shuffled cold", roots={slug: "Science" for slug in SLUGS})
    cold.roots = [
        ("Science", [1.0, 0.0]),
        ("Science", [1.0, 0.0]),
        ("Science", [1.0, 0.0]),
    ]
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Duplicate roots (exact)"].status == "fail"
    assert by_name["Duplicate roots (cosine)"].status == "fail"


def test_one_near_duplicate_root_pair_fails() -> None:
    cold = make_round("Cold")
    warm = make_round("Warm")
    shuffled = make_round("Shuffled cold")
    angle = math.acos(0.95)
    cold.roots = [
        ("Science", [1.0, 0.0]),
        ("Life Sciences", [math.cos(angle), math.sin(angle)]),
    ]
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    gate = {item.name: item for item in gates}["Duplicate roots (cosine)"]
    assert gate.status == "fail"
    assert "allowed 0" in gate.detail


def test_max_depth_and_runtime_gates_fail() -> None:
    cold = make_round("Cold", elapsed_s=601.0)
    cold.records[0].path_names = ["A", "B", "C", "D", "E"]
    warm = make_round("Warm")
    shuffled = make_round("Shuffled cold")
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Max depth"].status == "fail"
    assert by_name["Runtime (cold)"].status == "fail"


def test_warm_stability_and_robustness_fail() -> None:
    cold = make_round("Cold")
    warm = make_round("Warm", paths={slug: ["Different", "Path"] for slug in SLUGS})
    shuffled = make_round("Shuffled cold")
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Warm stability (root)"].status == "fail"
    assert by_name["Warm stability (path)"].status == "fail"

    broken = make_round("Cold", error_slug=SLUGS[0])
    gates = evaluate_automated(
        cold=broken,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    assert {gate.name: gate for gate in gates}["Robustness (crashes)"].status == "fail"


def test_order_independence_is_diagnostic_only() -> None:
    cold = make_round("Cold")
    warm = make_round("Warm")
    shuffled = make_round("Shuffled cold", roots={slug: "Other" for slug in SLUGS})
    gates = evaluate_automated(
        cold=cold,
        warm=warm,
        shuffled=shuffled,
        repeat=None,
        expected=len(SLUGS),
    )
    diagnostic = {gate.name: gate for gate in gates}["Order independence"]
    assert diagnostic.status == "fail"
    assert diagnostic.blocking is False
    assert overall_status(gates) == "pass"


def test_human_gates_pending_until_every_dataset_is_scored() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=4, path=4, naming=4, sibling_distinctness=True)
        for slug in SLUGS[:-1]
    }
    gates = evaluate_human(cold=cold, scores=scores, expected=len(SLUGS))
    assert all(gate.status == "pending" for gate in gates)
    assert overall_status(gates) == "pending"


def test_human_gates_pass_with_good_scores() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=4, path=4, naming=4, sibling_distinctness=True)
        for slug in SLUGS
    }
    gates = evaluate_human(cold=cold, scores=scores, expected=len(SLUGS))
    assert {gate.status for gate in gates} == {"pass"}


def test_human_gates_fail_on_low_scores() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=3, path=2, naming=3, sibling_distinctness=False)
        for slug in SLUGS
    }
    gates = evaluate_human(cold=cold, scores=scores, expected=len(SLUGS))
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Full-path plausibility"].status == "fail"
    assert by_name["Naming (human)"].status == "fail"
    assert by_name["Sibling distinctness"].status == "fail"


def test_human_scores_at_rubric_minimum_use_the_documented_stricter_criteria() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(
            root=HUMAN_MIN_ACCEPTABLE_SCORE,
            path=HUMAN_MIN_ACCEPTABLE_SCORE,
            naming=int(HUMAN_NAMING_MEAN_THRESHOLD),
            sibling_distinctness=True,
        )
        for slug in SLUGS
    }
    gates = evaluate_human(cold=cold, scores=scores, expected=len(SLUGS))
    by_name = {gate.name: gate for gate in gates}
    # The documented criteria require means of 3.5, not merely every score >= 3.
    assert by_name["Top-level plausibility"].status == "fail"
    assert by_name["Full-path plausibility"].status == "fail"
    assert by_name["Naming (human)"].status == "pass"
    assert by_name["Sibling distinctness"].status == "pass"
    assert f"{HUMAN_MEAN_THRESHOLD:.2f}" in by_name["Top-level plausibility"].detail


def test_human_naming_can_be_assumed_from_the_automated_gate() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=4, path=4, naming=None, sibling_distinctness=True)
        for slug in SLUGS
    }
    gates = evaluate_human(
        cold=cold,
        scores=scores,
        expected=len(SLUGS),
        assume_automated_naming=True,
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Naming (human)"].status == "pass"
    assert "assumed from automated naming" in by_name["Naming (human)"].detail


def test_human_naming_missing_stays_pending_without_the_assumption() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=4, path=4, naming=None, sibling_distinctness=True)
        for slug in SLUGS
    }
    gates = evaluate_human(
        cold=cold,
        scores=scores,
        expected=len(SLUGS),
        assume_automated_naming=False,
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Naming (human)"].status == "pending"


def test_explicit_human_naming_can_still_fail_when_auto_is_trusted() -> None:
    cold = make_round("Cold")
    scores = {
        slug: HumanScore(root=4, path=4, naming=2, sibling_distinctness=True)
        for slug in SLUGS
    }
    gates = evaluate_human(
        cold=cold,
        scores=scores,
        expected=len(SLUGS),
        assume_automated_naming=True,
    )
    by_name = {gate.name: gate for gate in gates}
    assert by_name["Naming (human)"].status == "fail"


def test_naming_gate_allows_the_classifier_fallback_root() -> None:
    entry = make_entry("fallback")
    cold = make_naming_round("fallback", ["General Research", "General Topics"])
    gates = evaluate_naming(cold=cold, entries=[entry])
    assert gates[0].status == "pass"


def test_naming_gate_rejects_a_genuinely_banned_root() -> None:
    entry = make_entry("banned")
    cold = make_naming_round("banned", ["Miscellaneous", "Subfield"])
    gates = evaluate_naming(cold=cold, entries=[entry])
    assert gates[0].status == "fail"
    assert "banned root" in gates[0].detail


def test_naming_gate_rejects_forbidden_labels_as_whole_tokens() -> None:
    entry = make_entry("labels", forbidden=("palmer penguins", "uci"))
    cold = make_naming_round("labels", ["UCI Air Quality", "Palmer Penguins Data"])
    gates = evaluate_naming(cold=cold, entries=[entry])
    assert gates[0].status == "fail"
    assert "Palmer Penguins Data" in gates[0].detail


def test_naming_gate_does_not_flag_incidental_substrings() -> None:
    entry = make_entry("substring", forbidden=("uci",))
    cold = make_naming_round("substring", ["Biology", "Mucilage and Biomass"])
    gates = evaluate_naming(cold=cold, entries=[entry])
    assert gates[0].status == "pass"


def test_score_template_documents_the_same_criteria_as_the_gates(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "scores.json"
    write_scores_template(destination, DATASETS)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    instructions = payload["_instructions"]
    assert HUMAN_PASS_CRITERIA in instructions
    assert f"{HUMAN_MEAN_THRESHOLD:.1f}" in instructions
    assert f"{HUMAN_NAMING_MEAN_THRESHOLD:.1f}" in instructions
    assert "both at least 3" not in instructions


def test_score_template_round_trips_as_unscored(tmp_path: Path) -> None:
    destination = tmp_path / "scores.json"
    write_scores_template(destination, DATASETS)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["_rubric"]["5"]
    assert payload["penguins"]["root"] is None
    assert load_scores(destination, DATASETS) == {}


def test_load_scores_allows_missing_naming(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps(
            {
                "penguins": {
                    "root": 4,
                    "path": 4,
                    "naming": None,
                    "sibling_distinctness": True,
                }
            }
        ),
        encoding="utf-8",
    )
    scores = load_scores(path, DATASETS)
    assert scores["penguins"].naming is None


def test_load_scores_rejects_missing_required_values_or_invalid_scores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps(
            {
                "penguins": {
                    "root": None,
                    "path": 4,
                    "naming": 4,
                    "sibling_distinctness": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provide root, path, and sibling_distinctness"):
        load_scores(path, DATASETS)

    path.write_text(
        json.dumps(
            {
                "penguins": {
                    "root": 6,
                    "path": 4,
                    "naming": 4,
                    "sibling_distinctness": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="1 to 5"):
        load_scores(path, DATASETS)

    path.write_text(
        json.dumps(
            {
                "penguins": {
                    "root": 4,
                    "path": 4,
                    "naming": 6,
                    "sibling_distinctness": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="naming must be"):
        load_scores(path, DATASETS)


def test_parse_datasets_validates_slugs() -> None:
    assert _parse_datasets(None) == list(DATASETS)
    assert [entry.slug for entry in _parse_datasets("penguins,california")] == [
        "penguins",
        "california",
    ]
    with pytest.raises(ValueError, match="unknown dataset"):
        _parse_datasets("penguins,nope")


def test_overall_status_prefers_failure_over_pending() -> None:
    from eval.field_classification import GateResult

    assert overall_status([GateResult("a", "pass", "")]) == "pass"
    assert (
        overall_status(
            [
                GateResult("a", "pending", ""),
                GateResult("b", "pass", ""),
            ]
        )
        == "pending"
    )
    assert (
        overall_status(
            [
                GateResult("a", "pending", ""),
                GateResult("b", "fail", ""),
            ]
        )
        == "fail"
    )
