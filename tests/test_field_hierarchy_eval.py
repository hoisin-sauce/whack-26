from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.field_hierarchy import (
    SYNTHETIC_DATASET,
    CreatedField,
    EvaluationDataset,
    FieldHierarchy,
    FieldNode,
    GateThresholds,
    PlacementPrediction,
    PredictionSet,
    PromotionEvent,
    SyntheticCase,
    SyntheticDocument,
    build_predictions,
    evaluate_predictions,
    main,
    run_evaluation,
)


def _document(document_id: str, *keywords: str) -> SyntheticDocument:
    return SyntheticDocument(
        id=document_id,
        title=document_id.replace("_", " "),
        summary="Synthetic test document.",
        keywords=tuple(keywords),
    )


def _math_dataset() -> EvaluationDataset:
    hierarchy = FieldHierarchy(
        (
            FieldNode("root", "Root"),
            FieldNode("a", "A", "root"),
            FieldNode("a1", "A1", "a"),
            FieldNode("b", "B", "root"),
        )
    )
    cases = tuple(
        SyntheticCase(_document(document_id), expected)
        for document_id, expected in (
            ("exact", "a1"),
            ("ancestor", "a1"),
            ("over", "a"),
            ("under", "a1"),
            ("wrong", "a1"),
            ("missing", "a1"),
        )
    )
    return EvaluationDataset(
        name="metric-math",
        hierarchy=hierarchy,
        cases=cases,
        accepted_branch_ids=frozenset({"new"}),
        expected_promotions=(
            PromotionEvent("new", "a", "root"),
        ),
    )


def _math_predictions() -> PredictionSet:
    return PredictionSet(
        placements={
            "exact": PlacementPrediction("exact", "a1", ("root", "a", "a1")),
            "ancestor": PlacementPrediction("ancestor", "a", ("root", "a")),
            "over": PlacementPrediction("over", "a1", ("root", "a", "a1")),
            "under": PlacementPrediction("under", "root", ("root",)),
            "wrong": PlacementPrediction("wrong", "b", ("root", "b")),
        },
        branches=(
            CreatedField("new", "a", "New"),
            CreatedField("bogus", "a", "Bogus"),
        ),
        promotions=(
            PromotionEvent("new", "a", "root"),
            PromotionEvent("bogus", "a", "root"),
        ),
    )


def test_metric_math_counts_each_relation_and_lifecycle_event() -> None:
    report = evaluate_predictions(
        _math_dataset(),
        _math_predictions(),
        strategy="custom",
        generated_at="2026-01-01T00:00:00Z",
    )
    metrics = report.metrics

    assert metrics.total == 6
    assert metrics.exact == 1
    assert metrics.ancestor == 2
    assert metrics.exact_or_ancestor == 3
    assert metrics.over_specific == 1
    assert metrics.under_specific == 2
    assert metrics.path_consistent == 5
    assert metrics.branches_predicted == 2
    assert metrics.branches_accepted == 1
    assert metrics.promotions_predicted == 2
    assert metrics.promotions_accepted == 1

    assert metrics.exact_rate == pytest.approx(1 / 6)
    assert metrics.ancestor_rate == pytest.approx(2 / 6)
    assert metrics.exact_or_ancestor_rate == pytest.approx(3 / 6)
    assert metrics.over_specificity_rate == pytest.approx(1 / 6)
    assert metrics.under_specificity_rate == pytest.approx(2 / 6)
    assert metrics.path_consistency_rate == pytest.approx(5 / 6)
    assert metrics.branch_precision == pytest.approx(1 / 2)
    assert metrics.promotion_precision == pytest.approx(1 / 2)

    assert [case.relation for case in report.cases] == [
        "exact",
        "ancestor",
        "descendant",
        "ancestor",
        "unrelated",
        "missing",
    ]


def test_path_consistency_rejects_a_path_that_does_not_match_the_tree() -> None:
    dataset = EvaluationDataset(
        name="path-math",
        hierarchy=FieldHierarchy(
            (
                FieldNode("root", "Root"),
                FieldNode("a", "A", "root"),
                FieldNode("a1", "A1", "a"),
                FieldNode("b", "B", "root"),
            )
        ),
        cases=(SyntheticCase(_document("bad_path"), "a1"),),
        accepted_branch_ids=frozenset(),
        expected_promotions=(),
    )
    predictions = PredictionSet(
        placements={
            "bad_path": PlacementPrediction(
                "bad_path",
                "a1",
                ("root", "b", "a1"),
            )
        }
    )

    report = evaluate_predictions(dataset, predictions, strategy="custom")

    assert report.metrics.exact == 1
    assert report.metrics.path_consistent == 0
    assert report.cases[0].path_consistent is False
    assert {gate.name for gate in report.gates if gate.status == "fail"} >= {
        "Path consistency"
    }


def test_perfect_run_is_fully_correct_and_passes_every_gate() -> None:
    report = run_evaluation(strategy="perfect", generated_at="2026-01-01T00:00:00Z")
    metrics = report.metrics

    assert 30 <= metrics.total <= 50
    assert metrics.exact == metrics.total
    assert metrics.ancestor == 0
    assert metrics.over_specific == 0
    assert metrics.under_specific == 0
    assert metrics.path_consistent == metrics.total
    assert metrics.branches_predicted >= 1
    assert metrics.branches_accepted == metrics.branches_predicted
    assert metrics.promotions_predicted >= 1
    assert metrics.promotions_accepted == metrics.promotions_predicted
    assert metrics.branch_precision == pytest.approx(1.0)
    assert metrics.promotion_precision == pytest.approx(1.0)
    assert report.status == "pass"
    assert {gate.status for gate in report.gates if gate.status != "skip"} == {
        "pass"
    }


def test_keyword_baseline_matches_the_labels_and_passes() -> None:
    report = run_evaluation(strategy="keyword", generated_at="2026-01-01T00:00:00Z")

    assert report.metrics.exact == report.metrics.total
    assert report.metrics.over_specific == 0
    assert report.metrics.path_consistency_rate == pytest.approx(1.0)
    assert report.metrics.branch_precision == pytest.approx(1.0)
    assert report.status == "pass"


def test_over_specific_run_fails_the_over_specificity_gate() -> None:
    report = run_evaluation(
        strategy="over-specific", generated_at="2026-01-01T00:00:00Z"
    )

    assert report.metrics.over_specific > 0
    assert report.metrics.over_specificity_rate > 0
    assert report.metrics.path_consistency_rate == pytest.approx(1.0)
    assert report.status == "fail"
    gate = {item.name: item for item in report.gates}["Over-specific attachment"]
    assert gate.status == "fail"
    assert gate.blocking is True


def test_under_specific_run_is_counted_and_flagged_as_a_diagnostic() -> None:
    report = run_evaluation(
        strategy="under-specific", generated_at="2026-01-01T00:00:00Z"
    )

    assert report.metrics.under_specific > 0
    assert report.metrics.exact < report.metrics.total
    assert report.metrics.over_specific == 0
    assert report.metrics.exact_or_ancestor_rate == pytest.approx(1.0)
    assert report.metrics.path_consistency_rate == pytest.approx(1.0)
    # The spec's blocking gate accepts ancestors, so the run passes overall...
    assert report.status == "pass"
    # ...but the diagnostic makes the broader-than-labelled placements visible.
    diagnostic = {
        item.name: item for item in report.gates
    }["Under-specificity"]
    assert diagnostic.status == "fail"
    assert diagnostic.blocking is False


def test_under_specificity_can_be_made_blocking() -> None:
    report = run_evaluation(
        strategy="under-specific",
        thresholds=GateThresholds(
            max_under_specificity=0.20,
            under_specificity_blocking=True,
        ),
        generated_at="2026-01-01T00:00:00Z",
    )

    assert report.metrics.under_specificity_rate > 0.20
    assert report.status == "fail"


def test_bad_branch_and_promotion_predictions_fail_precision_gates() -> None:
    report = evaluate_predictions(
        _math_dataset(),
        _math_predictions(),
        strategy="custom",
        generated_at="2026-01-01T00:00:00Z",
    )
    gates = {item.name: item for item in report.gates}

    assert gates["Branch precision"].status == "fail"
    assert gates["Promotion precision"].status == "fail"
    assert report.status == "fail"


def test_default_strategy_builds_a_complete_prediction_set() -> None:
    predictions = build_predictions(SYNTHETIC_DATASET, "keyword")

    assert set(predictions.placements) == {
        case.document.id for case in SYNTHETIC_DATASET.cases
    }
    for prediction in predictions.placements.values():
        assert prediction.path_ids
        assert prediction.path_ids[-1] == prediction.leaf_field_id


def test_main_writes_both_reports_and_returns_the_gate_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report.md"
    json_output = tmp_path / "report.json"

    exit_code = main(
        [
            "--strategy",
            "perfect",
            "--output",
            str(output),
            "--json-output",
            str(json_output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    assert json_output.is_file()
    markdown = output.read_text(encoding="utf-8")
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert "# Field hierarchy evaluation" in markdown
    assert "Overall status: **PASS**" in markdown
    assert payload["status"] == "pass"
    assert payload["strategy"] == "perfect"
    assert payload["metrics"]["exact_rate"] == pytest.approx(1.0)
    captured = capsys.readouterr()
    assert "# Field hierarchy evaluation" in captured.out

    exit_code = main(
        [
            "--strategy",
            "over-specific",
            "--output",
            str(output),
            "--json-output",
            str(json_output),
        ]
    )

    assert exit_code == 1
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["metrics"]["over_specific"] > 0
    captured = capsys.readouterr()
    assert "gate failed" in captured.err


def test_main_can_print_json_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--strategy",
            "perfect",
            "--format",
            "json",
            "--output",
            str(tmp_path / "report.md"),
            "--json-output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "pass"
    assert payload["metrics"]["total"] == len(SYNTHETIC_DATASET.cases)
