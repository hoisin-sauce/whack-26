"""Field-hierarchy placement evaluation gate.

This harness is deliberately independent of ``whack26.hierarchy.place`` and the
other H1-H3 modules so it can run before those implementations land. It ships
with a small, hand-labelled synthetic hierarchy, a deterministic keyword
placer, and a perfect/over-specific/under-specific strategy used to prove that
the metrics and gate behave as intended.

The evaluation is offline: it never calls a network service or a live LLM.

Usage:
    python eval/field_hierarchy.py
    python eval/field_hierarchy.py --strategy perfect --format json
    python eval/field_hierarchy.py --strategy over-specific
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "eval" / "field-hierarchy-results.md"

Relation = Literal["exact", "ancestor", "descendant", "unrelated", "missing"]
GateStatus = Literal["pass", "fail", "skip"]
StrategyName = Literal["keyword", "perfect", "over_specific", "under_specific"]


# ---------------------------------------------------------------------------
# Hierarchy model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldNode:
    """One synthetic field."""

    id: str
    name: str
    parent_id: str | None = None
    keywords: tuple[str, ...] = ()


class FieldHierarchy:
    """A small validated field tree."""

    def __init__(self, nodes: Iterable[FieldNode]) -> None:
        self._nodes: dict[str, FieldNode] = {}
        for node in nodes:
            if node.id in self._nodes:
                raise ValueError(f"duplicate field id: {node.id}")
            self._nodes[node.id] = node
        if not self._nodes:
            raise ValueError("a hierarchy needs at least one field")

        for node in self._nodes.values():
            if node.parent_id is not None and node.parent_id not in self._nodes:
                raise ValueError(
                    f"field {node.id!r} has unknown parent {node.parent_id!r}"
                )
        # Force every path to be walked so cycles and disconnected nodes fail
        # immediately instead of during a metric calculation.
        for field_id in self._nodes:
            self.path(field_id)

        sibling_names: dict[tuple[str | None, str], str] = {}
        for node in self._nodes.values():
            key = (node.parent_id, _normalize_text(node.name))
            if key in sibling_names:
                raise ValueError(
                    f"fields {sibling_names[key]!r} and {node.id!r} share a sibling name"
                )
            sibling_names[key] = node.id

    @property
    def nodes(self) -> tuple[FieldNode, ...]:
        return tuple(self._nodes.values())

    def __contains__(self, field_id: object) -> bool:
        return field_id in self._nodes

    def get(self, field_id: str) -> FieldNode:
        return self._nodes[field_id]

    def roots(self) -> tuple[str, ...]:
        return tuple(
            sorted(node.id for node in self._nodes.values() if node.parent_id is None)
        )

    def children(self, field_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                node.id
                for node in self._nodes.values()
                if node.parent_id == field_id
            )
        )

    def level(self, field_id: str) -> int:
        return len(self.path(field_id)) - 1

    def path(self, field_id: str) -> tuple[str, ...]:
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = field_id
        while current is not None:
            if current in seen:
                raise ValueError(f"cycle in hierarchy at field {current!r}")
            if current not in self._nodes:
                raise KeyError(f"unknown field id: {current}")
            seen.add(current)
            chain.append(current)
            current = self._nodes[current].parent_id
        return tuple(reversed(chain))

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        if ancestor_id == descendant_id:
            return False
        try:
            return ancestor_id in self.path(descendant_id)[:-1]
        except KeyError:
            return False

    def is_descendant(self, descendant_id: str, ancestor_id: str) -> bool:
        return self.is_ancestor(ancestor_id, descendant_id)


@dataclass(frozen=True)
class CreatedField:
    """A field the system under test proposes to create."""

    field_id: str
    parent_id: str
    name: str


@dataclass(frozen=True)
class PromotionEvent:
    """A proposed promote/swap: ``old_parent`` moves under ``new_field``."""

    new_field_id: str
    old_parent_id: str
    new_parent_id: str | None


# ---------------------------------------------------------------------------
# Synthetic labelled data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticDocument:
    id: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    kind: str = "paper"


@dataclass(frozen=True)
class SyntheticCase:
    document: SyntheticDocument
    expected_leaf_id: str
    expected_branch: CreatedField | None = None


@dataclass(frozen=True)
class BranchRule:
    """Deterministic keyword rule used by the offline ``keyword`` placer."""

    keywords: tuple[str, ...]
    branch: CreatedField


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    hierarchy: FieldHierarchy
    cases: tuple[SyntheticCase, ...]
    accepted_branch_ids: frozenset[str]
    expected_promotions: tuple[PromotionEvent, ...]
    branch_rules: tuple[BranchRule, ...] = ()

    def __post_init__(self) -> None:
        seen_documents: set[str] = set()
        for case in self.cases:
            document_id = case.document.id
            if document_id in seen_documents:
                raise ValueError(f"duplicate document id: {document_id}")
            seen_documents.add(document_id)

            if case.expected_branch is not None:
                if case.expected_leaf_id != case.expected_branch.field_id:
                    raise ValueError(
                        f"{document_id}: expected leaf must equal the created field id"
                    )
                if case.expected_branch.parent_id not in self.hierarchy:
                    raise ValueError(
                        f"{document_id}: branch parent "
                        f"{case.expected_branch.parent_id!r} does not exist"
                    )
            elif case.expected_leaf_id not in self.hierarchy:
                raise ValueError(
                    f"{document_id}: unknown expected leaf {case.expected_leaf_id!r}"
                )

    def expected_path(self, case: SyntheticCase) -> tuple[str, ...]:
        if case.expected_branch is not None:
            return self.hierarchy.path(case.expected_branch.parent_id) + (
                case.expected_branch.field_id,
            )
        return self.hierarchy.path(case.expected_leaf_id)


@dataclass(frozen=True)
class PlacementPrediction:
    document_id: str
    leaf_field_id: str
    path_ids: tuple[str, ...] = ()
    created_fields: tuple[CreatedField, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class PredictionSet:
    placements: Mapping[str, PlacementPrediction]
    branches: tuple[CreatedField, ...] = ()
    promotions: tuple[PromotionEvent, ...] = ()


# ---------------------------------------------------------------------------
# Metrics and gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    document_id: str
    expected_leaf_id: str
    predicted_leaf_id: str | None
    relation: Relation
    exact: bool
    ancestor: bool
    over_specific: bool
    under_specific: bool
    path_consistent: bool
    path_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalMetrics:
    total: int
    exact: int
    ancestor: int
    exact_or_ancestor: int
    over_specific: int
    under_specific: int
    path_consistent: int
    branches_predicted: int
    branches_accepted: int
    promotions_predicted: int
    promotions_accepted: int

    @staticmethod
    def _rate(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
        if denominator == 0:
            return empty
        return numerator / denominator

    @property
    def exact_rate(self) -> float:
        return self._rate(self.exact, self.total)

    @property
    def ancestor_rate(self) -> float:
        return self._rate(self.ancestor, self.total)

    @property
    def exact_or_ancestor_rate(self) -> float:
        return self._rate(self.exact_or_ancestor, self.total)

    @property
    def over_specificity_rate(self) -> float:
        return self._rate(self.over_specific, self.total)

    @property
    def under_specificity_rate(self) -> float:
        return self._rate(self.under_specific, self.total)

    @property
    def path_consistency_rate(self) -> float:
        return self._rate(self.path_consistent, self.total)

    @property
    def branch_precision(self) -> float:
        return self._rate(self.branches_accepted, self.branches_predicted, empty=1.0)

    @property
    def promotion_precision(self) -> float:
        return self._rate(
            self.promotions_accepted, self.promotions_predicted, empty=1.0
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "exact": self.exact,
            "exact_rate": self.exact_rate,
            "ancestor": self.ancestor,
            "ancestor_rate": self.ancestor_rate,
            "exact_or_ancestor": self.exact_or_ancestor,
            "exact_or_ancestor_rate": self.exact_or_ancestor_rate,
            "over_specific": self.over_specific,
            "over_specificity_rate": self.over_specificity_rate,
            "under_specific": self.under_specific,
            "under_specificity_rate": self.under_specificity_rate,
            "path_consistent": self.path_consistent,
            "path_consistency_rate": self.path_consistency_rate,
            "branches_predicted": self.branches_predicted,
            "branches_accepted": self.branches_accepted,
            "branch_precision": self.branch_precision,
            "promotions_predicted": self.promotions_predicted,
            "promotions_accepted": self.promotions_accepted,
            "promotion_precision": self.promotion_precision,
        }


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    detail: str
    blocking: bool = True


@dataclass(frozen=True)
class GateThresholds:
    """Gate configuration.

    The two blocking gates from the spec are ``max_over_specific_rate`` and
    ``min_exact_or_ancestor``. Path consistency is an invariant and is also
    blocking. Branch/promotion precision are blocking when the run actually
    proposes lifecycle changes. Under-specificity is diagnostic by default
    because the spec's exact/ancestor gate intentionally accepts broader
    attachments; set ``under_specificity_blocking`` to make it a hard gate.
    """

    min_exact_or_ancestor: float = 0.80
    max_over_specific_rate: float = 0.0
    min_path_consistency: float = 1.0
    min_branch_precision: float = 0.80
    min_promotion_precision: float = 0.80
    max_under_specificity: float = 0.50
    under_specificity_blocking: bool = False


@dataclass(frozen=True)
class EvalReport:
    dataset_name: str
    strategy: str
    generated_at: str
    metrics: EvalMetrics
    gates: tuple[GateResult, ...]
    cases: tuple[CaseResult, ...]
    predicted_branches: tuple[CreatedField, ...]
    accepted_branches: tuple[CreatedField, ...]
    predicted_promotions: tuple[PromotionEvent, ...]
    expected_promotions: tuple[PromotionEvent, ...]
    warnings: tuple[str, ...]

    @property
    def status(self) -> Literal["pass", "fail"]:
        return overall_status(self.gates)


def overall_status(gates: Sequence[GateResult]) -> Literal["pass", "fail"]:
    if any(gate.blocking and gate.status == "fail" for gate in gates):
        return "fail"
    return "pass"


def _normalize_text(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value.lower()).split()
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    text_tokens = _normalize_text(text).split()
    phrase_tokens = _normalize_text(phrase).split()
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def _extended_hierarchy(
    hierarchy: FieldHierarchy, created_fields: Iterable[CreatedField]
) -> FieldHierarchy:
    nodes = list(hierarchy.nodes)
    existing = {node.id for node in nodes}
    for created in created_fields:
        # A prediction that reuses an existing id or names an unknown parent is
        # malformed. Keep evaluating the remaining cases instead of crashing;
        # path consistency will flag the malformed prediction.
        if created.field_id in existing:
            continue
        if created.parent_id not in existing:
            continue
        nodes.append(
            FieldNode(
                id=created.field_id,
                name=created.name,
                parent_id=created.parent_id,
            )
        )
        existing.add(created.field_id)
    return FieldHierarchy(nodes)


def _relation(
    hierarchy: FieldHierarchy, predicted: str, expected: str
) -> Relation:
    if predicted == expected:
        return "exact"
    if predicted not in hierarchy or expected not in hierarchy:
        return "unrelated"
    if hierarchy.is_ancestor(predicted, expected):
        return "ancestor"
    if hierarchy.is_ancestor(expected, predicted):
        return "descendant"
    return "unrelated"


def _path_is_consistent(
    hierarchy: FieldHierarchy, prediction: PlacementPrediction
) -> bool:
    path = prediction.path_ids
    if not path or path[-1] != prediction.leaf_field_id:
        return False
    if len(set(path)) != len(path):
        return False
    if path[0] not in hierarchy:
        return False
    if hierarchy.get(path[0]).parent_id is not None:
        return False
    for parent_id, child_id in zip(path, path[1:]):
        if child_id not in hierarchy:
            return False
        if hierarchy.get(child_id).parent_id != parent_id:
            return False
    return True


def _dedupe_branches(branches: Iterable[CreatedField]) -> tuple[CreatedField, ...]:
    seen: set[str] = set()
    unique: list[CreatedField] = []
    for branch in branches:
        if branch.field_id in seen:
            continue
        seen.add(branch.field_id)
        unique.append(branch)
    return tuple(unique)


def _dedupe_promotions(
    promotions: Iterable[PromotionEvent],
) -> tuple[PromotionEvent, ...]:
    seen: set[PromotionEvent] = set()
    unique: list[PromotionEvent] = []
    for promotion in promotions:
        if promotion in seen:
            continue
        seen.add(promotion)
        unique.append(promotion)
    return tuple(unique)


def _ratio_gate(
    name: str,
    matched: int,
    total: int,
    *,
    minimum: float,
    blocking: bool = True,
    description: str,
) -> GateResult:
    if total == 0:
        return GateResult(
            name=name,
            status="fail" if blocking else "skip",
            detail=f"{description}: no cases to evaluate",
            blocking=blocking,
        )
    rate = matched / total
    return GateResult(
        name=name,
        status="pass" if rate >= minimum else "fail",
        detail=(
            f"{description}: {matched}/{total} ({rate:.1%}); "
            f"required {minimum:.0%}"
        ),
        blocking=blocking,
    )


def _precision_gate(
    name: str,
    accepted: int,
    predicted: int,
    *,
    minimum: float,
    description: str,
) -> GateResult:
    if predicted == 0:
        return GateResult(
            name=name,
            status="skip",
            detail=f"{description}: no predictions",
            blocking=False,
        )
    precision = accepted / predicted
    return GateResult(
        name=name,
        status="pass" if precision >= minimum else "fail",
        detail=(
            f"{description}: {accepted}/{predicted} ({precision:.1%}); "
            f"required {minimum:.0%}"
        ),
        blocking=True,
    )


def _build_gates(
    metrics: EvalMetrics, thresholds: GateThresholds
) -> tuple[GateResult, ...]:
    return (
        GateResult(
            name="Over-specific attachment",
            status=(
                "pass"
                if metrics.total > 0
                and metrics.over_specificity_rate <= thresholds.max_over_specific_rate
                else "fail"
            ),
            detail=(
                f"{metrics.over_specific}/{metrics.total} "
                f"({metrics.over_specificity_rate:.1%}); allowed "
                f"{thresholds.max_over_specific_rate:.0%}"
                if metrics.total
                else "no cases to evaluate"
            ),
            blocking=True,
        ),
        _ratio_gate(
            "Exact or ancestor match",
            metrics.exact_or_ancestor,
            metrics.total,
            minimum=thresholds.min_exact_or_ancestor,
            description="placements at or above the labelled field",
        ),
        _ratio_gate(
            "Path consistency",
            metrics.path_consistent,
            metrics.total,
            minimum=thresholds.min_path_consistency,
            description="predicted root-to-leaf paths",
        ),
        _precision_gate(
            "Branch precision",
            metrics.branches_accepted,
            metrics.branches_predicted,
            minimum=thresholds.min_branch_precision,
            description="accepted new fields",
        ),
        _precision_gate(
            "Promotion precision",
            metrics.promotions_accepted,
            metrics.promotions_predicted,
            minimum=thresholds.min_promotion_precision,
            description="promotions that improve the hierarchy",
        ),
        _ratio_gate(
            "Under-specificity",
            metrics.total - metrics.under_specific,
            metrics.total,
            minimum=1.0 - thresholds.max_under_specificity,
            blocking=thresholds.under_specificity_blocking,
            description="placements no broader than the labelled field",
        ),
    )


def evaluate_predictions(
    dataset: EvaluationDataset,
    predictions: PredictionSet,
    *,
    strategy: str = "custom",
    thresholds: GateThresholds | None = None,
    generated_at: str | None = None,
) -> EvalReport:
    """Score placements and lifecycle predictions against the labels."""

    thresholds = thresholds or GateThresholds()
    generated_at = generated_at or _now()

    created_fields: list[CreatedField] = list(predictions.branches)
    for prediction in predictions.placements.values():
        created_fields.extend(prediction.created_fields)
    unique_created = _dedupe_branches(created_fields)
    hierarchy = _extended_hierarchy(dataset.hierarchy, unique_created)

    warnings: list[str] = []
    case_results: list[CaseResult] = []
    exact = ancestor = exact_or_ancestor = over_specific = under_specific = 0
    path_consistent = 0

    for case in dataset.cases:
        document_id = case.document.id
        prediction = predictions.placements.get(document_id)
        if prediction is None:
            warnings.append(f"{document_id}: no placement prediction")
            case_results.append(
                CaseResult(
                    document_id=document_id,
                    expected_leaf_id=case.expected_leaf_id,
                    predicted_leaf_id=None,
                    relation="missing",
                    exact=False,
                    ancestor=False,
                    over_specific=False,
                    under_specific=False,
                    path_consistent=False,
                )
            )
            continue

        relation = _relation(
            hierarchy, prediction.leaf_field_id, case.expected_leaf_id
        )
        is_exact = relation == "exact"
        is_ancestor = relation == "ancestor"
        is_over = relation == "descendant"
        is_under = relation == "ancestor"
        is_path_consistent = _path_is_consistent(hierarchy, prediction)

        exact += int(is_exact)
        ancestor += int(is_ancestor)
        exact_or_ancestor += int(is_exact or is_ancestor)
        over_specific += int(is_over)
        under_specific += int(is_under)
        path_consistent += int(is_path_consistent)

        case_results.append(
            CaseResult(
                document_id=document_id,
                expected_leaf_id=case.expected_leaf_id,
                predicted_leaf_id=prediction.leaf_field_id,
                relation=relation,
                exact=is_exact,
                ancestor=is_ancestor,
                over_specific=is_over,
                under_specific=is_under,
                path_consistent=is_path_consistent,
                path_ids=prediction.path_ids,
            )
        )

    predicted_branches = unique_created
    accepted_branches = tuple(
        branch
        for branch in predicted_branches
        if branch.field_id in dataset.accepted_branch_ids
    )

    predicted_promotions = _dedupe_promotions(predictions.promotions)
    expected_promotions = set(dataset.expected_promotions)
    accepted_promotions = tuple(
        promotion
        for promotion in predicted_promotions
        if promotion in expected_promotions
    )

    metrics = EvalMetrics(
        total=len(dataset.cases),
        exact=exact,
        ancestor=ancestor,
        exact_or_ancestor=exact_or_ancestor,
        over_specific=over_specific,
        under_specific=under_specific,
        path_consistent=path_consistent,
        branches_predicted=len(predicted_branches),
        branches_accepted=len(accepted_branches),
        promotions_predicted=len(predicted_promotions),
        promotions_accepted=len(accepted_promotions),
    )
    gates = _build_gates(metrics, thresholds)
    return EvalReport(
        dataset_name=dataset.name,
        strategy=strategy,
        generated_at=generated_at,
        metrics=metrics,
        gates=gates,
        cases=tuple(case_results),
        predicted_branches=predicted_branches,
        accepted_branches=accepted_branches,
        predicted_promotions=predicted_promotions,
        expected_promotions=tuple(dataset.expected_promotions),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Deterministic strategies
# ---------------------------------------------------------------------------


def _document_text(document: SyntheticDocument) -> str:
    # The deterministic baseline uses the explicit title/keyword signal only;
    # the summary is free text and would make keyword matching harder to keep
    # stable across edits.
    return " ".join((document.title, *document.keywords))


def _best_keyword_field(dataset: EvaluationDataset, document: SyntheticDocument) -> str:
    text = _document_text(document)
    scored: list[tuple[int, int, str]] = []
    for node in dataset.hierarchy.nodes:
        score = sum(
            1 for keyword in node.keywords if _contains_phrase(text, keyword)
        )
        if score:
            scored.append((score, dataset.hierarchy.level(node.id), node.id))
    if not scored:
        return dataset.hierarchy.roots()[0]
    # Highest score wins; deeper fields break ties in favour of specificity,
    # and the id keeps the result deterministic.
    return max(scored)[2]


def _matching_branch_rule(
    dataset: EvaluationDataset, document: SyntheticDocument
) -> BranchRule | None:
    text = _document_text(document)
    for rule in dataset.branch_rules:
        if all(_contains_phrase(text, keyword) for keyword in rule.keywords):
            return rule
    return None


def _perfect_predictions(
    dataset: EvaluationDataset, *, include_promotions: bool = True
) -> PredictionSet:
    placements: dict[str, PlacementPrediction] = {}
    branches: list[CreatedField] = []
    for case in dataset.cases:
        created = (case.expected_branch,) if case.expected_branch else ()
        if case.expected_branch is not None:
            branches.append(case.expected_branch)
        placements[case.document.id] = PlacementPrediction(
            document_id=case.document.id,
            leaf_field_id=case.expected_leaf_id,
            path_ids=dataset.expected_path(case),
            created_fields=created,
            reason="perfect strategy: use the labelled field",
        )
    return PredictionSet(
        placements=placements,
        branches=tuple(branches),
        promotions=tuple(dataset.expected_promotions) if include_promotions else (),
    )


def _keyword_predictions(dataset: EvaluationDataset) -> PredictionSet:
    placements: dict[str, PlacementPrediction] = {}
    branches: list[CreatedField] = []
    for case in dataset.cases:
        rule = _matching_branch_rule(dataset, case.document)
        if rule is not None:
            created = rule.branch
            branches.append(created)
            leaf = created.field_id
            path = dataset.hierarchy.path(created.parent_id) + (created.field_id,)
            created_fields = (created,)
        else:
            leaf = _best_keyword_field(dataset, case.document)
            path = dataset.hierarchy.path(leaf)
            created_fields = ()
        placements[case.document.id] = PlacementPrediction(
            document_id=case.document.id,
            leaf_field_id=leaf,
            path_ids=path,
            created_fields=created_fields,
            reason="keyword strategy",
        )
    return PredictionSet(placements=placements, branches=tuple(branches))


def _perturb_predictions(
    dataset: EvaluationDataset,
    base: PredictionSet,
    *,
    mode: Literal["over_specific", "under_specific"],
) -> PredictionSet:
    placements: dict[str, PlacementPrediction] = {}
    for case in dataset.cases:
        prediction = base.placements[case.document.id]
        leaf = prediction.leaf_field_id
        path = prediction.path_ids
        created_fields = prediction.created_fields

        if mode == "over_specific":
            if leaf in dataset.hierarchy:
                children = dataset.hierarchy.children(leaf)
                if children:
                    leaf = children[0]
                    path = dataset.hierarchy.path(leaf)
        else:
            if leaf in dataset.hierarchy:
                parent_id = dataset.hierarchy.get(leaf).parent_id
            else:
                parent_id = next(
                    (
                        created.parent_id
                        for created in created_fields
                        if created.field_id == leaf
                    ),
                    None,
                )
            if parent_id is not None:
                leaf = parent_id
                path = dataset.hierarchy.path(parent_id)

        placements[case.document.id] = PlacementPrediction(
            document_id=case.document.id,
            leaf_field_id=leaf,
            path_ids=path,
            created_fields=created_fields,
            reason=f"{mode} strategy",
        )
    return PredictionSet(
        placements=placements,
        branches=base.branches,
        promotions=base.promotions,
    )


def build_predictions(
    dataset: EvaluationDataset, strategy: str = "keyword"
) -> PredictionSet:
    normalized = strategy.replace("-", "_")
    if normalized == "perfect":
        return _perfect_predictions(dataset)
    if normalized == "keyword":
        return _keyword_predictions(dataset)
    if normalized in {"over_specific", "under_specific"}:
        base = _perfect_predictions(dataset)
        return _perturb_predictions(dataset, base, mode=normalized)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown strategy {strategy!r}; choose keyword, perfect, "
        "over-specific, or under-specific"
    )


def run_evaluation(
    *,
    strategy: str = "keyword",
    dataset: EvaluationDataset | None = None,
    thresholds: GateThresholds | None = None,
    generated_at: str | None = None,
) -> EvalReport:
    dataset = dataset or SYNTHETIC_DATASET
    normalized = strategy.replace("-", "_")
    predictions = build_predictions(dataset, normalized)
    return evaluate_predictions(
        dataset,
        predictions,
        strategy=normalized,
        thresholds=thresholds,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Labelled synthetic set
# ---------------------------------------------------------------------------


_AGROECOLOGY = CreatedField(
    field_id="field_agroecology",
    parent_id="field_ecology",
    name="Agroecology",
)


def _doc(
    document_id: str,
    title: str,
    summary: str,
    keywords: tuple[str, ...],
) -> SyntheticDocument:
    return SyntheticDocument(
        id=document_id,
        title=title,
        summary=summary,
        keywords=keywords,
    )


def _case(
    document: SyntheticDocument,
    leaf: str,
    *,
    branch: CreatedField | None = None,
) -> SyntheticCase:
    return SyntheticCase(
        document=document,
        expected_leaf_id=leaf,
        expected_branch=branch,
    )


def build_synthetic_dataset() -> EvaluationDataset:
    hierarchy = FieldHierarchy(
        (
            FieldNode("field_science", "Science", None, ("science", "research")),
            FieldNode(
                "field_biology",
                "Biology",
                "field_science",
                ("biology", "biological", "organism", "species"),
            ),
            FieldNode(
                "field_ecology",
                "Ecology",
                "field_biology",
                ("ecology", "ecosystem", "habitat", "population", "breeding", "biodiversity"),
            ),
            FieldNode(
                "field_conservation",
                "Conservation Biology",
                "field_ecology",
                ("conservation", "protected area", "endangered", "restoration"),
            ),
            FieldNode(
                "field_genetics",
                "Genetics",
                "field_biology",
                ("genetics", "genetic", "genome", "gene", "inheritance"),
            ),
            FieldNode(
                "field_economics",
                "Economics",
                "field_science",
                ("economics", "economic", "economy", "market"),
            ),
            FieldNode(
                "field_labor",
                "Labor Economics",
                "field_economics",
                ("labor", "labour", "employment", "unemployment", "wage"),
            ),
            FieldNode(
                "field_housing",
                "Housing Economics",
                "field_economics",
                ("housing", "mortgage", "rent", "real estate", "house price"),
            ),
            FieldNode(
                "field_cs",
                "Computer Science",
                "field_science",
                ("computer science", "computing", "software", "algorithm"),
            ),
            FieldNode(
                "field_ml",
                "Machine Learning",
                "field_cs",
                ("machine learning", "neural network", "classifier", "deep learning", "training"),
            ),
            FieldNode(
                "field_systems",
                "Computer Systems",
                "field_cs",
                ("operating system", "distributed system", "database", "networking", "compiler"),
            ),
            FieldNode(
                "field_health",
                "Public Health",
                "field_science",
                ("public health", "health", "disease", "mortality", "vaccination"),
            ),
            FieldNode(
                "field_epi",
                "Epidemiology",
                "field_health",
                ("epidemiology", "epidemic", "outbreak", "infection", "surveillance"),
            ),
        )
    )

    cases = (
        _case(
            _doc(
                "doc_penguin_ecology",
                "Breeding colonies of Adelie penguins",
                "Population ecology and habitat use in a colonial seabird.",
                ("ecology", "breeding", "habitat"),
            ),
            "field_ecology",
        ),
        _case(
            _doc(
                "doc_coral_restoration",
                "Coral reef restoration",
                "Restoration and endangered-species recovery on coral reefs.",
                ("conservation", "restoration", "endangered"),
            ),
            "field_conservation",
        ),
        _case(
            _doc(
                "doc_maize_genetics",
                "Genetic diversity in maize",
                "Genome-wide inheritance and genetic variation in maize.",
                ("genetics", "genome", "inheritance"),
            ),
            "field_genetics",
        ),
        _case(
            _doc(
                "doc_mammal_biology",
                "Mammalian physiology and evolution",
                "Organism-level biology across mammal species.",
                ("biology", "organism", "species"),
            ),
            "field_biology",
        ),
        _case(
            _doc(
                "doc_minimum_wage",
                "Minimum wage and employment",
                "Labor-market effects of wage floors on employment.",
                ("labor", "employment", "wage"),
            ),
            "field_labor",
        ),
        _case(
            _doc(
                "doc_mortgage_rates",
                "Mortgage rates and housing demand",
                "Housing demand under changing mortgage rates.",
                ("housing", "mortgage", "house price"),
            ),
            "field_housing",
        ),
        _case(
            _doc(
                "doc_economic_growth",
                "Drivers of economic growth",
                "Cross-country economic growth and market institutions.",
                ("economics", "economic", "economy"),
            ),
            "field_economics",
        ),
        _case(
            _doc(
                "doc_neural_classifier",
                "Neural network classifiers",
                "Machine-learning classifiers trained on image data.",
                ("machine learning", "neural network", "classifier"),
            ),
            "field_ml",
        ),
        _case(
            _doc(
                "doc_database_networking",
                "Distributed database networking",
                "Networking and consistency in distributed database systems.",
                ("database", "distributed system", "networking"),
            ),
            "field_systems",
        ),
        _case(
            _doc(
                "doc_algorithm_design",
                "Algorithm design and software",
                "Computer-science algorithms and software engineering.",
                ("computer science", "algorithm", "software"),
            ),
            "field_cs",
        ),
        _case(
            _doc(
                "doc_outbreak_surveillance",
                "Outbreak surveillance systems",
                "Epidemiology surveillance for infectious outbreaks.",
                ("epidemiology", "outbreak", "surveillance"),
            ),
            "field_epi",
        ),
        _case(
            _doc(
                "doc_vaccination_mortality",
                "Vaccination and mortality",
                "Public-health effects of vaccination on mortality.",
                ("public health", "vaccination", "mortality"),
            ),
            "field_health",
        ),
        _case(
            _doc(
                "doc_science_method",
                "Reproducibility in science",
                "Research methods and reproducibility across science.",
                ("science", "research"),
            ),
            "field_science",
        ),
        _case(
            _doc(
                "doc_ecosystem_services",
                "Ecosystem services",
                "Ecology of ecosystem services and habitat value.",
                ("ecology", "ecosystem"),
            ),
            "field_ecology",
        ),
        _case(
            _doc(
                "doc_genetic_variation",
                "Genetic variation",
                "Genetic and genome-wide sources of variation.",
                ("genetic", "genome"),
            ),
            "field_genetics",
        ),
        _case(
            _doc(
                "doc_unemployment",
                "Unemployment dynamics",
                "Labour-market transitions into and out of unemployment.",
                ("unemployment", "labour"),
            ),
            "field_labor",
        ),
        _case(
            _doc(
                "doc_real_estate",
                "Real estate rents",
                "Rent levels in real-estate markets.",
                ("real estate", "rent"),
            ),
            "field_housing",
        ),
        _case(
            _doc(
                "doc_model_training",
                "Model training at scale",
                "Training classifiers on large datasets.",
                ("classifier", "training"),
            ),
            "field_ml",
        ),
        _case(
            _doc(
                "doc_compiler_systems",
                "Compiler and operating system design",
                "Compiler construction and operating-system internals.",
                ("compiler", "operating system"),
            ),
            "field_systems",
        ),
        _case(
            _doc(
                "doc_epidemic_models",
                "Epidemic surveillance",
                "Surveillance models for epidemic spread.",
                ("surveillance", "epidemic"),
            ),
            "field_epi",
        ),
        _case(
            _doc(
                "doc_health_disparities",
                "Health disparities",
                "Population health and disease disparities.",
                ("health", "disease"),
            ),
            "field_health",
        ),
        _case(
            _doc(
                "doc_biodiversity",
                "Biodiversity and ecology",
                "Biodiversity patterns in ecological communities.",
                ("biodiversity", "ecology"),
            ),
            "field_ecology",
        ),
        _case(
            _doc(
                "doc_agroecology",
                "Agroecology and soil health",
                "Farming practices and soil ecology at the field scale.",
                ("agroecology", "farming", "soil"),
            ),
            _AGROECOLOGY.field_id,
            branch=_AGROECOLOGY,
        ),
        _case(
            _doc(
                "doc_public_health_systems",
                "Public health systems",
                "Health-system organisation and public-health delivery.",
                ("public health", "health"),
            ),
            "field_health",
        ),
        _case(
            _doc(
                "doc_software_testing",
                "Software testing",
                "Software-quality algorithms and testing methods.",
                ("software", "algorithm"),
            ),
            "field_cs",
        ),
        _case(
            _doc(
                "doc_protected_areas",
                "Protected areas and endangered species",
                "Conservation planning for protected areas.",
                ("conservation", "protected area", "endangered"),
            ),
            "field_conservation",
        ),
        _case(
            _doc(
                "doc_macro_economics",
                "Macroeconomic policy",
                "Economic policy at the level of the economy.",
                ("economy", "economic"),
            ),
            "field_economics",
        ),
        _case(
            _doc(
                "doc_housing_markets",
                "Housing markets",
                "Mortgage finance and housing-market dynamics.",
                ("housing", "mortgage"),
            ),
            "field_housing",
        ),
        _case(
            _doc(
                "doc_deep_learning",
                "Deep learning systems",
                "Machine-learning systems built from neural networks.",
                ("neural network", "machine learning"),
            ),
            "field_ml",
        ),
        _case(
            _doc(
                "doc_species_richness",
                "Species richness",
                "Species counts and organismal biology.",
                ("biology", "species"),
            ),
            "field_biology",
        ),
    )

    return EvaluationDataset(
        name="synthetic-field-hierarchy-v1",
        hierarchy=hierarchy,
        cases=cases,
        accepted_branch_ids=frozenset({_AGROECOLOGY.field_id}),
        expected_promotions=(
            PromotionEvent(
                new_field_id="field_data_science",
                old_parent_id="field_ml",
                new_parent_id="field_cs",
            ),
            PromotionEvent(
                new_field_id="field_computational_biology",
                old_parent_id="field_genetics",
                new_parent_id="field_biology",
            ),
        ),
        branch_rules=(
            BranchRule(
                keywords=("agroecology", "farming", "soil"),
                branch=_AGROECOLOGY,
            ),
        ),
    )


SYNTHETIC_DATASET = build_synthetic_dataset()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_rate(value: float) -> str:
    return f"{value:.1%}"


def _branch_payload(branch: CreatedField) -> dict[str, str]:
    return {
        "field_id": branch.field_id,
        "parent_id": branch.parent_id,
        "name": branch.name,
    }


def _promotion_payload(promotion: PromotionEvent) -> dict[str, str | None]:
    return {
        "new_field_id": promotion.new_field_id,
        "old_parent_id": promotion.old_parent_id,
        "new_parent_id": promotion.new_parent_id,
    }


def build_json(report: EvalReport) -> dict[str, object]:
    metrics = report.metrics
    return {
        "schema_version": "1.0.0",
        "generated_at": report.generated_at,
        "dataset": report.dataset_name,
        "strategy": report.strategy,
        "status": report.status,
        "metrics": metrics.as_dict(),
        "gates": [
            {
                "name": gate.name,
                "status": gate.status,
                "detail": gate.detail,
                "blocking": gate.blocking,
            }
            for gate in report.gates
        ],
        "cases": [
            {
                "document_id": case.document_id,
                "expected_leaf_id": case.expected_leaf_id,
                "predicted_leaf_id": case.predicted_leaf_id,
                "relation": case.relation,
                "exact": case.exact,
                "ancestor": case.ancestor,
                "over_specific": case.over_specific,
                "under_specific": case.under_specific,
                "path_consistent": case.path_consistent,
                "path_ids": list(case.path_ids),
            }
            for case in report.cases
        ],
        "branches": {
            "predicted": [
                _branch_payload(branch) for branch in report.predicted_branches
            ],
            "accepted": [
                _branch_payload(branch) for branch in report.accepted_branches
            ],
            "precision": metrics.branch_precision,
        },
        "promotions": {
            "predicted": [
                _promotion_payload(promotion)
                for promotion in report.predicted_promotions
            ],
            "expected": [
                _promotion_payload(promotion)
                for promotion in report.expected_promotions
            ],
            "precision": metrics.promotion_precision,
        },
        "warnings": list(report.warnings),
    }


def render_markdown(report: EvalReport) -> str:
    metrics = report.metrics
    lines = [
        "# Field hierarchy evaluation",
        "",
        f"Generated: {report.generated_at}",
        f"Dataset: `{report.dataset_name}`",
        f"Strategy: `{report.strategy}`",
        f"Cases: {metrics.total}",
        f"Overall status: **{report.status.upper()}**",
        "",
        "## Metrics",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Exact field match | {metrics.exact}/{metrics.total} | {_format_rate(metrics.exact_rate)} |",
        f"| Ancestor match (strict) | {metrics.ancestor}/{metrics.total} | {_format_rate(metrics.ancestor_rate)} |",
        f"| Exact or ancestor match | {metrics.exact_or_ancestor}/{metrics.total} | {_format_rate(metrics.exact_or_ancestor_rate)} |",
        f"| Over-specificity | {metrics.over_specific}/{metrics.total} | {_format_rate(metrics.over_specificity_rate)} |",
        f"| Under-specificity | {metrics.under_specific}/{metrics.total} | {_format_rate(metrics.under_specificity_rate)} |",
        f"| Path consistency | {metrics.path_consistent}/{metrics.total} | {_format_rate(metrics.path_consistency_rate)} |",
        f"| Branch precision | {metrics.branches_accepted}/{metrics.branches_predicted} | {_format_rate(metrics.branch_precision)} |",
        f"| Promotion precision | {metrics.promotions_accepted}/{metrics.promotions_predicted} | {_format_rate(metrics.promotion_precision)} |",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---|---|",
    ]
    for gate in report.gates:
        suffix = "" if gate.blocking or gate.status == "skip" else " (diagnostic)"
        lines.append(
            f"| {_escape(gate.name)}{suffix} | {gate.status} | "
            f"{_escape(gate.detail)} |"
        )

    lines.extend(
        [
            "",
            "## Placements",
            "",
            "| Document | Expected | Predicted | Relation | Path consistent |",
            "|---|---|---|---|---|",
        ]
    )
    for case in report.cases:
        lines.append(
            f"| {_escape(case.document_id)} | {_escape(case.expected_leaf_id)} | "
            f"{_escape(case.predicted_leaf_id or '(missing)')} | "
            f"{case.relation} | {'yes' if case.path_consistent else 'no'} |"
        )

    lines.extend(["", "## Lifecycle events", ""])
    if report.predicted_branches:
        lines.extend(
            [
                "| Predicted branch | Parent | Accepted |",
                "|---|---|---|",
            ]
        )
        accepted_ids = {branch.field_id for branch in report.accepted_branches}
        for branch in report.predicted_branches:
            lines.append(
                f"| {_escape(branch.field_id)} ({_escape(branch.name)}) | "
                f"{_escape(branch.parent_id)} | "
                f"{'yes' if branch.field_id in accepted_ids else 'no'} |"
            )
    else:
        lines.append("- No branches predicted.")
    lines.append("")
    if report.predicted_promotions:
        lines.extend(
            [
                "| Predicted promotion | Old parent | New parent | Accepted |",
                "|---|---|---|---|",
            ]
        )
        expected = set(report.expected_promotions)
        for promotion in report.predicted_promotions:
            lines.append(
                f"| {_escape(promotion.new_field_id)} | "
                f"{_escape(promotion.old_parent_id)} | "
                f"{_escape(promotion.new_parent_id or '(root)')} | "
                f"{'yes' if promotion in expected else 'no'} |"
            )
    else:
        lines.append("- No promotions predicted.")

    lines.extend(["", "## Warnings", ""])
    if report.warnings:
        lines.extend(f"- {_escape(warning)}" for warning in report.warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Offline and deterministic: no network calls and no live LLM.",
            "- `keyword` is the default baseline; `perfect`, `over-specific`, and "
            "`under-specific` are diagnostic strategies used to exercise the metrics.",
            "- The blocking gate is no over-specific attachment, at least 80% "
            "exact/ancestor match, and path consistency. Branch and promotion "
            "precision are blocking when lifecycle changes are predicted.",
            "- Under-specificity is reported as a diagnostic because the spec's "
            "exact/ancestor gate intentionally accepts broader attachments.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline field-hierarchy placement evaluation gate."
    )
    parser.add_argument(
        "--strategy",
        choices=("keyword", "perfect", "over-specific", "under-specific"),
        default="keyword",
        help="deterministic placement strategy to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown report path",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON report path (default: output with a .json suffix)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="stdout format; both files are always written",
    )
    parser.add_argument(
        "--min-exact-or-ancestor",
        type=float,
        default=0.80,
        help="minimum exact-or-ancestor rate (default: 0.80)",
    )
    parser.add_argument(
        "--max-over-specific-rate",
        type=float,
        default=0.0,
        help="maximum over-specificity rate (default: 0.0)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = GateThresholds(
        min_exact_or_ancestor=args.min_exact_or_ancestor,
        max_over_specific_rate=args.max_over_specific_rate,
    )
    report = run_evaluation(strategy=args.strategy, thresholds=thresholds)
    markdown = render_markdown(report)
    payload = build_json(report)
    json_text = json.dumps(payload, indent=2) + "\n"

    output = args.output
    json_output = args.json_output or output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output.write_text(json_text, encoding="utf-8")

    if args.format == "json":
        print(json_text, end="")
    else:
        print(markdown)
        print(f"\nstatus={report.status} report={output} json={json_output}")
    if report.status != "pass":
        print(
            f"field-hierarchy gate failed: {report.status}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
