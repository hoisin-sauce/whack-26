from __future__ import annotations

import numpy as np
import pandas as pd

from whack26.model_search.candidates import generate_candidates

FEATURE_KINDS = {
    "x1": "numeric",
    "x2": "numeric",
    "x3": "numeric",
    "g": "categorical",
    "row_id": "id",
    "leak": "leakage_risk",
}


def _frame(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.gamma(2.0, 0.5, size=n)  # non-negative
    x2 = rng.normal(size=n)
    x3 = rng.uniform(0.5, 3.0, size=n)  # non-negative
    group = rng.choice(["a", "b", "c"], size=n).astype(object)
    score = 1.0 + 0.6 * x1 + 0.3 * x3 + 0.2 * x2
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "g": group,
            "row_id": np.arange(n),
            "leak": score * 2.0,
            "y_continuous": score + rng.normal(scale=0.1, size=n),
            "y_binary": (score > np.median(score)).astype(int),
            "y_multiclass": pd.cut(
                score,
                bins=3,
                labels=["low", "mid", "high"],
            ).astype(str),
            "y_count": rng.poisson(np.exp(0.3 + 0.2 * x1)),
        }
    )


def _generate(frame: pd.DataFrame, target: str, kind: str, **kwargs):
    return generate_candidates(
        target=target,
        target_kind=kind,
        frame=frame,
        feature_kinds=FEATURE_KINDS,
        **kwargs,
    )


def _families(specs) -> set[str]:
    return {spec.family for spec in specs}


def test_generates_baseline_first_for_every_target_kind() -> None:
    frame = _frame()
    for target, kind in (
        ("y_continuous", "continuous"),
        ("y_binary", "binary"),
        ("y_multiclass", "multiclass"),
        ("y_count", "count"),
    ):
        specs = _generate(frame, target, kind)
        assert specs, kind
        assert specs[0].family == "baseline"
        assert specs[0].target_kind == kind


def test_continuous_pool_has_linear_transforms_interactions_and_groups() -> None:
    frame = _frame()
    specs = _generate(frame, "y_continuous", "continuous")
    families = _families(specs)

    assert {"baseline", "linear", "interaction", "group_comparison"} <= families

    linear = [spec for spec in specs if spec.family == "linear"]
    assert {spec.target_transform for spec in linear} >= {
        "none",
        "log1p",
        "sqrt",
    }
    assert any(spec.feature_transforms for spec in linear)
    feature_transforms = {
        transform
        for spec in linear
        for transform in spec.feature_transforms.values()
    }
    assert {"log1p", "sqrt"} <= feature_transforms
    assert any(spec.regularization is not None for spec in linear)

    interactions = [spec for spec in specs if spec.family == "interaction"]
    assert interactions
    for spec in interactions:
        assert spec.interactions
        for left, right in spec.interactions:
            assert {left, right} <= {"x1", "x2", "x3"}
            assert set(spec.features) >= {left, right}


def test_binary_pool_has_logistic_and_rates() -> None:
    frame = _frame()
    specs = _generate(frame, "y_binary", "binary")
    families = _families(specs)
    assert {"baseline", "logistic", "interaction", "group_comparison"} <= families
    assert all(
        spec.target_transform == "none"
        for spec in specs
        if spec.family in {"logistic", "interaction", "group_comparison"}
    )

    group_specs = [spec for spec in specs if spec.family == "group_comparison"]
    assert any(
        spec.hyperparameters.get("statistic") == "rate" for spec in group_specs
    )


def test_multiclass_pool_has_multinomial() -> None:
    frame = _frame()
    specs = _generate(frame, "y_multiclass", "multiclass")
    families = _families(specs)
    assert {"baseline", "multinomial", "group_comparison"} <= families
    assert "logistic" not in families


def test_count_pool_has_poisson_and_negative_binomial() -> None:
    frame = _frame()
    specs = _generate(frame, "y_count", "count")
    families = _families(specs)
    assert {"baseline", "poisson", "negative_binomial", "group_comparison"} <= families
    assert "linear" in families


def test_candidate_cap_keeps_baseline_and_spreads_families() -> None:
    frame = _frame()
    specs = _generate(frame, "y_continuous", "continuous", max_candidates=4)
    assert len(specs) == 4
    assert specs[0].family == "baseline"
    assert len({spec.family for spec in specs}) == 4


def test_target_and_unusable_columns_are_never_features() -> None:
    frame = _frame()
    specs = _generate(frame, "y_continuous", "continuous")
    for spec in specs:
        assert "y_continuous" not in spec.features
        assert "row_id" not in spec.features
        assert "leak" not in spec.features


def test_feature_scores_drive_the_interaction_pair_order() -> None:
    frame = _frame()
    specs = _generate(
        frame,
        "y_continuous",
        "continuous",
        feature_scores={"x1": 0.1, "x2": 5.0, "x3": 0.2},
        max_features=3,
    )
    interactions = [spec for spec in specs if spec.family == "interaction"]
    assert interactions
    assert interactions[0].interactions == [("x2", "x3")]


def test_non_negative_features_can_be_declared_without_a_frame() -> None:
    specs = generate_candidates(
        target="y",
        target_kind="continuous",
        features=["a", "b"],
        feature_kinds={"a": "numeric", "b": "numeric"},
        non_negative_features=["a"],
    )
    transformed = [
        spec
        for spec in specs
        if spec.feature_transforms.get("a") in {"log1p", "sqrt"}
    ]
    assert transformed
    assert all("b" not in spec.feature_transforms for spec in transformed)


def test_invalid_target_kind_raises() -> None:
    try:
        generate_candidates(target="y", target_kind="ordinal", features=[])
    except ValueError as exc:
        assert "target kind" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
