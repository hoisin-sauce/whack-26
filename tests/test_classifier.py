from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from whack26.classify import FIELD_CHOICE_SCHEMA, Classifier, FieldPath
from whack26.config import Settings
from whack26.profile import ColumnProfile, DatasetProfile, TargetChoice
from whack26.store import FieldStore, FieldStoreError


class FakeEmbeddings:
    """Tiny keyword embedding so classifier tests are deterministic."""

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "housing" in lowered:
                vectors.append([1.0, 0.0])
            elif "biology" in lowered or "penguin" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.8, 0.6])
        return vectors


class ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "median_income": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "median_house_value": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "ocean_proximity": ["INLAND", "NEAR BAY", "INLAND", "ISLAND", "NEAR OCEAN", "INLAND"],
        }
    )


def make_profile(
    dataset_id: str = "ds_aaaaaaaaaaaa",
    description: str = "California housing prices by block group.",
    target_name: str | None = "median_house_value",
) -> DatasetProfile:
    frame = make_frame()
    columns = [
        ColumnProfile(
            name="median_income",
            dtype="float64",
            role="feature",
            missing_pct=0.0,
            cardinality=6,
            sample_values=["1.0", "2.0"],
            numeric={"min": 1.0, "max": 6.0, "mean": 3.5, "std": 1.87, "zeros": 0, "negatives": 0},
            top_values=None,
            datetime=False,
        ),
        ColumnProfile(
            name="median_house_value",
            dtype="float64",
            role="target",
            missing_pct=0.0,
            cardinality=6,
            sample_values=["100.0", "200.0"],
            numeric={"min": 100.0, "max": 600.0, "mean": 350.0, "std": 187.1, "zeros": 0, "negatives": 0},
            top_values=None,
            datetime=False,
        ),
        ColumnProfile(
            name="ocean_proximity",
            dtype="object",
            role="feature",
            missing_pct=0.0,
            cardinality=4,
            sample_values=["INLAND", "NEAR BAY"],
            numeric=None,
            top_values=[("INLAND", 3), ("NEAR BAY", 1)],
            datetime=False,
        ),
    ]
    target = (
        TargetChoice(name=target_name, source="user", confidence=1.0)
        if target_name
        else None
    )
    return DatasetProfile(
        dataset_id=dataset_id,
        source_path=f"data/{dataset_id}.csv",
        sha256="a" * 64,
        format="csv",
        rows=len(frame),
        columns=columns,
        description=description,
        target=target,
        time_column=None,
        warnings=[],
    )


def make_classifier(
    tmp_path: Path, responses: list[dict[str, Any]]
) -> tuple[Classifier, FieldStore, ScriptedLLM]:
    settings = Settings(embed_dims=2, max_depth=4, refine_confidence=0.62)
    store = FieldStore(tmp_path / "fields.db", dims=2, max_depth=4)
    llm = ScriptedLLM(responses)
    classifier = Classifier(
        llm=llm,
        embeddings=FakeEmbeddings(),
        store=store,
        settings=settings,
    )
    return classifier, store, llm


def create_response(name: str, description: str) -> dict[str, Any]:
    return {
        "action": "create",
        "name": name,
        "description": description,
        "reason": "test fixture",
    }


def test_field_choice_schema_caps_string_lengths() -> None:
    properties = FIELD_CHOICE_SCHEMA["properties"]
    assert properties["name"]["maxLength"] == 80
    assert properties["description"]["maxLength"] == 500
    assert properties["reason"]["maxLength"] == 300
    assert FIELD_CHOICE_SCHEMA["propertyOrdering"][0] == "action"


def test_cold_classification_creates_root_and_child(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    llm_responses = [
        create_response("Housing", "Housing markets and residential property."),
        create_response("Housing Markets", "Prices and demand in housing markets."),
        {"action": "stop", "reason": "the dataset belongs here"},
    ]
    classifier, store, llm = make_classifier(tmp_path, llm_responses)
    try:
        path = classifier.classify(profile, frame)
        assert path.names == ["Housing", "Housing Markets"]
        assert path.depth == 2
        assert len(path.created_fields) == 2
        assert path.confidence == pytest.approx(0.5)
        assert store.roots()[0].name == "Housing"
        assert len(store.children(store.roots()[0].id)) == 1
        assert store.get_dataset(profile.dataset_id) is not None
        assert store.dataset_path(profile.dataset_id)[-1].name == "Housing Markets"
        assert "root level" in llm.calls[0]["user"]
        assert "Housing" in llm.calls[1]["user"]
        assert llm.calls[0]["max_tokens"] == 1024
    finally:
        store.close()


def test_stop_is_rejected_at_first_level(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            {"action": "stop", "reason": "too shallow"},
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        path = classifier.classify(profile, frame)
        assert path.names == ["Housing", "Housing Markets"]
        assert "`stop` is not allowed here" in llm.calls[2]["user"]
    finally:
        store.close()


def test_warm_classification_reuses_existing_path(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    settings = Settings(embed_dims=2, max_depth=4)
    store = FieldStore(tmp_path / "fields.db", dims=2, max_depth=4)
    try:
        root = store.create_field(
            parent_id=None,
            name="Housing",
            description="Housing markets and residential property.",
            description_embedding=[1.0, 0.0],
        )
        child = store.create_field(
            parent_id=root.id,
            name="Housing Markets",
            description="Prices and demand in housing markets.",
            description_embedding=[1.0, 0.0],
        )
        llm = ScriptedLLM(
            [
                {"action": "assign", "field_id": root.id, "reason": "root fits"},
                {"action": "assign", "field_id": child.id, "reason": "child fits"},
                {"action": "stop", "reason": "leaf"},
            ]
        )
        classifier = Classifier(
            llm=llm,
            embeddings=FakeEmbeddings(),
            store=store,
            settings=settings,
        )
        path = classifier.classify(profile, frame)
        assert path.ids == [root.id, child.id]
        assert path.created_fields == []
        assert path.confidence == pytest.approx(1.0)
    finally:
        store.close()


def test_warm_classify_reuses_committed_path_without_llm(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        second = classifier.classify(profile, frame)
        assert second.ids == first.ids
        assert second.confidence == first.confidence
        assert second.created_fields == []
        assert len(llm.calls) == calls_after_cold
    finally:
        store.close()


def test_warm_classify_recomputes_after_description_change(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    changed = profile.model_copy(update={"description": "A different description."})
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf again"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        second = classifier.classify(changed, frame)
        assert second.ids == first.ids
        assert len(llm.calls) > calls_after_cold
    finally:
        store.close()


def test_warm_classify_reuses_path_for_same_content_at_a_different_path(
    tmp_path: Path,
) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        moved = profile.model_copy(
            update={"source_path": "archive/relocated/housing.csv"}
        )
        second = classifier.classify(moved, frame)
        assert second.ids == first.ids
        assert second.created_fields == []
        assert len(llm.calls) == calls_after_cold
    finally:
        store.close()


def test_warm_classify_ignores_warning_text_changes(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        warned = profile.model_copy(update={"warnings": ["a new warning"]})
        second = classifier.classify(warned, frame)
        assert second.ids == first.ids
        assert len(llm.calls) == calls_after_cold
    finally:
        store.close()


def test_warm_classify_recomputes_after_schema_change(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        changed_columns = [
            profile.columns[0].model_copy(update={"dtype": "int64"}),
            *profile.columns[1:],
        ]
        changed = profile.model_copy(update={"columns": changed_columns})
        llm.responses.extend(
            [
                create_response("Housing", "Housing markets and residential property."),
                create_response("Housing Markets", "Prices and demand in housing markets."),
                {"action": "stop", "reason": "leaf again"},
            ]
        )
        second = classifier.classify(changed, frame)
        assert second.ids == first.ids
        assert len(llm.calls) > calls_after_cold
    finally:
        store.close()


def test_warm_classify_recomputes_after_settings_change(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, llm = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )
    try:
        first = classifier.classify(profile, frame)
        calls_after_cold = len(llm.calls)
        classifier.settings = Settings(
            embed_dims=2,
            max_depth=4,
            duplicate_cosine=0.5,
        )
        llm.responses.extend(
            [
                create_response("Housing", "Housing markets and residential property."),
                create_response("Housing Markets", "Prices and demand in housing markets."),
                {"action": "stop", "reason": "leaf again"},
            ]
        )
        second = classifier.classify(profile, frame)
        assert second.ids == first.ids
        assert len(llm.calls) > calls_after_cold
    finally:
        store.close()


def test_duplicate_root_is_merged(tmp_path: Path) -> None:
    frame = make_frame()
    profile_one = make_profile("ds_aaaaaaaaaaaa")
    profile_two = make_profile("ds_bbbbbbbbbbbb", description="housing prices again")
    classifier, store, _ = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "the dataset belongs here"},
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "merged path"},
        ],
    )
    try:
        first = classifier.classify(profile_one, frame)
        assert first.depth == 2
        second = classifier.classify(profile_two, frame)
        assert second.names == ["Housing", "Housing Markets"]
        assert second.created_fields == []
        assert any("duplicate_merge" in warning for warning in second.warnings)
        assert len(store.roots()) == 1
        assert store.roots()[0].member_count == 2
    finally:
        store.close()


def test_invalid_assign_falls_back_to_top_candidate(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    settings = Settings(embed_dims=2, max_depth=4)
    store = FieldStore(tmp_path / "fields.db", dims=2, max_depth=4)
    try:
        root = store.create_field(
            parent_id=None,
            name="Housing",
            description="Housing markets and residential property.",
            description_embedding=[1.0, 0.0],
        )
        llm = ScriptedLLM(
            [
                {"action": "assign", "field_id": "fld_missing", "reason": "bad"},
                {"action": "assign", "field_id": "fld_missing", "reason": "bad again"},
                create_response("Housing Markets", "Prices and demand in housing markets."),
                {"action": "stop", "reason": "leaf"},
            ]
        )
        classifier = Classifier(
            llm=llm,
            embeddings=FakeEmbeddings(),
            store=store,
            settings=settings,
        )
        path = classifier.classify(profile, frame)
        assert path.depth == 2
        assert path.ids[0] == root.id
        assert any("classification_fallback" in warning for warning in path.warnings)
    finally:
        store.close()


def test_stop_at_root_falls_back_to_general_root(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, _ = make_classifier(
        tmp_path,
        [
            {"action": "stop", "reason": "bad root stop"},
            {"action": "stop", "reason": "bad root stop again"},
            create_response("General Topics", "Broad topics that span many domains."),
            {"action": "stop", "reason": "stop at fallback child"},
        ],
    )
    try:
        path = classifier.classify(profile, frame)
        assert path.names == ["General Research", "General Topics"]
        assert len(path.created_fields) == 2
        assert any("classification_fallback" in warning for warning in path.warnings)
    finally:
        store.close()


def test_classifier_can_create_four_levels(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, _ = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            create_response("Housing Prices", "Price levels for residential housing."),
            create_response("Housing Price Levels", "Detailed residential price levels."),
        ],
    )
    try:
        path = classifier.classify(profile, frame)
        assert path.depth == 4
        assert path.names == [
            "Housing",
            "Housing Markets",
            "Housing Prices",
            "Housing Price Levels",
        ]
        assert len(path.created_fields) == 4
    finally:
        store.close()


def test_refinement_can_move_to_another_existing_root(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    settings = Settings(embed_dims=2, max_depth=4)
    store = FieldStore(tmp_path / "fields.db", dims=2, max_depth=4)
    try:
        housing = store.create_field(
            parent_id=None,
            name="Housing",
            description="Housing markets and residential property.",
            description_embedding=[1.0, 0.0],
        )
        housing_child = store.create_field(
            parent_id=housing.id,
            name="Housing Markets",
            description="Prices and demand in housing markets.",
            description_embedding=[1.0, 0.0],
        )
        economics = store.create_field(
            parent_id=None,
            name="Economics",
            description="Economic systems, markets, and prices.",
            description_embedding=[0.0, 1.0],
        )
        # Keep the economics root alive: empty fields are pruned after every
        # classification, so a real alternative root would have a member.
        store.commit_path(
            dataset_id="ds_otherdataset",
            sha256="b" * 64,
            profile_json="{}",
            summary="other dataset",
            embedding=[0.0, 1.0],
            path_json=f'["{economics.id}"]',
            confidence=1.0,
            refined_after_research=False,
            field_ids=[economics.id],
        )
        store.recompute_centroids([economics.id])
        llm = ScriptedLLM(
            [
                {"action": "assign", "field_id": housing.id, "reason": "start"},
                {"action": "assign", "field_id": housing_child.id, "reason": "child"},
                {"action": "stop", "reason": "leaf"},
            ]
        )
        classifier = Classifier(
            llm=llm,
            embeddings=FakeEmbeddings(),
            store=store,
            settings=settings,
        )
        original = classifier.classify(profile, frame)
        assert original.ids == [housing.id, housing_child.id]

        llm.responses = [
            {"action": "assign", "field_id": economics.id, "reason": "research points to economics"}
        ]
        refined = classifier.refine(
            profile,
            frame,
            original,
            research_titles=["Housing prices and the macroeconomy"],
            keywords=["housing", "prices", "macroeconomy"],
            force=True,
        )
        assert refined.ids == [economics.id]
        assert refined.refined_after_research is True
        assert refined.refinement_reason
        assert store.dataset_path(profile.dataset_id)[-1].id == economics.id
    finally:
        store.close()


def test_failed_commit_prunes_new_fields(tmp_path: Path) -> None:
    frame = make_frame()
    profile = make_profile()
    classifier, store, _ = make_classifier(
        tmp_path,
        [
            create_response("Housing", "Housing markets and residential property."),
            create_response("Housing Markets", "Prices and demand in housing markets."),
            {"action": "stop", "reason": "leaf"},
        ],
    )

    def fail_commit(**kwargs: Any) -> list[str]:
        raise FieldStoreError("commit failed")

    store.commit_path = fail_commit  # type: ignore[method-assign]
    try:
        with pytest.raises(FieldStoreError, match="commit failed"):
            classifier.classify(profile, frame)
        assert store.all_fields() == []
        assert store.stats()["orphan_fields"] == 0
    finally:
        store.close()


def test_should_refine_uses_confidence_threshold() -> None:
    settings = Settings(embed_dims=2, refine_confidence=0.62)
    classifier = Classifier(
        llm=ScriptedLLM([]),
        embeddings=FakeEmbeddings(),
        store=None,  # type: ignore[arg-type]
        settings=settings,
    )
    low = FieldPath(fields=[], confidence=0.4)
    high = FieldPath(fields=[], confidence=0.9)
    assert classifier.should_refine(low) is True
    assert classifier.should_refine(high) is False
