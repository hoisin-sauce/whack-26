from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tests.contracts.fakes import make_document, make_model_search_result
from whack26.factors import Factor
from whack26.hypotheses.context import (
    CONTEXT_MAX_CHARS,
    build_context_pack,
    render_context_pack,
)
from whack26.io import read_dataset
from whack26.profile import build_profile
from whack26.screen import build_screen
from whack26.store import FieldStore

DIMS = 2
TIMESTAMP = "2026-11-08T10:00:00Z"


def make_profile_and_frame(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "group": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "y": [2.0, 4.5, 5.5, 8.5, 9.0, 12.5, 13.0, 16.5],
        }
    )
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    loaded = read_dataset(path, target="y")
    return build_profile(loaded, target="y"), loaded.frame


def make_field(store: FieldStore, name: str, *, parent_id: str | None = None):
    return store.create_field(
        parent_id=parent_id,
        name=name,
        description=f"{name} description.",
        description_embedding=[1.0, 0.0],
    )


def build_store(tmp_path: Path, profile, frame):
    store = FieldStore(tmp_path / "fields.db", dims=DIMS)
    root = make_field(store, "Root")
    domain = make_field(store, "Domain", parent_id=root.id)
    leaf = make_field(store, "Leaf", parent_id=domain.id)
    sibling = make_field(store, "Sibling", parent_id=root.id)

    store.commit_path(
        dataset_id=profile.dataset_id,
        sha256=profile.sha256,
        profile_json=profile.model_dump_json(),
        summary="Dataset about prices; contact owner@example.com",
        embedding=[1.0, 0.0],
        path_json=json.dumps([root.id, domain.id, leaf.id]),
        confidence=0.9,
        refined_after_research=False,
        field_ids=[root.id, domain.id, leaf.id],
    )

    documents = {
        "root": make_document(
            "doc_root",
            title="Root ancestor",
            summary="Root ancestor summary for person@example.com",
            embedding=[1.0, 0.0],
        ).model_copy(
            update={
                "credibility_score": 0.9,
                "published_at": "2024-01-01",
                "authors": ["Ann Author"],
            }
        ),
        "domain": make_document(
            "doc_domain",
            title="Domain ancestor",
            summary="Domain ancestor summary",
            embedding=[0.0, 1.0],
        ).model_copy(
            update={"credibility_score": 0.7, "published_at": "2022-01-01"}
        ),
        "leaf": make_document(
            "doc_leaf",
            title="Leaf ancestor",
            summary="Leaf ancestor summary at 10.9.8.7",
            embedding=[1.0, 0.0],
        ).model_copy(
            update={"credibility_score": 0.5, "published_at": "2020-01-01"}
        ),
        "sibling": make_document(
            "doc_sibling",
            title="Sibling paper",
            summary="Sibling summary",
            embedding=[1.0, 0.0],
        ).model_copy(update={"credibility_score": 1.0}),
    }
    for index, field in enumerate((root, domain, leaf)):
        document = documents[("root", "domain", "leaf")[index]]
        store.upsert_document(document)
        store.assign_document_path(document.id, [field.id])
    store.upsert_document(documents["sibling"])
    store.assign_document_path(documents["sibling"].id, [sibling.id])
    return store, (root, domain, leaf, sibling), documents


def make_factors() -> list[Factor]:
    return [
        Factor(
            id="fac_income",
            name="Income",
            description="Higher income predicts a higher price.",
            direction="positive",
            claim_type="association",
            evidence_ids=["doc_root"],
            mapped_column="x",
        ),
        Factor(
            id="fac_group",
            name="Group effect",
            description="Groups differ in the mean outcome.",
            direction="unclear",
            claim_type="descriptive",
            evidence_ids=["doc_domain"],
            mapped_column="group",
        ),
    ]


def test_preloaded_statistics_documents_are_consumed(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, (root, domain, leaf, sibling), _documents = build_store(
        tmp_path, profile, frame
    )
    preloaded = make_document(
        "pap_preloaded",
        title="Leakage and validation",
        summary="A preloaded statistics document about data leakage.",
        embedding=[1.0, 0.0],
    ).model_copy(
        update={
            "preloaded": True,
            "credible": True,
            "credibility_score": 0.9,
        }
    )
    try:
        store.upsert_document(preloaded)
        pack = build_context_pack(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            factors=[],
            screen=build_screen(profile, frame),
        )
    finally:
        store.close()

    assert any(
        "Statistics corpus document" in chunk
        and "preloaded statistics document" in chunk
        for chunk in pack.critic_chunks
    )


def test_ancestor_inclusion_and_sibling_exclusion(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, (root, domain, leaf, sibling), documents = build_store(
        tmp_path, profile, frame
    )
    try:
        pack = build_context_pack(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            factors=make_factors(),
            critic_chunks=["### S0-how-to-use\nUse associational language."],
            model_candidates=make_model_search_result(),
            screen=build_screen(profile, frame),
        )
    finally:
        store.close()

    assert pack.field_path == ["Root", "Domain", "Leaf"]
    ancestor_ids = [document.document_id for document in pack.ancestor_documents]
    assert "doc_root" in ancestor_ids
    assert "doc_domain" in ancestor_ids
    assert "doc_leaf" in ancestor_ids
    assert "doc_sibling" not in ancestor_ids
    # Highest similarity then credibility: root, leaf, domain.
    assert ancestor_ids == ["doc_root", "doc_leaf", "doc_domain"]
    assert pack.ancestor_documents[0].similarity == 1.0
    assert pack.ancestor_documents[0].field_name == "Root"
    assert pack.ancestor_stats["documents_per_field"][root.id] == 1
    assert pack.ancestor_stats["documents_per_field"][domain.id] == 1
    assert pack.ancestor_stats["documents_per_field"][leaf.id] == 1
    assert sibling.id not in pack.ancestor_stats["documents_per_field"]
    assert pack.ancestor_stats["factors_by_field"][root.id] == ["Income"]
    assert pack.ancestor_stats["factors_by_field"][domain.id] == ["Group effect"]


def test_field_path_names_resolve_from_store(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, _fields, _documents = build_store(tmp_path, profile, frame)
    try:
        pack = build_context_pack(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            field_path=["Root", "Domain"],
        )
    finally:
        store.close()

    assert pack.field_path == ["Root", "Domain"]
    ancestor_ids = [document.document_id for document in pack.ancestor_documents]
    assert "doc_root" in ancestor_ids
    assert "doc_domain" in ancestor_ids
    assert "doc_leaf" not in ancestor_ids
    assert "doc_sibling" not in ancestor_ids


def test_context_pack_includes_factors_models_screen_and_chunks(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, _fields, _documents = build_store(tmp_path, profile, frame)
    try:
        pack = build_context_pack(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            factors=make_factors(),
            model_candidates=make_model_search_result(),
            screen=build_screen(profile, frame),
            critic_chunks=["### S0-how-to-use\nUse associational language."],
        )
    finally:
        store.close()

    assert [factor.id for factor in pack.factors] == ["fac_group", "fac_income"]
    assert pack.factors[0].description == "Groups differ in the mean outcome."
    assert pack.ancestor_stats["target"]["name"] == "y"
    assert pack.model_candidates[0].spec.family == "linear"
    assert pack.screen["target"]["name"] == "y"
    assert pack.critic_chunks[0].startswith("### S0-how-to-use")
    assert pack.token_estimate > 0

    rendered = render_context_pack(pack)
    assert "## Dataset summary" in rendered
    assert "## Field path" in rendered
    assert "fac_income" in rendered
    assert "doc_root" in rendered
    assert "linear" in rendered


def test_context_pack_empty_ancestors(tmp_path: Path) -> None:
    profile, _frame = make_profile_and_frame(tmp_path)

    pack = build_context_pack(
        profile=profile,
        field_path=["Root"],
        ancestor_documents=[],
    )

    assert pack.ancestor_documents == []
    assert pack.field_path == ["Root"]
    assert pack.ancestor_stats["counts"]["documents"] == 0
    assert "## Ancestor documents\n(none)" in render_context_pack(pack)


def test_context_pack_respects_token_cap(tmp_path: Path) -> None:
    profile, _frame = make_profile_and_frame(tmp_path)
    long_summary = "long summary " * 2_000
    long_document = make_document(
        "doc_long",
        title="Long paper",
        summary="paper text " * 2_000,
        embedding=[1.0, 0.0],
    )

    pack = build_context_pack(
        profile=profile,
        summary=long_summary,
        field_path=["Root", "Domain", "Leaf"],
        ancestor_documents=[long_document],
        factors=make_factors(),
        critic_chunks=["### S1-assumptions\n" + "assumption " * 1_000],
        max_chars=1_200,
    )

    rendered = render_context_pack(pack, max_chars=1_200)
    assert len(rendered) <= 1_200
    assert pack.token_estimate <= 300
    assert pack.summary != long_summary


def test_context_pack_is_deterministic(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, _fields, _documents = build_store(tmp_path, profile, frame)
    try:
        kwargs = dict(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            factors=make_factors(),
            critic_chunks=["### S0-how-to-use\nUse associational language."],
            model_candidates=make_model_search_result(),
            screen=build_screen(profile, frame),
        )
        first = build_context_pack(**kwargs)
        second = build_context_pack(**kwargs)
    finally:
        store.close()

    assert first.model_dump_json() == second.model_dump_json()
    assert render_context_pack(first) == render_context_pack(second)


def test_context_pack_redacts_pii_and_secrets(tmp_path: Path) -> None:
    profile, frame = make_profile_and_frame(tmp_path)
    store, _fields, _documents = build_store(
        tmp_path, profile, frame
    )
    model = SimpleNamespace(
        document_id="doc_model",
        family="linear",
        outcome="y",
        predictors=["x"],
        evidence="Contact secret.person@example.com",
        confidence=0.8,
        applied=True,
    )
    try:
        pack = build_context_pack(
            profile=profile,
            store=store,
            dataset_id=profile.dataset_id,
            factors=[
                Factor(
                    id="fac_pii",
                    name="Phone factor",
                    description="Call +1 555 010 9999 for details.",
                    evidence_ids=["doc_root"],
                )
            ],
            research_models=[model],
            critic_chunks=[
                "### S0-how-to-use\nAuthorization: Bearer topsecret "
                "and api_key=abc123"
            ],
        )
    finally:
        store.close()

    payload = pack.model_dump_json()
    assert "owner@example.com" not in payload
    assert "person@example.com" not in payload
    assert "secret.person@example.com" not in payload
    assert "10.9.8.7" not in payload
    assert "+1 555 010 9999" not in payload
    assert "topsecret" not in payload
    assert "abc123" not in payload
    assert "[redacted]" in payload


def test_research_models_are_duck_typed(tmp_path: Path) -> None:
    profile, _frame = make_profile_and_frame(tmp_path)
    object_model = SimpleNamespace(
        document_id="doc_a",
        family="linear_model",
        outcome="y",
        predictors=["x", "group"],
        target_transform="log1p",
        feature_transforms={"x": "sqrt"},
        interactions=[("x", "group")],
        assumptions=["linearity"],
        reported_metrics={"r2": 0.4},
        evidence="A quoted result.",
        mapped_columns=["x", "group"],
        confidence=0.6,
        applied=False,
    )
    dict_model = {
        "document_id": "doc_b",
        "family": "group_comparison",
        "outcome": "y",
        "predictors": ["group"],
        "confidence": 0.9,
        "applied": True,
    }

    pack = build_context_pack(
        profile=profile,
        field_path=["Root"],
        research_models=[object_model, dict_model],
    )

    assert [model.document_id for model in pack.research_models] == [
        "doc_b",
        "doc_a",
    ]
    assert pack.research_models[1].interactions == [("x", "group")]
    assert pack.research_models[1].feature_transforms == {"x": "sqrt"}
    assert pack.research_models[0].applied is True


def test_context_default_cap_constant_matches_plan() -> None:
    assert CONTEXT_MAX_CHARS == 20_000
