from __future__ import annotations

from typing import Any

import pytest

from whack26.config import Settings
from whack26.research.scrape import pipeline
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

DOCUMENT_ID = "pap_hierarchy_fallback"
FIELD_ID = "fld_statistics"


def make_settings(**overrides: object) -> Settings:
    return Settings.from_env({}, mock=True).with_overrides(**overrides)


def make_document() -> Document:
    return Document(
        id=DOCUMENT_ID,
        kind="paper",
        source="openalex",
        source_id="W-fallback",
        title="Housing prices and income",
        summary="A study of housing prices and income.",
        status="summarized",
    )


def test_hierarchy_failure_warns_and_falls_back_to_classify_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failing_hierarchical(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("hierarchical")
        raise RuntimeError("hierarchy unavailable")

    def fallback_classify(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("classic")
        return [FIELD_ID]

    monkeypatch.setattr(
        "whack26.research.scrape.classify.classify_document_hierarchically",
        failing_hierarchical,
    )
    monkeypatch.setattr(
        "whack26.research.scrape.classify.classify_document",
        fallback_classify,
    )

    store = FieldStore(":memory:", dims=2)
    warnings: list[str] = []
    try:
        classified = pipeline._classify_document(
            make_document(),
            store=store,
            llm=None,
            embeddings=None,
            settings=make_settings(hierarchy_placement_enabled=True),
            warnings=warnings,
        )
    finally:
        store.close()

    assert calls == ["hierarchical", "classic"]
    assert any(
        warning.startswith(f"scrape_hierarchy_placement_failed:{DOCUMENT_ID}:")
        for warning in warnings
    )
    assert any("RuntimeError: hierarchy unavailable" in warning for warning in warnings)
    assert classified.status == "classified"


def test_disabled_hierarchy_uses_classify_document_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def unexpected_hierarchical(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("hierarchical")
        raise AssertionError("hierarchy placement must not run when disabled")

    def classic_classify(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("classic")
        return [FIELD_ID]

    monkeypatch.setattr(
        "whack26.research.scrape.classify.classify_document_hierarchically",
        unexpected_hierarchical,
    )
    monkeypatch.setattr(
        "whack26.research.scrape.classify.classify_document",
        classic_classify,
    )

    store = FieldStore(":memory:", dims=2)
    warnings: list[str] = []
    try:
        classified = pipeline._classify_document(
            make_document(),
            store=store,
            llm=None,
            embeddings=None,
            settings=make_settings(hierarchy_placement_enabled=False),
            warnings=warnings,
        )
    finally:
        store.close()

    assert calls == ["classic"]
    assert not any(
        warning.startswith("scrape_hierarchy_placement_failed")
        for warning in warnings
    )
    assert classified.status == "classified"
