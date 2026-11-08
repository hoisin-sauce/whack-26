from __future__ import annotations

from collections.abc import Iterator

import pytest

from whack26.config import Settings
from whack26.llm import LLMError, MockLLMClient
from whack26.research.scrape.compare import (
    COMPARISON_KINDS,
    compare_documents,
    compare_with_dataset,
    deterministic_similarity,
)
from whack26.research.scrape.models import Document
from whack26.store import FieldStore

DIMS = 4
SETTINGS = Settings.from_env({}, mock=True)

COMPARISON_FIXTURE = {
    "verdict": "supports",
    "rationale": "Both documents use comparable cohort designs.",
    "evidence": ["same cohort design", "comparable outcome definition"],
}


def vec(*values: float) -> list[float]:
    assert len(values) <= DIMS
    return [*values, *([0.0] * (DIMS - len(values)))]


def make_document(
    document_id: str,
    *,
    kind: str = "paper",
    embedding: list[float] | None = None,
    summary: str | None = None,
) -> Document:
    return Document(
        id=document_id,
        kind=kind,  # type: ignore[arg-type]
        source="test",
        title=f"Title for {document_id}",
        summary=summary if summary is not None else f"Summary for {document_id}",
        summary_embedding=vec(1.0) if embedding is None else embedding,
        embedding_model="mock",
    )


@pytest.fixture
def store() -> Iterator[FieldStore]:
    with FieldStore(":memory:", dims=DIMS) as opened:
        yield opened


def add_documents(store: FieldStore) -> tuple[Document, Document]:
    left = store.upsert_document(make_document("doc_alpha"))
    right = store.upsert_document(make_document("doc_beta"))
    return left, right


def test_cache_hit_makes_no_second_llm_call(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    first = compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )
    second = compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert llm.calls == ["document_comparison"]
    assert second == first
    assert first.verdict == "supports"
    assert first.similarity == pytest.approx(1.0)
    assert store.get_comparison("doc_alpha", "doc_beta", "finding_agreement") == first


def test_symmetric_pair_normalization_uses_one_cache_entry(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    forward = compare_documents(
        left,
        right,
        "method_overlap",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )
    reverse = compare_documents(
        right,
        left,
        "method_overlap",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert llm.calls == ["document_comparison"]
    assert forward == reverse
    assert (forward.left_id, forward.right_id) == ("doc_alpha", "doc_beta")


@pytest.mark.parametrize("kind", COMPARISON_KINDS)
def test_every_comparison_kind_is_accepted(store: FieldStore, kind: str) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    result = compare_documents(
        left,
        right,
        kind,
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert result.kind == kind
    assert store.get_comparison("doc_alpha", "doc_beta", kind) is not None


def test_invalid_json_returns_insufficient_without_caching(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(
        fixtures={
            "document_comparison": {
                "verdict": "probably",
                "rationale": "not a valid verdict",
                "evidence": "not a list",
            }
        }
    )

    result = compare_documents(
        left,
        right,
        "contradiction",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert result.verdict == "insufficient"
    assert "could not be completed" in result.rationale
    assert llm.calls == ["document_comparison"]
    assert store.get_comparison("doc_alpha", "doc_beta", "contradiction") is None


def test_llm_failure_returns_insufficient(store: FieldStore) -> None:
    left, right = add_documents(store)

    class FailingLLM:
        def generate_json(self, **kwargs: object) -> dict[str, object]:
            raise LLMError("provider down", purpose="document_comparison")

    result = compare_documents(
        left,
        right,
        "population_overlap",
        llm=FailingLLM(),
        store=store,
        settings=SETTINGS,
    )

    assert result.verdict == "insufficient"
    assert "LLMError" in result.rationale


def test_compare_with_dataset_uses_dataset_relevance(store: FieldStore) -> None:
    paper = store.upsert_document(make_document("doc_paper", embedding=vec(1.0)))
    dataset = store.upsert_document(
        make_document("ds_data", kind="dataset", embedding=vec(1.0))
    )
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    result = compare_with_dataset(
        paper,
        dataset,
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert result.kind == "dataset_relevance"
    assert result.right_id == "ds_data"
    assert result.similarity == pytest.approx(1.0)
    assert store.get_comparison(
        "doc_paper", "ds_data", "dataset_relevance"
    ) == result


def test_embedding_change_invalidates_the_cache(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )
    updated_right = right.model_copy(update={"summary_embedding": vec(0.0, 1.0)})
    store.upsert_document(updated_right)

    result = compare_documents(
        left,
        updated_right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert llm.calls == ["document_comparison", "document_comparison"]
    assert result.similarity == pytest.approx(0.0)


def test_method_version_change_invalidates_the_cache(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )
    newer = SETTINGS.with_overrides(scrape_method_version="2.0.0")
    result = compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=newer,
    )

    assert llm.calls == ["document_comparison", "document_comparison"]
    assert result.method_version == "2.0.0"
    assert store.get_comparison(
        "doc_alpha", "doc_beta", "finding_agreement", "2.0.0"
    ) == result


def test_missing_embeddings_fall_back_to_zero_similarity(store: FieldStore) -> None:
    left = store.upsert_document(make_document("doc_alpha", embedding=vec()))
    right = store.upsert_document(make_document("doc_beta", embedding=vec()))
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    result = compare_documents(
        left,
        right,
        "finding_agreement",
        llm=llm,
        store=store,
        settings=SETTINGS,
    )

    assert deterministic_similarity(left, right) == pytest.approx(0.0)
    assert result.similarity == pytest.approx(0.0)


def test_explicit_similarity_overrides_the_local_cosine(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient(fixtures={"document_comparison": COMPARISON_FIXTURE})

    result = compare_documents(
        left,
        right,
        "dataset_relevance",
        llm=llm,
        store=store,
        settings=SETTINGS,
        similarity=0.42,
    )

    assert result.similarity == pytest.approx(0.42)


def test_unknown_kind_is_rejected(store: FieldStore) -> None:
    left, right = add_documents(store)
    llm = MockLLMClient()

    with pytest.raises(ValueError):
        compare_documents(
            left,
            right,
            "vibes",
            llm=llm,
            store=store,
            settings=SETTINGS,
        )
    assert llm.calls == []
