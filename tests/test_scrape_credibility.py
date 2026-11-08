from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from whack26.config import Settings
from whack26.research.scrape.credibility import (
    PENALTY_NO_DATE,
    PENALTY_PREDATORY,
    PENALTY_RETRACTED,
    WEIGHTS,
    apply_credibility,
    score_credibility,
)
from whack26.research.scrape.models import Document


def _days_ago_iso(days: int) -> str:
    today = datetime.now(timezone.utc).date()
    return (today - timedelta(days=days)).isoformat()


def _document(**overrides: object) -> Document:
    values: dict[str, object] = {
        "id": "doc_test",
        "kind": "paper",
        "source": "openalex",
        "title": "A study",
        "authors": ["Ada Lovelace"],
        "publisher": "Journal of Examples",
        "published_at": _days_ago_iso(1),
        "metadata": {},
    }
    values.update(overrides)
    return Document(**values)


@pytest.fixture
def settings() -> Settings:
    return Settings()


def test_peer_reviewed_journal_scores_above_blog(settings: Settings) -> None:
    journal = _document(
        publisher="Nature",
        metadata={
            "peer_reviewed": True,
            "venue_type": "journal",
            "cited_by_count": 100,
        },
    )
    blog = _document(
        id="doc_blog",
        source="web",
        publisher="Example Blog",
        metadata={"venue_type": "blog"},
    )

    journal_result = score_credibility(journal, settings)
    blog_result = score_credibility(blog, settings)

    assert journal_result.components["venue"] == 1.0
    assert blog_result.components["venue"] == 0.2
    assert journal_result.credible is True
    assert blog_result.credible is False
    assert journal_result.score > blog_result.score


def test_retracted_paper_is_penalised_and_clamped(settings: Settings) -> None:
    clean = _document(
        metadata={
            "peer_reviewed": True,
            "venue_type": "journal",
            "cited_by_count": 100,
        }
    )
    retracted = _document(
        metadata={
            "peer_reviewed": True,
            "venue_type": "journal",
            "cited_by_count": 100,
            "retracted": True,
        }
    )

    clean_result = score_credibility(clean, settings)
    retracted_result = score_credibility(retracted, settings)

    assert clean_result.credible is True
    assert retracted_result.components["penalties"] == PENALTY_RETRACTED
    assert retracted_result.score == 0.0
    assert retracted_result.credible is False
    assert retracted_result.score < clean_result.score
    assert any("retracted" in reason for reason in retracted_result.reasons)


def test_is_retracted_alias_is_penalised(settings: Settings) -> None:
    result = score_credibility(
        _document(metadata={"venue_type": "journal", "is_retracted": True}),
        settings,
    )

    assert result.components["penalties"] == PENALTY_RETRACTED
    assert result.credible is False


def test_unknown_author_defaults(settings: Settings) -> None:
    no_authors = score_credibility(_document(authors=[]), settings)
    unverified = score_credibility(_document(authors=["Anonymous"]), settings)

    assert no_authors.components["author"] == 0.2
    assert unverified.components["author"] == 0.5
    assert any("no authors" in reason for reason in no_authors.reasons)
    assert any("not verified" in reason for reason in unverified.reasons)


def test_author_metrics_raise_the_author_component(settings: Settings) -> None:
    document = _document()

    by_h_index = score_credibility(document, settings, h_index=20)
    by_citations = score_credibility(document, settings, citation_count=500)

    assert by_h_index.components["author"] == pytest.approx(0.75)
    expected = 0.5 + 0.5 * math.log1p(500) / math.log1p(10_000)
    assert by_citations.components["author"] == pytest.approx(expected)
    # The optional arguments are author-level signals, not document citations.
    assert by_citations.components["citation"] == 0.0


def test_recency_decays_with_age(settings: Settings) -> None:
    recent = score_credibility(_document(published_at=_days_ago_iso(0)), settings)
    old = score_credibility(_document(published_at=_days_ago_iso(3652)), settings)

    assert recent.components["recency"] == pytest.approx(1.0)
    assert old.components["recency"] == pytest.approx(math.exp(-2.0), abs=0.01)
    assert recent.components["recency"] > old.components["recency"]
    assert recent.score > old.score


def test_missing_date_is_penalised(settings: Settings) -> None:
    result = score_credibility(_document(published_at=None), settings)

    assert result.components["recency"] == pytest.approx(math.exp(-1.0))
    assert result.components["penalties"] == PENALTY_NO_DATE
    assert any("missing publication date" in reason for reason in result.reasons)


def test_threshold_boundary_is_inclusive() -> None:
    document = _document(source="web", metadata={"venue_type": "blog"})
    baseline = score_credibility(document, Settings())
    assert baseline.score < 1.0

    at_threshold = Settings(scrape_credibility_threshold=baseline.score)
    above_threshold = Settings(
        scrape_credibility_threshold=min(1.0, baseline.score + 0.01)
    )

    assert score_credibility(document, at_threshold).credible is True
    assert score_credibility(document, above_threshold).credible is False


def test_components_reconstruct_the_score_and_stay_in_range(settings: Settings) -> None:
    document = _document(
        metadata={
            "peer_reviewed": True,
            "venue_type": "journal",
            "cited_by_count": 1000,
        }
    )
    result = score_credibility(document, settings)

    weighted = sum(WEIGHTS[name] * result.components[name] for name in WEIGHTS)
    expected = min(1.0, max(0.0, weighted + result.components["penalties"]))

    assert result.score == pytest.approx(expected, abs=2e-6)
    assert set(result.components) == {
        "author",
        "venue",
        "source_type",
        "recency",
        "citation",
        "penalties",
    }
    assert 0.0 <= result.score <= 1.0

    maximum = score_credibility(
        _document(
            metadata={
                "peer_reviewed": True,
                "venue_type": "journal",
                "cited_by_count": 10_000,
            }
        ),
        settings,
        h_index=500,
    )
    assert 0.0 <= maximum.score <= 1.0


def test_apply_credibility_returns_a_copy_without_mutating_input(
    settings: Settings,
) -> None:
    document = _document(
        metadata={"peer_reviewed": True, "cited_by_count": 25}
    )
    before = document.model_dump()
    result = score_credibility(document, settings)

    updated = apply_credibility(document, result)

    assert updated is not document
    assert updated.credibility_score == result.score
    assert updated.credible == result.credible
    assert updated.credibility["score"] == result.score
    assert updated.credibility["components"] == result.components
    assert updated.credibility["reasons"] == result.reasons
    assert updated.metadata is not document.metadata

    assert document.model_dump() == before
    assert document.credibility_score == 0.0
    assert document.credible is False
    assert document.credibility == {}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"peer_reviewed": True}, 1.0),
        ({"venue_type": "journal"}, 1.0),
        ({"venue_type": "book chapter"}, 0.8),
        ({"venue_type": "preprint"}, 0.6),
        ({"venue_type": "dataset"}, 0.7),
        ({"venue_type": "news"}, 0.4),
        ({"venue_type": "blog"}, 0.2),
    ],
)
def test_metadata_drives_venue_signal(
    settings: Settings, metadata: dict[str, object], expected: float
) -> None:
    result = score_credibility(_document(metadata=metadata), settings)
    assert result.components["venue"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("openalex", 0.8),
        ("arxiv", 0.8),
        ("zenodo", 0.7),
        ("hf", 0.7),
        ("huggingface", 0.7),
        ("wikipedia", 0.5),
        ("web", 0.3),
        ("unknown-source", 0.3),
    ],
)
def test_source_type_signal(
    settings: Settings, source: str, expected: float
) -> None:
    result = score_credibility(_document(source=source), settings)
    assert result.components["source_type"] == pytest.approx(expected)


def test_predatory_venue_is_penalised(settings: Settings) -> None:
    flagged = score_credibility(
        _document(metadata={"peer_reviewed": True, "predatory": True}),
        settings,
    )
    typed = score_credibility(
        _document(metadata={"venue_type": "predatory"}),
        settings,
    )

    assert flagged.components["penalties"] == PENALTY_PREDATORY
    assert typed.components["penalties"] == PENALTY_PREDATORY
    assert any("predatory" in reason for reason in flagged.reasons)
