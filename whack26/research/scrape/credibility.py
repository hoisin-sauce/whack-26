"""Deterministic document credibility scoring.

The score is the plan's weighted sum of five transparent signals plus a small
set of explicit penalties:

    score = 0.35 * author
          + 0.25 * venue
          + 0.20 * source_type
          + 0.10 * recency
          + 0.10 * citation
          - penalties

Nothing here calls an LLM or the network. Every signal is derived from the
document itself (authors, publisher, source, published date, metadata) and, for
author-level citation metrics, the optional ``citation_count``/``h_index``
arguments. When a signal is unavailable we use the conservative defaults from
section 9 of ``docs/spec/semantic-scraping.md`` rather than guessing.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone

from whack26.config import Settings

from .models import CredibilityResult, Document

# Weights from the plan. They intentionally sum to 1.0.
WEIGHTS: dict[str, float] = {
    "author": 0.35,
    "venue": 0.25,
    "source_type": 0.20,
    "recency": 0.10,
    "citation": 0.10,
}

PENALTY_RETRACTED = -1.0
PENALTY_PREDATORY = -0.5
PENALTY_NO_DATE = -0.1

_AUTHOR_NONE = 0.2
_AUTHOR_UNKNOWN = 0.5
_AUTHOR_H_INDEX_FLOOR = 0.5
_AUTHOR_H_INDEX_SCALE = 40.0
_AUTHOR_CITATION_SCALE = 10_000.0

_VENUE_SIGNALS: dict[str, float] = {
    "journal": 1.0,
    "journal_article": 1.0,
    "peer_reviewed": 1.0,
    "peer_reviewed_journal": 1.0,
    "conference": 1.0,
    "conference_paper": 1.0,
    "proceedings": 1.0,
    "book": 0.8,
    "book_chapter": 0.8,
    "chapter": 0.8,
    "preprint": 0.6,
    "dataset": 0.7,
    "repository": 0.7,
    "data_repository": 0.7,
    "news": 0.4,
    "newspaper": 0.4,
    "news_article": 0.4,
    "blog": 0.2,
    "web": 0.2,
    "website": 0.2,
    "unknown": 0.2,
}

_SOURCE_SIGNALS: dict[str, float] = {
    "openalex": 0.8,
    "arxiv": 0.8,
    "zenodo": 0.7,
    "hf": 0.7,
    "huggingface": 0.7,
    "wikipedia": 0.5,
    "web": 0.3,
}

_PREPRINT_MARKERS = (
    "arxiv",
    "biorxiv",
    "medrxiv",
    "preprint",
    "research square",
    "ssrn",
)
_REPOSITORY_MARKERS = (
    "zenodo",
    "figshare",
    "huggingface",
    "dryad",
    "dataverse",
    "mendeley data",
)
_NEWS_MARKERS = (
    "news",
    "newspaper",
    "reuters",
    "associated press",
    "bbc",
    "cnn",
    "npr",
    "guardian",
    "washington post",
    "new york times",
    "the economist",
    "tribune",
    "herald",
)
_BLOG_MARKERS = (
    "blog",
    "medium",
    "substack",
    "wordpress",
    "blogspot",
    "tumblr",
)
_SCHOLARLY_MARKERS = (
    "nature",
    "science",
    "cell press",
    "elsevier",
    "springer",
    "wiley",
    "taylor & francis",
    "sage",
    "oxford university press",
    "cambridge university press",
    "ieee",
    "acm",
    "plos",
    "bmj",
    "lancet",
    "nejm",
    "jama",
    "annual reviews",
)


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _token(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _parse_date(value: object) -> date | None:
    """Parse the date shapes found in research metadata.

    Accepts datetimes, ``YYYY-MM-DD``, ``YYYY-MM``, and a bare year. A bare
    year is anchored to mid-year so the recency signal is not biased by the
    arbitrary choice of January.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if len(text) >= 10:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), 15)
        except ValueError:
            return None
    match = re.fullmatch(r"(\d{4})", text)
    if match:
        try:
            return date(int(match.group(1)), 7, 1)
        except ValueError:
            return None
    return None


def _published_date(document: Document) -> date | None:
    published = _parse_date(document.published_at)
    if published is not None:
        return published
    metadata = document.metadata or {}
    for key in ("published_at", "publication_date", "year", "publication_year"):
        published = _parse_date(metadata.get(key))
        if published is not None:
            return published
    return None


def _author_signal(
    document: Document,
    citation_count: int | None,
    h_index: int | None,
) -> tuple[float, str]:
    authors = [str(author).strip() for author in document.authors if str(author).strip()]
    if not authors:
        return _AUTHOR_NONE, "no authors listed"

    metadata = document.metadata or {}
    resolved_h_index = _as_int(h_index)
    if resolved_h_index is None:
        resolved_h_index = _as_int(metadata.get("author_h_index"))
    if resolved_h_index is None:
        resolved_h_index = _as_int(metadata.get("h_index"))

    if resolved_h_index is not None:
        capped = min(max(resolved_h_index, 0) / _AUTHOR_H_INDEX_SCALE, 1.0)
        signal = _AUTHOR_H_INDEX_FLOOR + (1.0 - _AUTHOR_H_INDEX_FLOOR) * capped
        return signal, f"author h-index {resolved_h_index}"

    resolved_citations = _as_int(citation_count)
    if resolved_citations is None:
        resolved_citations = _as_int(metadata.get("author_citation_count"))
    if resolved_citations is not None:
        capped = min(
            math.log1p(max(resolved_citations, 0)) / math.log1p(_AUTHOR_CITATION_SCALE),
            1.0,
        )
        signal = _AUTHOR_H_INDEX_FLOOR + (1.0 - _AUTHOR_H_INDEX_FLOOR) * capped
        return signal, f"author citation count {resolved_citations}"

    if _as_bool(metadata.get("author_known")) is True:
        return 0.8, "known author"

    return _AUTHOR_UNKNOWN, "author identity not verified"


def _venue_from_publisher(
    document: Document, *, include_scholarly: bool
) -> tuple[float, str] | None:
    haystack = " ".join(
        part for part in (document.publisher, document.source) if part
    ).lower()
    if not haystack:
        return None
    if any(marker in haystack for marker in _PREPRINT_MARKERS):
        return 0.6, "preprint publisher"
    if any(marker in haystack for marker in _REPOSITORY_MARKERS):
        return 0.7, "dataset repository"
    if any(marker in haystack for marker in _NEWS_MARKERS):
        return 0.4, "news publisher"
    if any(marker in haystack for marker in _BLOG_MARKERS):
        return 0.2, "blog publisher"
    if include_scholarly and any(marker in haystack for marker in _SCHOLARLY_MARKERS):
        return 0.8, "scholarly publisher"
    return None


def _venue_signal(document: Document) -> tuple[float, str, bool]:
    metadata = document.metadata or {}
    venue_type = _token(metadata.get("venue_type") or metadata.get("venue"))

    predatory = (
        _as_bool(metadata.get("predatory")) is True
        or _as_bool(metadata.get("predatory_venue")) is True
        or venue_type == "predatory"
    )
    if venue_type == "predatory":
        return 0.2, "predatory venue", True
    if venue_type in _VENUE_SIGNALS:
        return _VENUE_SIGNALS[venue_type], f"venue type {venue_type}", predatory

    peer_reviewed = _as_bool(metadata.get("peer_reviewed"))
    if peer_reviewed is None:
        peer_reviewed = _as_bool(metadata.get("is_peer_reviewed"))
    if peer_reviewed is True:
        return 1.0, "peer-reviewed venue", predatory

    inferred = _venue_from_publisher(
        document, include_scholarly=peer_reviewed is None
    )
    if inferred is not None:
        return inferred[0], inferred[1], predatory
    if peer_reviewed is False:
        return 0.2, "venue not peer reviewed", predatory
    return 0.2, "venue unknown", predatory


def _source_type_signal(document: Document) -> tuple[float, str]:
    source = _token(document.source)
    if source in _SOURCE_SIGNALS:
        return _SOURCE_SIGNALS[source], f"source type {source}"
    return 0.3, f"unknown source type {source or 'unset'}"


def _recency_signal(document: Document, today: date) -> tuple[float, str, bool]:
    published = _published_date(document)
    if published is None:
        # A missing date is treated like a five-year-old document, matching the
        # existing research normalizer, and also carries the explicit penalty.
        return math.exp(-1.0), "no publication date; assumed five years old", True
    age_years = max(0.0, (today - published).days / 365.25)
    return math.exp(-age_years / 5.0), f"published {age_years:.1f} years ago", False


def _citation_signal(document: Document) -> tuple[float, str]:
    metadata = document.metadata or {}
    cited_by = _as_int(metadata.get("cited_by_count"))
    if cited_by is None:
        return 0.0, "no citation count"
    cited_by = max(cited_by, 0)
    signal = min(math.log1p(cited_by) / math.log1p(1000), 1.0)
    return signal, f"{cited_by} cited-by count"


def score_credibility(
    document: Document,
    settings: Settings,
    *,
    citation_count: int | None = None,
    h_index: int | None = None,
) -> CredibilityResult:
    """Score a document with deterministic, decomposable signals.

    ``citation_count`` and ``h_index`` are author-level metrics (for example
    from OpenAlex or Semantic Scholar). Document-level citations come from
    ``document.metadata["cited_by_count"]``.
    """
    reasons: list[str] = []

    author, author_reason = _author_signal(document, citation_count, h_index)
    venue, venue_reason, predatory = _venue_signal(document)
    source_type, source_reason = _source_type_signal(document)
    recency, recency_reason, missing_date = _recency_signal(
        document, datetime.now(timezone.utc).date()
    )
    citation, citation_reason = _citation_signal(document)

    components: dict[str, float] = {
        "author": author,
        "venue": venue,
        "source_type": source_type,
        "recency": recency,
        "citation": citation,
    }

    reasons.append(f"author {author:.3f}: {author_reason}")
    reasons.append(f"venue {venue:.3f}: {venue_reason}")
    reasons.append(f"source_type {source_type:.3f}: {source_reason}")
    reasons.append(f"recency {recency:.3f}: {recency_reason}")
    reasons.append(f"citation {citation:.3f}: {citation_reason}")

    metadata = document.metadata or {}
    retracted = _as_bool(metadata.get("retracted"))
    if retracted is None:
        retracted = _as_bool(metadata.get("is_retracted"))

    penalties = 0.0
    if retracted is True:
        penalties += PENALTY_RETRACTED
        reasons.append(f"penalty {PENALTY_RETRACTED:.2f}: retracted")
    if predatory:
        penalties += PENALTY_PREDATORY
        reasons.append(f"penalty {PENALTY_PREDATORY:.2f}: predatory venue")
    if missing_date:
        penalties += PENALTY_NO_DATE
        reasons.append(f"penalty {PENALTY_NO_DATE:.2f}: missing publication date")
    components["penalties"] = penalties

    weighted = sum(WEIGHTS[name] * components[name] for name in WEIGHTS)
    raw_score = weighted + penalties
    score = round(min(1.0, max(0.0, raw_score)), 6)
    threshold = settings.scrape_credibility_threshold
    credible = score >= threshold
    reasons.append(
        f"score {score:.3f} {'>=' if credible else '<'} threshold "
        f"{threshold:.3f}: {'credible' if credible else 'not credible'}"
    )

    return CredibilityResult(
        score=score,
        credible=credible,
        components=components,
        reasons=reasons,
    )


def apply_credibility(document: Document, result: CredibilityResult) -> Document:
    """Return a copy of ``document`` carrying the credibility result.

    The input document and its nested metadata are never mutated.
    """
    updated = document.model_copy(deep=True)
    updated.credibility_score = result.score
    updated.credible = result.credible
    updated.credibility = result.model_dump()
    return updated


__all__ = [
    "PENALTY_NO_DATE",
    "PENALTY_PREDATORY",
    "PENALTY_RETRACTED",
    "WEIGHTS",
    "apply_credibility",
    "score_credibility",
]
