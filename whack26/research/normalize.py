"""Normalization, identity keys, deduplication, and ranking for research records."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from typing import Literal, Sequence

from ..embeddings import EmbeddingClient, EmbeddingError, cosine
from .base import (
    DATASET_SOURCES,
    PAPER_SOURCES,
    SOURCE_PRIOR,
    SOURCE_RANK,
    ResearchRecord,
    clean_text,
)

_DOI_PREFIXES: tuple[str, ...] = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
_DOI_TRAILING_PUNCTUATION = ".,;:"

# New-style arXiv ids: 2401.01234 (v2). Old-style: hep-th/9901001.
_ARXIV_NEW_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(
    r"(?<![a-z0-9])([a-z][a-z-]*(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?![a-z0-9])",
    re.IGNORECASE,
)
_URL_MARKERS: tuple[str, ...] = (
    "openalex.org/",
    "huggingface.co/datasets/",
    "huggingface.co/",
    "zenodo.org/records/",
    "zenodo.org/record/",
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

logger = logging.getLogger(__name__)


def normalize_doi(value: str | None) -> str | None:
    """Return a bare, lower-case DOI, or None when no DOI remains."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in _DOI_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                changed = True
                break
    cleaned = cleaned.rstrip(_DOI_TRAILING_PUNCTUATION).strip()
    while cleaned.endswith(")") and cleaned.count("(") < cleaned.count(")"):
        cleaned = cleaned[:-1].rstrip(_DOI_TRAILING_PUNCTUATION).strip()
    return cleaned or None


def _bare_source_id(value: str) -> str:
    """Strip a known repository URL prefix while preserving case."""
    text = clean_text(value)
    lowered = text.lower()
    for marker in _URL_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            text = text[index + len(marker) :]
            text = text.split("?", 1)[0].split("#", 1)[0].strip("/")
            break
    return text


def _extract_arxiv_id(
    values: Sequence[str | None], *, allow_old_style: bool
) -> str | None:
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        match = _ARXIV_NEW_RE.search(text)
        if match:
            return match.group(1).lower()
        if allow_old_style or "arxiv" in text.lower():
            match = _ARXIV_OLD_RE.search(text)
            if match:
                return match.group(1).lower()
    return None


def _title_key(record: ResearchRecord) -> str:
    title = clean_text(record.title).casefold()
    author = clean_text(record.authors[0]).casefold() if record.authors else ""
    year = str(record.year) if record.year is not None else ""
    digest = hashlib.sha256(f"{title}|{author}|{year}".encode("utf-8")).hexdigest()
    return f"title:{digest}"


def record_identity_keys(record: ResearchRecord) -> list[str]:
    """Return dedup keys in priority order, primary key first."""
    keys: list[str] = []

    doi = normalize_doi(record.doi)
    if doi:
        keys.append(f"doi:{doi}")

    arxiv_values: list[str | None] = [record.id, record.source_id, record.url]
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv_values.append(doi)
    arxiv_id = _extract_arxiv_id(
        arxiv_values, allow_old_style=record.source == "arxiv"
    )
    if arxiv_id is None and record.source == "arxiv":
        fallback = _bare_source_id(record.source_id)
        if fallback:
            arxiv_id = fallback.removeprefix("arxiv:").strip().lower()
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")

    source_id = _bare_source_id(record.source_id)
    if record.source == "openalex" and source_id:
        keys.append(f"openalex:{source_id}")
    elif record.source == "huggingface" and source_id:
        keys.append(f"hf:{source_id}")
    elif record.source == "zenodo" and source_id:
        keys.append(f"zenodo:{source_id}")

    keys.append(_title_key(record))
    return list(dict.fromkeys(keys))


def _winner_sort_key(record: ResearchRecord) -> tuple[int, int, int, int, int, str]:
    """Rank duplicate records by metadata quality, then source, then id.

    Abstract length is the strongest signal because a real abstract is the
    most reusable content downstream. Authors, year, and DOI follow; source
    rank only breaks genuine ties so a richer arXiv record can beat a thinner
    OpenAlex one. Note that identifiers are still derived from the winner.
    """
    abstract_length = len(record.abstract or "")
    author_count = len(record.authors)
    year = record.year if record.year is not None else -1
    has_doi = 1 if normalize_doi(record.doi) else 0
    return (
        -abstract_length,
        -author_count,
        -year,
        -has_doi,
        -SOURCE_RANK.get(record.source, 0),
        record.id,
    )


def _best_abstract(group: Sequence[ResearchRecord]) -> str | None:
    candidates = [record for record in group if record.abstract]
    if not candidates:
        return None
    return min(candidates, key=_winner_sort_key).abstract


def _best_authors(group: Sequence[ResearchRecord]) -> list[str]:
    candidates = [record for record in group if record.authors]
    if not candidates:
        return []
    winner = min(
        candidates,
        key=lambda record: (-len(record.authors), *_winner_sort_key(record)),
    )
    return list(winner.authors)


def _merge_group(group: Sequence[ResearchRecord]) -> ResearchRecord:
    winner = min(group, key=_winner_sort_key)
    losers = [record for record in group if record is not winner]

    # Merge the best abstract even when the winner already has one, so a
    # richer duplicate is not discarded just because it lost the winner sort.
    abstract = _best_abstract(group)
    authors = _best_authors(group)
    year = winner.year
    doi = normalize_doi(winner.doi)
    license_name = winner.license

    seen: list[str] = []

    def add_seen(value: str | None) -> None:
        if value and value != winner.id and value not in seen:
            seen.append(value)

    for value in winner.also_seen_in:
        add_seen(value)
    for loser in losers:
        if year is None and loser.year is not None:
            year = loser.year
        if doi is None and loser.doi:
            doi = normalize_doi(loser.doi)
        if license_name is None and loser.license:
            license_name = loser.license
        add_seen(loser.id)
        for value in loser.also_seen_in:
            add_seen(value)

    return winner.model_copy(
        deep=True,
        update={
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "doi": doi,
            "license": license_name,
            "also_seen_in": seen,
        },
    )


def deduplicate(records: Sequence[ResearchRecord]) -> list[ResearchRecord]:
    """Merge records that share any identity key, one record per group."""
    items = list(records)
    if not items:
        return []

    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[str, int] = {}
    for index, record in enumerate(items):
        for key in record_identity_keys(record):
            existing = owners.get(key)
            if existing is None:
                owners[key] = index
            else:
                union(existing, index)

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)

    ordered_groups = sorted(groups.values(), key=min)
    return [_merge_group([items[index] for index in group]) for group in ordered_groups]


def record_kind(record: ResearchRecord) -> Literal["paper", "dataset", "web"]:
    if record.source in PAPER_SOURCES:
        return "paper"
    if record.source in DATASET_SOURCES:
        return "dataset"
    return "web"


def _recency(year: int | None, now: datetime) -> float:
    effective_year = year if year is not None else now.year - 5
    age_years = max(0, now.year - effective_year)
    return math.exp(-age_years / 5.0)


def _embedding_text(record: ResearchRecord) -> str:
    abstract = (record.abstract or "").strip()
    return f"{record.title} {abstract}" if abstract else record.title


def _lexical_similarity(summary: str, text: str) -> float:
    """Documented lexical fallback when the embedding provider fails.

    Returns the fraction of distinct summary tokens that also occur in the
    record text, in ``[0, 1]``. It is deliberately crude; it preserves an
    ordering signal without pretending the embeddings were computed.
    """
    summary_tokens = set(_TOKEN_RE.findall(summary.lower()))
    if not summary_tokens:
        return 0.0
    text_tokens = set(_TOKEN_RE.findall(text.lower()))
    return len(summary_tokens & text_tokens) / len(summary_tokens)


def _ensure_kind(
    selected: list[ResearchRecord],
    ranked: Sequence[ResearchRecord],
    kind: Literal["paper", "dataset", "web"],
    *,
    top_k: int,
    protected: frozenset[str],
) -> list[ResearchRecord]:
    if any(record_kind(record) == kind for record in selected):
        return selected
    candidate = next(
        (record for record in ranked if record_kind(record) == kind), None
    )
    if candidate is None:
        return selected
    if len(selected) < top_k:
        return [*selected, candidate]
    if not selected:
        return selected

    replacement_index = next(
        (
            index
            for index in range(len(selected) - 1, -1, -1)
            if record_kind(selected[index]) not in protected
        ),
        None,
    )
    if replacement_index is None:
        # With a single slot a paper and a dataset cannot both fit; the
        # protected item (a paper) wins.
        if len(selected) <= 1:
            return selected
        replacement_index = len(selected) - 1

    replacement = list(selected)
    replacement[replacement_index] = candidate
    return replacement


def rank_records(
    records: Sequence[ResearchRecord],
    *,
    summary: str,
    embeddings: EmbeddingClient,
    top_k: int = 20,
    now: datetime | None = None,
    warnings: list[str] | None = None,
) -> list[ResearchRecord]:
    """Score, sort, and truncate records while keeping paper/dataset coverage.

    When embeddings fail, the ranking surfaces a warning (through ``warnings``
    when provided, and always through logging) and falls back to a documented
    lexical score based on summary-token coverage. It never silently treats a
    provider outage as zero similarity.
    """
    items = list(records)
    if not items or top_k <= 0:
        return []

    current = now or datetime.now(timezone.utc)
    texts = [summary, *(_embedding_text(record) for record in items)]
    try:
        vectors = embeddings.embed(texts)
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedding returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        similarities = [
            cosine(vectors[0], vector) for vector in vectors[1:]
        ]
    except EmbeddingError as exc:
        message = (
            "research_ranking_embeddings_failed: "
            f"{type(exc).__name__}: {exc}; fell back to lexical similarity"
        )
        if warnings is not None:
            warnings.append(message)
        logger.warning("%s", message)
        similarities = [
            _lexical_similarity(summary, _embedding_text(record))
            for record in items
        ]

    scored: list[ResearchRecord] = []
    for record, similarity in zip(items, similarities):
        score = (
            0.55 * similarity
            + 0.25 * SOURCE_PRIOR.get(record.source, 0.0)
            + 0.20 * _recency(record.year, current)
        )
        scored.append(record.model_copy(update={"relevance": round(score, 6)}))

    scored.sort(key=lambda record: (-record.relevance, record.id))
    selected = list(scored[:top_k])
    selected = _ensure_kind(
        selected, scored, "paper", top_k=top_k, protected=frozenset()
    )
    selected = _ensure_kind(
        selected, scored, "dataset", top_k=top_k, protected=frozenset({"paper"})
    )

    deduped: list[ResearchRecord] = []
    seen_ids: set[str] = set()
    for record in selected:
        if record.id in seen_ids:
            continue
        seen_ids.add(record.id)
        deduped.append(record)

    deduped.sort(key=lambda record: (-record.relevance, record.id))
    return deduped
