"""Extract followable link candidates from document metadata.

API records describe their neighbours in several shapes: plain URLs, dicts
with ``url``/``href``/``link``, OpenAlex work ids, arXiv ids, and DOIs. This
module normalizes all of them into :class:`LinkCandidate` rows and turns a
candidate into a pending :class:`Document`. It never mutates the source
document.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterator

from pydantic import BaseModel, Field

from .identity import canonicalize_url, metadata_identity, url_hash
from .models import Document, DocumentKind

LINK_METADATA_KEYS = (
    "links",
    "references",
    "referenced_works",
    "urls",
    "citations",
)

_ID_PREFIXES: dict[str, str] = {
    "dataset": "ds_",
    "search": "srch_",
    "paper": "pap_",
    "web": "web_",
}

_DOI_RE = re.compile(r"^(?:doi:\s*)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_DOI_URL_RE = re.compile(
    r"^https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)$", re.IGNORECASE
)
_OPENALEX_RE = re.compile(r"^(?:https?://(?:api\.)?openalex\.org/(?:works/)?)?(W\d+)$")
_ARXIV_RE = re.compile(
    r"^(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)?"
    r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)

_RECOGNIZED_KEYS = frozenset(
    {
        "url",
        "href",
        "link",
        "id",
        "doi",
        "arxiv",
        "arxiv_id",
        "openalex",
        "openalex_id",
        "title",
        "name",
        "source",
        "provider",
        "source_id",
        "abstract",
        "summary",
        "description",
        "authors",
        "author",
        "year",
        "publication_year",
        "relation",
        "metadata",
    }
)


class LinkCandidate(BaseModel):
    """A potential neighbour discovered in a document's metadata."""

    url: str
    title: str | None = None
    relation: str = "references"
    source: str | None = None
    source_id: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _coerce_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1000 <= value <= 3000 else None
    text = str(value).strip()
    match = re.match(r"^(\d{4})", text)
    if match:
        year = int(match.group(1))
        return year if 1000 <= year <= 3000 else None
    return None


def _coerce_authors(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    authors: list[str] = []
    for entry in value:
        if isinstance(entry, dict):
            name = entry.get("display_name") or entry.get("name")
            if name:
                authors.append(str(name).strip())
        elif entry is not None:
            text = str(entry).strip()
            if text:
                authors.append(text)
    return authors


def _normalize_doi(raw: str) -> str | None:
    text = raw.strip()
    match = _DOI_URL_RE.match(text) or _DOI_RE.match(text)
    if not match:
        return None
    doi = match.group(1).rstrip(".,;:)")
    return doi.lower() if doi else None


def _normalize_url(raw: str) -> str | None:
    """Return a canonical followable URL for any supported identifier."""
    text = (raw or "").strip()
    if not text:
        return None

    doi = _normalize_doi(text)
    if doi is not None:
        return canonicalize_url(f"https://doi.org/{doi}")

    match = _OPENALEX_RE.match(text)
    if match:
        work = match.group(1).upper()
        return canonicalize_url(f"https://openalex.org/{work}")

    match = _ARXIV_RE.match(text)
    if match:
        arxiv_id = match.group(1)
        return canonicalize_url(f"https://arxiv.org/abs/{arxiv_id}")

    try:
        return canonicalize_url(text)
    except ValueError:
        return None


def _item_url(item: dict[str, Any]) -> str | None:
    for key in ("url", "href", "link", "openalex_id", "openalex", "arxiv_id", "arxiv"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_url(value)
            if normalized:
                return normalized
    doi = item.get("doi")
    if isinstance(doi, str) and doi.strip():
        normalized = _normalize_doi(doi)
        if normalized:
            return canonicalize_url(f"https://doi.org/{normalized}")
    for key in ("id", "source_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            normalized = _normalize_url(value)
            if normalized:
                return normalized
    return None


def _candidate_from_item(item: Any, document: Document) -> LinkCandidate | None:
    if isinstance(item, str):
        url = _normalize_url(item)
        if url is None:
            return None
        doi = _normalize_doi(item) or _normalize_doi(url)
        return LinkCandidate(
            url=url,
            source=document.source or None,
            doi=doi,
        )

    if not isinstance(item, dict):
        return None

    url = _item_url(item)
    if url is None:
        return None

    doi = item.get("doi")
    doi_text = _normalize_doi(doi) if isinstance(doi, str) else None
    if doi_text is None:
        doi_text = _normalize_doi(url)

    title = item.get("title") or item.get("name")
    abstract = item.get("abstract") or item.get("summary") or item.get("description")
    authors = _coerce_authors(item.get("authors") or item.get("author"))
    year = _coerce_year(item.get("year") or item.get("publication_year"))
    relation = item.get("relation") or "references"
    source = item.get("source") or item.get("provider") or document.source or None
    source_id = item.get("source_id") or item.get("id")
    if source_id is not None:
        source_id = str(source_id)

    metadata = copy.deepcopy(dict(item.get("metadata") or {}))
    for key, value in item.items():
        if key not in _RECOGNIZED_KEYS and key not in metadata:
            metadata[key] = value

    return LinkCandidate(
        url=url,
        title=str(title).strip() if title else None,
        relation=str(relation),
        source=str(source) if source else None,
        source_id=source_id,
        abstract=str(abstract).strip() if abstract else None,
        authors=authors,
        year=year,
        doi=doi_text,
        metadata=metadata,
    )


def _iter_items(value: Any) -> Iterator[Any]:
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for entry in value:
            yield from _iter_items(entry)
        return
    yield value


def _merge(existing: LinkCandidate, incoming: LinkCandidate) -> LinkCandidate:
    """Keep the first URL but fill gaps from a later, richer duplicate."""
    merged = existing.model_copy(deep=True)
    merged.title = merged.title or incoming.title
    merged.source = merged.source or incoming.source
    merged.source_id = merged.source_id or incoming.source_id
    merged.abstract = merged.abstract or incoming.abstract
    if not merged.authors:
        merged.authors = list(incoming.authors)
    merged.year = merged.year if merged.year is not None else incoming.year
    merged.doi = merged.doi or incoming.doi
    if merged.relation == "references" and incoming.relation != "references":
        merged.relation = incoming.relation
    for key, value in incoming.metadata.items():
        merged.metadata.setdefault(key, value)
    return merged


def extract_links(document: Document, *, max_links: int = 25) -> list[LinkCandidate]:
    """Return deduplicated link candidates found in ``document.metadata``.

    ``document`` is not modified. Unsupported or malformed entries are
    skipped. The result preserves discovery order and is capped at
    ``max_links`` after deduplication.
    """
    if max_links <= 0:
        return []

    metadata = document.metadata or {}
    candidates: list[LinkCandidate] = []
    seen: dict[str, int] = {}

    for key in LINK_METADATA_KEYS:
        for item in _iter_items(metadata.get(key)):
            candidate = _candidate_from_item(item, document)
            if candidate is None:
                continue
            try:
                identity = canonicalize_url(candidate.url)
            except ValueError:
                continue
            candidate.url = identity
            if identity in seen:
                index = seen[identity]
                candidates[index] = _merge(candidates[index], candidate)
                continue
            seen[identity] = len(candidates)
            candidates.append(candidate)

    return candidates[:max_links]


def candidate_to_document(
    candidate: LinkCandidate, *, kind: DocumentKind = "paper"
) -> Document:
    """Create a pending document for a link candidate.

    The id is derived from the canonical URL hash so the same target always
    maps to the same document. The document has no summary yet and is marked
    ``new``; metadata is deep-copied from the candidate.
    """
    url = canonicalize_url(candidate.url)
    digest = url_hash(url)
    prefix = _ID_PREFIXES.get(kind, "doc_")
    metadata = copy.deepcopy(dict(candidate.metadata or {}))

    for key, value in (
        ("doi", candidate.doi),
        ("year", candidate.year),
        ("abstract", candidate.abstract),
    ):
        if value is not None and key not in metadata:
            metadata[key] = value

    title = candidate.title or url
    content_hash = metadata_identity(
        title=title,
        abstract=candidate.abstract,
        authors=candidate.authors,
        year=candidate.year,
        doi=candidate.doi,
        canonical_url=url,
    )

    return Document(
        id=f"{prefix}{digest}",
        kind=kind,
        source=candidate.source or "web",
        source_id=candidate.source_id,
        title=title,
        summary="",
        canonical_url=url,
        url_hash=digest,
        content_sha512=content_hash,
        authors=list(candidate.authors),
        metadata=metadata,
        status="new",
    )
