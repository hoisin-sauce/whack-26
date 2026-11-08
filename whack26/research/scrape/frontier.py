"""Breadth-first link frontier for semantic scraping.

The frontier walks documents by depth. A document whose summary is close
enough to the target allows all of its links to be queued; otherwise every
link is fetched and summarized before only the close-enough ones are queued.
Every run is bounded by depth, document, link fan-out, and fetch budgets.
"""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

from ...config import Settings
from ...embeddings import cosine
from .identity import canonicalize_url, url_hash as compute_url_hash
from .links import LinkCandidate, candidate_to_document, extract_links
from .models import Document, DocumentLink, ScrapeResult
from .store import DocumentStore

ProcessDocument = Callable[[Document], Document]
FetchLink = Callable[[LinkCandidate], Document | None]

_ALLOWED_RELATIONS = frozenset(
    {
        "references",
        "cites",
        "linked_from",
        "same_as",
        "dataset_paper",
        "search_result",
    }
)
_DEFAULT_RELATION = "references"
_YEAR_RE = re.compile(r"(\d{4})")


def _normalize_relation(relation: str | None) -> str:
    value = (relation or "").strip()
    return value if value in _ALLOWED_RELATIONS else _DEFAULT_RELATION


def _similarity(document: Document, target_embedding: list[float]) -> float:
    """Target similarity, tolerating missing or mismatched vectors."""
    if not document.summary_embedding or not target_embedding:
        return 0.0
    if len(document.summary_embedding) != len(target_embedding):
        return 0.0
    return cosine(document.summary_embedding, target_embedding)


def _recency(document: Document) -> float:
    year: int | None = None
    for candidate in (
        document.published_at,
        (document.metadata or {}).get("year"),
    ):
        if candidate is None:
            continue
        match = _YEAR_RE.search(str(candidate))
        if match:
            year = int(match.group(1))
            break
    if year is None:
        return 1.0
    age = max(0, datetime.now(timezone.utc).year - year)
    return math.exp(-age / 5.0)


def _priority(document: Document, target_embedding: list[float]) -> float:
    """``relevance * credibility * recency``; zero vectors sort last."""
    relevance = max(0.0, _similarity(document, target_embedding))
    credibility = document.credibility_score
    if credibility <= 0.0:
        credibility = 0.5
    return relevance * credibility * _recency(document)


@dataclass
class _QueueItem:
    document: Document
    depth: int
    processed: bool = False
    parent_id: str | None = None
    relation: str = _DEFAULT_RELATION
    url: str | None = None


class _Expansion:
    """Mutable state for one ``expand_frontier`` call."""

    def __init__(
        self,
        *,
        target_embedding: list[float],
        store: DocumentStore,
        process: ProcessDocument,
        fetch: FetchLink,
        settings: Settings,
        summary_threshold: float,
        link_threshold: float,
    ) -> None:
        self.target_embedding = list(target_embedding)
        self.store = store
        self.process = process
        self.fetch = fetch
        self.settings = settings
        self.summary_threshold = summary_threshold
        self.link_threshold = link_threshold

        self.max_depth = settings.scrape_max_depth
        self.max_resources = settings.scrape_max_resources
        self.max_links = settings.scrape_max_links_per_resource
        self.max_fetches = settings.scrape_max_fetches

        self.queue: list[tuple[int, float, int, _QueueItem]] = []
        self.order = 0
        self.visited: set[str] = set()
        self.documents: dict[str, Document] = {}
        self.links: dict[tuple[str, str, str], DocumentLink] = {}
        self.persisted: set[str] = set()
        self.known: dict[str, Document] = {}
        self.scheduled_by_url: dict[str, str] = {}
        self.scheduled_by_content: dict[str, str] = {}
        self.warnings: list[str] = []
        self.processed_count = 0
        self.fetch_count = 0
        self.duplicate_count = 0
        self.skipped_count = 0
        self.max_depth_reached = 0
        self.stop = False

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _known_document(self, document_id: str) -> Document | None:
        document = self.known.get(document_id)
        if document is not None:
            return document
        return self.store.get_document(document_id)

    def _find(
        self, document: Document, *, ignore_queued_self: bool = False
    ) -> tuple[Document | None, str | None]:
        """Find an already-known copy: URL hit first, then content hit."""
        by_url: Document | None = None
        if document.url_hash:
            by_url = self.store.find_document_by_url(document.url_hash)
            if by_url is None and document.url_hash in self.scheduled_by_url:
                scheduled = self._known_document(
                    self.scheduled_by_url[document.url_hash]
                )
                if scheduled is not None and not (
                    ignore_queued_self and scheduled.id == document.id
                ):
                    by_url = scheduled
            if by_url is not None:
                return by_url, "url"

        if document.content_sha512:
            by_content = self.store.find_document_by_content(document.content_sha512)
            if by_content is None and (
                document.content_sha512 in self.scheduled_by_content
            ):
                scheduled = self._known_document(
                    self.scheduled_by_content[document.content_sha512]
                )
                if scheduled is not None and not (
                    ignore_queued_self and scheduled.id == document.id
                ):
                    by_content = scheduled
            if by_content is not None:
                return by_content, "content"
        return None, None

    def _register(self, document: Document) -> None:
        self.known.setdefault(document.id, document)
        if document.url_hash:
            self.scheduled_by_url.setdefault(document.url_hash, document.id)
        if document.content_sha512:
            self.scheduled_by_content.setdefault(
                document.content_sha512, document.id
            )

    def _record_duplicate(self, document: Document, kind: str) -> None:
        self.duplicate_count += 1
        label = "content" if kind == "content" else "url"
        self.warn(f"scrape_duplicate_{label}:{document.canonical_url or document.id}")

    def _prepare(self, document: Document) -> Document:
        """Copy a document and fill canonical URL/url-hash identity."""
        prepared = document.model_copy(deep=True)
        if prepared.canonical_url:
            try:
                canonical = canonicalize_url(prepared.canonical_url)
            except ValueError:
                canonical = None
            if canonical:
                prepared.canonical_url = canonical
                prepared.url_hash = compute_url_hash(canonical)
        return prepared

    def _apply_process(self, document: Document) -> Document:
        processed = self.process(document.model_copy(deep=True))
        if not isinstance(processed, Document):
            raise TypeError("process() must return a Document")

        updates: dict[str, object] = {}
        if not processed.id:
            updates["id"] = document.id
        if not processed.canonical_url and document.canonical_url:
            updates["canonical_url"] = document.canonical_url
        if not processed.url_hash and document.url_hash:
            updates["url_hash"] = document.url_hash
        if not processed.content_sha512 and document.content_sha512:
            updates["content_sha512"] = document.content_sha512
        if not processed.title and document.title:
            updates["title"] = document.title
        if not processed.source:
            updates["source"] = document.source
        if updates:
            processed = processed.model_copy(update=updates)
        return processed

    def _persist(self, document: Document) -> Document:
        if document.id in self.persisted:
            return self.documents[document.id]
        stored = self.store.upsert_document(document)
        self.persisted.add(stored.id)
        self.documents[stored.id] = stored
        self.known[stored.id] = stored
        self.processed_count += 1
        return stored

    def _push(self, item: _QueueItem) -> None:
        priority = _priority(item.document, self.target_embedding)
        heapq.heappush(
            self.queue,
            (item.depth, -priority, self.order, item),
        )
        self.order += 1

    def _record_link(
        self,
        from_id: str,
        to_id: str,
        relation: str | None,
        url: str | None,
    ) -> None:
        normalized = _normalize_relation(relation)
        key = (from_id, to_id, normalized)
        if key in self.links:
            return
        self.links[key] = DocumentLink(
            from_id=from_id,
            to_id=to_id,
            relation=normalized,
            url=url,
        )

    # ------------------------------------------------------------------
    # frontier phases
    # ------------------------------------------------------------------
    def _enqueue_seed(self, seed: Document) -> None:
        prepared = self._prepare(seed)
        existing, hit = self._find(prepared)
        if existing is not None:
            self._record_duplicate(existing, hit or "url")
            document = existing
            already_processed = True
        else:
            self._register(prepared)
            document = prepared
            already_processed = False
        if document.id in self.visited or any(
            entry[3].document.id == document.id for entry in self.queue
        ):
            return
        self._register(document)
        self._push(
            _QueueItem(document=document, depth=0, processed=already_processed)
        )

    def _expand(self, parent: Document, depth: int) -> None:
        if depth >= self.max_depth or self.stop:
            return
        candidates = extract_links(parent, max_links=self.max_links)
        similarity = _similarity(parent, self.target_embedding)
        if similarity >= self.summary_threshold:
            self._follow_all(parent, candidates, depth)
        else:
            self._follow_filtered(parent, candidates, depth)

    def _follow_all(
        self, parent: Document, candidates: Iterable[LinkCandidate], depth: int
    ) -> None:
        for candidate in candidates:
            document = candidate_to_document(candidate)
            existing, hit = self._find(document)
            if existing is not None:
                self._record_duplicate(existing, hit or "url")
                self._record_link(
                    parent.id, existing.id, candidate.relation, candidate.url
                )
                continue

            self._register(document)
            self._record_link(
                parent.id, document.id, candidate.relation, candidate.url
            )
            self._push(
                _QueueItem(
                    document=document,
                    depth=depth + 1,
                    processed=False,
                    parent_id=parent.id,
                    relation=candidate.relation,
                    url=candidate.url,
                )
            )

    def _follow_filtered(
        self, parent: Document, candidates: Iterable[LinkCandidate], depth: int
    ) -> None:
        for candidate in candidates:
            if self.stop:
                break

            document = candidate_to_document(candidate)
            existing, hit = self._find(document)
            if existing is not None:
                self._record_duplicate(existing, hit or "url")
                self._record_link(
                    parent.id, existing.id, candidate.relation, candidate.url
                )
                continue

            if self.fetch_count >= self.max_fetches:
                self.warn("scrape_fetch_budget_exhausted")
                self.stop = True
                break

            self.fetch_count += 1
            try:
                fetched = self.fetch(candidate.model_copy(deep=True))
            except Exception as exc:  # noqa: BLE001 - an adapter failure is a warning
                self.warn(
                    f"scrape_fetch_failed:{candidate.url}:{type(exc).__name__}"
                )
                continue

            if fetched is None:
                self.warn(f"scrape_fetch_failed:{candidate.url}")
                continue

            fetched = self._prepare(fetched)
            existing, hit = self._find(fetched)
            if existing is not None:
                self._record_duplicate(existing, hit or "url")
                relation = (
                    "same_as" if hit == "content" else candidate.relation
                )
                self._record_link(parent.id, existing.id, relation, candidate.url)
                continue

            if self.processed_count >= self.max_resources:
                self.warn("scrape_budget_exhausted")
                self.stop = True
                break

            processed = self._apply_process(fetched)
            stored = self._persist(processed)
            self._register(stored)
            self._record_link(parent.id, stored.id, candidate.relation, candidate.url)

            if _similarity(stored, self.target_embedding) >= self.link_threshold:
                self._push(
                    _QueueItem(
                        document=stored,
                        depth=depth + 1,
                        processed=True,
                        parent_id=parent.id,
                        relation=candidate.relation,
                        url=candidate.url,
                    )
                )
            else:
                self.skipped_count += 1
                self.warn(
                    f"scrape_skipped_threshold:{candidate.url}"
                )

    def run(self, seeds: Sequence[Document]) -> ScrapeResult:
        for seed in seeds:
            self._enqueue_seed(seed)

        while self.queue:
            _, _, _, item = heapq.heappop(self.queue)
            document = item.document
            if document.id in self.visited:
                continue

            if self.processed_count >= self.max_resources:
                self.warn("scrape_budget_exhausted")
                break

            self.visited.add(document.id)
            self.max_depth_reached = max(self.max_depth_reached, item.depth)

            if not item.processed:
                # Another document processed earlier in this run may have
                # resolved the same content; never summarize it twice.
                existing, hit = self._find(document, ignore_queued_self=True)
                if existing is not None and existing.id != document.id:
                    self._record_duplicate(existing, hit or "url")
                    if item.parent_id:
                        relation = (
                            "same_as" if hit == "content" else item.relation
                        )
                        self._record_link(
                            item.parent_id, existing.id, relation, item.url
                        )
                    if existing.id in self.persisted:
                        continue
                    document = existing
                    item = _QueueItem(
                        document=existing,
                        depth=item.depth,
                        processed=True,
                        parent_id=item.parent_id,
                        relation=item.relation,
                        url=item.url,
                    )

            if item.processed:
                if document.id not in self.persisted:
                    document = self._persist(document)
            else:
                document = self._apply_process(document)
                document = self._persist(document)

            self._expand(document, item.depth)
            if self.stop:
                break

        persisted_links: list[DocumentLink] = []
        for link in self.links.values():
            if (
                self.store.get_document(link.from_id) is None
                or self.store.get_document(link.to_id) is None
            ):
                self.warn(f"scrape_link_target_missing:{link.to_id}")
                continue
            self.store.add_document_link(link)
            persisted_links.append(link)

        stats: dict[str, int | float] = {
            "documents": len(self.documents),
            "links": len(persisted_links),
            "processed": self.processed_count,
            "fetches": self.fetch_count,
            "duplicates": self.duplicate_count,
            "skipped_threshold": self.skipped_count,
            "warnings": len(self.warnings),
            "max_depth_reached": self.max_depth_reached,
        }
        return ScrapeResult(
            documents=list(self.documents.values()),
            links=persisted_links,
            warnings=self.warnings,
            stats=stats,
        )


def expand_frontier(
    seeds: Sequence[Document],
    *,
    target_embedding: list[float],
    store: DocumentStore,
    process: ProcessDocument,
    fetch: FetchLink,
    settings: Settings,
    summary_threshold: float | None = None,
    link_threshold: float | None = None,
) -> ScrapeResult:
    """Expand ``seeds`` into a deduplicated, relevance-gated link frontier."""
    summary_value = (
        settings.scrape_summary_threshold
        if summary_threshold is None
        else summary_threshold
    )
    link_value = (
        settings.scrape_link_threshold
        if link_threshold is None
        else link_threshold
    )
    for name, value in (
        ("summary_threshold", summary_value),
        ("link_threshold", link_value),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    expansion = _Expansion(
        target_embedding=target_embedding,
        store=store,
        process=process,
        fetch=fetch,
        settings=settings,
        summary_threshold=float(summary_value),
        link_threshold=float(link_value),
    )
    return expansion.run(list(seeds))
