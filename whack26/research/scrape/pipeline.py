"""Semantic-scraping orchestrator.

The orchestrator turns the one-hop research records into documents, expands
their link frontier with the configured budgets and relevance gates, and
persists the resulting catalog.  The heavy lifting lives in the sibling
modules; this file only sequences them and keeps a failing resource from
aborting the run.

The imports of the sibling modules are deliberately inside
:func:`run_semantic_scraping`: :mod:`whack26.store` imports this package's
models, so importing the classifiers and frontier at module load time would
create a cycle through the field store.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ...common import redact_secrets
from ...config import Settings
from ..base import DATASET_SOURCES, ResearchRecord
from .identity import canonicalize_url, metadata_identity, title_identity, url_hash
from .models import (
    Document,
    DocumentComparison,
    DocumentCritique,
    ScrapeResult,
)

ID_PREFIXES = {"dataset": "ds_", "search": "srch_", "paper": "pap_", "web": "web_"}
DATASET_DOCUMENT_SOURCES = frozenset({"user", "local", "dataset"})

__all__ = ["run_semantic_scraping"]


def _llm_call_count(llm: Any) -> int:
    calls = getattr(llm, "calls", None)
    return len(calls) if isinstance(calls, list) else 0


def _error(exc: BaseException) -> str:
    return redact_secrets(f"{type(exc).__name__}: {exc}")


def _document_id(kind: str, canonical: str | None, content_hash: str | None) -> str:
    digest: str | None = None
    if canonical:
        try:
            digest = url_hash(canonical)
        except ValueError:
            digest = None
    if digest is None:
        digest = content_hash
    if digest is None:  # pragma: no cover - seed documents always have a title
        digest = "unknown"
    return f"{ID_PREFIXES.get(kind, 'doc_')}{digest}"


def _record_to_document(record: ResearchRecord) -> Document:
    """Convert one one-hop research record into a pending document."""
    try:
        canonical = canonicalize_url(record.url)
    except ValueError:
        canonical = None

    content_hash = metadata_identity(
        title=record.title,
        abstract=record.abstract,
        authors=record.authors,
        year=record.year,
        doi=record.doi,
        canonical_url=canonical,
    )
    kind = "dataset" if record.source in DATASET_SOURCES else "paper"
    metadata: dict[str, Any] = {
        "query": record.query,
        "relevance": float(record.relevance),
        "retrieved_at": record.retrieved_at,
    }
    if record.abstract:
        metadata["abstract"] = record.abstract
    if record.year is not None:
        metadata["year"] = record.year
    if record.doi:
        metadata["doi"] = record.doi
    if record.also_seen_in:
        metadata["also_seen_in"] = list(record.also_seen_in)

    return Document(
        id=_document_id(kind, canonical, content_hash),
        kind=kind,  # type: ignore[arg-type]
        source=record.source,
        source_id=record.source_id,
        title=record.title,
        summary="",
        canonical_url=canonical,
        url_hash=url_hash(canonical) if canonical else None,
        content_sha512=content_hash,
        authors=list(record.authors),
        published_at=str(record.year) if record.year is not None else None,
        license=record.license,
        metadata=metadata,
        status="new",
    )


def _normalise_document(document: Document) -> Document:
    """Fill missing canonical URL, URL hash, and content hash fields."""
    updates: dict[str, Any] = {}
    canonical = document.canonical_url
    if canonical:
        try:
            canonical = canonicalize_url(canonical)
        except ValueError:
            canonical = None
    if canonical:
        updates["canonical_url"] = canonical
        if not document.url_hash:
            updates["url_hash"] = url_hash(canonical)

    if not document.content_sha512:
        year: int | None = None
        raw_year = (document.metadata or {}).get("year")
        if isinstance(raw_year, int) and not isinstance(raw_year, bool):
            year = raw_year
        elif document.published_at:
            try:
                year = int(str(document.published_at)[:4])
            except ValueError:
                year = None
        updates["content_sha512"] = metadata_identity(
            title=document.title,
            abstract=(document.metadata or {}).get("abstract"),
            authors=document.authors,
            year=year,
            doi=(document.metadata or {}).get("doi"),
            canonical_url=canonical,
        )

    if not document.id:
        updates["id"] = _document_id(
            document.kind,
            updates.get("canonical_url", canonical),
            updates.get("content_sha512", document.content_sha512),
        )

    return document.model_copy(update=updates) if updates else document


def _seed_to_document(seed: ResearchRecord | Document) -> Document:
    if isinstance(seed, Document):
        return _normalise_document(seed)
    if isinstance(seed, ResearchRecord):
        return _record_to_document(seed)
    raise TypeError(
        "seeds must be ResearchRecord or Document instances, "
        f"got {type(seed).__name__}"
    )


def _target_dataset_document(
    documents: Sequence[Document],
) -> Document | None:
    """Return the seed that represents the user's own dataset, if any."""
    explicit: list[Document] = []
    for document in documents:
        metadata = document.metadata or {}
        for key in ("dataset_document", "target_dataset_document", "dataset"):
            inline = metadata.get(key)
            if isinstance(inline, Document):
                return _normalise_document(inline)
        if document.kind != "dataset":
            continue
        if (
            document.source in DATASET_DOCUMENT_SOURCES
            or metadata.get("is_target_dataset") is True
        ):
            explicit.append(document)
    return explicit[0] if explicit else None


def _fallback_summary(document: Document) -> str:
    metadata = document.metadata or {}
    for key in ("abstract", "description", "summary"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:1200]
    return " ".join((document.title or document.id).split())[:1200]


def _merge_summary_metadata(document: Document, summary: Any) -> Document:
    metadata = dict(document.metadata or {})
    for key, value in (
        ("claims", summary.claims),
        ("methods", summary.methods),
        ("population", summary.population),
        ("time_period", summary.time_period),
        ("credibility_signals", summary.credibility_signals),
    ):
        if value and key not in metadata:
            metadata[key] = value

    signals = summary.credibility_signals or {}
    for key in ("peer_reviewed", "retracted"):
        if key in signals and key not in metadata:
            metadata[key] = signals[key]

    updates: dict[str, Any] = {"metadata": metadata}
    publisher = signals.get("publisher")
    if not document.publisher and isinstance(publisher, str) and publisher.strip():
        updates["publisher"] = publisher.strip()
    return document.model_copy(update=updates)


def _apply_credibility(
    document: Document, *, settings: Settings, warnings: list[str]
) -> Document:
    from .credibility import apply_credibility, score_credibility

    try:
        return apply_credibility(document, score_credibility(document, settings))
    except Exception as exc:  # noqa: BLE001 - credibility is advisory
        warnings.append(f"scrape_credibility_failed:{document.id}: {_error(exc)}")
        return document


def _process_document(
    document: Document,
    *,
    target: str,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    warnings: list[str],
) -> Document:
    """Summarize, embed, and credibility-score one document."""
    from .process import process_document
    from .summarize import summarize_document
    from .models import SummaryResult

    try:
        summary = summarize_document(document, target, llm, settings)
    except Exception as exc:  # noqa: BLE001 - fall back to local metadata
        warnings.append(f"scrape_summary_failed:{document.id}: {_error(exc)}")
        summary = SummaryResult(summary=_fallback_summary(document))

    try:
        processed = process_document(
            document,
            target,
            llm,
            embeddings,
            settings,
            summary=summary,
        )
    except Exception as exc:  # noqa: BLE001 - keep the document with its summary
        warnings.append(f"scrape_embedding_failed:{document.id}: {_error(exc)}")
        processed = document.model_copy(
            update={"summary": summary.summary, "status": "summarized"}
        )

    processed = _merge_summary_metadata(processed, summary)
    processed = _apply_credibility(processed, settings=settings, warnings=warnings)
    return processed.model_copy(update={"status": "summarized"})


def _classify_document(
    document: Document,
    *,
    store: Any,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    warnings: list[str],
) -> Document:
    from .classify import classify_document

    try:
        path = classify_document(
            document,
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - a missing path must not stop the run
        warnings.append(f"scrape_classification_failed:{document.id}: {_error(exc)}")
        path = []

    status = "classified" if path else document.status
    try:
        return store.upsert_document(document.model_copy(update={"status": status}))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"scrape_store_failed:{document.id}: {_error(exc)}")
        return document


def _compare_with_dataset(
    document: Document,
    dataset_document: Document,
    *,
    store: Any,
    llm: Any,
    settings: Settings,
    warnings: list[str],
) -> tuple[DocumentComparison | None, bool]:
    from .compare import (
        SIMILARITY_TOLERANCE,
        canonical_pair,
        compare_with_dataset,
        deterministic_similarity,
    )

    left, right = canonical_pair(document.id, dataset_document.id)
    try:
        cached = store.get_comparison(
            left, right, "dataset_relevance", settings.scrape_method_version
        )
    except Exception:  # noqa: BLE001 - a cache read failure only costs an LLM call
        cached = None

    cache_hit = False
    if cached is not None:
        similarity = deterministic_similarity(document, dataset_document)
        cache_hit = abs(float(cached.similarity) - similarity) <= SIMILARITY_TOLERANCE

    try:
        comparison = compare_with_dataset(
            document, dataset_document, llm=llm, store=store, settings=settings
        )
    except Exception as exc:  # noqa: BLE001 - one comparison must not stop the run
        warnings.append(f"scrape_comparison_failed:{document.id}: {_error(exc)}")
        return None, False
    return comparison, cache_hit


def _run_document_critics(
    documents: Sequence[Document],
    *,
    settings: Settings,
    llm: Any,
    embeddings: Any,
    store: Any,
    research: Sequence[Document],
    warnings: list[str],
) -> list[DocumentCritique]:
    from ...critic.corpus import CorpusError, load_corpus
    from ...critic.document_critic import run_document_critic

    try:
        corpus = load_corpus(settings.corpus_path)
    except CorpusError as exc:
        warnings.append(f"scrape_corpus_failed: {_error(exc)}")
        corpus = None
    if corpus is None:
        return []

    critiques: list[DocumentCritique] = []
    for document in documents:
        if document.kind not in {"paper", "web"} or not document.credible:
            continue
        try:
            critique = run_document_critic(
                document,
                corpus=corpus,
                llm=llm,
                settings=settings,
                research=[
                    item for item in research if item.id != document.id
                ][:8],
                store=store,
                embeddings=embeddings,
            )
        except Exception as exc:  # noqa: BLE001 - a critic failure is a warning
            warnings.append(f"scrape_critique_failed:{document.id}: {_error(exc)}")
            continue
        critiques.append(critique)
    return critiques


def _default_fetch(candidate: Any) -> Document | None:
    """Build a document from link metadata without crawling the web.

    General web crawling stays disabled until a legal review; a candidate that
    already carries a title or abstract from an API can still be processed.
    """
    from .links import candidate_to_document

    if candidate.title or candidate.abstract:
        return candidate_to_document(candidate)
    return None


def _dedupe_seeds(
    documents: Iterable[Document],
    *,
    target: str,
    store: Any,
    dataset_id: str | None,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    warnings: list[str],
) -> list[Document]:
    unique: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        is_target_dataset = document.kind == "dataset" and (
            document.source in DATASET_DOCUMENT_SOURCES
            or (document.metadata or {}).get("is_target_dataset") is True
        )
        if document.id == dataset_id or is_target_dataset:
            continue
        keys = [key for key in (document.url_hash, document.content_sha512) if key]
        if not keys:
            keys = [
                title_identity(
                    document.title,
                    document.authors,
                    (document.metadata or {}).get("year"),
                )
            ]
        if any(key in seen for key in keys):
            continue
        seen.update(keys)

        existing: Document | None = None
        try:
            if document.url_hash:
                existing = store.find_document_by_url(document.url_hash)
            if existing is None and document.content_sha512:
                existing = store.find_document_by_content(document.content_sha512)
        except Exception as exc:  # noqa: BLE001 - a lookup failure is a warning
            warnings.append(f"scrape_identity_lookup_failed:{document.id}: {_error(exc)}")

        if existing is not None:
            if not existing.summary or not existing.summary_embedding:
                existing = _process_document(
                    existing,
                    target=target,
                    llm=llm,
                    embeddings=embeddings,
                    settings=settings,
                    warnings=warnings,
                )
                try:
                    existing = store.upsert_document(existing)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"scrape_store_failed:{existing.id}: {_error(exc)}"
                    )
            unique.append(existing)
        else:
            unique.append(document)
    return unique


def _ensure_dataset_document(
    document: Document | None, *, store: Any, warnings: list[str]
) -> Document | None:
    """Persist the target dataset only when it is not already catalogued."""
    if document is None:
        return None
    try:
        existing = store.get_document(document.id)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"scrape_dataset_lookup_failed:{document.id}: {_error(exc)}")
        existing = None
    if existing is not None:
        return existing
    try:
        return store.upsert_document(_normalise_document(document))
    except Exception as exc:  # noqa: BLE001 - comparisons are optional
        warnings.append(f"scrape_dataset_store_failed:{document.id}: {_error(exc)}")
        return None


def _target_embedding(
    target: str | Sequence[float],
    *,
    dataset_document: Document | None,
    embeddings: Any,
    warnings: list[str],
) -> list[float] | None:
    from .summarize import embed_summary

    if not isinstance(target, str):
        try:
            vector = [float(value) for value in target]
        except (TypeError, ValueError) as exc:
            warnings.append(f"scrape_target_embedding_failed: {_error(exc)}")
            return None
        if not vector:
            warnings.append("scrape_target_missing")
            return None
        return vector

    text = " ".join((target or "").split())
    if not text and dataset_document is not None:
        text = " ".join(
            (dataset_document.summary or dataset_document.title or "").split()
        )
    if not text:
        warnings.append("scrape_target_missing")
        return None
    try:
        return embed_summary(text, embeddings)
    except Exception as exc:  # noqa: BLE001 - no target means no gated expansion
        warnings.append(f"scrape_target_embedding_failed: {_error(exc)}")
        return None


def _prepare_seeds(
    seeds: Sequence[ResearchRecord | Document],
    *,
    target: str,
    store: Any,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    warnings: list[str],
) -> tuple[list[Document], Document | None]:
    converted: list[Document] = []
    for seed in seeds:
        try:
            converted.append(_seed_to_document(seed))
        except Exception as exc:  # noqa: BLE001 - one bad seed is a warning
            warnings.append(f"scrape_seed_failed: {_error(exc)}")

    dataset_document = _target_dataset_document(converted)
    dataset_document = _ensure_dataset_document(
        dataset_document, store=store, warnings=warnings
    )

    unique = _dedupe_seeds(
        converted,
        target=target,
        store=store,
        dataset_id=dataset_document.id if dataset_document else None,
        llm=llm,
        embeddings=embeddings,
        settings=settings,
        warnings=warnings,
    )

    prepared: list[Document] = []
    for document in unique:
        if document.summary and document.summary_embedding:
            prepared.append(
                _apply_credibility(document, settings=settings, warnings=warnings)
            )
            continue
        prepared.append(
            _process_document(
                document,
                target=target,
                llm=llm,
                embeddings=embeddings,
                settings=settings,
                warnings=warnings,
            )
        )
    return prepared, dataset_document


def run_semantic_scraping(
    *,
    seeds: Sequence[ResearchRecord | Document],
    target: str | Sequence[float],
    store: Any,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    client: Any = None,
    fetch: Any = None,
) -> ScrapeResult:
    """Expand research seeds into a semantic-scraping document catalog.

    ``seeds`` may be one-hop :class:`ResearchRecord` rows or already-built
    :class:`Document` rows.  ``target`` may be the target summary text or a
    precomputed embedding.  Scraping is opt-in: when
    ``settings.scrape_enabled`` is false this returns immediately with the
    ``scrape_disabled`` warning and makes no network or provider calls.
    """
    if not settings.scrape_enabled:
        return ScrapeResult(
            warnings=["scrape_disabled"],
            stats={
                "documents": 0,
                "links": 0,
                "comparisons": 0,
                "critiques": 0,
                "llm_calls": 0,
                "cache_hits": 0,
                "warnings": 1,
            },
        )

    warnings: list[str] = []
    llm_before = _llm_call_count(llm)
    summary_target = target if isinstance(target, str) else ""

    prepared_seeds, dataset_document = _prepare_seeds(
        seeds,
        target=summary_target,
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=settings,
        warnings=warnings,
    )

    if not prepared_seeds:
        warnings.append("scrape_no_seeds")
        return ScrapeResult(
            warnings=_dedupe_warnings(warnings),
            stats={
                "documents": 0,
                "links": 0,
                "comparisons": 0,
                "critiques": 0,
                "llm_calls": max(0, _llm_call_count(llm) - llm_before),
                "cache_hits": 0,
                "warnings": len(_dedupe_warnings(warnings)),
            },
        )

    target_embedding = _target_embedding(
        target,
        dataset_document=dataset_document,
        embeddings=embeddings,
        warnings=warnings,
    )
    if target_embedding is None:
        return ScrapeResult(
            warnings=_dedupe_warnings(warnings),
            stats={
                "documents": 0,
                "links": 0,
                "comparisons": 0,
                "critiques": 0,
                "llm_calls": max(0, _llm_call_count(llm) - llm_before),
                "cache_hits": 0,
                "warnings": len(_dedupe_warnings(warnings)),
            },
        )

    from .frontier import expand_frontier

    def process_callback(document: Document) -> Document:
        if document.summary and document.summary_embedding:
            return _apply_credibility(
                document, settings=settings, warnings=warnings
            )
        return _process_document(
            document,
            target=summary_target,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
            warnings=warnings,
        )

    fetcher = None
    if fetch is not None:
        fetch_callback = fetch
    elif settings.scrape_fetch_enabled:
        from .fetch import Fetcher

        fetcher = Fetcher(
            settings=settings,
            store=store,
            client=client,
            warnings=warnings,
        )
        fetch_callback = fetcher.fetch_document
    else:
        fetch_callback = _default_fetch

    try:
        frontier_result = expand_frontier(
            prepared_seeds,
            target_embedding=target_embedding,
            store=store,
            process=process_callback,
            fetch=fetch_callback,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - surface a warning, keep stored rows
        warnings.append(f"scrape_frontier_failed: {_error(exc)}")
        return ScrapeResult(
            warnings=_dedupe_warnings(warnings),
            stats={
                "documents": 0,
                "links": 0,
                "comparisons": 0,
                "critiques": 0,
                "llm_calls": max(0, _llm_call_count(llm) - llm_before),
                "cache_hits": 0,
                "warnings": len(_dedupe_warnings(warnings)),
            },
        )
    finally:
        if fetcher is not None:
            fetcher.close()

    warnings.extend(frontier_result.warnings)
    documents: list[Document] = []
    classified: set[str] = set()

    for document in frontier_result.documents:
        document = _apply_credibility(
            document, settings=settings, warnings=warnings
        )
        if document.id not in classified:
            existing_path: list[Any] = []
            try:
                existing_path = store.document_path(document.id)
            except Exception:  # noqa: BLE001 - classification is best effort
                existing_path = []
            if existing_path:
                classified.add(document.id)
            else:
                document = _classify_document(
                    document,
                    store=store,
                    llm=llm,
                    embeddings=embeddings,
                    settings=settings,
                    warnings=warnings,
                )
                classified.add(document.id)
        documents.append(document)

    comparisons: list[DocumentComparison] = []
    cache_hits = 0
    compared_ids: set[str] = set()
    if dataset_document is not None:
        for document in documents:
            if not document.credible or document.id == dataset_document.id:
                continue
            comparison, cached = _compare_with_dataset(
                document,
                dataset_document,
                store=store,
                llm=llm,
                settings=settings,
                warnings=warnings,
            )
            if comparison is not None:
                comparisons.append(comparison)
                compared_ids.add(document.id)
                if cached:
                    cache_hits += 1

    if compared_ids:
        updated: list[Document] = []
        for document in documents:
            if document.id in compared_ids:
                try:
                    document = store.upsert_document(
                        document.model_copy(update={"status": "compared"})
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"scrape_store_failed:{document.id}: {_error(exc)}"
                    )
            updated.append(document)
        documents = updated

    critiques = _run_document_critics(
        documents,
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        store=store,
        research=prepared_seeds,
        warnings=warnings,
    )

    stats: dict[str, int | float] = dict(frontier_result.stats or {})
    stats.update(
        {
            "documents": len(documents),
            "credible": sum(1 for document in documents if document.credible),
            "links": len(frontier_result.links),
            "comparisons": len(comparisons),
            "critiques": len(critiques),
            "llm_calls": max(0, _llm_call_count(llm) - llm_before),
            "cache_hits": cache_hits,
            "warnings": len(_dedupe_warnings(warnings)),
        }
    )

    return ScrapeResult(
        documents=documents,
        links=list(frontier_result.links),
        comparisons=comparisons,
        critiques=critiques,
        warnings=_dedupe_warnings(warnings),
        stats=stats,
    )


def _dedupe_warnings(warnings: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        result.append(warning)
    return result
