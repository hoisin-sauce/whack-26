"""Preload the layer-1 statistics and validity corpus.

The preload is a small, bounded discovery run: thirteen seed queries are sent
to OpenAlex and arXiv, the top records are normalized into :class:`Document`
rows, deduplicated by URL/content/title identity, summarized, embedded,
classified under a ``Statistics and Validity`` root, scored for credibility,
and marked ``preloaded``. It never raises on a source failure or offline cache
miss; those become run warnings so that the static
``docs/corpus/statistical-pitfalls.md`` file remains the fallback corpus.
"""

from __future__ import annotations

from typing import Any, Iterable

import httpx

from ...common import redact_secrets
from ...config import Settings
from ..arxiv import arXivAdapter
from ..base import DATASET_SOURCES, ResearchRecord, SourceError
from ..cache import ResearchCache
from ..metrics import metrics_metadata
from ..openalex import OpenAlexAdapter
from .identity import canonicalize_url, metadata_identity, title_identity, url_hash
from .models import Document, ScrapeResult

PRELOAD_ROOT_NAME = "Statistics and Validity"
PRELOAD_ROOT_DESCRIPTION = (
    "Statistical validity, reproducibility, causal identification, and "
    "research-method standards used to review evidence."
)
PRELOAD_TARGET = "statistical validity, reproducibility, and research methods"
PRELOAD_CACHE_DATASET_ID = "preload-statistics"
PRELOAD_SOURCE_LIMIT = 10
PRELOAD_ID_PREFIXES = {"dataset": "ds_", "paper": "pap_"}

STATISTICS_QUERIES: tuple[str, ...] = (
    "statistical validity and reproducibility",
    "p-hacking and publication bias",
    "multiple comparisons and false discovery rate",
    "causal inference assumptions and identification",
    "regression diagnostics and heteroskedasticity",
    "missing data mechanisms and imputation",
    "measurement error and misclassification",
    "sample size and statistical power",
    "time-series stationarity and dependence",
    "data leakage and train/test contamination",
    "external validity and distribution shift",
    "model selection and overfitting",
    "pre-registration and reporting standards",
)

__all__ = [
    "PRELOAD_CACHE_DATASET_ID",
    "PRELOAD_ROOT_DESCRIPTION",
    "PRELOAD_ROOT_NAME",
    "PRELOAD_SOURCE_LIMIT",
    "PRELOAD_TARGET",
    "STATISTICS_QUERIES",
    "preload_statistics",
]


def _llm_call_count(llm: Any) -> int:
    calls = getattr(llm, "calls", None)
    return len(calls) if isinstance(calls, list) else 0


def _error(exc: BaseException) -> str:
    return redact_secrets(f"{type(exc).__name__}: {exc}")


def _document_id(kind: str, canonical_url: str | None, content_hash: str | None) -> str:
    digest: str | None = None
    if canonical_url:
        try:
            digest = url_hash(canonical_url)
        except ValueError:
            digest = None
    if digest is None:
        digest = content_hash
    if digest is None:  # pragma: no cover - every record has a URL or identity
        digest = "unknown"
    prefix = PRELOAD_ID_PREFIXES.get(kind, "doc_")
    return f"{prefix}{digest}"


def _record_to_document(record: ResearchRecord, *, preloaded: bool = True) -> Document:
    """Convert one research record into a pending document."""
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
    metadata.update(metrics_metadata(record))

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
        publisher=record.publisher,
        published_at=str(record.year) if record.year is not None else None,
        license=record.license,
        metadata=metadata,
        preloaded=preloaded,
        status="new",
    )


def _ensure_root(store: Any, embeddings: Any, warnings: list[str]) -> Any | None:
    root = store.find_sibling(None, PRELOAD_ROOT_NAME)
    if root is not None:
        return root

    from .summarize import embed_summary

    try:
        vector = embed_summary(PRELOAD_ROOT_DESCRIPTION, embeddings)
    except Exception as exc:  # noqa: BLE001 - embeddings may be unavailable
        warnings.append(f"preload_root_embedding_failed: {_error(exc)}")
        return None

    try:
        return store.create_field(
            parent_id=None,
            name=PRELOAD_ROOT_NAME,
            description=PRELOAD_ROOT_DESCRIPTION,
            description_embedding=vector,
        )
    except Exception as exc:  # noqa: BLE001 - a concurrent create is not fatal
        existing = store.find_sibling(None, PRELOAD_ROOT_NAME)
        if existing is not None:
            return existing
        warnings.append(f"preload_root_create_failed: {_error(exc)}")
        return None


def _identity_keys(document: Document) -> tuple[str | None, str | None, str | None]:
    year: int | None = None
    if document.published_at:
        try:
            year = int(str(document.published_at)[:4])
        except ValueError:
            year = None
    if year is None:
        raw_year = (document.metadata or {}).get("year")
        if isinstance(raw_year, int) and not isinstance(raw_year, bool):
            year = raw_year
    return (
        document.url_hash,
        document.content_sha512,
        title_identity(document.title, document.authors, year),
    )


def _select_candidates(
    records: Iterable[ResearchRecord],
    *,
    store: Any,
    limit: int,
    warnings: list[str],
    seen: set[str],
) -> tuple[list[Document], list[Document]]:
    """Deduplicate records and return ``(new, reused)`` documents."""
    new_documents: list[Document] = []
    reused_documents: list[Document] = []

    for record in records:
        if len(new_documents) + len(reused_documents) >= limit:
            break
        try:
            document = _record_to_document(record)
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop preload
            warnings.append(f"preload_record_failed: {_error(exc)}")
            continue

        keys = [key for key in _identity_keys(document) if key]
        if any(key in seen for key in keys):
            continue
        seen.update(keys)

        existing: Document | None = None
        try:
            if document.url_hash:
                existing = store.find_document_by_url(document.url_hash)
            if existing is None and document.content_sha512:
                existing = store.find_document_by_content(document.content_sha512)
        except Exception as exc:  # noqa: BLE001 - a store lookup failure is a warning
            warnings.append(f"preload_identity_lookup_failed: {_error(exc)}")

        if existing is not None:
            reused_documents.append(existing)
        else:
            new_documents.append(document)

    return new_documents, reused_documents


def _process_document(
    document: Document,
    *,
    store: Any,
    root: Any,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    warnings: list[str],
) -> Document | None:
    from .classify import classify_document
    from .credibility import apply_credibility, score_credibility
    from .process import process_document
    from .summarize import summarize_document

    try:
        summary = summarize_document(document, PRELOAD_TARGET, llm, settings)
        processed = process_document(
            document,
            PRELOAD_TARGET,
            llm,
            embeddings,
            settings,
            summary=summary,
        )
    except Exception as exc:  # noqa: BLE001 - skip one document, keep the corpus
        warnings.append(f"preload_document_failed:{document.id}: {_error(exc)}")
        return None

    metadata = dict(processed.metadata)
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
    if not processed.publisher and isinstance(signals.get("publisher"), str):
        processed = processed.model_copy(update={"publisher": signals["publisher"]})
    processed = processed.model_copy(
        update={"metadata": metadata, "preloaded": True, "status": "summarized"}
    )

    try:
        processed = apply_credibility(processed, score_credibility(processed, settings))
    except Exception as exc:  # noqa: BLE001 - credibility is advisory
        warnings.append(f"preload_credibility_failed:{document.id}: {_error(exc)}")

    try:
        processed = store.upsert_document(processed)
    except Exception as exc:  # noqa: BLE001 - keep the run alive
        warnings.append(f"preload_store_failed:{document.id}: {_error(exc)}")
        return None

    path: list[str] = []
    try:
        path = classify_document(
            processed,
            store=store,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - assignment falls back to the root
        warnings.append(f"preload_classification_failed:{document.id}: {_error(exc)}")

    if not path or path[0] != root.id:
        if path:
            warnings.append(f"preload_classification_outside_root:{document.id}")
        try:
            store.assign_document_path(processed.id, [root.id], confidence=1.0)
        except Exception as exc:  # noqa: BLE001 - a missing path is not fatal
            warnings.append(f"preload_assign_failed:{document.id}: {_error(exc)}")

    try:
        return store.upsert_document(
            processed.model_copy(update={"status": "classified", "preloaded": True})
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"preload_store_failed:{document.id}: {_error(exc)}")
        return processed


def _reuse_document(
    document: Document,
    *,
    store: Any,
    root: Any,
    warnings: list[str],
) -> Document | None:
    """Mark an already-known document as preloaded and ensure it has a path."""
    if not document.preloaded:
        try:
            document = store.upsert_document(
                document.model_copy(update={"preloaded": True})
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"preload_store_failed:{document.id}: {_error(exc)}")
            return None

    try:
        path = store.document_path(document.id)
    except Exception:  # noqa: BLE001 - older stores may not expose document_path
        path = []
    if not path:
        try:
            store.assign_document_path(document.id, [root.id], confidence=1.0)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"preload_assign_failed:{document.id}: {_error(exc)}")
    return document


def _fetch_records(
    *,
    settings: Settings,
    client: httpx.Client,
    cache: ResearchCache,
    limit: int,
    store: Any,
    warnings: list[str],
) -> tuple[list[Document], list[Document]]:
    adapters = [
        OpenAlexAdapter(
            api_key=settings.openalex_api_key,
            cache=cache,
            dataset_id=PRELOAD_CACHE_DATASET_ID,
            offline=settings.offline,
        ),
        arXivAdapter(
            cache=cache,
            dataset_id=PRELOAD_CACHE_DATASET_ID,
            offline=settings.offline,
        ),
    ]

    seen: set[str] = set()
    new_documents: list[Document] = []
    reused_documents: list[Document] = []

    for query in STATISTICS_QUERIES:
        for adapter in adapters:
            if len(new_documents) + len(reused_documents) >= limit:
                break
            try:
                records = adapter.search(query, PRELOAD_SOURCE_LIMIT, client)
            except SourceError as exc:
                warnings.append(f"preload_source_failed: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one source must not stop preload
                warnings.append(
                    f"preload_source_failed:{adapter.name}:{query!r}: {_error(exc)}"
                )
                continue
            for adapter_warning in getattr(adapter, "warnings", []):
                if adapter_warning not in warnings:
                    warnings.append(str(adapter_warning))
            found, reused = _select_candidates(
                records,
                store=store,
                limit=limit - len(new_documents) - len(reused_documents),
                warnings=warnings,
                seen=seen,
            )
            new_documents.extend(found)
            reused_documents.extend(reused)
        if len(new_documents) + len(reused_documents) >= limit:
            break

    return new_documents, reused_documents


def preload_statistics(
    *,
    store: Any,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    client: httpx.Client | None = None,
    limit: int = 200,
) -> ScrapeResult:
    """Fetch, process, and catalog the statistics/validity seed corpus.

    The call is offline-safe: an offline cache miss, an unreachable API, and a
    malformed record all become warnings. ``limit`` caps the number of unique
    documents considered, and the static pitfalls corpus is never touched.
    """
    warnings: list[str] = []
    llm_before = _llm_call_count(llm)

    if limit <= 0:
        return ScrapeResult(warnings=["preload_limit_zero"], stats={"documents": 0})

    root = _ensure_root(store, embeddings, warnings)
    if root is None:
        return ScrapeResult(
            warnings=warnings or ["preload_root_unavailable"],
            stats={"documents": 0, "new": 0, "reused": 0, "warnings": len(warnings)},
        )

    owned_client = client is None
    cache: ResearchCache | None = None
    try:
        cache = ResearchCache(settings.research_db, offline=settings.offline)
        if client is None:
            client = httpx.Client(
                timeout=httpx.Timeout(settings.http_timeout),
                follow_redirects=True,
                headers={"User-Agent": settings.scrape_user_agent},
            )

        new_documents, reused_documents = _fetch_records(
            settings=settings,
            client=client,
            cache=cache,
            limit=limit,
            store=store,
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001 - preload must degrade, not crash
        warnings.append(f"preload_unavailable: {_error(exc)}")
        new_documents, reused_documents = [], []
    finally:
        if cache is not None:
            cache.close()
        if owned_client and client is not None:
            client.close()

    documents: list[Document] = []
    reused_count = 0
    new_count = 0
    for document in reused_documents:
        reused = _reuse_document(
            document, store=store, root=root, warnings=warnings
        )
        if reused is not None:
            documents.append(reused)
            reused_count += 1

    for document in new_documents:
        if len(documents) >= limit:
            break
        processed = _process_document(
            document,
            store=store,
            root=root,
            llm=llm,
            embeddings=embeddings,
            settings=settings,
            warnings=warnings,
        )
        if processed is not None:
            documents.append(processed)
            new_count += 1

    credible = sum(1 for document in documents if document.credible)
    stats: dict[str, int | float] = {
        "documents": len(documents),
        "new": new_count,
        "reused": reused_count,
        "credible": credible,
        "queries": len(STATISTICS_QUERIES),
        "llm_calls": max(0, _llm_call_count(llm) - llm_before),
        "warnings": len(warnings),
    }
    return ScrapeResult(documents=documents, warnings=warnings, stats=stats)
