"""Search-session documents and the search/dataset graph.

Every executed query is recorded as one ``Document(kind="search")``.  The
document id is derived from the query, the source, the session, and the
(redacted) parameters, so re-running the same query updates the existing row
instead of creating a second one.  Results, follow-up queries, and explicit
dataset relationships are stored as ordinary ``document_links`` rows; no new
table is required.

Secrets never enter a search document.  Queries and parameter values are
redacted before they are hashed, rendered, or persisted.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, field_validator

from ..common import redact_secrets, utc_now_iso
from .scrape.models import Document, DocumentLink, DocumentStatus, LinkRelation

SEARCH_ID_PREFIX = "srch_"
SEARCH_KIND = "search"

_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|secret|token|password|passwd|"
    r"credential|authorization)"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|passwd)\b"
    r"\s*[:=]\s*[^\s,;\"']+"
)
_REDACTED = "[redacted]"

__all__ = [
    "SEARCH_ID_PREFIX",
    "SEARCH_KIND",
    "SearchSession",
    "SearchSessionError",
    "build_search_document",
    "create_search_document",
    "link_dataset_document",
    "link_search_follow_up",
    "link_search_result",
    "record_search_session",
    "search_document_id",
]


class SearchSessionError(RuntimeError):
    """Raised when a search document or one of its links cannot be recorded."""


class SearchSession(BaseModel):
    """One query executed against one source within one research session."""

    query: str = Field(min_length=1)
    source: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_ids: list[str] = Field(default_factory=list)
    status: DocumentStatus = "classified"
    timestamp: str = ""

    @field_validator("query", "source", "session_id")
    @classmethod
    def _require_non_blank(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("must not be blank")
        return text


# ---------------------------------------------------------------------------
# Redaction and identity
# ---------------------------------------------------------------------------
def _redact_text(value: str) -> str:
    """Redact URL credentials and ``key=value`` credentials from text."""
    text = redact_secrets(str(value))
    return _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}={_REDACTED}", text
    )


def _sanitize(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe copy with credential-like values removed."""
    if key is not None and _SECRET_KEY_RE.search(key):
        return _REDACTED
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_sanitize(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(repr(value))


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _normalized_query(query: str) -> str:
    return _redact_text(" ".join(str(query).split())).casefold()


def _identity_payload(session: SearchSession) -> dict[str, Any]:
    return {
        "query": _normalized_query(session.query),
        "source": _redact_text(session.source.strip()).casefold(),
        "session_id": _redact_text(session.session_id.strip()),
        "parameters": _sanitize(session.parameters),
    }


def search_document_id(session: SearchSession) -> str:
    """Return the deterministic ``srch_<hash>`` id for a search session."""
    digest = hashlib.sha256(
        _stable_json(_identity_payload(session)).encode("utf-8")
    ).hexdigest()[:24]
    return f"{SEARCH_ID_PREFIX}{digest}"


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def _unique_ids(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        identifier = str(value)
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        result.append(identifier)
    return result


def build_search_document(
    session: SearchSession,
    *,
    result_count: int | None = None,
    embedding: Sequence[float] | None = None,
    embedding_model: str = "",
    timestamp: str | None = None,
) -> Document:
    """Build the search document for ``session`` without persisting it."""
    query = _redact_text(" ".join(session.query.split()))
    source = _redact_text(session.source.strip())
    parameters = _sanitize(session.parameters)
    if not isinstance(parameters, dict):  # pragma: no cover - defensive
        parameters = {}
    count = (
        len(_unique_ids(session.result_ids))
        if result_count is None
        else result_count
    )
    count = max(0, int(count))
    resolved_timestamp = _redact_text(
        timestamp or session.timestamp or utc_now_iso()
    )
    identifier = search_document_id(session)
    summary = (
        f"query={query} source={source} results={count} "
        f"parameters={_stable_json(parameters)}"
    )
    metadata = {
        "query": query,
        "source": source,
        "parameters": parameters,
        "result_count": count,
        "session_id": _redact_text(session.session_id.strip()),
        "timestamp": resolved_timestamp,
    }
    return Document(
        id=identifier,
        kind=SEARCH_KIND,
        source=source,
        title=query,
        summary=summary,
        summary_embedding=list(embedding or []),
        embedding_model=embedding_model,
        metadata=metadata,
        content_sha256=identifier.removeprefix(SEARCH_ID_PREFIX),
        status=session.status,
    )


def _put_document(store: Any, document: Document) -> Document:
    """Persist through either ``put_document`` or the concrete store method."""
    putter = getattr(store, "put_document", None)
    if callable(putter):
        return putter(document)
    return store.upsert_document(document)


def _require_document(store: Any, document_id: str) -> Document | None:
    getter = getattr(store, "get_document", None)
    if not callable(getter):
        return None
    return getter(document_id)


def _supports_lookup(store: Any) -> bool:
    return callable(getattr(store, "get_document", None))


def _link_url_and_anchor(
    store: Any,
    document_id: str,
    *,
    url: str | None,
    anchor_text: str | None,
) -> tuple[str | None, str | None]:
    if url is not None and anchor_text is not None:
        return url, anchor_text
    target = _require_document(store, document_id)
    if target is None:
        return url, anchor_text
    return (
        url if url is not None else target.canonical_url,
        anchor_text if anchor_text is not None else target.title,
    )


def link_search_result(
    store: Any,
    search_id: str,
    result_id: str,
    *,
    url: str | None = None,
    anchor_text: str | None = None,
    relation: LinkRelation = "search_result",
) -> DocumentLink:
    """Link a search document to one of its result documents."""
    if search_id == result_id:
        raise SearchSessionError("a search document cannot link to itself")
    if _supports_lookup(store):
        if _require_document(store, search_id) is None:
            raise SearchSessionError(f"search document not found: {search_id}")
        if _require_document(store, result_id) is None:
            raise SearchSessionError(f"result document not found: {result_id}")
    resolved_url, resolved_anchor = _link_url_and_anchor(
        store, result_id, url=url, anchor_text=anchor_text
    )
    link = DocumentLink(
        from_id=search_id,
        to_id=result_id,
        relation=relation,
        url=resolved_url,
        anchor_text=resolved_anchor,
    )
    store.add_document_link(link)
    return link


def link_search_follow_up(
    store: Any,
    prior_search_id: str,
    follow_up_search_id: str,
    *,
    relation: LinkRelation = "search_result",
) -> DocumentLink:
    """Link an earlier search to a follow-up search.

    ``prior_search_id`` is the source of the link and ``follow_up_search_id``
    is its target, so ``relation="search_result"`` reads as "the follow-up was
    produced by the earlier search".  ``linked_from`` is also accepted when a
    caller wants the inverse vocabulary.
    """
    if prior_search_id == follow_up_search_id:
        raise SearchSessionError("a search cannot be its own follow-up")
    follow_up = _require_document(store, follow_up_search_id)
    if _supports_lookup(store):
        prior = _require_document(store, prior_search_id)
        if prior is None or prior.kind != SEARCH_KIND:
            raise SearchSessionError(f"prior search not found: {prior_search_id}")
        if follow_up is None or follow_up.kind != SEARCH_KIND:
            raise SearchSessionError(
                f"follow-up search not found: {follow_up_search_id}"
            )
    link = DocumentLink(
        from_id=prior_search_id,
        to_id=follow_up_search_id,
        relation=relation,
        url=follow_up.canonical_url if follow_up is not None else None,
        anchor_text=follow_up.title if follow_up is not None else None,
    )
    store.add_document_link(link)
    return link


def link_dataset_document(
    store: Any,
    dataset_id: str,
    document_id: str,
    *,
    relation: LinkRelation | None = None,
) -> DocumentLink:
    """Link a dataset to an explicitly related search or paper document."""
    dataset = _require_document(store, dataset_id)
    target = _require_document(store, document_id)
    if _supports_lookup(store):
        if dataset is None:
            raise SearchSessionError(f"dataset document not found: {dataset_id}")
        if dataset.kind != "dataset":
            raise SearchSessionError(
                f"document {dataset_id!r} is not a dataset "
                f"(kind={dataset.kind!r})"
            )
        if target is None:
            raise SearchSessionError(f"related document not found: {document_id}")
    if relation is None:
        relation = (
            "search_result"
            if target is not None and target.kind == SEARCH_KIND
            else "dataset_paper"
        )
    link = DocumentLink(
        from_id=dataset_id,
        to_id=document_id,
        relation=relation,
        url=target.canonical_url if target is not None else None,
        anchor_text=target.title if target is not None else None,
    )
    store.add_document_link(link)
    return link


def record_search_session(
    store: Any,
    session: SearchSession | None = None,
    *,
    query: str | None = None,
    source: str | None = None,
    session_id: str | None = None,
    parameters: dict[str, Any] | None = None,
    result_ids: Sequence[str] | None = None,
    result_count: int | None = None,
    dataset_ids: Sequence[str] = (),
    follow_up_of: str | None = None,
    follow_up_relation: LinkRelation = "search_result",
    dataset_relation: LinkRelation | None = None,
    embedding: Sequence[float] | None = None,
    embedding_model: str = "",
    timestamp: str | None = None,
) -> Document:
    """Create or update one search document and all of its graph links.

    Either pass a :class:`SearchSession` or the ``query``/``source``/
    ``session_id`` keyword arguments.  The returned document is the persisted
    row.  Re-running the same query with the same session and parameters is
    idempotent: the document id and links are stable.
    """
    if session is None:
        if query is None or source is None or session_id is None:
            raise SearchSessionError(
                "pass a SearchSession or query, source, and session_id"
            )
        session = SearchSession(
            query=query,
            source=source,
            session_id=session_id,
            parameters=parameters or {},
            result_ids=list(result_ids or []),
        )
    elif result_ids is not None or parameters is not None:
        updates: dict[str, Any] = {}
        if result_ids is not None:
            updates["result_ids"] = list(result_ids)
        if parameters is not None:
            updates["parameters"] = parameters
        session = session.model_copy(update=updates)

    identifier = search_document_id(session)
    existing = _require_document(store, identifier)

    # Validate relationships before writing anything so a missing result does
    # not leave a half-written search graph behind.  Minimal fake stores may
    # not implement lookups; in that case the link writer is the only guard.
    unique_results = [
        item for item in _unique_ids(session.result_ids) if item != identifier
    ]
    unique_datasets = _unique_ids(dataset_ids)
    if _supports_lookup(store):
        for result_id in unique_results:
            if _require_document(store, result_id) is None:
                raise SearchSessionError(f"result document not found: {result_id}")
        for dataset_id in unique_datasets:
            dataset = _require_document(store, dataset_id)
            if dataset is None:
                raise SearchSessionError(f"dataset document not found: {dataset_id}")
            if dataset.kind != "dataset":
                raise SearchSessionError(
                    f"document {dataset_id!r} is not a dataset "
                    f"(kind={dataset.kind!r})"
                )
        if follow_up_of is not None:
            prior = _require_document(store, follow_up_of)
            if prior is None or prior.kind != SEARCH_KIND:
                raise SearchSessionError(f"prior search not found: {follow_up_of}")

    if not session.timestamp and existing is not None:
        previous = (existing.metadata or {}).get("timestamp")
        if isinstance(previous, str) and previous:
            session = session.model_copy(update={"timestamp": previous})

    document = build_search_document(
        session,
        result_count=result_count,
        embedding=embedding,
        embedding_model=embedding_model,
        timestamp=timestamp,
    )
    stored = _put_document(store, document)

    for result_id in unique_results:
        link_search_result(store, stored.id, result_id)
    if follow_up_of is not None:
        link_search_follow_up(
            store,
            follow_up_of,
            stored.id,
            relation=follow_up_relation,
        )
    for dataset_id in unique_datasets:
        link_dataset_document(
            store,
            dataset_id,
            stored.id,
            relation=dataset_relation,
        )
    return stored


# ``create_search_document`` reads naturally at call sites that do not think in
# terms of a session lifecycle; it is the same idempotent operation.
create_search_document = record_search_session
