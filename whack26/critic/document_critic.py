"""Corpus-grounded method critique for scraped research documents.

Unlike the hypothesis critic, this reviewer never receives runnable Python code:
it reads a document's title, summary, and method metadata, retrieves the
relevant chunks from the statistical-pitfalls corpus, and returns a compact
method verdict.  Performance, p-values, effect sizes, and out-of-sample results
are explicitly out of scope; a separate validator owns them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from ..common import redact_secrets, utc_now_iso
from ..config import Settings
from ..embeddings import EmbeddingError, build_embedding_client
from ..llm import LLMError
from ..research.scrape.models import Document, DocumentCritique
from .corpus import Corpus, CorpusError, load_corpus
from .critic import (
    _BANNED_PERFORMANCE_RE,
    _SENTENCE_RE,
    CRITIC_SCHEMA,
    CRITIC_SYSTEM_PROMPT,
    LLMCriticPayload,
)

DOCUMENT_CRITIC_SYSTEM_PROMPT = (
    CRITIC_SYSTEM_PROMPT
    + " The input is a research document summary and its metadata, not runnable "
    "Python code: do not ask for or expect code, and do not treat the absence of "
    "code as a flaw. Judge the method and reporting described in the summary. "
    "Do not evaluate model performance, p-values, effect sizes, accuracy, AUC, "
    "RMSE, calibration, or out-of-sample results; a separate validator owns "
    "those. Cite a chunk id from the supplied corpus for every issue."
)

DOCUMENT_CRITIC_PROMPT = (
    "Document metadata and summary:\n"
    "{document}\n\n"
    "{research_block}"
    "Retrieved corpus chunks:\n"
    "{chunks}\n\n"
    "Review the document's method, assumptions, and reporting for validity "
    "problems. No Python code or derivation trace is supplied; do not ask for "
    "code. Do not assess performance metrics, p-values, effect sizes, or "
    "out-of-sample results. Every issue must cite a chunk id from the supplied "
    "chunks. Use `insufficient` when the summary or metadata does not contain "
    "enough information to judge a check, and list the missing details in "
    "`missing_information`. Return JSON matching the schema."
)

_MAX_TEXT = 1_800


class _UnavailableEmbeddings:
    """Stand-in that lets ``Corpus.retrieve`` use its deterministic fallback."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("embedding provider unavailable for document critic")


def _model_name(settings: Settings) -> str:
    return settings.llm_model_name


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _shorten(text: str, limit: int = _MAX_TEXT) -> str:
    cleaned = redact_secrets(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _document_block(document: Document) -> str:
    metadata = document.metadata or {}
    lines = [
        f"id: {document.id}",
        f"kind: {document.kind}",
        f"source: {document.source}",
        f"title: {_shorten(document.title, 300)}",
    ]
    if document.authors:
        lines.append("authors: " + ", ".join(document.authors[:12]))
    if document.published_at:
        lines.append(f"published_at: {document.published_at}")
    if document.publisher:
        lines.append(f"publisher: {_shorten(document.publisher, 200)}")
    lines.append("summary: " + (_shorten(document.summary) or "(no summary available)"))

    claims = _as_list(metadata.get("claims"))
    methods = _as_list(metadata.get("methods"))
    population = metadata.get("population")
    time_period = metadata.get("time_period")
    if claims:
        lines.append(
            "claims:\n"
            + "\n".join(f"- {_shorten(item, 300)}" for item in claims[:10])
        )
    if methods:
        lines.append(
            "methods:\n"
            + "\n".join(f"- {_shorten(item, 300)}" for item in methods[:10])
        )
    if population:
        lines.append(f"population: {_shorten(str(population), 300)}")
    if time_period:
        lines.append(f"time_period: {_shorten(str(time_period), 300)}")
    return _shorten("\n".join(lines), 6_000)


def _research_block(research: Sequence[Any]) -> str:
    lines: list[str] = []
    for record in list(research)[:8]:
        title = _shorten(str(getattr(record, "title", "") or ""), 240)
        summary = _shorten(str(getattr(record, "summary", "") or ""), 400)
        if not title and not summary:
            continue
        line = f"- {title}"
        if summary and summary != title:
            line += f": {summary}"
        lines.append(line)
    if not lines:
        return ""
    return (
        "Additional statistics documents (context only; not evidence about this "
        "specific document):\n" + "\n".join(lines) + "\n\n"
    )


def _chunks_text(chunks: Sequence[Any]) -> str:
    blocks = [
        f"### {chunk.id} — {chunk.title}\n{chunk.text}" for chunk in chunks
    ]
    return "\n\n".join(blocks) or "(no chunks retrieved)"


def _has_performance_language(text: str) -> bool:
    return bool(_BANNED_PERFORMANCE_RE.search(text or ""))


def _clean_text(text: str) -> tuple[str, bool]:
    if not _has_performance_language(text):
        return text, False
    sentences = _SENTENCE_RE.split(text)
    kept = [
        sentence
        for sentence in sentences
        if not _has_performance_language(sentence)
    ]
    return " ".join(kept).strip(), True


def _issue_text(issue: dict[str, Any]) -> str:
    return " ".join(
        str(issue.get(field, ""))
        for field in ("finding", "evidence", "why", "fix")
    )


def _merge_verdict(
    *,
    issues: Sequence[dict[str, Any]],
    missing_information: Sequence[str],
    llm_verdict: str,
) -> str:
    if any(issue.get("severity") == "fatal" for issue in issues):
        return "reject"
    if missing_information or llm_verdict == "insufficient":
        return "insufficient"
    if any(issue.get("severity") == "major" for issue in issues):
        return "refine"
    return "pass"


def _store(store: Any | None, critique: DocumentCritique) -> DocumentCritique:
    if store is not None:
        store.put_critique(critique)
    return critique


def _corpus_missing_critique(
    document: Document,
    *,
    settings: Settings,
    store: Any | None,
) -> DocumentCritique:
    critique = DocumentCritique(
        document_id=document.id,
        corpus_version="missing",
        verdict="insufficient",
        issues=[
            {
                "chunk": "S0-how-to-use",
                "severity": "major",
                "finding": "The statistical-pitfalls corpus is missing or unreadable.",
                "evidence": "critic.corpus_missing",
                "why": (
                    "The document critic cannot ground its method review "
                    "without the corpus."
                ),
                "fix": (
                    "Restore docs/corpus/statistical-pitfalls.md and rerun "
                    "the critic."
                ),
            }
        ],
        missing_information=["statistical-pitfalls corpus"],
        summary=(
            "The document method critic could not run because the preloaded "
            "corpus is missing. The document's method is unassessed."
        ),
        model=_model_name(settings),
        created_at=utc_now_iso(),
    )
    return _store(store, critique)


def _unavailable_critique(
    document: Document,
    *,
    corpus_version: str,
    retrieved_ids: Sequence[str],
    settings: Settings,
    reason: str,
    store: Any | None,
) -> DocumentCritique:
    critique = DocumentCritique(
        document_id=document.id,
        corpus_version=corpus_version,
        verdict="insufficient",
        issues=[
            {
                "chunk": retrieved_ids[0] if retrieved_ids else "S0-how-to-use",
                "severity": "major",
                "finding": (
                    "The document method critic could not produce a valid "
                    "review."
                ),
                "evidence": f"critic.unavailable: {_shorten(reason, 400)}",
                "why": (
                    "The LLM response was missing, invalid JSON, or failed "
                    "validation."
                ),
                "fix": "Rerun the critic or inspect the document's methods manually.",
            }
        ],
        missing_information=["method critique review"],
        summary=(
            "The document method critic did not run successfully, so the "
            "document's method is unassessed."
        ),
        model=_model_name(settings),
        created_at=utc_now_iso(),
    )
    return _store(store, critique)


def run_document_critic(
    document: Document,
    *,
    corpus: Corpus | None,
    llm: Any,
    settings: Settings,
    research: Sequence[Any] = (),
    store: Any | None = None,
    embeddings: Any | None = None,
) -> DocumentCritique:
    """Critique a document's method against the preloaded pitfalls corpus.

    ``corpus=None`` attempts to load the configured corpus and then degrades to
    an ``insufficient`` critique.  Invalid LLM output or a provider failure is
    handled the same way, so callers never see an exception for a degraded
    review.  When ``store`` is supplied, the critique is persisted with
    ``store.put_critique``.
    """
    if corpus is None:
        try:
            corpus = load_corpus(settings.corpus_path)
        except CorpusError:
            corpus = None
    if corpus is None:
        return _corpus_missing_critique(document, settings=settings, store=store)

    metadata = document.metadata or {}
    query_parts = [
        document.title,
        document.summary,
        " ".join(_as_list(metadata.get("claims"))),
        " ".join(_as_list(metadata.get("methods"))),
        str(metadata.get("population") or ""),
        str(metadata.get("time_period") or ""),
        " ".join(
            str(getattr(record, "title", "") or "")
            for record in list(research)[:8]
        ),
    ]
    query = " ".join(part for part in query_parts if part).strip() or document.title

    if embeddings is None:
        try:
            embeddings = build_embedding_client(settings)
        except (EmbeddingError, ValueError):
            embeddings = _UnavailableEmbeddings()
    chunks, _retrieval_warnings = corpus.retrieve(query, embeddings)
    retrieved_ids = [chunk.id for chunk in chunks]
    valid_chunk_ids = {chunk.id for chunk in corpus.chunks}

    try:
        payload = llm.generate_json(
            system=DOCUMENT_CRITIC_SYSTEM_PROMPT,
            user=DOCUMENT_CRITIC_PROMPT.format(
                document=_document_block(document),
                research_block=_research_block(research),
                chunks=_chunks_text(chunks),
            ),
            schema=CRITIC_SCHEMA,
            max_tokens=4096,
            temperature=0.0,
            purpose="critic",
        )
        if not isinstance(payload, dict):
            raise ValueError("document critic returned a non-object response")
        parsed = LLMCriticPayload.model_validate(payload)
    except (LLMError, ValidationError, ValueError, TypeError) as exc:
        return _unavailable_critique(
            document,
            corpus_version=corpus.version,
            retrieved_ids=retrieved_ids,
            settings=settings,
            reason=f"{type(exc).__name__}: {exc}",
            store=store,
        )

    issues: list[dict[str, Any]] = []
    removed_performance = False
    dropped_citations = 0
    for raw_issue in parsed.issues:
        if raw_issue.chunk not in valid_chunk_ids:
            dropped_citations += 1
            continue
        issue = raw_issue.model_dump()
        if _has_performance_language(_issue_text(issue)):
            removed_performance = True
            continue
        issues.append(issue)

    summary_text, removed_summary = _clean_text(parsed.summary)
    removed_performance = removed_performance or removed_summary

    missing_information: list[str] = []
    for item in parsed.missing_information:
        text = str(item).strip()
        if not text or _has_performance_language(text):
            removed_performance = removed_performance or bool(text)
            continue
        missing_information.append(text)

    if removed_performance and not summary_text:
        summary_text = (
            "The review contained out-of-scope performance language, which was "
            "removed; the method remains unassessed."
        )
    if dropped_citations and not summary_text:
        summary_text = (
            "The review cited chunks that were not supplied; no valid issue "
            "remains."
        )

    verdict = _merge_verdict(
        issues=issues,
        missing_information=missing_information,
        llm_verdict=parsed.verdict,
    )
    critique = DocumentCritique(
        document_id=document.id,
        corpus_version=corpus.version,
        verdict=verdict,  # type: ignore[arg-type]
        issues=issues,
        missing_information=missing_information,
        summary=summary_text,
        model=_model_name(settings),
        created_at=utc_now_iso(),
    )
    return _store(store, critique)


__all__ = [
    "DOCUMENT_CRITIC_PROMPT",
    "DOCUMENT_CRITIC_SYSTEM_PROMPT",
    "run_document_critic",
]
