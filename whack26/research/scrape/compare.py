"""Cached pairwise document comparisons.

Comparisons are expensive LLM calls, so every result is cached in
``document_comparisons`` under a canonical, symmetric key.  The deterministic
cosine similarity is computed locally from the stored summary embeddings; the
LLM only supplies the verdict, rationale, and supporting evidence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ...common import redact_secrets, utc_now_iso
from ...config import Settings
from ...embeddings import cosine
from ...llm import LLMError
from .models import ComparisonVerdict, Document, DocumentComparison

COMPARISON_KINDS: tuple[str, ...] = (
    "dataset_relevance",
    "method_overlap",
    "finding_agreement",
    "population_overlap",
    "contradiction",
)

COMPARISON_VERDICTS: tuple[str, ...] = (
    "supports",
    "contradicts",
    "orthogonal",
    "insufficient",
)

DEFAULT_METHOD_VERSION = "1.0.0"
SIMILARITY_TOLERANCE = 1e-6
_MAX_DOCUMENT_CHARS = 2_000

COMPARISON_SYSTEM_PROMPT = (
    "You compare two research documents for a modelling research pipeline. "
    "Use only the supplied titles, metadata, and summaries; do not use outside "
    "knowledge and do not invent details. Choose exactly one verdict from "
    "`supports`, `contradicts`, `orthogonal`, or `insufficient`. Use "
    "`insufficient` when the summaries are too thin to decide, and state what "
    "information would be needed. Every piece of evidence must be a short "
    "quote or close paraphrase taken from one of the two documents. The task "
    "kind tells you which relationship to judge. Return JSON matching the "
    "schema."
)

COMPARISON_PROMPT = (
    "Comparison kind: {kind}\n"
    "{kind_instruction}\n\n"
    "Document A:\n{left}\n\n"
    "Document B:\n{right}\n\n"
    "Compare the two documents for this kind only. A deterministic cosine "
    "similarity of their summaries is supplied so you can calibrate your "
    "verdict: {similarity:.4f} (0.0 means unrelated or unavailable).\n\n"
    "Return a verdict, a concise rationale, and evidence drawn from the "
    "documents. Use `insufficient` if either summary is missing or too thin."
)

COMPARISON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": list(COMPARISON_VERDICTS),
        },
        "rationale": {"type": "string", "maxLength": 1200},
        "evidence": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
            "maxItems": 6,
        },
    },
    "required": ["verdict", "rationale", "evidence"],
    "propertyOrdering": ["verdict", "rationale", "evidence"],
}

_KIND_INSTRUCTIONS = {
    "dataset_relevance": (
        "Judge how the documents relate to the dataset under study. "
        "`supports` means the other document informs the dataset's population, "
        "variables, or modelling context; `contradicts` means it conflicts "
        "with the dataset's stated scope or measurements; `orthogonal` means "
        "it is unrelated; `insufficient` means the summaries do not say."
    ),
    "method_overlap": (
        "Judge whether the two documents use the same or comparable methods. "
        "`supports` means the methods overlap or corroborate each other; "
        "`contradicts` means the methods are incompatible or answer the same "
        "question with a conflicting design; `orthogonal` means the methods "
        "differ but are compatible; `insufficient` means the method is unclear."
    ),
    "finding_agreement": (
        "Judge whether the two documents report compatible findings. "
        "`supports` means the findings agree; `contradicts` means the findings "
        "conflict; `orthogonal` means they address different outcomes; "
        "`insufficient` means the findings are not stated clearly enough."
    ),
    "population_overlap": (
        "Judge whether the two documents study the same population or setting. "
        "`supports` means the populations overlap; `contradicts` means they are "
        "incompatible in a way that matters; `orthogonal` means different but "
        "unrelated populations; `insufficient` means the population is unclear."
    ),
    "contradiction": (
        "Judge whether the documents' claims are mutually inconsistent. "
        "`contradicts` means the claims cannot both hold; `supports` means they "
        "are consistent; `orthogonal` means they address unrelated questions; "
        "`insufficient` means too little of the claims is reported."
    ),
}


class LLMComparisonPayload(BaseModel):
    """Validated shape of the LLM's comparison response."""

    verdict: ComparisonVerdict = "insufficient"
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)


def canonical_pair(left_id: str, right_id: str) -> tuple[str, str]:
    """Return the pair in the canonical (sorted) cache-key order."""
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


def deterministic_similarity(left: Document, right: Document) -> float:
    """Cosine similarity of the summary embeddings, or ``0.0`` if unavailable."""
    left_vector = left.summary_embedding
    right_vector = right.summary_embedding
    if not left_vector or not right_vector:
        return 0.0
    try:
        return float(cosine(left_vector, right_vector))
    except (TypeError, ValueError):
        return 0.0


def _method_version(settings: Settings) -> str:
    return str(
        getattr(settings, "scrape_method_version", DEFAULT_METHOD_VERSION)
        or DEFAULT_METHOD_VERSION
    )


def _shorten(text: str, limit: int = _MAX_DOCUMENT_CHARS) -> str:
    cleaned = redact_secrets(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _document_block(document: Document) -> str:
    details = [
        f"id: {document.id}",
        f"kind: {document.kind}",
        f"source: {document.source}",
        f"title: {_shorten(document.title, 300)}",
    ]
    if document.authors:
        details.append("authors: " + ", ".join(document.authors[:12]))
    if document.published_at:
        details.append(f"published_at: {document.published_at}")
    if document.publisher:
        details.append(f"publisher: {_shorten(document.publisher, 200)}")
    summary = _shorten(document.summary)
    details.append("summary: " + (summary or "(no summary available)"))
    return "\n".join(details)


def comparison_user_prompt(
    left: Document,
    right: Document,
    kind: str,
    *,
    similarity: float,
) -> str:
    """Render the user prompt for one comparison."""
    return COMPARISON_PROMPT.format(
        kind=kind,
        kind_instruction=_KIND_INSTRUCTIONS[kind],
        left=_document_block(left),
        right=_document_block(right),
        similarity=similarity,
    )


def _cached_matches_similarity(cached: DocumentComparison, similarity: float) -> bool:
    return abs(float(cached.similarity) - float(similarity)) <= SIMILARITY_TOLERANCE


def _unavailable_comparison(
    *,
    left_id: str,
    right_id: str,
    kind: str,
    similarity: float,
    method_version: str,
    reason: str,
) -> DocumentComparison:
    detail = _shorten(reason, 400)
    return DocumentComparison(
        left_id=left_id,
        right_id=right_id,
        kind=kind,  # type: ignore[arg-type]
        similarity=similarity,
        verdict="insufficient",
        rationale=f"The comparison could not be completed: {detail}",
        evidence=[],
        method_version=method_version,
        created_at=utc_now_iso(),
    )


def compare_documents(
    left: Document,
    right: Document,
    kind: str,
    *,
    llm: Any,
    store: Any,
    settings: Settings,
    similarity: float | None = None,
) -> DocumentComparison:
    """Compare two documents, returning a cached result whenever possible.

    The pair is canonicalised by sorted document id so ``compare(a, b)`` and
    ``compare(b, a)`` share one cache entry.  A cached result is reused only
    while its deterministic similarity still matches the current embeddings;
    changing an embedding or the configured method version triggers a fresh
    LLM call.
    """
    if kind not in COMPARISON_KINDS:
        raise ValueError(
            f"unknown comparison kind {kind!r}; expected one of {COMPARISON_KINDS}"
        )

    left_id, right_id = canonical_pair(left.id, right.id)
    by_id = {left.id: left, right.id: right}
    ordered_left = by_id[left_id]
    ordered_right = by_id[right_id]

    score = (
        float(similarity)
        if similarity is not None
        else deterministic_similarity(ordered_left, ordered_right)
    )
    method_version = _method_version(settings)

    cached = store.get_comparison(left_id, right_id, kind, method_version)
    if cached is not None and _cached_matches_similarity(cached, score):
        return cached

    try:
        payload = llm.generate_json(
            system=COMPARISON_SYSTEM_PROMPT,
            user=comparison_user_prompt(
                ordered_left,
                ordered_right,
                kind,
                similarity=score,
            ),
            schema=COMPARISON_SCHEMA,
            max_tokens=1536,
            temperature=0.0,
            purpose="document_comparison",
        )
        if not isinstance(payload, dict):
            raise ValueError("comparison returned a non-object response")
        parsed = LLMComparisonPayload.model_validate(payload)
    except (LLMError, ValidationError, ValueError, TypeError) as exc:
        return _unavailable_comparison(
            left_id=left_id,
            right_id=right_id,
            kind=kind,
            similarity=score,
            method_version=method_version,
            reason=f"{type(exc).__name__}: {exc}",
        )

    comparison = DocumentComparison(
        left_id=left_id,
        right_id=right_id,
        kind=kind,  # type: ignore[arg-type]
        similarity=score,
        verdict=parsed.verdict,
        rationale=_shorten(parsed.rationale, 1200),
        evidence=[_shorten(item, 400) for item in parsed.evidence if item],
        method_version=method_version,
        created_at=utc_now_iso(),
    )
    store.put_comparison(comparison)
    return comparison


def compare_with_dataset(
    document: Document,
    dataset_document: Document,
    *,
    llm: Any,
    store: Any,
    settings: Settings,
) -> DocumentComparison:
    """Compare a document against the dataset currently being processed."""
    return compare_documents(
        document,
        dataset_document,
        "dataset_relevance",
        llm=llm,
        store=store,
        settings=settings,
    )


__all__ = [
    "COMPARISON_KINDS",
    "COMPARISON_PROMPT",
    "COMPARISON_SCHEMA",
    "COMPARISON_SYSTEM_PROMPT",
    "COMPARISON_VERDICTS",
    "LLMComparisonPayload",
    "canonical_pair",
    "compare_documents",
    "compare_with_dataset",
    "comparison_user_prompt",
    "deterministic_similarity",
]
