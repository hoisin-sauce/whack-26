"""Summarize a scraped document with one structured LLM call.

The summary is the semantic handle for a resource: the frontier compares its
embedding to the target, and the comparison/critique stages read its claims,
methods, and credibility signals. Keep the prompt bounded: title, metadata, the
target, and at most ``max_input_chars`` characters of public text.
"""

from __future__ import annotations

import math
from typing import Any

from ...common import redact_secrets
from ...config import Settings
from ...embeddings import EmbeddingClient, l2_normalize
from ...llm import LLMClient
from .models import Document, SummaryResult

MAX_SUMMARY_CHARS = 1200

SUMMARY_SYSTEM_PROMPT = (
    "You summarize scientific papers, datasets, searches, and public web pages "
    "for a research assistant. Use only the supplied metadata and public text. "
    "Do not invent findings, methods, populations, dates, or links. Write a "
    "neutral 2-4 sentence summary, then list the claims the resource makes, the "
    "methods it uses, the population and time period when stated, any links it "
    "explicitly references, and the credibility signals you can observe. Use "
    "null for an unknown population or time period. Return JSON matching the "
    "schema."
)

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "maxLength": MAX_SUMMARY_CHARS,
            "description": "A neutral 2-4 sentence summary of the resource.",
        },
        "claims": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
        },
        "methods": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
        },
        "population": {"type": "string", "maxLength": 400},
        "time_period": {"type": "string", "maxLength": 200},
        "links": {
            "type": "array",
            "items": {"type": "string", "maxLength": 2048},
        },
        "credibility_signals": {
            "type": "object",
            "properties": {
                "peer_reviewed": {"type": "boolean"},
                "publisher": {"type": "string", "maxLength": 400},
                "retracted": {"type": "boolean"},
            },
        },
    },
    "required": [
        "summary",
        "claims",
        "methods",
        "population",
        "time_period",
        "links",
        "credibility_signals",
    ],
    "propertyOrdering": [
        "summary",
        "claims",
        "methods",
        "population",
        "time_period",
        "links",
        "credibility_signals",
    ],
}

_ABSTRACT_KEYS = ("abstract", "description")
_PUBLIC_TEXT_KEYS = ("public_text", "text", "content", "body")
_TEXT_KEYS = frozenset({*_ABSTRACT_KEYS, *_PUBLIC_TEXT_KEYS, "links"})


def _first_text(metadata: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata_lines(document: Document) -> list[str]:
    """Render scalar metadata without dumping full text or link lists."""
    lines: list[str] = []
    for key in sorted(document.metadata):
        if key in _TEXT_KEYS:
            continue
        value = document.metadata[key]
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        rendered = str(value).strip()
        if rendered:
            lines.append(f"{key}: {rendered}")
    return lines


def _build_user_prompt(
    document: Document, target: str, max_input_chars: int
) -> str:
    abstract = _first_text(document.metadata, _ABSTRACT_KEYS)
    public_text = _first_text(document.metadata, _PUBLIC_TEXT_KEYS)
    public_text = public_text[:max_input_chars]

    lines = [
        "Summarize the following research resource for semantic scraping.",
        "",
        f"Title: {document.title}",
        f"Kind: {document.kind}",
        f"Source: {document.source}",
    ]
    if document.source_id:
        lines.append(f"Source id: {document.source_id}")
    if document.authors:
        lines.append("Authors: " + ", ".join(str(author) for author in document.authors))
    if document.publisher:
        lines.append(f"Publisher: {document.publisher}")
    if document.published_at:
        lines.append(f"Published: {document.published_at}")
    if document.license:
        lines.append(f"License: {document.license}")
    if document.canonical_url:
        lines.append(f"URL: {document.canonical_url}")
    lines.extend(_metadata_lines(document))

    lines.extend(
        [
            "",
            "Abstract:",
            abstract or "(none)",
            "",
            f"Public text (first {len(public_text)} characters):",
            public_text or "(none)",
            "",
            "Target the summary should be relevant to:",
            target.strip() or "(none)",
            "",
            "Return JSON matching the schema.",
        ]
    )
    return "\n".join(lines)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for entry in value:
            text = str(entry).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def summarize_document(
    document: Document,
    target: str,
    llm: LLMClient,
    settings: Settings,
    *,
    max_input_chars: int = 2000,
) -> SummaryResult:
    """Return a validated summary for ``document``.

    Raises ``ValueError`` when the model returns no summary or a summary longer
    than :data:`MAX_SUMMARY_CHARS`. The optional ``settings`` argument is kept
    for interface symmetry with the rest of the scraping stage.
    """
    if max_input_chars < 0:
        raise ValueError("max_input_chars must be non-negative")
    del settings  # reserved for model selection and prompt tuning

    prompt = redact_secrets(_build_user_prompt(document, target, max_input_chars))
    payload = llm.generate_json(
        system=SUMMARY_SYSTEM_PROMPT,
        user=prompt,
        schema=SUMMARY_SCHEMA,
        purpose="document_summary",
    )

    raw_summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(raw_summary, str) or not raw_summary.strip():
        raise ValueError(
            "document summary is empty; a non-empty summary is required"
        )
    summary_text = raw_summary.strip()
    if len(summary_text) > MAX_SUMMARY_CHARS:
        raise ValueError(
            "document summary is too long: "
            f"{len(summary_text)} characters exceeds the maximum of "
            f"{MAX_SUMMARY_CHARS}"
        )

    credibility = payload.get("credibility_signals")
    if not isinstance(credibility, dict):
        credibility = {}

    return SummaryResult(
        summary=summary_text,
        claims=_as_string_list(payload.get("claims")),
        methods=_as_string_list(payload.get("methods")),
        population=_as_optional_string(payload.get("population")),
        time_period=_as_optional_string(payload.get("time_period")),
        links=_as_string_list(payload.get("links")),
        credibility_signals=dict(credibility),
    )


def embed_summary(summary: str, embeddings: EmbeddingClient) -> list[float]:
    """Embed one summary and return a finite, L2-normalized vector."""
    text = (summary or "").strip()
    if not text:
        raise ValueError("cannot embed an empty summary")

    batch = embeddings.embed([text])
    if not isinstance(batch, (list, tuple)) or len(batch) != 1:
        count = len(batch) if isinstance(batch, (list, tuple)) else type(batch).__name__
        raise ValueError(
            f"expected exactly one embedding vector, got {count}"
        )

    try:
        vector = [float(value) for value in batch[0]]
    except TypeError as exc:
        raise ValueError(
            "embedding vector must be a sequence of numbers"
        ) from exc
    except (ValueError, OverflowError) as exc:
        raise ValueError("embedding vector contains non-numeric values") from exc
    if not vector:
        raise ValueError("embedding vector is empty")
    for value in vector:
        if not math.isfinite(value):
            raise ValueError("embedding vector contains a non-finite value")
    return l2_normalize(vector)
