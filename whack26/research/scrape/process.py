"""Turn a fetched document into a summarized, embedded document.

The function is deliberately pure with respect to its input: the caller keeps
the staged document, and the returned copy is what gets stored. Extracted links
live in ``metadata["links"]`` so the frontier can expand them without a second
parse.
"""

from __future__ import annotations

import copy
from typing import Any

from ...config import Settings
from ...embeddings import EmbeddingClient
from ...llm import LLMClient
from .models import Document, SummaryResult
from .summarize import embed_summary, summarize_document


def embedding_model_name(settings: Settings) -> str:
    """Return the model identifier recorded next to an embedding."""
    if settings.embed_provider == "gemini":
        return settings.gemini_embed_model
    return settings.embed_provider


def _merge_links(existing: Any, extracted: list[str]) -> list[Any]:
    """Preserve existing links, append new ones, and drop exact duplicates."""
    merged: list[Any] = []
    for source in (existing, extracted):
        if source is None:
            continue
        items = source if isinstance(source, (list, tuple)) else [source]
        for item in items:
            if item not in merged:
                merged.append(item)
    return merged


def process_document(
    document: Document,
    target: str,
    llm: LLMClient,
    embeddings: EmbeddingClient,
    settings: Settings,
    *,
    summary: SummaryResult | dict[str, Any] | None = None,
) -> Document:
    """Return a summarized copy of ``document``.

    When ``summary`` is supplied the LLM is not called; this is the cache/dedup
    path used by the frontier. All fields other than the summary, embedding,
    embedding model, status, and extracted links are preserved.
    """
    if summary is None:
        resolved = summarize_document(document, target, llm, settings)
    elif isinstance(summary, SummaryResult):
        resolved = summary
    elif isinstance(summary, dict):
        resolved = SummaryResult.model_validate(summary)
    else:
        raise TypeError("summary must be a SummaryResult, a mapping, or None")

    summary_embedding = embed_summary(resolved.summary, embeddings)

    metadata = copy.deepcopy(document.metadata)
    metadata["links"] = _merge_links(metadata.get("links"), resolved.links)

    return document.model_copy(
        deep=True,
        update={
            "summary": resolved.summary,
            "summary_embedding": summary_embedding,
            "embedding_model": embedding_model_name(settings),
            "metadata": metadata,
            "status": "summarized",
        }
    )
