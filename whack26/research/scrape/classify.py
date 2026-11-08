"""Classify research documents into the existing dynamic field tree.

This is the document counterpart to :mod:`whack26.classify`.  Section 7.4 of
``docs/spec/semantic-scraping.md`` asks for the same top-k plus LLM choice
pattern, but with a research-document prompt.  A document is assigned to one
path through the existing fields; this module never creates a new root, and it
falls back deterministically when the LLM cannot produce a valid choice.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from pydantic import BaseModel, ValidationError, field_validator

from ...config import Settings
from ...embeddings import cosine
from ...llm import LLMError
from ...store import Field, FieldStore
from .models import Document

__all__ = [
    "DOCUMENT_CHOICE_SCHEMA",
    "DOCUMENT_SYSTEM_PROMPT",
    "DocumentChoice",
    "classify_document",
    "classify_document_hierarchically",
]

DOCUMENT_CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["assign", "create", "stop"]},
        "field_id": {"type": "string", "maxLength": 64},
        "name": {"type": "string", "maxLength": 80},
        "description": {"type": "string", "maxLength": 500},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["action", "reason"],
    "propertyOrdering": ["action", "name", "description", "field_id", "reason"],
}

DOCUMENT_SYSTEM_PROMPT = (
    "You are classifying a research document (a dataset, search, paper, or web "
    "resource) into a dynamic topic hierarchy. This is a research document, not "
    "a dataset: choose the field that best describes the document's topic and "
    "method, not its columns. The hierarchy is general-domain and is built from "
    "data; there is no fixed list. Choose an existing field only when it "
    "genuinely fits. Propose a new child only when no existing sibling is a "
    "reasonable fit and the child would be more specific than the current field. "
    "Never create a new root. Names must be human-readable, at most five words, "
    "and must not restate the document title. Descriptions must be one or two "
    "sentences and must distinguish the field from its siblings. Prefer `stop` "
    "when the current field already describes the document well."
)


class DocumentChoice(BaseModel):
    """The validated shape of one document-classification response."""

    action: Literal["assign", "create", "stop"]
    field_id: str | None = None
    name: str | None = None
    description: str | None = None
    reason: str = ""

    @field_validator("field_id", "name", "description", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _document_vector(
    document: Document, embeddings: Any, settings: Settings
) -> list[float]:
    """Return the stored summary embedding, or embed the document summary.

    The returned list is always a copy so the caller's :class:`Document` is
    never mutated.
    """
    stored = list(document.summary_embedding or [])
    if stored and len(stored) == settings.embed_dims:
        return stored

    text = document.summary.strip() or document.title.strip()
    if not text:
        raise ValueError(
            f"document {document.id!r} has no summary embedding, summary, or title"
        )
    vectors = embeddings.embed([text])
    if not vectors:
        raise ValueError("embedding provider returned no vectors")
    vector = [float(value) for value in vectors[0]]
    if len(vector) != settings.embed_dims:
        raise ValueError(
            f"embedding dimension mismatch: expected {settings.embed_dims}, "
            f"got {len(vector)}"
        )
    return vector


def _field_score(
    document_vector: Sequence[float], item: Field, settings: Settings
) -> float:
    description_similarity = cosine(list(document_vector), item.description_embedding)
    if item.member_count > 0 and item.centroid:
        centroid_similarity = cosine(list(document_vector), item.centroid)
    else:
        centroid_similarity = description_similarity
    return (
        settings.desc_weight * description_similarity
        + settings.centroid_weight * centroid_similarity
    )


def _candidates(
    store: FieldStore,
    document_vector: Sequence[float],
    current: Field | None,
    settings: Settings,
) -> list[tuple[Field, float]]:
    fields = store.children(current.id if current else None)
    ranked = [(item, _field_score(document_vector, item, settings)) for item in fields]
    ranked.sort(key=lambda pair: (-pair[1], pair[0].id))
    return ranked[: settings.top_k]


def _candidate_by_id(
    store: FieldStore,
    candidates: Sequence[tuple[Field, float]],
    field_id: str | None,
) -> tuple[Field | None, float]:
    for item, score in candidates:
        if item.id == field_id:
            # Re-read so callers that mutate the tree during a walk still see
            # current member counts and centroids.
            return store.get_field(item.id) or item, score
    return None, 0.0


def _format_path(path: Sequence[Field]) -> str:
    if not path:
        return "(root; no field assigned yet)"
    return "\n".join(
        f"- Level {item.level}: {item.name} — {item.description}" for item in path
    )


def _format_candidates(candidates: Sequence[tuple[Field, float]]) -> str:
    if not candidates:
        return "(no existing fields at this level)"
    return "\n".join(
        f"{index}. id={item.id} | name={item.name} | score={score:.3f} | "
        f"members={item.member_count} | description={item.description}"
        for index, (item, score) in enumerate(candidates, start=1)
    )


def _choice_prompt(
    *,
    document: Document,
    path: Sequence[Field],
    current: Field | None,
    candidates: Sequence[tuple[Field, float]],
    max_depth: int,
) -> str:
    level = 1 if current is None else current.level + 1
    if current is None:
        position = (
            "You are at the root level. `stop` and `create` are not allowed here: "
            "assign the document to one of the existing roots."
        )
    elif current.level >= max_depth:
        position = (
            "This is the maximum depth. Assign the document to an existing child "
            "or `stop` at the current field; `create` is not allowed."
        )
    else:
        position = (
            "Choose an existing child when it fits. Prefer `stop` when the current "
            "field already describes the document well; create a child only when "
            "the current field is clearly too broad."
        )
    summary = " ".join(document.summary.split())[:1200] or "(no summary)"
    return (
        f"Research document:\n"
        f"- id: {document.id}\n"
        f"- kind: {document.kind}\n"
        f"- source: {document.source}\n"
        f"- title: {document.title}\n"
        f"- summary: {summary}\n\n"
        f"Current field path:\n{_format_path(path)}\n\n"
        f"Current depth: {level} of {max_depth}\n"
        f"Candidates (children of the current field):\n"
        f"{_format_candidates(candidates)}\n\n"
        f"{position}\n"
        "Return one JSON object with `action`, and either `field_id` for `assign`, "
        "or `name` and `description` for `create`. Always include a short `reason`. "
        "Keep the response small: `reason` at most 25 words, `name` at most 5 "
        "words, and `description` at most 2 sentences. Do not repeat the document "
        "summary."
    )


def _validate_choice(
    choice: DocumentChoice,
    *,
    current: Field | None,
    candidates: Sequence[tuple[Field, float]],
    level: int,
    max_depth: int,
) -> str | None:
    if choice.action == "stop":
        if current is None:
            return (
                "`stop` is not allowed at the root; assign the document to one of "
                "the existing roots"
            )
        return None
    if choice.action == "assign":
        candidate_ids = {item.id for item, _ in candidates}
        if not choice.field_id or choice.field_id not in candidate_ids:
            return (
                "`field_id` must be one of the supplied candidates: "
                f"{sorted(candidate_ids)}"
            )
        return None
    # create
    if current is None:
        return "creating a new root is not allowed; assign to an existing root"
    if level > max_depth:
        return (
            f"`create` is not allowed at level {level}; the maximum depth is "
            f"{max_depth}"
        )
    if not choice.name or not choice.description:
        return "`create` requires both `name` and `description`"
    name = choice.name.strip()
    if len(name.split()) > 5:
        return "field names must be at most five words"
    if len(choice.description.strip()) < 10:
        return "field descriptions must be at least 10 characters"
    return None


def _ask_valid(
    *,
    llm: Any,
    prompt: str,
    current: Field | None,
    candidates: Sequence[tuple[Field, float]],
    level: int,
    max_depth: int,
) -> tuple[DocumentChoice | None, str | None]:
    last_error = "invalid response"
    for _ in range(2):
        try:
            payload = llm.generate_json(
                system=DOCUMENT_SYSTEM_PROMPT,
                user=prompt,
                schema=DOCUMENT_CHOICE_SCHEMA,
                max_tokens=1024,
                purpose="document_classification",
            )
        except LLMError:
            # A transient provider failure must not abort the scrape run; the
            # deterministic fallback below still produces a usable path.
            return None, "LLM call failed"

        try:
            choice = DocumentChoice.model_validate(payload)
        except ValidationError as exc:
            last_error = f"schema error: {exc}"
            prompt = (
                prompt
                + f"\n\nYour previous response was invalid: {last_error}. "
                "Return a valid JSON object matching the schema."
            )
            continue

        error = _validate_choice(
            choice,
            current=current,
            candidates=candidates,
            level=level,
            max_depth=max_depth,
        )
        if error is None:
            return choice, None
        last_error = error
        prompt = (
            prompt
            + f"\n\nYour previous response was invalid: {error}. "
            "Return a valid JSON object matching the schema."
        )
    return None, last_error


def _fallback_choice(
    current: Field | None, candidates: Sequence[tuple[Field, float]]
) -> DocumentChoice | None:
    if candidates:
        item, _ = candidates[0]
        return DocumentChoice(
            action="assign",
            field_id=item.id,
            reason="fallback to the top-scoring candidate",
        )
    if current is not None:
        return DocumentChoice(
            action="stop",
            reason="fallback to the current field",
        )
    return None


def _embed_one(embeddings: Any, text: str, settings: Settings) -> list[float]:
    vectors = embeddings.embed([text])
    if not vectors:
        raise ValueError("embedding provider returned no vectors")
    vector = [float(value) for value in vectors[0]]
    if len(vector) != settings.embed_dims:
        raise ValueError(
            f"embedding dimension mismatch: expected {settings.embed_dims}, "
            f"got {len(vector)}"
        )
    return vector


def _create_field(
    *,
    store: FieldStore,
    embeddings: Any,
    settings: Settings,
    parent: Field,
    name: str,
    description: str,
    document_vector: Sequence[float],
    parent_score: float,
) -> tuple[Field, float]:
    """Create a child, merging into an existing sibling when it is a duplicate."""
    exact = store.find_sibling(parent.id, name)
    if exact is not None:
        return exact, _field_score(document_vector, exact, settings)

    candidate_vector = _embed_one(embeddings, f"{name}. {description}", settings)
    siblings = store.children(parent.id)
    best: Field | None = None
    best_similarity = -1.0
    for sibling in siblings:
        similarity = cosine(candidate_vector, sibling.description_embedding)
        if similarity > best_similarity:
            best = sibling
            best_similarity = similarity
    if best is not None and best_similarity >= settings.duplicate_cosine:
        return best, _field_score(document_vector, best, settings)

    created = store.create_field(
        parent_id=parent.id,
        name=name,
        description=description,
        description_embedding=candidate_vector,
    )
    return created, parent_score


def classify_document(
    document: Document,
    *,
    store: FieldStore,
    llm: Any,
    embeddings: Any,
    settings: Settings,
) -> list[str]:
    """Assign one document to a field path and return the ids root → leaf.

    An empty list is returned when the field tree has no roots: this function
    never creates a new root.  The document must already be persisted in the
    store so that ``document_fields`` has a row to point at.
    """
    if not store.roots():
        return []

    effective_max_depth = min(
        settings.max_depth, getattr(store, "max_depth", settings.max_depth)
    )
    document_vector = _document_vector(document, embeddings, settings)
    current: Field | None = None
    path: list[Field] = []
    scores: list[float] = []

    while True:
        candidates = _candidates(store, document_vector, current, settings)
        level = 1 if current is None else current.level + 1
        prompt = _choice_prompt(
            document=document,
            path=path,
            current=current,
            candidates=candidates,
            max_depth=effective_max_depth,
        )
        choice, _ = _ask_valid(
            llm=llm,
            prompt=prompt,
            current=current,
            candidates=candidates,
            level=level,
            max_depth=effective_max_depth,
        )
        if choice is None:
            choice = _fallback_choice(current, candidates)
            if choice is None:
                break

        if choice.action == "stop":
            break

        if choice.action == "assign":
            selected, score = _candidate_by_id(store, candidates, choice.field_id)
        elif current is None or current.level >= effective_max_depth:
            # Validation should have rejected this, but never create a root or
            # a field past the maximum depth, even on a malformed response.
            fallback = _fallback_choice(current, candidates)
            if fallback is None or fallback.action == "stop":
                break
            selected, score = _candidate_by_id(store, candidates, fallback.field_id)
        else:
            parent_score = scores[-1] if scores else 0.5
            selected, score = _create_field(
                store=store,
                embeddings=embeddings,
                settings=settings,
                parent=current,
                name=choice.name or "",
                description=choice.description or "",
                document_vector=document_vector,
                parent_score=parent_score,
            )

        if selected is None:
            fallback = _fallback_choice(current, candidates)
            if fallback is None or fallback.action == "stop":
                break
            selected, score = _candidate_by_id(store, candidates, fallback.field_id)
            if selected is None:
                break

        path.append(selected)
        scores.append(score)
        current = selected
        if current.level >= effective_max_depth:
            break

    if not path:
        return []

    field_ids = [item.id for item in path]
    confidence = _clamp(sum(scores) / len(scores))
    store.assign_document_path(document.id, field_ids, confidence=confidence)
    return field_ids


def classify_document_hierarchically(
    document: Document,
    *,
    store: FieldStore,
    llm: Any,
    embeddings: Any,
    settings: Settings,
) -> list[str]:
    """Place one document through the dynamic hierarchy package.

    Unlike :func:`classify_document`, this path uses the devplan's
    highest-directly-related placement, branching, promotion, subtree
    re-levelling, and document reassignment. It requires the document to be
    persisted already. Failures are raised so the caller can fall back to the
    legacy create/assign walk.
    """
    from ...hierarchy.place import place_document

    result = place_document(
        document,
        store=store,
        llm=llm,
        embeddings=embeddings,
        settings=settings,
    )
    return list(result.path_ids)
