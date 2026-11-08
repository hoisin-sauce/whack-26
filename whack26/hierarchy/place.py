"""Place documents at the highest field that directly covers them.

The walk starts at the roots and descends only while a child is directly
related *and* the document is more specific than the current field.  When no
existing field expresses the document's concept, a child (or root) is branched
under the highest related ancestor.
"""

from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..embeddings import cosine
from ..research.scrape.models import Document
from ..store import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    Field,
    FieldStore,
)
from .types import HierarchyError, PlacementResult

logger = logging.getLogger(__name__)

SpecificityRelation = Literal[
    "attach_here",
    "needs_child",
    "new_field_is_parent",
    "new_field_is_child",
]

SPECIFICITY_RELATIONS: tuple[str, ...] = (
    "attach_here",
    "needs_child",
    "new_field_is_parent",
    "new_field_is_child",
)

SPECIFICITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": list(SPECIFICITY_RELATIONS),
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["relation", "confidence", "reason"],
    "propertyOrdering": ["relation", "confidence", "reason"],
}

SPECIFICITY_SYSTEM_PROMPT = (
    "You are deciding where a document belongs in a dynamic topic hierarchy. "
    "Prefer the highest field that genuinely covers the document. Only descend "
    "when the document is clearly about a narrower concept. Never invent a "
    "concept that the summary does not support. Never create a root when an "
    "existing root already covers the document. The current field is the field "
    "under consideration; the candidate is the direct child with the best "
    "similarity. Return `attach_here` when the document belongs at the current "
    "field, `needs_child` when it needs a child under the current field, "
    "`new_field_is_parent` when the current/new field is more general than the "
    "candidate, and `new_field_is_child` when the current/new field is narrower "
    "than the candidate."
)

FIELD_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": MAX_NAME_LENGTH},
        "description": {"type": "string", "maxLength": MAX_DESCRIPTION_LENGTH},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["name", "description", "reason"],
    "propertyOrdering": ["name", "description", "reason"],
}

FIELD_PROPOSAL_SYSTEM_PROMPT = (
    "You are proposing one field in a dynamic topic hierarchy. The new field "
    "must describe the supplied document without repeating its filename or a "
    "raw column name. Names must be human-readable and at most five words. "
    "Descriptions must be one or two sentences and must distinguish the field "
    "from its siblings. Prefer a broad, reusable concept over a one-off title."
)


class SpecificityJudgement(BaseModel):
    """The LLM's relation verdict for one placement ambiguity."""

    relation: SpecificityRelation
    confidence: float = 0.0
    reason: str = ""


class FieldProposal(BaseModel):
    """A proposed new field name and description."""

    name: str
    description: str
    reason: str = ""


def effective_max_depth(store: FieldStore, settings: Settings) -> int:
    """Return the depth cap enforced by both the store and the settings."""
    return min(int(store.max_depth), int(settings.max_depth))


def threshold_for_level(settings: Settings, level: int) -> float:
    """Return the directly-related threshold for ``level`` (1-based)."""
    bounded = max(1, min(int(level), 4))
    return float(getattr(settings, f"hierarchy_tau_{bounded}"))


def field_score(
    embedding: Sequence[float], field: Field, settings: Settings
) -> float:
    """Blend the description similarity and the member centroid."""
    vector = list(embedding)
    desc_similarity = cosine(vector, field.description_embedding)
    if field.centroid:
        centroid_similarity = cosine(vector, field.centroid)
    else:
        centroid_similarity = desc_similarity
    return (
        settings.desc_weight * desc_similarity
        + settings.centroid_weight * centroid_similarity
    )


def ancestor_chain(store: FieldStore, field: Field) -> list[Field]:
    """Return the root-to-``field`` chain, rejecting cycles and gaps."""
    chain: list[Field] = []
    current: Field | None = field
    seen: set[str] = set()
    while current is not None:
        if current.id in seen:
            raise HierarchyError(f"field tree contains a cycle at {current.id}")
        seen.add(current.id)
        chain.append(current)
        if current.parent_id is None:
            break
        parent = store.get_field(current.parent_id)
        if parent is None:
            raise HierarchyError(
                f"field {current.id} references missing parent {current.parent_id}"
            )
        current = parent
    chain.reverse()
    return chain


def ancestor_ids(store: FieldStore, field: Field) -> list[str]:
    """Return the root-to-leaf id path for ``field``."""
    return [item.id for item in ancestor_chain(store, field)]


def subtree_fields(store: FieldStore, field_id: str) -> list[Field]:
    """Return ``field_id`` and every descendant in breadth-first order."""
    root = store.get_field(field_id)
    if root is None:
        return []
    fields = store.all_fields()
    children: dict[str | None, list[Field]] = {}
    for item in fields:
        children.setdefault(item.parent_id, []).append(item)
    ordered: list[Field] = [root]
    queue = [root]
    seen = {root.id}
    while queue:
        current = queue.pop(0)
        for child in children.get(current.id, []):
            if child.id in seen:
                continue
            seen.add(child.id)
            ordered.append(child)
            queue.append(child)
    return ordered


def model_name(llm: Any, settings: Settings) -> str | None:
    """Return the model label recorded on audit rows."""
    override = getattr(settings, "hierarchy_judge_model", None)
    if override:
        return str(override)
    value = getattr(llm, "model", None)
    return str(value) if value else None


def _embed_text(
    embeddings: Any,
    store: FieldStore,
    text: str,
    *,
    fallback: Sequence[float] | None = None,
) -> list[float]:
    if embeddings is not None and hasattr(embeddings, "embed"):
        try:
            vectors = embeddings.embed([text])
        except Exception as exc:  # pragma: no cover - provider defensive path
            logger.warning("hierarchy_embedding_failed: %s", exc)
            vectors = []
        if vectors:
            vector = [float(value) for value in vectors[0]]
            if len(vector) == store.dims:
                return vector
    if fallback is not None and len(fallback) == store.dims:
        return [float(value) for value in fallback]
    raise HierarchyError("could not embed a field description at the store dimensions")


def document_embedding(
    document: Document, *, store: FieldStore, embeddings: Any
) -> list[float]:
    """Return the document embedding, embedding the summary when missing."""
    vector = [float(value) for value in document.summary_embedding]
    if not vector:
        text = document.summary or document.title
        if embeddings is None or not hasattr(embeddings, "embed"):
            raise HierarchyError(
                f"document {document.id} has no embedding and no embedding client"
            )
        vectors = embeddings.embed([text])
        if not vectors:
            raise HierarchyError(
                f"embedding provider returned no vector for document {document.id}"
            )
        vector = [float(value) for value in vectors[0]]
    if len(vector) != store.dims:
        raise HierarchyError(
            f"document {document.id} embedding has {len(vector)} dimensions; "
            f"the field store expects {store.dims}"
        )
    return vector


def specificity_judge(
    summary: str,
    *,
    current: Field,
    candidate: Field | None,
    llm: Any,
    settings: Settings,
    purpose: str = "hierarchy_specificity",
) -> SpecificityJudgement | None:
    """Ask the LLM whether the document is more specific than ``current``.

    A failed or invalid response returns ``None`` so callers fall back to the
    deterministic margin rule and record ``hierarchy_judge_fallback``.
    """
    if llm is None:
        return None
    candidate_text = (
        "(none; the document reached a leaf field)" if candidate is None else
        f"- Level {candidate.level}: {candidate.name} — {candidate.description}"
    )
    user = (
        f"Document summary:\n{summary.strip() or '(empty summary)'}\n\n"
        f"Current field:\n- Level {current.level}: {current.name} — "
        f"{current.description}\n\n"
        f"Best direct child:\n{candidate_text}\n\n"
        "Decide the relation for this document. Prefer the highest field that "
        "genuinely covers it; only descend for a clearly narrower concept. "
        "Return one JSON object with `relation`, `confidence`, and a short "
        "`reason`."
    )
    try:
        payload = llm.generate_json(
            system=SPECIFICITY_SYSTEM_PROMPT,
            user=user,
            schema=SPECIFICITY_SCHEMA,
            model=getattr(settings, "hierarchy_judge_model", None),
            max_tokens=512,
            purpose=purpose,
        )
    except Exception as exc:  # noqa: BLE001 - provider-neutral fallback
        logger.warning("hierarchy_judge_fallback: LLM call failed: %s", exc)
        return None
    try:
        return SpecificityJudgement.model_validate(payload)
    except ValidationError as exc:
        logger.warning("hierarchy_judge_fallback: invalid judge response: %s", exc)
        return None


def is_more_specific(
    document: Document,
    current: Field,
    best: Field | None = None,
    *,
    current_score: float | None = None,
    best_score: float | None = None,
    llm: Any = None,
    settings: Settings | None = None,
) -> bool:
    """Return True when the document belongs below ``current``.

    ``best`` is the best directly-related child.  With no child, the
    deterministic fallback treats a weakly-related current field as evidence
    that the document names a narrower concept.  With a child, score gaps
    larger than ``specificity_margin`` decide without an LLM call; only the
    ambiguous band calls the specificity judge.
    """
    if settings is None:
        settings = Settings()
    if current_score is None:
        vector = [float(value) for value in document.summary_embedding]
        if vector:
            current_score = field_score(vector, current, settings)
    if best is None:
        if current_score is None:
            return False
        tau = threshold_for_level(settings, current.level)
        return (current_score - tau) < settings.specificity_margin

    if best_score is None:
        vector = [float(value) for value in document.summary_embedding]
        if vector:
            best_score = field_score(vector, best, settings)
    if current_score is None or best_score is None:
        return False

    difference = best_score - current_score
    if difference > settings.specificity_margin:
        return True
    if -difference > settings.specificity_margin:
        return False

    judgement = specificity_judge(
        document.summary or document.title,
        current=current,
        candidate=best,
        llm=llm,
        settings=settings,
    )
    if judgement is None:
        logger.info("hierarchy_judge_fallback: deterministic margin used")
        return best_score > current_score
    return judgement.relation in {"needs_child", "new_field_is_child"}


def _clean_name(name: str, *, parent: Field | None, document: Document) -> str:
    cleaned = " ".join((name or "").split())
    if not cleaned:
        if parent is None:
            return "General Research"
        words = (document.title or document.summary or "").split()
        cleaned = " ".join(words[:5]) or "General Topics"
    if len(cleaned) > MAX_NAME_LENGTH:
        cleaned = cleaned[:MAX_NAME_LENGTH].rstrip()
    if not any(character.isalnum() for character in cleaned):
        cleaned = "General Research" if parent is None else "General Topics"
    return cleaned


def _clean_description(description: str, *, name: str, document: Document) -> str:
    cleaned = " ".join((description or "").split())
    if not cleaned:
        cleaned = (document.summary or document.title or "").strip()
    if not cleaned:
        cleaned = f"Documents about {name}."
    if len(cleaned) > MAX_DESCRIPTION_LENGTH:
        cleaned = cleaned[:MAX_DESCRIPTION_LENGTH].rstrip()
    if len(cleaned) < 10:
        cleaned = f"Documents about {name}."
    return cleaned


def propose_field(
    document: Document,
    *,
    parent: Field | None,
    store: FieldStore,
    llm: Any,
    embeddings: Any,
    settings: Settings,
) -> tuple[str, str, list[float]]:
    """Propose a new field name, description, and description embedding.

    The LLM is advisory: an unavailable or invalid response falls back to a
    deterministic name derived from the document title/summary.
    """
    name = ""
    description = ""
    if llm is not None:
        user = (
            f"Document title: {document.title}\n\n"
            f"Document summary:\n{document.summary}\n\n"
            f"Intended parent field: "
            + (
                "(root level; no parent)"
                if parent is None
                else f"Level {parent.level}: {parent.name} — {parent.description}"
            )
            + "\n\nReturn one JSON object with `name`, `description`, and a short "
            "`reason`."
        )
        try:
            payload = llm.generate_json(
                system=FIELD_PROPOSAL_SYSTEM_PROMPT,
                user=user,
                schema=FIELD_PROPOSAL_SCHEMA,
                model=getattr(settings, "hierarchy_judge_model", None),
                max_tokens=512,
                purpose="hierarchy_field",
            )
            proposal = FieldProposal.model_validate(payload)
            name = proposal.name.strip()
            description = proposal.description.strip()
        except Exception as exc:  # noqa: BLE001 - deterministic fallback
            logger.warning("hierarchy_field_fallback: %s", exc)

    cleaned_name = _clean_name(name, parent=parent, document=document)
    cleaned_description = _clean_description(
        description, name=cleaned_name, document=document
    )
    vector = _embed_text(
        embeddings,
        store,
        f"{cleaned_name}. {cleaned_description}",
        fallback=document.summary_embedding,
    )
    return cleaned_name, cleaned_description, vector


def _paths_module() -> Any | None:
    """Return the optional ``hierarchy.paths`` package module, if it exists."""
    try:
        return importlib.import_module("whack26.hierarchy.paths")
    except ImportError:
        return None


def _refresh_dataset_path(
    store: FieldStore,
    document: Document,
    path_ids: Sequence[str],
    *,
    run_id: str | None = None,
) -> None:
    """Keep ``datasets.path_json`` in sync with a document's field path."""
    if document.kind != "dataset":
        return
    paths = _paths_module()
    recompute = getattr(paths, "recompute_paths", None) if paths is not None else None
    if callable(recompute):
        try:
            recompute(store, field_ids=path_ids, run_id=run_id)
            return
        except Exception as exc:  # noqa: BLE001 - fall through to the direct write
            logger.warning("hierarchy_paths_fallback: %s", exc)

    connection = getattr(store, "_connection", None)
    transaction = getattr(store, "_transaction", None)
    if not callable(connection) or not callable(transaction):
        return
    try:
        with transaction():
            connection().execute(
                "UPDATE datasets SET path_json = ? WHERE document_id = ?",
                (json.dumps(list(path_ids)), document.id),
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("hierarchy_dataset_path_update_failed: %s", exc)


def update_document_path(
    document: Document,
    field: Field,
    *,
    store: FieldStore,
    confidence: float = 1.0,
    reason: str = "",
    operation: str = "reassign_leaf",
    run_id: str | None = None,
    model: str | None = None,
    log: bool = True,
) -> list[str]:
    """Write ``document``'s leaf, membership rows, paths, and centroids."""
    path_ids = ancestor_ids(store, field)
    affected = store.assign_document_path(
        document.id, path_ids, confidence=float(confidence)
    )
    store.set_leaf(document.id, field.id)
    _refresh_dataset_path(store, document, path_ids, run_id=run_id)
    if affected:
        store.recompute_centroids(affected)
    if log:
        store.log_field_operation(
            operation=operation,
            field_id=field.id,
            document_id=document.id,
            affected_documents=1,
            reason=reason,
            model=model,
            run_id=run_id,
        )
    return path_ids


def attach_document(
    document: Document,
    field: Field,
    *,
    store: FieldStore,
    settings: Settings,
    confidence: float | None = None,
    created_fields: Sequence[str] = (),
    promoted: bool = False,
    reason: str = "",
    llm: Any = None,
    run_id: str | None = None,
) -> PlacementResult:
    """Attach ``document`` to ``field`` and return the placement result."""
    if confidence is None:
        vector = [float(value) for value in document.summary_embedding]
        if vector and len(vector) == store.dims:
            confidence = field_score(vector, field, settings)
        else:
            confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    path_ids = update_document_path(
        document,
        field,
        store=store,
        confidence=confidence,
        reason=reason or "attached to the highest directly related field",
        operation="reassign_leaf",
        run_id=run_id,
        model=model_name(llm, settings),
    )
    store.bump_hierarchy_version()
    return PlacementResult(
        document_id=document.id,
        leaf_field_id=field.id,
        path_ids=path_ids,
        confidence=confidence,
        created_fields=list(created_fields),
        promoted=promoted,
        reason=reason,
    )


def _branch_and_attach(
    document: Document,
    branch: Any,
    *,
    store: FieldStore,
    settings: Settings,
    llm: Any,
    reason: str,
    run_id: str | None = None,
) -> PlacementResult:
    field = store.get_field(branch.field_id)
    if field is None:
        raise HierarchyError(f"branched field disappeared: {branch.field_id}")
    created = [branch.field_id] if branch.created else []
    return attach_document(
        document,
        field,
        store=store,
        settings=settings,
        created_fields=created,
        promoted=bool(branch.promoted),
        reason=branch.reason or reason,
        llm=llm,
        run_id=run_id,
    )


def place_document(
    document: Document,
    *,
    store: FieldStore,
    llm: Any,
    embeddings: Any,
    settings: Settings,
    run_id: str | None = None,
) -> PlacementResult:
    """Place ``document`` at the highest directly related field."""
    # Imported lazily: operations imports the placement helpers at module load.
    from .operations import branch_field

    embedding = document_embedding(document, store=store, embeddings=embeddings)
    max_depth = effective_max_depth(store, settings)
    current: Field | None = None
    current_score: float | None = None

    while True:
        candidates = store.children(current.id if current else None)
        scored = [
            (field_score(embedding, candidate, settings), candidate)
            for candidate in candidates
        ]
        direct = [
            (score, candidate)
            for score, candidate in scored
            if score >= threshold_for_level(settings, candidate.level)
        ]

        if not direct:
            if current is None:
                roots = store.roots()
                if not roots:
                    name, description, vector = propose_field(
                        document,
                        parent=None,
                        store=store,
                        llm=llm,
                        embeddings=embeddings,
                        settings=settings,
                    )
                    branch = branch_field(
                        parent=None,
                        name=name,
                        description=description,
                        description_embedding=vector,
                        document=document,
                        store=store,
                        llm=llm,
                        settings=settings,
                        run_id=run_id,
                    )
                    return _branch_and_attach(
                        document,
                        branch,
                        store=store,
                        settings=settings,
                        llm=llm,
                        reason="created a root field",
                        run_id=run_id,
                    )
                # Existing roots are a better fallback than creating a new
                # top-level domain for every weakly related document. A new
                # root is only created when the tree is empty.
                nearest = max(scored, key=lambda item: item[0])[1]
                return attach_document(
                    document,
                    nearest,
                    store=store,
                    settings=settings,
                    confidence=field_score(embedding, nearest, settings),
                    reason="no root was directly related; attached to the nearest root",
                    llm=llm,
                    run_id=run_id,
                )

            if (
                current.level < max_depth
                and is_more_specific(
                    document,
                    current,
                    current_score=current_score,
                    llm=llm,
                    settings=settings,
                )
            ):
                name, description, vector = propose_field(
                    document,
                    parent=current,
                    store=store,
                    llm=llm,
                    embeddings=embeddings,
                    settings=settings,
                )
                branch = branch_field(
                    parent=current,
                    name=name,
                    description=description,
                    description_embedding=vector,
                    document=document,
                    store=store,
                    llm=llm,
                    settings=settings,
                    run_id=run_id,
                )
                return _branch_and_attach(
                    document,
                    branch,
                    store=store,
                    settings=settings,
                    llm=llm,
                    reason="branched under the highest related field",
                    run_id=run_id,
                )
            return attach_document(
                document,
                current,
                store=store,
                settings=settings,
                confidence=current_score,
                reason="attached at the deepest directly related concept",
                llm=llm,
                run_id=run_id,
            )

        best_score, best_field = max(direct, key=lambda item: item[0])
        if (
            current is not None
            and current_score is not None
            and not is_more_specific(
                document,
                current,
                best_field,
                current_score=current_score,
                best_score=best_score,
                llm=llm,
                settings=settings,
            )
        ):
            return attach_document(
                document,
                current,
                store=store,
                settings=settings,
                confidence=current_score,
                reason="current field is the highest directly related field",
                llm=llm,
                run_id=run_id,
            )

        if best_field.level >= max_depth:
            return attach_document(
                document,
                best_field,
                store=store,
                settings=settings,
                confidence=best_score,
                reason="maximum depth reached; attached to the best direct field",
                llm=llm,
                run_id=run_id,
            )

        current = best_field
        current_score = best_score


__all__ = [
    "FIELD_PROPOSAL_SCHEMA",
    "FIELD_PROPOSAL_SYSTEM_PROMPT",
    "SPECIFICITY_SCHEMA",
    "SPECIFICITY_SYSTEM_PROMPT",
    "SpecificityJudgement",
    "ancestor_chain",
    "ancestor_ids",
    "attach_document",
    "document_embedding",
    "effective_max_depth",
    "field_score",
    "is_more_specific",
    "model_name",
    "place_document",
    "propose_field",
    "specificity_judge",
    "subtree_fields",
    "threshold_for_level",
    "update_document_path",
]
