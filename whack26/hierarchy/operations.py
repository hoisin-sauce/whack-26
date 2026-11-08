"""Branch and promote operations for the dynamic field hierarchy.

Branching creates a child under the highest related field, merges a duplicate
sibling instead of creating another, and asks the specificity judge whether the
new field is actually the parent of its intended parent.  Promotion swaps the
two nodes, re-levels the old parent's subtree, and reassigns documents that are
closer to the new, more general field.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..common import utc_now_iso
from ..config import Settings
from ..embeddings import cosine
from ..research.scrape.models import Document
from ..store import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    Field,
    FieldStore,
)
from .place import (
    ancestor_ids,
    attach_document,
    effective_max_depth,
    field_score,
    model_name,
    specificity_judge,
    subtree_fields,
    threshold_for_level,
    update_document_path,
)
from .types import BranchResult, HierarchyError, PromotionResult


def _merge_with_sibling(
    *,
    store: FieldStore,
    sibling: Field,
    parent: Field | None,
    document: Document | None,
    llm: Any,
    settings: Settings,
    run_id: str | None,
    reason: str,
) -> BranchResult:
    store.log_field_operation(
        operation="merge",
        field_id=sibling.id,
        new_parent_id=parent.id if parent is not None else None,
        new_level=sibling.level,
        document_id=document.id if document is not None else None,
        affected_documents=0,
        reason=reason,
        model=model_name(llm, settings),
        run_id=run_id,
    )
    return BranchResult(
        field_id=sibling.id,
        path_ids=ancestor_ids(store, sibling),
        created=False,
        merged_into=sibling.id,
        reason=reason,
    )


def branch_field(
    *,
    parent: Field | None,
    name: str,
    description: str,
    description_embedding: Sequence[float],
    document: Document | None = None,
    store: FieldStore,
    llm: Any = None,
    settings: Settings,
    run_id: str | None = None,
) -> BranchResult:
    """Create or merge a child of ``parent`` and promote it when appropriate."""
    display_name = (name or "").strip()
    display_description = (description or "").strip()
    if not display_name:
        raise HierarchyError("field name must not be empty")
    if len(display_name) > MAX_NAME_LENGTH:
        raise HierarchyError(
            f"field name must be at most {MAX_NAME_LENGTH} characters"
        )
    if not display_description:
        raise HierarchyError("field description must not be empty")
    if len(display_description) > MAX_DESCRIPTION_LENGTH:
        raise HierarchyError(
            f"field description must be at most {MAX_DESCRIPTION_LENGTH} characters"
        )

    max_depth = effective_max_depth(store, settings)
    parent_id = parent.id if parent is not None else None
    if parent is not None:
        if parent.level >= max_depth:
            raise HierarchyError(
                f"cannot branch below level {parent.level}; "
                f"max_depth is {max_depth}"
            )
        if store.get_field(parent.id) is None:
            raise HierarchyError(f"parent field not found: {parent.id}")
    elif len(store.roots()) >= settings.max_roots:
        raise HierarchyError(
            f"cannot create a root; max_roots is {settings.max_roots}"
        )

    exact = store.find_sibling(parent_id, display_name)
    if exact is not None:
        return _merge_with_sibling(
            store=store,
            sibling=exact,
            parent=parent,
            document=document,
            llm=llm,
            settings=settings,
            run_id=run_id,
            reason=f"duplicate sibling name {display_name!r}; merged",
        )

    candidate_vector = [float(value) for value in description_embedding]
    siblings = store.children(parent_id)
    best_sibling: Field | None = None
    best_similarity = -1.0
    for sibling in siblings:
        try:
            similarity = cosine(candidate_vector, sibling.description_embedding)
        except ValueError:
            continue
        if similarity > best_similarity:
            best_sibling = sibling
            best_similarity = similarity
    if best_sibling is not None and best_similarity >= settings.duplicate_cosine:
        return _merge_with_sibling(
            store=store,
            sibling=best_sibling,
            parent=parent,
            document=document,
            llm=llm,
            settings=settings,
            run_id=run_id,
            reason=(
                f"duplicate sibling cosine {best_similarity:.3f} >= "
                f"{settings.duplicate_cosine:.3f}; merged"
            ),
        )

    if len(siblings) >= settings.max_children:
        raise HierarchyError(
            f"cannot create another child; max_children is {settings.max_children}"
        )

    created = store.create_field(
        parent_id=parent_id,
        name=display_name,
        description=display_description,
        description_embedding=candidate_vector,
    )
    store.log_field_operation(
        operation="create",
        field_id=created.id,
        new_parent_id=parent_id,
        new_level=created.level,
        document_id=document.id if document is not None else None,
        affected_documents=0,
        reason=f"created field {created.name!r}",
        model=model_name(llm, settings),
        run_id=run_id,
    )
    store.bump_hierarchy_version()

    if parent is None:
        return BranchResult(
            field_id=created.id,
            path_ids=ancestor_ids(store, created),
            created=True,
            reason="created a root field",
        )

    judgement = specificity_judge(
        f"{display_name}. {display_description}",
        current=created,
        candidate=parent,
        llm=llm,
        settings=settings,
    )
    if judgement is not None and judgement.relation == "new_field_is_parent":
        promotion = promote_field(
            created,
            parent,
            store=store,
            llm=llm,
            settings=settings,
            run_id=run_id,
        )
        if promotion.rejected:
            return BranchResult(
                field_id=created.id,
                path_ids=ancestor_ids(store, created),
                created=True,
                promoted=False,
                reason=f"promotion_rejected_depth: {promotion.reason}",
            )
        refreshed = store.get_field(created.id) or created
        return BranchResult(
            field_id=created.id,
            path_ids=ancestor_ids(store, refreshed),
            created=True,
            promoted=True,
            reason=f"promoted above {parent.name!r}",
        )

    relation = "no judge verdict" if judgement is None else judgement.relation
    return BranchResult(
        field_id=created.id,
        path_ids=ancestor_ids(store, created),
        created=True,
        promoted=False,
        reason=f"kept as a child ({relation})",
    )


def _log_promotion(
    *,
    store: FieldStore,
    new_field: Field,
    old_parent: Field,
    grandparent_id: str | None,
    old_level: int,
    new_level: int,
    affected_documents: int,
    reason: str,
    llm: Any,
    settings: Settings,
    run_id: str | None,
) -> None:
    store.log_field_operation(
        operation="promote",
        field_id=new_field.id,
        old_parent_id=old_parent.id,
        new_parent_id=grandparent_id,
        old_level=old_level,
        new_level=new_level,
        affected_documents=affected_documents,
        reason=reason,
        model=model_name(llm, settings),
        run_id=run_id,
    )


def _documents_in_subtree(store: FieldStore, field_ids: Sequence[str]) -> list[str]:
    ids = [field_id for field_id in dict.fromkeys(field_ids) if field_id]
    if not ids:
        return []
    connection = getattr(store, "_connection", None)
    if not callable(connection):
        return []
    placeholders = ", ".join("?" for _ in ids)
    rows = connection().execute(
        "SELECT DISTINCT document_id FROM document_fields "
        f"WHERE field_id IN ({placeholders}) ORDER BY document_id",
        ids,
    ).fetchall()
    return [row["document_id"] for row in rows]


def _derived_leaf(store: FieldStore, document_id: str) -> str | None:
    connection = getattr(store, "_connection", None)
    if not callable(connection):
        return None
    row = connection().execute(
        "SELECT field_id FROM document_fields WHERE document_id = ? "
        "ORDER BY ordinal DESC LIMIT 1",
        (document_id,),
    ).fetchone()
    return None if row is None else row["field_id"]


def _reject_promotion(
    *,
    store: FieldStore,
    new_field: Field,
    old_parent: Field,
    grandparent_id: str | None,
    old_level: int,
    reason: str,
    llm: Any,
    settings: Settings,
    run_id: str | None,
) -> PromotionResult:
    _log_promotion(
        store=store,
        new_field=new_field,
        old_parent=old_parent,
        grandparent_id=grandparent_id,
        old_level=old_level,
        new_level=new_field.level,
        affected_documents=0,
        reason=reason,
        llm=llm,
        settings=settings,
        run_id=run_id,
    )
    return PromotionResult(
        new_field_id=new_field.id,
        old_parent_id=old_parent.id,
        rejected=True,
        reason=reason,
    )


def promote_field(
    new_field: Field,
    old_parent: Field,
    *,
    store: FieldStore,
    llm: Any = None,
    settings: Settings,
    run_id: str | None = None,
) -> PromotionResult:
    """Move ``new_field`` above ``old_parent`` and re-level its subtree."""
    if new_field.id == old_parent.id:
        raise HierarchyError("a field cannot be promoted above itself")
    if new_field.parent_id != old_parent.id:
        raise HierarchyError(
            f"field {new_field.id} is not a child of {old_parent.id}"
        )
    if store.children(new_field.id):
        raise HierarchyError(
            f"field {new_field.id} has children and cannot be promoted"
        )

    grandparent_id = old_parent.parent_id
    old_level = new_field.level
    descendants = [
        item
        for item in subtree_fields(store, old_parent.id)
        if item.id not in {old_parent.id, new_field.id}
    ]
    max_descendant_depth = max(
        (item.level - old_parent.level for item in descendants), default=0
    )
    final_max_level = old_parent.level + 1 + max_descendant_depth
    max_depth = effective_max_depth(store, settings)
    if final_max_level > max_depth:
        return _reject_promotion(
            store=store,
            new_field=new_field,
            old_parent=old_parent,
            grandparent_id=grandparent_id,
            old_level=old_level,
            reason=(
                "promotion_rejected_depth: moving "
                f"{old_parent.name!r} down would reach level {final_max_level}; "
                f"max_depth is {max_depth}"
            ),
            llm=llm,
            settings=settings,
            run_id=run_id,
        )

    existing = store.find_sibling(grandparent_id, new_field.name)
    if existing is not None:
        return _reject_promotion(
            store=store,
            new_field=new_field,
            old_parent=old_parent,
            grandparent_id=grandparent_id,
            old_level=old_level,
            reason=(
                "promotion_rejected_duplicate: a sibling named "
                f"{new_field.name!r} already exists at the target level"
            ),
            llm=llm,
            settings=settings,
            run_id=run_id,
        )

    moved_ids = {old_parent.id, new_field.id}
    moved_ids.update(item.id for item in descendants)
    document_ids = _documents_in_subtree(store, sorted(moved_ids))

    now = utc_now_iso()
    connection = store._connection()
    with store._transaction():
        connection.execute(
            "UPDATE fields SET parent_id = ?, level = ?, updated_at = ? "
            "WHERE id = ?",
            (grandparent_id, old_parent.level, now, new_field.id),
        )
        connection.execute(
            "UPDATE fields SET parent_id = ?, level = ?, updated_at = ? "
            "WHERE id = ?",
            (new_field.id, old_parent.level + 1, now, old_parent.id),
        )
        for descendant in descendants:
            connection.execute(
                "UPDATE fields SET level = ?, updated_at = ? WHERE id = ?",
                (descendant.level + 1, now, descendant.id),
            )

    promoted_field = store.get_field(new_field.id) or new_field
    moved_parent = store.get_field(old_parent.id) or old_parent
    levels_changed = 2 + len(descendants)

    affected_documents = 0
    reassigned_documents = 0
    affected_fields: set[str] = set(moved_ids)

    for document_id in document_ids:
        document = store.get_document(document_id)
        if document is None:
            continue
        leaf_id = store.get_leaf(document_id) or _derived_leaf(store, document_id)
        if leaf_id not in moved_ids:
            continue
        affected_documents += 1
        leaf_field = store.get_field(leaf_id)
        if leaf_field is None:
            continue

        embedding = [float(value) for value in document.summary_embedding]
        target = leaf_field
        reassigned = False
        if leaf_id != promoted_field.id and embedding:
            sim_new = field_score(embedding, promoted_field, settings)
            sim_old = field_score(embedding, moved_parent, settings)
            tau = threshold_for_level(settings, promoted_field.level)
            if sim_new >= tau and sim_new > sim_old + settings.promotion_margin:
                target = promoted_field
                reassigned = True
            elif (
                sim_new >= tau
                and abs(sim_new - sim_old) <= settings.promotion_margin
            ):
                judgement = specificity_judge(
                    document.summary or document.title,
                    current=moved_parent,
                    candidate=promoted_field,
                    llm=llm,
                    settings=settings,
                    purpose="hierarchy_promotion",
                )
                if judgement is not None and judgement.relation in {
                    "attach_here",
                    "new_field_is_parent",
                }:
                    target = promoted_field
                    reassigned = True

        if embedding and len(embedding) == store.dims:
            confidence = max(
                0.0,
                min(1.0, field_score(embedding, target, settings)),
            )
        else:  # pragma: no cover - documents normally carry an embedding
            confidence = 1.0

        operation = "reassign_leaf" if reassigned else "recompute_path"
        update_document_path(
            document,
            target,
            store=store,
            confidence=confidence,
            reason=(
                f"promoted {promoted_field.id} above {moved_parent.id}"
                + ("; reassigned closer document" if reassigned else "")
            ),
            operation=operation,
            run_id=run_id,
            model=model_name(llm, settings),
        )
        if reassigned:
            reassigned_documents += 1

    if affected_fields:
        store.recompute_centroids(sorted(affected_fields))

    reason = (
        f"promoted {promoted_field.name!r} above {moved_parent.name!r}; "
        f"reassigned {reassigned_documents} of {affected_documents} documents"
    )
    _log_promotion(
        store=store,
        new_field=promoted_field,
        old_parent=moved_parent,
        grandparent_id=grandparent_id,
        old_level=old_level,
        new_level=promoted_field.level,
        affected_documents=affected_documents,
        reason=reason,
        llm=llm,
        settings=settings,
        run_id=run_id,
    )
    store.bump_hierarchy_version()
    return PromotionResult(
        new_field_id=promoted_field.id,
        old_parent_id=moved_parent.id,
        affected_documents=affected_documents,
        reassigned_documents=reassigned_documents,
        levels_changed=levels_changed,
        rejected=False,
        reason=reason,
    )


promote = promote_field


__all__ = ["attach_document", "branch_field", "promote", "promote_field"]
