"""Dynamic field classification.

The classifier builds a general-domain field tree from dataset summaries. It
never uses a pre-seeded taxonomy: fields are created only when the LLM decides
that no existing sibling fits.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ValidationError, field_validator

from .config import Settings
from .embeddings import cosine
from .llm import LLMError
from .profile import DatasetProfile, build_summary
from .store import Field, FieldStore, FieldStoreError


class ClassificationError(RuntimeError):
    """A classification failure that should stop the run with a clear message."""


class FieldChoice(BaseModel):
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


@dataclass
class FieldPath:
    fields: list[Field]
    confidence: float
    created_fields: list[str] = field(default_factory=list)
    refined_after_research: bool = False
    refinement_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [item.id for item in self.fields]

    @property
    def names(self) -> list[str]:
        return [item.name for item in self.fields]

    @property
    def depth(self) -> int:
        return len(self.fields)

    @property
    def leaf(self) -> Field | None:
        return self.fields[-1] if self.fields else None


FIELD_CHOICE_SCHEMA: dict[str, Any] = {
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

FIELD_SYSTEM_PROMPT = (
    "You are classifying a tabular dataset into a dynamic topic hierarchy. "
    "The hierarchy is general-domain and is built from data; there is no fixed "
    "list. Choose an existing field only when it genuinely fits. Propose a new "
    "field only when no existing sibling is a reasonable parent, and prefer "
    "stopping over creating a deeper field when the current field already "
    "describes the dataset well. Names must be human-readable, at most five "
    "words, and must not be the dataset's filename or a raw column name. "
    "Descriptions must be one or two sentences and must distinguish the field "
    "from its siblings. The maximum tree depth is 4; never propose a level-5 field."
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are refining the topic classification of a tabular dataset after "
    "reading research titles. You may assign the dataset to another existing "
    "field, create one child under the current leaf field, or stop at the "
    "current field. Do not create a new root. Names must be human-readable, at "
    "most five words, and must not be the dataset's filename or a raw column "
    "name. Descriptions must distinguish the field from its siblings."
)


def _normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class Classifier:
    """Create or reuse a field path for one dataset at a time."""

    def __init__(
        self,
        *,
        llm: Any,
        embeddings: Any,
        store: FieldStore,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.store = store
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API

    def classify(self, profile: DatasetProfile, frame: Any) -> FieldPath:
        """Classify a dataset, creating fields as needed and committing its path."""
        try:
            return self._classify(profile, frame)
        except Exception:
            self._prune_after_failure()
            raise

    def _classify(self, profile: DatasetProfile, frame: Any) -> FieldPath:
        cached = self._cached_path(profile)
        if cached is not None:
            return cached
        summary = build_summary(profile, frame)
        dataset_vector = self._embed_one(summary)
        current: Field | None = None
        path: list[Field] = []
        scores: list[float] = []
        created: list[str] = []
        warnings: list[str] = []

        while True:
            candidates = self._candidates(dataset_vector, current)
            user_prompt = self._choice_prompt(
                summary=summary,
                path=path,
                current=current,
                candidates=candidates,
            )
            level = 1 if current is None else current.level + 1
            choice, error = self._ask_valid(
                system=FIELD_SYSTEM_PROMPT,
                user=user_prompt,
                current=current,
                candidates=candidates,
                level=level,
                profile=profile,
                purpose="classification",
                # A root is a broad domain, not a dataset leaf. Require at
                # least two levels so every dataset has a usable subfield.
                allow_stop=current is not None and current.level >= 2,
            )
            if choice is None:
                choice = self._fallback_choice(current, candidates)
                warnings.append(
                    f"classification_fallback: {error or 'invalid LLM response'}; "
                    f"used {choice.action}"
                )

            if choice.action == "stop":
                break

            if choice.action == "assign":
                selected, score = self._candidate_by_id(candidates, choice.field_id)
                if selected is None:
                    # _ask_valid guarantees this, but keep the path safe.
                    choice = self._fallback_choice(current, candidates)
                    warnings.append("classification_fallback: unknown field id")
                    if choice.action == "stop":
                        break
                    selected, score = self._candidate_by_id(candidates, choice.field_id)
                    if selected is None:
                        raise ClassificationError("could not resolve a fallback field")
                current = selected
                path.append(selected)
                scores.append(score)
            else:
                parent_score = scores[-1] if scores else 0.5
                parent = current
                selected, score, was_created, merged_id = self._create_field(
                    parent=parent,
                    name=choice.name or "Unnamed Field",
                    description=choice.description or "No description supplied.",
                    dataset_vector=dataset_vector,
                    parent_score=parent_score,
                )
                current = selected
                path.append(selected)
                scores.append(score)
                if was_created:
                    if selected.id not in created:
                        created.append(selected.id)
                else:
                    warnings.append(
                        f"duplicate_merge: proposed field {choice.name!r} merged into "
                        f"{merged_id or selected.id}"
                    )

            if current.level >= self.settings.max_depth:
                break

        if not path:
            raise ClassificationError("classification produced an empty field path")

        confidence = _clamp(sum(scores) / len(scores))
        affected = self._commit(
            profile=profile,
            summary=summary,
            dataset_vector=dataset_vector,
            path=path,
            confidence=confidence,
            refined_after_research=False,
        )
        self._refresh_and_prune(affected)
        return FieldPath(
            fields=path,
            confidence=confidence,
            created_fields=created,
            refined_after_research=False,
            refinement_reason=None,
            warnings=warnings,
        )

    def _cached_path(self, profile: DatasetProfile) -> FieldPath | None:
        """Reuse a committed path when the same content and settings recur.

        Identity is the content hash plus the inputs that can change the
        summary or the walk: target, description, schema, time column, and the
        classification settings.  ``source_path`` and ``warnings`` are
        deliberately excluded: reading identical bytes from another location,
        or a change in warning text, must not force a second classification.
        """
        dataset = self.store.get_dataset(profile.dataset_id)
        if dataset is None:
            return None
        if dataset.get("cache_key") != self._cache_key(profile):
            return None
        try:
            stored_ids = json.loads(dataset["path_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(stored_ids, list):
            return None
        try:
            fields = self.store.dataset_path(profile.dataset_id)
        except FieldStoreError:
            # A deleted field makes the cached path unusable; classify again.
            return None
        if not fields or len(fields) != len(stored_ids):
            return None
        confidence = float(dataset.get("confidence") or 0.0)
        return FieldPath(
            fields=fields,
            confidence=confidence,
            created_fields=[],
            refined_after_research=bool(dataset.get("refined_after_research")),
            refinement_reason=None,
            warnings=[],
        )

    def _content_identity(self, profile: DatasetProfile) -> tuple:
        """Return the stable inputs that make two classifications equivalent."""
        target = profile.target.name if profile.target is not None else None
        schema = tuple(
            (
                column.name,
                column.dtype,
                column.role,
                column.cardinality,
                column.datetime,
            )
            for column in profile.columns
        )
        settings = (
            self.settings.embed_dims,
            self.settings.max_depth,
            self.settings.duplicate_cosine,
            self.settings.desc_weight,
            self.settings.centroid_weight,
        )
        return (
            profile.sha256,
            target,
            profile.description,
            profile.source_site,
            profile.rows,
            profile.time_column,
            schema,
            settings,
        )

    def _cache_key(self, profile: DatasetProfile) -> str:
        """Serialize the content identity stored alongside a committed path."""
        return json.dumps(self._content_identity(profile), sort_keys=True)

    def should_refine(self, path: FieldPath, research_titles: Sequence[str] = ()) -> bool:
        """Return True when the confidence trigger fires.

        Contradiction detection is intentionally left to the caller: research
        can force a refinement by calling `refine(..., force=True)`.
        """
        return path.confidence < self.settings.refine_confidence

    def refine(
        self,
        profile: DatasetProfile,
        frame: Any,
        path: FieldPath,
        *,
        research_titles: Sequence[str] = (),
        keywords: Sequence[str] = (),
        force: bool = False,
    ) -> FieldPath:
        """Run at most one post-research refinement pass."""
        try:
            return self._refine(
                profile,
                frame,
                path,
                research_titles=research_titles,
                keywords=keywords,
                force=force,
            )
        except Exception:
            self._prune_after_failure()
            raise

    def _refine(
        self,
        profile: DatasetProfile,
        frame: Any,
        path: FieldPath,
        *,
        research_titles: Sequence[str] = (),
        keywords: Sequence[str] = (),
        force: bool = False,
    ) -> FieldPath:
        if not force and not self.should_refine(path, research_titles):
            return path
        if not path.fields:
            raise ClassificationError("cannot refine an empty field path")

        summary = build_summary(profile, frame)
        dataset_vector = self._embed_one(summary)
        current = path.fields[-1]
        all_fields = sorted(
            self.store.all_fields(),
            key=lambda item: self._field_score(dataset_vector, item),
            reverse=True,
        )[:40]
        user_prompt = self._refinement_prompt(
            summary=summary,
            path=path.fields,
            current=current,
            candidates=all_fields,
            research_titles=research_titles,
            keywords=keywords,
        )
        choice, error = self._ask_valid(
            system=REFINEMENT_SYSTEM_PROMPT,
            user=user_prompt,
            current=current,
            candidates=[(item, self._field_score(dataset_vector, item)) for item in all_fields],
            level=current.level + 1,
            profile=profile,
            purpose="refinement",
            allow_stop=True,
            allow_root_create=False,
        )
        warnings = list(path.warnings)
        created = list(path.created_fields)
        new_path = list(path.fields)
        reason = choice.reason if choice else ""

        if choice is None:
            warnings.append(f"refinement_failed: {error or 'invalid LLM response'}; kept the path")
            reason = "refinement could not be applied; kept the original path"
        elif choice.action == "stop":
            reason = choice.reason or "research supported the current path"
        elif choice.action == "assign":
            selected = self.store.get_field(choice.field_id or "")
            if selected is None:
                warnings.append(
                    f"refinement_failed: unknown field id {choice.field_id!r}; kept the path"
                )
                reason = "refinement returned an unknown field; kept the original path"
            else:
                new_path = self._path_to_field(selected)
                reason = choice.reason or "research suggested an existing field"
        elif choice.action == "create":
            if current.level >= self.settings.max_depth:
                warnings.append("refinement_failed: cannot create below the maximum depth")
                reason = "refinement could not create below the maximum depth"
            else:
                selected, _, was_created, merged_id = self._create_field(
                    parent=current,
                    name=choice.name or "Unnamed Field",
                    description=choice.description or "No description supplied.",
                    dataset_vector=dataset_vector,
                    parent_score=self._field_score(dataset_vector, current),
                )
                new_path = list(path.fields) + [selected]
                if was_created:
                    if selected.id not in created:
                        created.append(selected.id)
                else:
                    warnings.append(
                        f"duplicate_merge: proposed field {choice.name!r} merged into "
                        f"{merged_id or selected.id}"
                    )
                reason = choice.reason or "research suggested a child field"

        confidence = self._path_confidence(dataset_vector, new_path)
        affected = self._commit(
            profile=profile,
            summary=summary,
            dataset_vector=dataset_vector,
            path=new_path,
            confidence=confidence,
            refined_after_research=True,
        )
        self._refresh_and_prune(affected)
        return FieldPath(
            fields=new_path,
            confidence=confidence,
            created_fields=created,
            refined_after_research=True,
            refinement_reason=reason or None,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Scoring and candidate formatting

    def _embed_one(self, text: str) -> list[float]:
        vectors = self.embeddings.embed([text])
        if not vectors:
            raise ClassificationError("embedding provider returned no vectors")
        vector = [float(value) for value in vectors[0]]
        if len(vector) != self.settings.embed_dims:
            raise ClassificationError(
                f"embedding dimension mismatch: expected {self.settings.embed_dims}, "
                f"got {len(vector)}"
            )
        return vector

    def _field_score(self, dataset_vector: list[float], item: Field) -> float:
        desc_sim = cosine(dataset_vector, item.description_embedding)
        if item.member_count > 0 and item.centroid:
            cent_sim = cosine(dataset_vector, item.centroid)
        else:
            cent_sim = desc_sim
        return (
            self.settings.desc_weight * desc_sim
            + self.settings.centroid_weight * cent_sim
        )

    def _candidates(
        self, dataset_vector: list[float], current: Field | None
    ) -> list[tuple[Field, float]]:
        fields = self.store.children(current.id if current else None)
        ranked = [(item, self._field_score(dataset_vector, item)) for item in fields]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[: self.settings.top_k]

    @staticmethod
    def _candidate_by_id(
        candidates: Sequence[tuple[Field, float]], field_id: str | None
    ) -> tuple[Field | None, float]:
        for item, score in candidates:
            if item.id == field_id:
                return item, score
        return None, 0.0

    @staticmethod
    def _format_path(path: Sequence[Field]) -> str:
        if not path:
            return "(root; no field assigned yet)"
        return "\n".join(
            f"- Level {item.level}: {item.name} — {item.description}" for item in path
        )

    @staticmethod
    def _format_candidates(candidates: Sequence[tuple[Field, float]]) -> str:
        if not candidates:
            return "(no existing fields at this level)"
        lines = []
        for index, (item, score) in enumerate(candidates, start=1):
            lines.append(
                f"{index}. id={item.id} | name={item.name} | score={score:.3f} | "
                f"members={item.member_count} | description={item.description}"
            )
        return "\n".join(lines)

    def _choice_prompt(
        self,
        *,
        summary: str,
        path: Sequence[Field],
        current: Field | None,
        candidates: Sequence[tuple[Field, float]],
    ) -> str:
        level = 1 if current is None else current.level + 1
        if current is None:
            position = (
                "You are at the root level. `stop` is not allowed here: choose an "
                "existing root or create a new root."
            )
        elif current.level == 1:
            position = (
                "This is a top-level domain, which is too broad to describe the "
                "dataset by itself. `stop` is not allowed here: choose an existing "
                "child or create a child."
            )
        elif level == self.settings.max_depth:
            position = (
                "This is the maximum depth. You may assign to an existing child, "
                "create a level-4 field, or `stop` at the current field."
            )
        else:
            position = (
                "Choose an existing child when it fits. Prefer `stop` when the "
                "current field already describes the dataset well; create a child "
                "only when the current field is clearly too broad."
            )
        return (
            f"Dataset summary:\n{summary}\n\n"
            f"Current field path:\n{self._format_path(path)}\n\n"
            f"Current depth: {level} of {self.settings.max_depth}\n"
            f"Candidates (children of the current field):\n"
            f"{self._format_candidates(candidates)}\n\n"
            f"{position}\n"
            "Return one JSON object with `action`, and either `field_id` for "
            "`assign`, or `name` and `description` for `create`. Always include a "
            "short `reason`. Keep the response small: `reason` at most 25 words, "
            "`name` at most 5 words, and `description` at most 2 sentences. Do not "
            "repeat the dataset summary."
        )

    def _refinement_prompt(
        self,
        *,
        summary: str,
        path: Sequence[Field],
        current: Field,
        candidates: Sequence[Field],
        research_titles: Sequence[str],
        keywords: Sequence[str],
    ) -> str:
        titles = "\n".join(f"- {title}" for title in research_titles[:5]) or "(none)"
        keyword_text = ", ".join(keywords[:12]) or "(none)"
        candidate_lines = []
        for index, item in enumerate(candidates, start=1):
            candidate_lines.append(
                f"{index}. id={item.id} | level={item.level} | name={item.name} | "
                f"description={item.description}"
            )
        return (
            f"Dataset summary:\n{summary}\n\n"
            f"Original field path:\n{self._format_path(path)}\n\n"
            f"Top research titles:\n{titles}\n\n"
            f"Research keywords: {keyword_text}\n\n"
            f"Existing fields you may assign to:\n"
            + ("\n".join(candidate_lines) or "(none)")
            + "\n\n"
            "Allowed actions: `assign` to an existing field id, `create` one child "
            f"under the current leaf (level {current.level + 1}), or `stop`. Do not "
            "create a new root. Return one JSON object with `action`, and either "
            "`field_id` for `assign`, or `name` and `description` for `create`. "
            "Always include a short `reason`. Keep the response small: `reason` at "
            "most 25 words, `name` at most 5 words, and `description` at most 2 "
            "sentences. Do not repeat the dataset summary."
        )

    # ------------------------------------------------------------------
    # Validation and fallbacks

    def _ask_valid(
        self,
        *,
        system: str,
        user: str,
        current: Field | None,
        candidates: Sequence[tuple[Field, float]],
        level: int,
        profile: DatasetProfile,
        purpose: str,
        allow_stop: bool = True,
        allow_root_create: bool = True,
    ) -> tuple[FieldChoice | None, str | None]:
        prompt = user
        last_error = "invalid response"
        for _ in range(2):
            try:
                payload = self.llm.generate_json(
                    system=system,
                    user=prompt,
                    schema=FIELD_CHOICE_SCHEMA,
                    max_tokens=1024,
                    purpose=purpose,
                )
            except LLMError as exc:
                raise ClassificationError(f"LLM classification failed: {exc}") from exc
            try:
                choice = FieldChoice.model_validate(payload)
            except ValidationError as exc:
                last_error = f"schema error: {exc}"
                prompt = (
                    user
                    + f"\n\nYour previous response was invalid: {last_error}. "
                    "Return a valid JSON object matching the schema."
                )
                continue
            error = self._validate_choice(
                choice,
                current=current,
                candidates=candidates,
                level=level,
                profile=profile,
                allow_stop=allow_stop,
                allow_root_create=allow_root_create,
            )
            if error is None:
                return choice, None
            last_error = error
            prompt = (
                user
                + f"\n\nYour previous response was invalid: {error}. "
                "Return a valid JSON object matching the schema."
            )
        return None, last_error

    def _validate_choice(
        self,
        choice: FieldChoice,
        *,
        current: Field | None,
        candidates: Sequence[tuple[Field, float]],
        level: int,
        profile: DatasetProfile,
        allow_stop: bool,
        allow_root_create: bool,
    ) -> str | None:
        if choice.action == "stop":
            if not allow_stop or current is None:
                return (
                    "`stop` is not allowed here; choose an existing child or "
                    "create a child"
                )
            return None
        if choice.action == "assign":
            candidate_ids = {item.id for item, _ in candidates}
            if not choice.field_id or choice.field_id not in candidate_ids:
                return f"`field_id` must be one of the supplied candidates: {sorted(candidate_ids)}"
            return None
        if choice.action == "create":
            if level > self.settings.max_depth:
                return f"`create` is not allowed at level {level}; the maximum depth is {self.settings.max_depth}"
            if current is None and not allow_root_create:
                return "creating a new root is not allowed during refinement"
            if not choice.name or not choice.description:
                return "`create` requires both `name` and `description`"
            name = choice.name.strip()
            if len(name.split()) > 5:
                return "field names must be at most five words"
            normalized = _normalize_name(name)
            if not normalized:
                return "field names must contain letters or digits"
            stem = _normalize_name(Path(profile.source_path).stem)
            if normalized == stem or normalized == _normalize_name(profile.dataset_id):
                return "field names must not be the dataset filename or id"
            column_names = {_normalize_name(column.name) for column in profile.columns}
            if normalized in column_names:
                return "field names must not be a raw column name"
            if len(choice.description.strip()) < 10:
                return "field descriptions must be at least 10 characters"
            return None
        return f"unknown action {choice.action!r}"

    def _fallback_choice(
        self,
        current: Field | None,
        candidates: Sequence[tuple[Field, float]],
    ) -> FieldChoice:
        if candidates:
            item, _ = candidates[0]
            return FieldChoice(
                action="assign",
                field_id=item.id,
                reason="fallback to the top-scoring candidate",
            )
        if current is not None and current.level >= 2:
            return FieldChoice(
                action="stop",
                reason="fallback to the current field",
            )
        if current is not None:
            return FieldChoice(
                action="create",
                name="General Topics",
                description=(
                    "Broad subtopics used when a more specific field cannot be "
                    "determined."
                ),
                reason="fallback child after an invalid classification response",
            )
        return FieldChoice(
            action="create",
            name="General Research",
            description=(
                "Datasets that do not yet fit a more specific research domain."
            ),
            reason="fallback root after an invalid classification response",
        )

    # ------------------------------------------------------------------
    # Field creation, commit, and pruning

    def _create_field(
        self,
        *,
        parent: Field | None,
        name: str,
        description: str,
        dataset_vector: list[float],
        parent_score: float,
    ) -> tuple[Field, float, bool, str | None]:
        parent_id = parent.id if parent else None
        siblings = self.store.children(parent_id)
        exact = self.store.find_sibling(parent_id, name)
        if exact is not None:
            return exact, self._field_score(dataset_vector, exact), False, exact.id

        candidate_vector = self._embed_one(f"{name}. {description}")
        best: Field | None = None
        best_similarity = -1.0
        for sibling in siblings:
            similarity = cosine(candidate_vector, sibling.description_embedding)
            if similarity > best_similarity:
                best = sibling
                best_similarity = similarity
        if best is not None and best_similarity >= self.settings.duplicate_cosine:
            return best, self._field_score(dataset_vector, best), False, best.id

        created = self.store.create_field(
            parent_id=parent_id,
            name=name,
            description=description,
            description_embedding=candidate_vector,
        )
        return created, parent_score, True, None

    def _path_to_field(self, selected: Field) -> list[Field]:
        path: list[Field] = [selected]
        current = selected
        seen = {selected.id}
        while current.parent_id:
            if current.parent_id in seen:
                raise ClassificationError("field tree contains a cycle")
            seen.add(current.parent_id)
            parent = self.store.get_field(current.parent_id)
            if parent is None:
                raise ClassificationError(
                    f"field {current.id} references missing parent {current.parent_id}"
                )
            path.append(parent)
            current = parent
        path.reverse()
        return path

    def _commit(
        self,
        *,
        profile: DatasetProfile,
        summary: str,
        dataset_vector: list[float],
        path: Sequence[Field],
        confidence: float,
        refined_after_research: bool,
    ) -> list[str]:
        if not path:
            raise ClassificationError("cannot commit an empty field path")
        field_ids = [item.id for item in path]
        if len(set(field_ids)) != len(field_ids):
            raise ClassificationError("field path contains duplicate fields")
        path_json = json.dumps(field_ids)
        return self.store.commit_path(
            dataset_id=profile.dataset_id,
            sha256=profile.sha256,
            profile_json=profile.model_dump_json(),
            summary=summary,
            embedding=dataset_vector,
            path_json=path_json,
            confidence=confidence,
            refined_after_research=refined_after_research,
            field_ids=field_ids,
            cache_key=self._cache_key(profile),
        )

    def _refresh_and_prune(self, affected: Iterable[str]) -> None:
        affected_ids = list(dict.fromkeys(affected))
        if affected_ids:
            self.store.recompute_centroids(affected_ids)
        self._prune_empty()

    def _prune_empty(self) -> None:
        while True:
            deleted = self.store.delete_empty_leaf_fields()
            if not deleted:
                break

    def _prune_after_failure(self) -> None:
        """Best-effort cleanup; never mask the original failure."""
        try:
            self._prune_empty()
        except Exception:
            pass

    def _path_confidence(self, dataset_vector: list[float], path: Sequence[Field]) -> float:
        if not path:
            return 0.0
        scores = [self._field_score(dataset_vector, item) for item in path]
        return _clamp(sum(scores) / len(scores))
