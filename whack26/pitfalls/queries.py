"""Build deterministic, field-scoped pitfall queries.

The queries deliberately ask for limitations rather than the field itself:
``"housing valuation limitations bias"`` returns a very different slice of the
literature from ``"housing valuation"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .types import FieldPitfallQuery

_QUERY_TEMPLATES: tuple[str, ...] = (
    "{name} limitations bias",
    "{name} measurement error confounding",
    "{name} common pitfalls",
)

_ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"openalex", "arxiv", "semantic_scholar", "websearch"}
)


def field_value(field: Any, name: str, default: Any = None) -> Any:
    """Read a field attribute from an object or a mapping."""
    if isinstance(field, Mapping):
        return field.get(name, default)
    return getattr(field, name, default)


def field_name(field: Any) -> str:
    if isinstance(field, str):
        value = " ".join(field.split())
    else:
        value = " ".join(str(field_value(field, "name", "") or "").split())
    if not value:
        return "dataset topic"
    # Field names are capped at five words by the classifier, but imported
    # hierarchies may be looser.  Search queries stay short regardless.
    return " ".join(value.split()[:5])


def field_id(field: Any) -> str | None:
    value = field_value(field, "id")
    return str(value) if value else None


def field_level(field: Any) -> int:
    value = field_value(field, "level", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def build_field_queries(
    fields: Sequence[Any],
    *,
    per_field: int = 1,
    sources: Sequence[str] = ("openalex", "arxiv"),
) -> list[FieldPitfallQuery]:
    """Return one or more pitfall queries per field, leaf-to-root order.

    The caller normally passes the root-to-leaf field path; this function
    reverses it so the most specific field is searched first and its results
    are preferred when the report is capped.
    """
    if per_field < 1:
        return []
    allowed = [str(source).strip().lower() for source in sources if str(source).strip()]
    allowed = [source for source in allowed if source in _ALLOWED_SOURCES]
    if not allowed:
        allowed = ["openalex"]

    queries: list[FieldPitfallQuery] = []
    for field in reversed(list(fields)):
        name = field_name(field)
        for index in range(min(int(per_field), len(_QUERY_TEMPLATES))):
            template = _QUERY_TEMPLATES[index]
            queries.append(
                FieldPitfallQuery(
                    field_id=field_id(field),
                    field_name=name,
                    field_level=field_level(field),
                    source=allowed[index % len(allowed)],
                    query=template.format(name=name),
                )
            )
    return queries


__all__ = [
    "build_field_queries",
    "field_id",
    "field_level",
    "field_name",
    "field_value",
]
