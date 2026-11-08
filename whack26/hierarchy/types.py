"""Shared result types for dynamic field hierarchy operations."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlacementResult(BaseModel):
    document_id: str
    leaf_field_id: str
    path_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_fields: list[str] = Field(default_factory=list)
    promoted: bool = False
    reason: str = ""


class BranchResult(BaseModel):
    field_id: str
    path_ids: list[str] = Field(default_factory=list)
    created: bool = True
    promoted: bool = False
    merged_into: str | None = None
    reason: str = ""


class PromotionResult(BaseModel):
    new_field_id: str
    old_parent_id: str
    affected_documents: int = 0
    reassigned_documents: int = 0
    levels_changed: int = 0
    rejected: bool = False
    reason: str = ""


class HierarchyError(RuntimeError):
    """Base error for hierarchy operations."""


class HierarchyIntegrityError(HierarchyError):
    """Raised when the tree or stored paths are inconsistent."""
