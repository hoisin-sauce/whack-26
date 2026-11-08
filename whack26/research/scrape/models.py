"""Semantic-scraping domain models.

The devplan calls the rows “documents”; the existing code and plan prose also
use “resources”. A document is a Dataset, Search, Paper, or Web row.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentKind = Literal["dataset", "search", "paper", "web"]
DocumentStatus = Literal[
    "new", "fetched", "summarized", "classified", "compared", "rejected"
]
LinkRelation = Literal[
    "references",
    "cites",
    "linked_from",
    "same_as",
    "dataset_paper",
    "search_result",
]
ComparisonKind = Literal[
    "dataset_relevance",
    "method_overlap",
    "finding_agreement",
    "population_overlap",
    "contradiction",
]
ComparisonVerdict = Literal["supports", "contradicts", "orthogonal", "insufficient"]
CriticVerdict = Literal["pass", "refine", "reject", "insufficient"]


class Document(BaseModel):
    id: str
    kind: DocumentKind
    source: str
    source_id: str | None = None
    title: str
    summary: str = ""
    summary_embedding: list[float] = Field(default_factory=list)
    embedding_model: str = ""
    canonical_url: str | None = None
    url_hash: str | None = None
    content_sha256: str | None = None
    content_sha512: str | None = None
    authors: list[str] = Field(default_factory=list)
    publisher: str | None = None
    published_at: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    credibility_score: float = 0.0
    credibility: dict[str, Any] = Field(default_factory=dict)
    credible: bool = False
    preloaded: bool = False
    status: DocumentStatus = "new"
    created_at: str = ""
    updated_at: str = ""


class DocumentLink(BaseModel):
    from_id: str
    to_id: str
    relation: LinkRelation
    url: str | None = None
    anchor_text: str | None = None
    discovered_at: str = ""


class DocumentComparison(BaseModel):
    left_id: str
    right_id: str
    kind: ComparisonKind
    similarity: float = 0.0
    verdict: ComparisonVerdict = "insufficient"
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)
    method_version: str = "1.0.0"
    created_at: str = ""


class DocumentCritique(BaseModel):
    document_id: str
    corpus_version: str
    verdict: CriticVerdict = "insufficient"
    issues: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    summary: str = ""
    model: str = ""
    created_at: str = ""


class SummaryResult(BaseModel):
    summary: str
    claims: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    population: str | None = None
    time_period: str | None = None
    links: list[str] = Field(default_factory=list)
    credibility_signals: dict[str, Any] = Field(default_factory=dict)


class CredibilityResult(BaseModel):
    score: float = 0.0
    credible: bool = False
    components: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class ScrapeResult(BaseModel):
    documents: list[Document] = Field(default_factory=list)
    links: list[DocumentLink] = Field(default_factory=list)
    comparisons: list[DocumentComparison] = Field(default_factory=list)
    critiques: list[DocumentCritique] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int | float] = Field(default_factory=dict)
