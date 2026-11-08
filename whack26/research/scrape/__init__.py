"""Semantic-scraping package.

The devplan's “documents” are datasets, searches, papers, and web resources.
The package separates identity, summarization, frontier expansion, credibility,
comparisons, and the preload corpus.
"""

from .models import (
    ComparisonKind,
    ComparisonVerdict,
    CredibilityResult,
    CriticVerdict,
    Document,
    DocumentComparison,
    DocumentCritique,
    DocumentKind,
    DocumentLink,
    DocumentStatus,
    LinkRelation,
    ScrapeResult,
    SummaryResult,
)

__all__ = [
    "ComparisonKind",
    "ComparisonVerdict",
    "CredibilityResult",
    "CriticVerdict",
    "Document",
    "DocumentComparison",
    "DocumentCritique",
    "DocumentKind",
    "DocumentLink",
    "DocumentStatus",
    "LinkRelation",
    "ScrapeResult",
    "SummaryResult",
]
