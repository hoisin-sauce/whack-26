"""Research sources, caching, and normalization for the slice."""

from .base import (
    DATASET_SOURCES,
    PAPER_SOURCES,
    SOURCE_PRIOR,
    BaseHttpAdapter,
    ResearchRecord,
    SourceAdapter,
    SourceError,
    excerpt,
    reconstruct_abstract,
    request_with_retries,
)
from .cache import CachedResponse, ResearchCache, cache_key

__all__ = [
    "DATASET_SOURCES",
    "PAPER_SOURCES",
    "SOURCE_PRIOR",
    "BaseHttpAdapter",
    "CachedResponse",
    "ResearchCache",
    "ResearchRecord",
    "SourceAdapter",
    "SourceError",
    "cache_key",
    "excerpt",
    "reconstruct_abstract",
    "request_with_retries",
]
