"""Research pipeline: query generation, fetching, normalization, and ranking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
from pydantic import BaseModel, Field, ValidationError

from ..common import utc_now_iso
from ..config import Settings
from ..profile import DatasetProfile
from .arxiv import arXivAdapter
from .base import (
    DATASET_SOURCES,
    DEFAULT_USER_AGENT,
    PAPER_SOURCES,
    ResearchRecord,
    SourceAdapter,
    SourceError,
)
from .cache import ResearchCache, cache_key
from .datacite import DataCiteAdapter
from .datagov import DataGovAdapter
from .dataverse import DataverseAdapter
from .huggingface import HuggingFaceAdapter
from .kaggle import KaggleAdapter
from .normalize import deduplicate, rank_records
from .openalex import OpenAlexAdapter
from .openml import OpenMLAdapter
from .semantic_scholar import SemanticScholarAdapter
from .websearch import WebSearchAdapter
from .wikipedia import WikipediaAdapter
from .zenodo import ZenodoAdapter

RESEARCH_TOP_K = 20

SOURCE_LIMITS: dict[str, int] = {
    "openalex": 10,
    "arxiv": 5,
    "huggingface": 5,
    "zenodo": 5,
    "wikipedia": 3,
    "kaggle": 5,
    "dataverse": 5,
    "openml": 5,
    "datagov": 5,
    "datacite": 5,
    "semantic_scholar": 5,
    "websearch": 10,
}

QuerySource = Literal[
    "openalex",
    "arxiv",
    "huggingface",
    "zenodo",
    "wikipedia",
    "kaggle",
    "dataverse",
    "openml",
    "datagov",
    "datacite",
    "semantic_scholar",
    "websearch",
]

QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": [
                            "openalex",
                            "arxiv",
                            "huggingface",
                            "zenodo",
                            "wikipedia",
                            "kaggle",
                            "dataverse",
                            "openml",
                            "datagov",
                            "datacite",
                            "semantic_scholar",
                            "websearch",
                        ],
                    },
                    "query": {"type": "string", "maxLength": 120},
                    "rationale": {"type": "string", "maxLength": 240},
                },
                "required": ["source", "query", "rationale"],
            },
            "maxItems": 6,
        }
    },
    "required": ["queries"],
    "propertyOrdering": ["queries"],
}

QUERY_SYSTEM_PROMPT = (
    "You write short literature and dataset search queries for a research "
    "pipeline. Use the dataset summary and its field path. Return 4 to 6 "
    "queries: at least two paper queries (openalex or arxiv), at least one "
    "dataset query (huggingface or zenodo), and at most one wikipedia query. "
    "Use 2 to 5 words per query. Focus on substantive domain factors that "
    "could explain or predict the target, such as economics, biology, "
    "environment, behaviour, or policy. Avoid generic machine-learning terms "
    "unless the target is inherently about machine learning. Dataset queries "
    "should be a short subject phrase, for example `california housing` rather "
    "than a sentence. Queries must be at most 12 words, must not contain "
    "boolean operators, and must not include personal data. Return only the "
    "requested JSON."
)

_BOOLEAN_RE = re.compile(r"\b(?:AND|OR|NOT)\b")
_MAX_QUERY_WORDS = 12
_WIKIPEDIA_LIMIT = 1


class ResearchQuery(BaseModel):
    source: QuerySource
    query: str
    rationale: str = ""


@dataclass
class ResearchResult:
    records: list[ResearchRecord] = field(default_factory=list)
    queries: list[ResearchQuery] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)
    limited_mode: bool = False
    offline: bool = False
    cache: dict[str, float | int] = field(default_factory=dict)


def _dedupe_queries(queries: Sequence[ResearchQuery]) -> list[ResearchQuery]:
    seen: set[tuple[str, str]] = set()
    result: list[ResearchQuery] = []
    wikipedia = 0
    for query in queries:
        normalized = " ".join(query.query.lower().split())
        key = (query.source, normalized)
        if not normalized or key in seen:
            continue
        if query.source == "wikipedia":
            if wikipedia >= _WIKIPEDIA_LIMIT:
                continue
            wikipedia += 1
        seen.add(key)
        result.append(query)
    return result


def _valid_query(query: ResearchQuery) -> bool:
    text = query.query.strip()
    if not text or len(text.split()) > _MAX_QUERY_WORDS:
        return False
    if _BOOLEAN_RE.search(text):
        return False
    return True


def _short_query(query: str, words: int = 2) -> str | None:
    """Trim a long query to its leading subject phrase for HF search.

    The Hugging Face Hub search endpoint matches the whole phrase, so a long
    natural-language query often returns nothing. Even a useful qualifier such
    as ``dataset`` can make the phrase miss: ``palmer penguins`` finds the
    dataset while ``palmer penguins dataset`` does not.
    """
    parts = query.split()
    if len(parts) <= words:
        return None
    return " ".join(parts[:words])


def _fallback_queries(
    path_names: Sequence[str], profile: DatasetProfile
) -> list[ResearchQuery]:
    leaf = path_names[-1] if path_names else "dataset topic"
    target = (
        profile.target.name.replace("_", " ").strip()
        if profile.target is not None
        else "target"
    )
    topic = f"{leaf} {target}".strip()
    return [
        ResearchQuery(
            source="openalex",
            query=f"{topic} determinants",
            rationale="fallback paper query",
        ),
        ResearchQuery(
            source="arxiv",
            query=f"{topic} prediction",
            rationale="fallback paper query",
        ),
        ResearchQuery(
            source="huggingface",
            query=f"{leaf} dataset",
            rationale="fallback dataset query",
        ),
        ResearchQuery(
            source="zenodo",
            query=f"{target} dataset",
            rationale="fallback dataset query",
        ),
        ResearchQuery(
            source="wikipedia",
            query=topic,
            rationale="fallback encyclopedic query",
        ),
    ]


def _parse_queries(payload: Any) -> list[ResearchQuery]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("queries")
    if not isinstance(raw, list):
        return []
    queries: list[ResearchQuery] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip().lower()
        if source == "hf":
            source = "huggingface"
        if source not in SOURCE_LIMITS:
            continue
        try:
            query = ResearchQuery(
                source=source,  # type: ignore[arg-type]
                query=str(item.get("query", "")).strip(),
                rationale=str(item.get("rationale", "")).strip(),
            )
        except ValidationError:
            continue
        if _valid_query(query):
            queries.append(query)
    return _dedupe_queries(queries)


def _meets_constraints(queries: Sequence[ResearchQuery]) -> bool:
    if not 4 <= len(queries) <= 6:
        return False
    papers = sum(1 for query in queries if query.source in PAPER_SOURCES)
    datasets = sum(1 for query in queries if query.source in DATASET_SOURCES)
    wikipedia = sum(1 for query in queries if query.source == "wikipedia")
    return papers >= 2 and datasets >= 1 and wikipedia <= 1


def _query_cache_key(
    profile: DatasetProfile, summary: str, path_names: Sequence[str]
) -> str:
    # Include the pre-research field path: a reclassification can change the
    # field names that drive query generation, and reusing a query set keyed
    # only on the summary would silently ignore that change. Tradeoff: every
    # path change invalidates the query cache and costs one LLM call.
    normalized_path = [" ".join(name.lower().split()) for name in path_names]
    return cache_key(
        profile.dataset_id,
        "queries",
        "",
        {
            "summary_sha256": hashlib.sha256(
                summary.encode("utf-8")
            ).hexdigest(),
            "field_path": normalized_path,
        },
    )


def generate_queries(
    *,
    summary: str,
    path_names: Sequence[str],
    profile: DatasetProfile,
    llm: Any,
    cache: ResearchCache | None = None,
) -> tuple[list[ResearchQuery], list[str]]:
    """Ask the LLM for queries, falling back to deterministic templates."""
    key = _query_cache_key(profile, summary, path_names)
    if cache is not None:
        cached = cache.get_queries(key)
        if cached:
            restored: list[ResearchQuery] = []
            for item in cached:
                try:
                    restored.append(ResearchQuery.model_validate(item))
                except ValidationError:
                    restored = []
                    break
            if (
                restored
                and all(_valid_query(query) for query in restored)
                and _meets_constraints(restored)
            ):
                return _dedupe_queries(restored), []

    target = profile.target.name if profile.target is not None else "none"
    prompt = (
        f"Dataset summary:\n{summary}\n\n"
        f"Field path: {' > '.join(path_names) or '(none)'}\n"
        f"Target: {target}\n\n"
        "Available sources:\n"
        "- openalex: scholarly papers with abstracts and citation counts\n"
        "- arxiv: preprints in physics, mathematics, computer science, and related fields\n"
        "- huggingface: public datasets and dataset metadata\n"
        "- zenodo: DOI-backed research datasets\n"
        "- wikipedia: encyclopedic background summaries (fallback only)\n\n"
        "Write queries that would find determinants, associations, or useful "
        "datasets for this specific prediction problem. Do not repeat the dataset "
        "filename. Prefer field-specific terminology."
    )
    warnings: list[str] = []
    generated = False
    try:
        payload = llm.generate_json(
            system=QUERY_SYSTEM_PROMPT,
            user=prompt,
            schema=QUERY_SCHEMA,
            max_tokens=1024,
            purpose="queries",
        )
        queries = _parse_queries(payload)
        generated = True
    except Exception as exc:  # noqa: BLE001 - research must continue
        warnings.append(f"query_generation_failed: {type(exc).__name__}: {exc}")
        queries = []
    if not _meets_constraints(queries):
        if queries:
            warnings.append(
                "query_generation_invalid: fell back to deterministic templates"
            )
        generated = False
        queries = _fallback_queries(path_names, profile)
    if generated and cache is not None:
        cache.put_queries(
            key,
            dataset_id=profile.dataset_id,
            queries=[query.model_dump(mode="json") for query in queries],
        )
    return queries, warnings


def _build_adapters(
    settings: Settings,
    *,
    cache: ResearchCache,
    dataset_id: str,
    offline: bool,
) -> dict[str, SourceAdapter]:
    import os

    common = {"cache": cache, "dataset_id": dataset_id, "offline": offline}
    adapters: dict[str, SourceAdapter] = {
        "openalex": OpenAlexAdapter(api_key=settings.openalex_api_key, **common),
        "arxiv": arXivAdapter(**common),
        "huggingface": HuggingFaceAdapter(token=settings.hf_token, **common),
        "zenodo": ZenodoAdapter(token=settings.zenodo_token, **common),
        "wikipedia": WikipediaAdapter(**common),
        "dataverse": DataverseAdapter(**common),
        "openml": OpenMLAdapter(**common),
        "datagov": DataGovAdapter(**common),
        "datacite": DataCiteAdapter(**common),
    }
    # Optional sources are only registered when configured; an unconfigured
    # optional source must not put the whole run into limited mode.
    kaggle = KaggleAdapter(**common)
    if not getattr(kaggle, "limited_mode", False):
        adapters["kaggle"] = kaggle
    semantic_scholar = SemanticScholarAdapter(
        api_key=os.environ.get("S2_API_KEY"), **common
    )
    if not getattr(semantic_scholar, "limited_mode", False):
        adapters["semantic_scholar"] = semantic_scholar
    websearch = WebSearchAdapter(**common)
    if not getattr(websearch, "limited_mode", False) and not getattr(
        websearch, "warnings", None
    ):
        adapters["websearch"] = websearch
    return adapters


def _research_keywords(records: Sequence[ResearchRecord], limit: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "are",
        "was",
        "were",
        "using",
        "based",
        "study",
        "analysis",
        "data",
        "dataset",
        "results",
        "research",
        "paper",
    }
    for record in records[:10]:
        text = f"{record.title} {record.abstract or ''}".lower()
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", text):
            if token in stop:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def run_research(
    *,
    profile: DatasetProfile,
    summary: str,
    path_names: Sequence[str],
    llm: Any,
    embeddings: Any,
    settings: Settings,
    client: httpx.Client | None = None,
    cache_path: Path | None = None,
    offline: bool | None = None,
    refresh: bool = False,
) -> ResearchResult:
    """Fetch and rank research for one dataset. Never raises on source failure."""
    offline = settings.offline if offline is None else offline

    owned_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(settings.http_timeout),
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )

    cache = ResearchCache(
        cache_path or settings.research_db,
        offline=offline,
        refresh=refresh,
    )
    records: list[ResearchRecord] = []
    sources_used: list[str] = []
    try:
        queries, warnings = generate_queries(
            summary=summary,
            path_names=path_names,
            profile=profile,
            llm=llm,
            cache=cache,
        )
        adapters = _build_adapters(
            settings,
            cache=cache,
            dataset_id=profile.dataset_id,
            offline=offline,
        )
        for query in queries:
            adapter = adapters.get(query.source)
            if adapter is None:
                warnings.append(f"unknown_source: {query.source}")
                continue
            limit = SOURCE_LIMITS.get(query.source, 5)
            try:
                found = adapter.search(query.query, limit, client)
                if not found and query.source == "huggingface":
                    short = _short_query(query.query)
                    if short is not None:
                        warnings.append(
                            "source_retry: huggingface query shortened to "
                            f"{short!r}"
                        )
                        found = adapter.search(short, limit, client)
            except SourceError as exc:
                warnings.append(f"source_failed: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one source must not kill the run
                warnings.append(
                    f"source_failed: {query.source} query {query.query!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for adapter_warning in getattr(adapter, "warnings", []):
                if adapter_warning not in warnings:
                    warnings.append(adapter_warning)
            if found:
                records.extend(found)
                if query.source not in sources_used:
                    sources_used.append(query.source)
        unique = deduplicate(records)
        ranked = rank_records(
            unique,
            summary=summary,
            embeddings=embeddings,
            top_k=RESEARCH_TOP_K,
            warnings=warnings,
        )
        limited = any(
            getattr(adapter, "limited_mode", False)
            for adapter in adapters.values()
        )
        keywords = _research_keywords(ranked)
        return ResearchResult(
            records=ranked,
            queries=queries,
            warnings=warnings,
            keywords=keywords,
            sources_used=sources_used,
            limited_mode=limited,
            offline=offline,
            cache=cache.stats(),
        )
    finally:
        cache.close()
        if owned_client and client is not None:
            client.close()


def refine_after_research(
    *,
    classifier: Any,
    profile: DatasetProfile,
    frame: Any,
    path: Any,
    result: ResearchResult,
) -> Any:
    """Apply the single post-research refinement when the trigger fires."""
    if not result.records:
        return path
    titles = [record.title for record in result.records[:5]]
    if not classifier.should_refine(path, titles):
        return path
    return classifier.refine(
        profile,
        frame,
        path,
        research_titles=titles,
        keywords=result.keywords,
        force=True,
    )


def _now_for_tests() -> str:
    return utc_now_iso()
