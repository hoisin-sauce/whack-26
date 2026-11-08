"""Configuration, environment loading, and shared thresholds.

All settings are immutable and come from environment variables. The `.env`
file is loaded once, without overriding variables that are already set.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

LLM_PROVIDERS = ("deepseek", "gemini", "cloudflare", "mock")
EMBED_PROVIDERS = ("gemini", "local", "mock")


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from `.env`, never overriding the process env."""
    env_path = path or REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _get(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    value = env.get(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    value = _get(env, key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _get(env, key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {value!r}") from exc


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = _get(env, key)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{key} must be one of {sorted(_TRUE | _FALSE)}, got {value!r}")


def _csv(
    env: Mapping[str, str], key: str, default: tuple[str, ...] = ()
) -> tuple[str, ...]:
    value = _get(env, key)
    if value is None:
        return default
    return tuple(
        item.strip().lower() for item in value.split(",") if item.strip()
    )


def _choice(env: Mapping[str, str], key: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (_get(env, key, default) or default).lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    return value


def _path(env: Mapping[str, str], key: str, default: Path, base_dir: Path) -> Path:
    value = _get(env, key)
    path = Path(value) if value else default
    if not path.is_absolute():
        path = base_dir / path
    return path


@dataclass(frozen=True)
class Settings:
    """Resolved, non-secret runtime configuration."""

    base_dir: Path = REPO_ROOT

    llm_provider: str = "deepseek"
    embed_provider: str = "gemini"
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-flash"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_embed_model: str = "gemini-embedding-2"
    local_embed_model: str = "minishlab/potion-base-8M"
    embed_dims: int = 768

    openalex_api_key: str | None = None
    research_metrics_enabled: bool = False
    hf_token: str | None = None
    zenodo_token: str | None = None

    field_pitfalls_enabled: bool = False
    field_pitfalls_max_queries_per_field: int = 1
    field_pitfalls_records_per_query: int = 5
    field_pitfalls_max_per_field: int = 3
    field_pitfalls_max_total: int = 12
    field_pitfalls_sources: tuple[str, ...] = ("openalex", "arxiv")

    offline: bool = False
    log_level: str = "INFO"

    http_timeout: float = 20.0
    llm_timeout: float = 60.0
    embed_timeout: float = 30.0

    cache_dir: Path = REPO_ROOT / "cache"
    runs_dir: Path = REPO_ROOT / "runs"
    data_dir: Path = REPO_ROOT / "data"
    corpus_path: Path = REPO_ROOT / "docs" / "corpus" / "statistical-pitfalls.md"

    top_k: int = 10
    duplicate_cosine: float = 0.90
    refine_confidence: float = 0.62
    max_depth: int = 4
    desc_weight: float = 0.5
    centroid_weight: float = 0.5
    factor_map_cosine: float = 0.72

    scrape_enabled: bool = False
    scrape_max_depth: int = 2
    scrape_max_resources: int = 100
    scrape_max_links_per_resource: int = 25
    scrape_max_fetches: int = 100
    scrape_summary_threshold: float = 0.62
    scrape_link_threshold: float = 0.68
    scrape_credibility_threshold: float = 0.50
    scrape_ttl_days: int = 7
    scrape_user_agent: str = (
        "whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)"
    )
    scrape_method_version: str = "1.0.0"

    model_search_enabled: bool = False
    variants_enabled: bool = False
    generation_validation_enabled: bool = False
    model_search_folds: int = 3
    model_search_test_fraction: float = 0.30
    model_search_final_holdout_fraction: float = 0.20
    model_search_max_candidates: int = 100
    model_search_top_n: int = 5
    model_search_seed: int = 0
    model_search_backend: str = "core"

    agent_loop_enabled: bool = False
    agent_max_attempts_per_family: int = 6
    agent_max_total_attempts: int = 18
    agent_max_seconds: int = 900

    statistical_backend: str = "auto"
    external_validator: str | None = None

    final_judge_enabled: bool = False
    final_judge_model: str | None = None
    final_judge_da_model: str | None = None
    final_judge_max_workers: int = 3
    final_judge_max_tokens: int = 2000
    final_judge_min_evaluated_fraction: float = 0.5

    hierarchy_placement_enabled: bool = True
    hierarchy_tau_1: float = 0.55
    hierarchy_tau_2: float = 0.60
    hierarchy_tau_3: float = 0.65
    hierarchy_tau_4: float = 0.70
    specificity_margin: float = 0.05
    promotion_margin: float = 0.05
    max_roots: int = 20
    max_children: int = 20

    scrape_fetch_enabled: bool = False
    scrape_allowed_domains: tuple[str, ...] = ()
    scrape_blocked_domains: tuple[str, ...] = ()
    scrape_rate_limit_seconds: float = 1.0
    scrape_max_bytes: int = 2_000_000
    scrape_max_text_chars: int = 4_000
    scrape_fetch_timeout: float = 15.0
    scrape_max_redirects: int = 3
    scrape_allow_private_networks: bool = False

    def __post_init__(self) -> None:
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"llm_provider must be one of {LLM_PROVIDERS}")
        if self.embed_provider not in EMBED_PROVIDERS:
            raise ValueError(f"embed_provider must be one of {EMBED_PROVIDERS}")
        if self.embed_dims <= 0:
            raise ValueError("embed_dims must be positive")
        if not 1 <= self.max_depth <= 4:
            raise ValueError("max_depth must be between 1 and 4")
        if not 0.0 <= self.duplicate_cosine <= 1.0:
            raise ValueError("duplicate_cosine must be between 0 and 1")
        if not 0.0 <= self.refine_confidence <= 1.0:
            raise ValueError("refine_confidence must be between 0 and 1")
        if not 0.0 <= self.factor_map_cosine <= 1.0:
            raise ValueError("factor_map_cosine must be between 0 and 1")
        if not 0.0 <= self.desc_weight <= 1.0:
            raise ValueError("desc_weight must be between 0 and 1")
        if not 0.0 <= self.centroid_weight <= 1.0:
            raise ValueError("centroid_weight must be between 0 and 1")
        weight_total = self.desc_weight + self.centroid_weight
        if not math.isclose(weight_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "desc_weight and centroid_weight must sum to 1 "
                f"(got {self.desc_weight} + {self.centroid_weight} = {weight_total})"
            )
        if self.scrape_max_depth < 0:
            raise ValueError("scrape_max_depth must be non-negative")
        if self.scrape_max_resources <= 0:
            raise ValueError("scrape_max_resources must be positive")
        if self.scrape_max_links_per_resource <= 0:
            raise ValueError("scrape_max_links_per_resource must be positive")
        if self.scrape_max_fetches <= 0:
            raise ValueError("scrape_max_fetches must be positive")
        for name in (
            "scrape_summary_threshold",
            "scrape_link_threshold",
            "scrape_credibility_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.scrape_ttl_days <= 0:
            raise ValueError("scrape_ttl_days must be positive")
        if self.model_search_folds < 1:
            raise ValueError("model_search_folds must be positive")
        if not 0.0 < self.model_search_test_fraction < 1.0:
            raise ValueError("model_search_test_fraction must be between 0 and 1")
        if not 0.0 <= self.model_search_final_holdout_fraction < 1.0:
            raise ValueError(
                "model_search_final_holdout_fraction must be in [0, 1)"
            )
        if self.model_search_max_candidates <= 0:
            raise ValueError("model_search_max_candidates must be positive")
        if self.model_search_top_n <= 0:
            raise ValueError("model_search_top_n must be positive")
        if self.model_search_backend not in {"core", "sklearn", "statsmodels", "full"}:
            raise ValueError(
                "model_search_backend must be core, sklearn, statsmodels, or full"
            )
        if self.agent_max_attempts_per_family <= 0:
            raise ValueError("agent_max_attempts_per_family must be positive")
        if self.agent_max_total_attempts <= 0:
            raise ValueError("agent_max_total_attempts must be positive")
        if self.agent_max_seconds <= 0:
            raise ValueError("agent_max_seconds must be positive")
        if self.statistical_backend not in {
            "auto",
            "in_slice",
            "stats_engine",
            "external",
            "none",
        }:
            raise ValueError(
                "statistical_backend must be auto, in_slice, stats_engine, "
                "external, or none"
            )
        if not 1 <= self.field_pitfalls_max_queries_per_field <= 3:
            raise ValueError(
                "field_pitfalls_max_queries_per_field must be between 1 and 3"
            )
        if self.field_pitfalls_records_per_query <= 0:
            raise ValueError("field_pitfalls_records_per_query must be positive")
        if self.field_pitfalls_max_per_field <= 0:
            raise ValueError("field_pitfalls_max_per_field must be positive")
        if self.field_pitfalls_max_total <= 0:
            raise ValueError("field_pitfalls_max_total must be positive")
        if not self.field_pitfalls_sources:
            raise ValueError("field_pitfalls_sources must not be empty")
        if not all(
            isinstance(source, str) and source
            for source in self.field_pitfalls_sources
        ):
            raise ValueError("field_pitfalls_sources must contain non-empty strings")
        allowed_pitfall_sources = {"openalex", "arxiv", "semantic_scholar", "websearch"}
        invalid_sources = set(self.field_pitfalls_sources) - allowed_pitfall_sources
        if invalid_sources:
            raise ValueError(
                "field_pitfalls_sources contains unsupported source(s): "
                + ", ".join(sorted(invalid_sources))
            )
        if self.final_judge_max_workers <= 0:
            raise ValueError("final_judge_max_workers must be positive")
        if self.final_judge_max_tokens <= 0:
            raise ValueError("final_judge_max_tokens must be positive")
        if not 0.0 <= self.final_judge_min_evaluated_fraction <= 1.0:
            raise ValueError(
                "final_judge_min_evaluated_fraction must be between 0 and 1"
            )
        for name in (
            "hierarchy_tau_1",
            "hierarchy_tau_2",
            "hierarchy_tau_3",
            "hierarchy_tau_4",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("specificity_margin", "promotion_margin"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_roots <= 0:
            raise ValueError("max_roots must be positive")
        if self.max_children <= 0:
            raise ValueError("max_children must be positive")
        if self.scrape_rate_limit_seconds < 0:
            raise ValueError("scrape_rate_limit_seconds must be non-negative")
        if self.scrape_max_bytes <= 0:
            raise ValueError("scrape_max_bytes must be positive")
        if self.scrape_max_text_chars <= 0:
            raise ValueError("scrape_max_text_chars must be positive")
        if self.scrape_fetch_timeout <= 0:
            raise ValueError("scrape_fetch_timeout must be positive")
        if self.scrape_max_redirects < 0:
            raise ValueError("scrape_max_redirects must be non-negative")
        for name in ("scrape_allowed_domains", "scrape_blocked_domains"):
            values = getattr(self, name)
            if not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{name} must contain non-empty strings")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        mock: bool = False,
        offline: bool | None = None,
    ) -> "Settings":
        if env is None:
            load_dotenv()
            env = os.environ

        base_dir = Path(_get(env, "WHACK26_BASE_DIR", str(REPO_ROOT)) or REPO_ROOT).resolve()
        llm_provider = _choice(env, "WHACK26_LLM_PROVIDER", "deepseek", LLM_PROVIDERS)
        embed_provider = _choice(env, "WHACK26_EMBED_PROVIDER", "gemini", EMBED_PROVIDERS)
        if mock:
            llm_provider = "mock"
            embed_provider = "mock"

        settings = cls(
            base_dir=base_dir,
            llm_provider=llm_provider,
            embed_provider=embed_provider,
            deepseek_api_key=_get(env, "DEEPSEEK_API_KEY"),
            deepseek_model=_get(env, "DEEPSEEK_MODEL", "deepseek-flash")
            or "deepseek-flash",
            gemini_api_key=_get(env, "GEMINI_API_KEY"),
            gemini_model=_get(env, "GEMINI_MODEL", "gemini-3.6-flash") or "gemini-3.6-flash",
            gemini_embed_model=_get(env, "GEMINI_EMBED_MODEL", "gemini-embedding-2")
            or "gemini-embedding-2",
            local_embed_model=_get(
                env, "LOCAL_EMBED_MODEL", "minishlab/potion-base-8M"
            )
            or "minishlab/potion-base-8M",
            embed_dims=_int(env, "EMBED_DIMS", 768),
            openalex_api_key=_get(env, "OPENALEX_API_KEY"),
            research_metrics_enabled=_bool(
                env, "WHACK26_RESEARCH_METRICS_ENABLED", False
            ),
            hf_token=_get(env, "HF_TOKEN"),
            zenodo_token=_get(env, "ZENODO_TOKEN"),
            field_pitfalls_enabled=_bool(
                env, "WHACK26_FIELD_PITFALLS_ENABLED", False
            ),
            field_pitfalls_max_queries_per_field=_int(
                env, "WHACK26_FIELD_PITFALLS_MAX_QUERIES_PER_FIELD", 1
            ),
            field_pitfalls_records_per_query=_int(
                env, "WHACK26_FIELD_PITFALLS_RECORDS_PER_QUERY", 5
            ),
            field_pitfalls_max_per_field=_int(
                env, "WHACK26_FIELD_PITFALLS_MAX_PER_FIELD", 3
            ),
            field_pitfalls_max_total=_int(
                env, "WHACK26_FIELD_PITFALLS_MAX_TOTAL", 12
            ),
            field_pitfalls_sources=_csv(
                env,
                "WHACK26_FIELD_PITFALLS_SOURCES",
                ("openalex", "arxiv"),
            ),
            offline=_bool(env, "WHACK26_OFFLINE", False) if offline is None else offline,
            log_level=(_get(env, "WHACK26_LOG_LEVEL", "INFO") or "INFO").upper(),
            http_timeout=_float(env, "WHACK26_HTTP_TIMEOUT", 20.0),
            llm_timeout=_float(env, "WHACK26_LLM_TIMEOUT", 60.0),
            embed_timeout=_float(env, "WHACK26_EMBED_TIMEOUT", 30.0),
            cache_dir=_path(env, "WHACK26_CACHE_DIR", base_dir / "cache", base_dir),
            runs_dir=_path(env, "WHACK26_RUNS_DIR", base_dir / "runs", base_dir),
            data_dir=_path(env, "WHACK26_DATA_DIR", base_dir / "data", base_dir),
            corpus_path=_path(
                env,
                "WHACK26_CORPUS_PATH",
                base_dir / "docs" / "corpus" / "statistical-pitfalls.md",
                base_dir,
            ),
            top_k=_int(env, "WHACK26_TOP_K", 10),
            duplicate_cosine=_float(env, "WHACK26_DUPLICATE_COSINE", 0.90),
            refine_confidence=_float(env, "WHACK26_REFINE_CONFIDENCE", 0.62),
            max_depth=_int(env, "WHACK26_MAX_DEPTH", 4),
            desc_weight=_float(env, "WHACK26_DESC_WEIGHT", 0.5),
            centroid_weight=_float(env, "WHACK26_CENTROID_WEIGHT", 0.5),
            factor_map_cosine=_float(env, "WHACK26_FACTOR_MAP_COSINE", 0.72),
            scrape_enabled=_bool(env, "WHACK26_SCRAPE_ENABLED", False),
            scrape_max_depth=_int(env, "WHACK26_SCRAPE_MAX_DEPTH", 2),
            scrape_max_resources=_int(env, "WHACK26_SCRAPE_MAX_RESOURCES", 100),
            scrape_max_links_per_resource=_int(
                env, "WHACK26_SCRAPE_MAX_LINKS_PER_RESOURCE", 25
            ),
            scrape_max_fetches=_int(env, "WHACK26_SCRAPE_MAX_FETCHES", 100),
            scrape_summary_threshold=_float(
                env, "WHACK26_SCRAPE_SUMMARY_THRESHOLD", 0.62
            ),
            scrape_link_threshold=_float(env, "WHACK26_SCRAPE_LINK_THRESHOLD", 0.68),
            scrape_credibility_threshold=_float(
                env, "WHACK26_SCRAPE_CREDIBILITY_THRESHOLD", 0.50
            ),
            scrape_ttl_days=_int(env, "WHACK26_SCRAPE_TTL_DAYS", 7),
            scrape_user_agent=_get(
                env,
                "WHACK26_SCRAPE_USER_AGENT",
                "whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)",
            )
            or "whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)",
            scrape_method_version=_get(
                env, "WHACK26_SCRAPE_METHOD_VERSION", "1.0.0"
            )
            or "1.0.0",
            scrape_fetch_enabled=_bool(
                env, "WHACK26_SCRAPE_FETCH_ENABLED", False
            ),
            scrape_allowed_domains=_csv(
                env, "WHACK26_SCRAPE_ALLOWED_DOMAINS"
            ),
            scrape_blocked_domains=_csv(
                env, "WHACK26_SCRAPE_BLOCKED_DOMAINS"
            ),
            scrape_rate_limit_seconds=_float(
                env, "WHACK26_SCRAPE_RATE_LIMIT_SECONDS", 1.0
            ),
            scrape_max_bytes=_int(
                env, "WHACK26_SCRAPE_MAX_BYTES", 2_000_000
            ),
            scrape_max_text_chars=_int(
                env, "WHACK26_SCRAPE_MAX_TEXT_CHARS", 4_000
            ),
            scrape_fetch_timeout=_float(
                env, "WHACK26_SCRAPE_FETCH_TIMEOUT", 15.0
            ),
            scrape_max_redirects=_int(
                env, "WHACK26_SCRAPE_MAX_REDIRECTS", 3
            ),
            scrape_allow_private_networks=_bool(
                env, "WHACK26_SCRAPE_ALLOW_PRIVATE_NETWORKS", False
            ),
            model_search_enabled=_bool(env, "WHACK26_MODEL_SEARCH_ENABLED", False),
            variants_enabled=_bool(env, "WHACK26_VARIANTS_ENABLED", False),
            generation_validation_enabled=_bool(
                env, "WHACK26_GENERATION_VALIDATION_ENABLED", False
            ),
            model_search_folds=_int(env, "WHACK26_MODEL_SEARCH_FOLDS", 3),
            model_search_test_fraction=_float(
                env, "WHACK26_MODEL_SEARCH_TEST_FRACTION", 0.30
            ),
            model_search_final_holdout_fraction=_float(
                env, "WHACK26_MODEL_SEARCH_FINAL_HOLDOUT_FRACTION", 0.20
            ),
            model_search_max_candidates=_int(
                env, "WHACK26_MODEL_SEARCH_MAX_CANDIDATES", 100
            ),
            model_search_top_n=_int(env, "WHACK26_MODEL_SEARCH_TOP_N", 5),
            model_search_seed=_int(env, "WHACK26_MODEL_SEARCH_SEED", 0),
            model_search_backend=_get(
                env, "WHACK26_MODEL_SEARCH_BACKEND", "core"
            )
            or "core",
            agent_loop_enabled=_bool(env, "WHACK26_AGENT_LOOP_ENABLED", False),
            agent_max_attempts_per_family=_int(
                env, "WHACK26_AGENT_MAX_ATTEMPTS_PER_FAMILY", 6
            ),
            agent_max_total_attempts=_int(
                env, "WHACK26_AGENT_MAX_TOTAL_ATTEMPTS", 18
            ),
            agent_max_seconds=_int(env, "WHACK26_AGENT_MAX_SECONDS", 900),
            statistical_backend=_get(env, "WHACK26_STATISTICAL_BACKEND", "auto")
            or "auto",
            external_validator=_get(env, "WHACK26_EXTERNAL_VALIDATOR"),
            final_judge_enabled=_bool(env, "WHACK26_FINAL_JUDGE_ENABLED", False),
            final_judge_model=_get(env, "WHACK26_FINAL_JUDGE_MODEL"),
            final_judge_da_model=_get(env, "WHACK26_FINAL_JUDGE_DA_MODEL"),
            final_judge_max_workers=_int(
                env, "WHACK26_FINAL_JUDGE_MAX_WORKERS", 3
            ),
            final_judge_max_tokens=_int(
                env, "WHACK26_FINAL_JUDGE_MAX_TOKENS", 2000
            ),
            final_judge_min_evaluated_fraction=_float(
                env, "WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION", 0.5
            ),
            hierarchy_placement_enabled=_bool(
                env, "WHACK26_HIERARCHY_PLACEMENT_ENABLED", True
            ),
            hierarchy_tau_1=_float(env, "WHACK26_HIERARCHY_TAU_1", 0.55),
            hierarchy_tau_2=_float(env, "WHACK26_HIERARCHY_TAU_2", 0.60),
            hierarchy_tau_3=_float(env, "WHACK26_HIERARCHY_TAU_3", 0.65),
            hierarchy_tau_4=_float(env, "WHACK26_HIERARCHY_TAU_4", 0.70),
            specificity_margin=_float(env, "WHACK26_SPECIFICITY_MARGIN", 0.05),
            promotion_margin=_float(env, "WHACK26_PROMOTION_MARGIN", 0.05),
            max_roots=_int(env, "WHACK26_MAX_ROOTS", 20),
            max_children=_int(env, "WHACK26_MAX_CHILDREN", 20),
        )
        if settings.top_k <= 0:
            raise ValueError("WHACK26_TOP_K must be positive")
        return settings

    @property
    def mock(self) -> bool:
        return self.llm_provider == "mock" and self.embed_provider == "mock"

    @property
    def llm_model_name(self) -> str:
        """Return the configured model id for the selected LLM provider."""
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        if self.llm_provider == "gemini":
            return self.gemini_model
        return self.llm_provider

    @property
    def embed_model_name(self) -> str:
        """Return the configured model id for the selected embedding provider."""
        if self.embed_provider == "gemini":
            return self.gemini_embed_model
        return self.embed_provider

    @property
    def llm_api_key(self) -> str | None:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return None

    @property
    def embed_api_key(self) -> str | None:
        if self.embed_provider == "gemini":
            return self.gemini_api_key
        return None

    @property
    def fields_db(self) -> Path:
        return self.cache_dir / "fields.db"

    @property
    def research_db(self) -> Path:
        return self.cache_dir / "research.db"

    @property
    def scrape_ttl_seconds(self) -> int:
        return int(self.scrape_ttl_days * 86400)

    def with_overrides(self, **changes: object) -> "Settings":
        return replace(self, **changes)
