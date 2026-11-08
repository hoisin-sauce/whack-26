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

LLM_PROVIDERS = ("gemini", "cloudflare", "mock")
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

    llm_provider: str = "gemini"
    embed_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_embed_model: str = "gemini-embedding-2"
    embed_dims: int = 768

    openalex_api_key: str | None = None
    hf_token: str | None = None
    zenodo_token: str | None = None

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
        llm_provider = _choice(env, "WHACK26_LLM_PROVIDER", "gemini", LLM_PROVIDERS)
        embed_provider = _choice(env, "WHACK26_EMBED_PROVIDER", "gemini", EMBED_PROVIDERS)
        if mock:
            llm_provider = "mock"
            embed_provider = "mock"

        settings = cls(
            base_dir=base_dir,
            llm_provider=llm_provider,
            embed_provider=embed_provider,
            gemini_api_key=_get(env, "GEMINI_API_KEY"),
            gemini_model=_get(env, "GEMINI_MODEL", "gemini-3.6-flash") or "gemini-3.6-flash",
            gemini_embed_model=_get(env, "GEMINI_EMBED_MODEL", "gemini-embedding-2")
            or "gemini-embedding-2",
            embed_dims=_int(env, "EMBED_DIMS", 768),
            openalex_api_key=_get(env, "OPENALEX_API_KEY"),
            hf_token=_get(env, "HF_TOKEN"),
            zenodo_token=_get(env, "ZENODO_TOKEN"),
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
        )
        if settings.top_k <= 0:
            raise ValueError("WHACK26_TOP_K must be positive")
        return settings

    @property
    def mock(self) -> bool:
        return self.llm_provider == "mock" and self.embed_provider == "mock"

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
