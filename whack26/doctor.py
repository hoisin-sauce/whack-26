"""Environment, key, corpus, and provider readiness checks."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import Settings
from .embeddings import build_embedding_client, l2_normalize
from .llm import build_llm_client

PYTHON_MIN = (3, 12)
PYTHON_MAX_EXCLUSIVE = (3, 14)
CORE_PACKAGES = ("pandas", "numpy", "httpx", "pydantic", "scipy")
OPTIONAL_PACKAGES = ("openpyxl", "xlrd", "pyarrow")
OPENALEX_PROBE = "https://api.openalex.org/works"
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
}
PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "cloudflare": "Cloudflare",
    "local": "Local",
    "mock": "Mock",
}


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | SKIP
    detail: str
    hint: str | None = None


def _version_of(module: str) -> str | None:
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if PYTHON_MIN <= sys.version_info[:2] < PYTHON_MAX_EXCLUSIVE:
        return Check("Python", "PASS", f"{version} ({sys.executable})")
    wanted = (
        f"{PYTHON_MIN[0]}.{PYTHON_MIN[1]} or "
        f"{PYTHON_MIN[0]}.{PYTHON_MAX_EXCLUSIVE[1] - 1}"
    )
    return Check(
        "Python",
        "FAIL",
        f"{version} ({sys.executable})",
        f"Use Python {wanted}; create the venv with `py -3.12 -m venv .venv`.",
    )


def _check_packages() -> Check:
    missing_core: list[str] = []
    missing_optional: list[str] = []
    versions: list[str] = []
    for module in CORE_PACKAGES:
        try:
            imported = importlib.import_module(module)
        except ImportError:
            missing_core.append(module)
            continue
        version = getattr(imported, "__version__", None)
        if version is None:
            version = _version_of(module)
        versions.append(f"{module} {version or '?'}")
    for module in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_optional.append(module)
    detail = ", ".join(versions) if versions else "no core packages found"
    if missing_core:
        return Check(
            "Packages",
            "FAIL",
            f"missing: {', '.join(missing_core)}",
            "Run `pip install -r requirements.txt` inside the Python 3.12 venv.",
        )
    if missing_optional:
        return Check(
            "Packages",
            "WARN",
            detail + f"; optional missing: {', '.join(missing_optional)}",
            "Install requirements.txt for Excel/Parquet inputs.",
        )
    return Check("Packages", "PASS", detail + "; optional formats present")


def _check_corpus(path: Path) -> Check:
    if not path.is_file():
        return Check(
            "Pitfalls corpus",
            "FAIL",
            str(path),
            "Restore docs/corpus/statistical-pitfalls.md before running the critic.",
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^version:\s*(\S+)", text, re.MULTILINE)
    version = match.group(1) if match else "unknown"
    if "S0-how-to-use" not in text:
        return Check(
            "Pitfalls corpus",
            "WARN",
            f"{path.name} ({len(text)} chars, version {version}); no S0 chunk id",
            "Regenerate the corpus from the research branch.",
        )
    return Check("Pitfalls corpus", "PASS", f"{path.name} ({len(text)} chars, version {version})")


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider.title())


def _looks_placeholder(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered.startswith(("your-", "replace-", "changeme")) or lowered in {
        "your-deepseek-api-key",
        "your-gemini-api-key",
        "your-openalex-api-key",
    }


def _key_check(name: str, value: str | None, env_name: str) -> Check:
    has_key = bool(value) and not _looks_placeholder(value)
    status = "PASS" if has_key else "FAIL"
    detail = f"{env_name} present" if has_key else f"{env_name} missing or placeholder"
    hint = None if has_key else f"Add {env_name} to .env or use --mock."
    return Check(name, status, detail, hint)


def _check_keys(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    if settings.offline:
        checks.append(Check("Provider keys", "SKIP", "offline mode"))
    else:
        if settings.llm_provider == "deepseek":
            checks.append(
                _key_check("DeepSeek key", settings.deepseek_api_key, "DEEPSEEK_API_KEY")
            )
        elif settings.llm_provider == "gemini":
            checks.append(
                _key_check("Gemini key", settings.gemini_api_key, "GEMINI_API_KEY")
            )
        else:
            checks.append(
                Check("LLM key", "SKIP", f"provider: {settings.llm_provider}")
            )

        if settings.embed_provider == "gemini":
            if settings.llm_provider != "gemini":
                checks.append(
                    _key_check("Gemini key", settings.gemini_api_key, "GEMINI_API_KEY")
                )
        else:
            checks.append(
                Check(
                    "Embedding key",
                    "SKIP",
                    f"provider: {settings.embed_provider}",
                )
            )

        if settings.mock:
            checks.append(Check("OpenAlex key", "SKIP", "mock mode"))
        else:
            has_key = bool(settings.openalex_api_key) and not _looks_placeholder(
                settings.openalex_api_key
            )
            status = "PASS" if has_key else "FAIL"
            detail = "OPENALEX_API_KEY present" if has_key else "OPENALEX_API_KEY missing or placeholder"
            hint = None if has_key else "Add OPENALEX_API_KEY to .env or use --mock."
            checks.append(Check("OpenAlex key", status, detail, hint))

    checks.append(
        Check(
            "Hugging Face token",
            "PASS" if settings.hf_token else "SKIP",
            "HF_TOKEN present" if settings.hf_token else "HF_TOKEN absent; public search still works",
        )
    )
    return checks


def _check_llm(settings: Settings) -> Check:
    label = f"{_provider_label(settings.llm_provider)} LLM"
    started = time.perf_counter()
    try:
        client = build_llm_client(settings)
        try:
            result = client.generate_json(
                system="Reply with JSON only.",
                user='Return {"ok": true}.',
                schema=PROBE_SCHEMA,
                max_tokens=64,
                purpose="doctor",
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # doctor must report, not crash
        return Check(
            label,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            "Check the key, model id, and network; `--mock` bypasses providers.",
        )
    elapsed = time.perf_counter() - started
    if result.get("ok") is not True:
        return Check(label, "FAIL", f"unexpected response: {result!r}")
    model = settings.llm_model_name
    if elapsed > 15.0:
        hint = (
            "Latency is high; `deepseek-flash` is the verified fast default."
            if settings.llm_provider == "deepseek"
            else "Latency is high; `gemini-3.6-flash` is the verified fast default."
        )
        return Check(label, "WARN", f"{model} responded in {elapsed:.1f}s", hint)
    return Check(label, "PASS", f"{model} responded in {elapsed:.1f}s")


def _check_embeddings(settings: Settings) -> Check:
    label = f"{_provider_label(settings.embed_provider)} embeddings"
    started = time.perf_counter()
    try:
        client = build_embedding_client(settings)
        try:
            vectors = client.embed(["whack26 doctor embedding probe"])
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        return Check(
            label,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            "Check the key and embedding model; run `doctor --mock` to bypass providers.",
        )
    elapsed = time.perf_counter() - started
    if not vectors or len(vectors[0]) != settings.embed_dims:
        actual = len(vectors[0]) if vectors else 0
        return Check(
            label,
            "FAIL",
            f"expected {settings.embed_dims} dims, got {actual}",
            "Set EMBED_DIMS to the provider output or reset the field store.",
        )
    norm = sum(value * value for value in vectors[0]) ** 0.5
    if not math.isfinite(norm):
        return Check(
            label,
            "FAIL",
            f"vector is not finite (norm={norm})",
            "Check the embedding model and provider response; NaN/inf vectors "
            "would corrupt similarity scores.",
        )
    if abs(norm - 1.0) > 1e-3:
        vectors[0] = l2_normalize(vectors[0])
        norm = sum(value * value for value in vectors[0]) ** 0.5
    if not math.isfinite(norm):
        return Check(
            label,
            "FAIL",
            f"vector is not finite after normalization (norm={norm})",
        )
    if abs(norm - 1.0) > 1e-3:
        return Check(label, "FAIL", f"vector is not normalized (norm={norm:.4f})")
    return Check(
        label,
        "PASS",
        f"{settings.embed_model_name}@{settings.embed_dims} in {elapsed:.1f}s",
    )


def _check_openalex(settings: Settings) -> Check:
    params: dict[str, object] = {"search": "housing", "per_page": 1}
    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key
    try:
        with httpx.Client(timeout=settings.http_timeout, follow_redirects=True) as client:
            response = client.get(OPENALEX_PROBE, params=params)
    except httpx.HTTPError as exc:
        return Check(
            "OpenAlex",
            "WARN",
            f"request failed: {type(exc).__name__}",
            "A cold run needs network or a warm cache; offline mode still works.",
        )
    if response.status_code == 200:
        try:
            count = response.json().get("meta", {}).get("count")
        except ValueError:
            count = None
        suffix = f"; {count:,} results" if isinstance(count, int) else ""
        return Check("OpenAlex", "PASS", f"authenticated search OK{suffix}")
    if response.status_code == 401:
        return Check(
            "OpenAlex",
            "FAIL",
            "HTTP 401 (invalid key)",
            "Fix OPENALEX_API_KEY or remove it to use keyless limited mode.",
        )
    if response.status_code == 403:
        return Check(
            "OpenAlex",
            "WARN",
            "HTTP 403 (key not activated or quota)",
            "The adapter falls back to keyless limited mode.",
        )
    return Check(
        "OpenAlex",
        "WARN",
        f"HTTP {response.status_code}",
        "Research will degrade; a warm cache or offline mode may still work.",
    )


def _check_cache_dir(path: Path) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix="doctor-", delete=False) as handle:
            handle.write(b"ok")
            temp_path = Path(handle.name)
        temp_path.unlink(missing_ok=True)
    except OSError as exc:
        return Check(
            "Cache directory",
            "FAIL",
            f"{path}: {exc}",
            "Set WHACK26_CACHE_DIR to a writable directory.",
        )
    return Check("Cache directory", "PASS", str(path))


def run_doctor(settings: Settings, *, network: bool = True) -> list[Check]:
    checks: list[Check] = [
        _check_python(),
        _check_packages(),
        _check_corpus(settings.corpus_path),
        _check_cache_dir(settings.cache_dir),
    ]
    checks.extend(_check_keys(settings))

    online = network and not settings.offline
    llm_label = f"{_provider_label(settings.llm_provider)} LLM"
    if (
        settings.llm_provider in {"deepseek", "gemini"}
        and settings.llm_api_key
        and online
    ):
        checks.append(_check_llm(settings))
    else:
        checks.append(Check(llm_label, "SKIP", "mock/offline or no key"))

    embed_label = f"{_provider_label(settings.embed_provider)} embeddings"
    if settings.embed_provider == "gemini" and settings.embed_api_key and online:
        checks.append(_check_embeddings(settings))
    else:
        checks.append(Check(embed_label, "SKIP", "mock/offline or no key"))

    if online and not settings.mock:
        checks.append(_check_openalex(settings))
    else:
        checks.append(Check("OpenAlex", "SKIP", "mock/offline mode"))

    return checks


def exit_code(checks: list[Check]) -> int:
    return 1 if any(check.status == "FAIL" for check in checks) else 0


def format_checks(checks: list[Check]) -> str:
    name_width = max([len("CHECK")] + [len(check.name) for check in checks])
    lines = [f"{'STATUS':<6}  {'CHECK':<{name_width}}  DETAIL"]
    for check in checks:
        lines.append(f"{check.status:<6}  {check.name:<{name_width}}  {check.detail}")
    hints = [check for check in checks if check.hint]
    if hints:
        lines.append("")
        lines.append("Hints:")
        for check in hints:
            lines.append(f"  - {check.name}: {check.hint}")
    return "\n".join(lines)
