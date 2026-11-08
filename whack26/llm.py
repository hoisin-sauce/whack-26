"""LLM clients: DeepSeek chat completions, native Gemini REST, and a mock provider."""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable, Protocol

import httpx

from .common import redact_secrets
from .config import Settings

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = (1.0, 4.0)


class LLMError(RuntimeError):
    """A provider-neutral LLM failure with the stage attached."""

    def __init__(self, message: str, *, purpose: str | None = None) -> None:
        self.purpose = purpose
        prefix = f"[{purpose}] " if purpose else ""
        super().__init__(redact_secrets(prefix + message))


class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]: ...


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        suffix = f" (blockReason={reason})" if reason else ""
        raise LLMError(f"Gemini returned no candidates{suffix}")
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text = "".join(
        str(part.get("text", "")) for part in parts if not part.get("thought")
    ).strip()
    if not text:
        finish = candidate.get("finishReason")
        raise LLMError(f"Gemini returned empty text (finishReason={finish})")
    return text


class GeminiLLMClient:
    """Small native-REST client with structured output and bounded retries."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-3.6-flash",
        base_url: str = GEMINI_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        disable_thinking: bool = True,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.disable_thinking = disable_thinking
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiLLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        if schema:
            generation_config["responseSchema"] = schema
        if self.disable_thinking:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        selected_model = model or self.model
        url = f"{self.base_url}/models/{selected_model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}

        last_error: str | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"Gemini request failed after {attempts} attempts: {last_error}",
                    purpose=purpose,
                ) from exc

            if (
                response.status_code == 400
                and "thinkingConfig" in generation_config
            ):
                # Some models reject thinkingConfig outright. Drop it for the
                # rest of the session and retry the same request once.
                generation_config.pop("thinkingConfig", None)
                self.disable_thinking = False
                payload["generationConfig"] = generation_config
                try:
                    response = self._client.post(url, headers=headers, json=payload)
                except httpx.TransportError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < self.max_retries:
                        self._sleep(
                            BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                        )
                        continue
                    raise LLMError(
                        f"Gemini request failed after {attempts} attempts: {last_error}",
                        purpose=purpose,
                    ) from exc

            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"Gemini request failed after {attempts} attempts: {last_error}",
                    purpose=purpose,
                )

            if response.status_code >= 400:
                raise LLMError(
                    f"Gemini HTTP {response.status_code}: {response.text[:500]}",
                    purpose=purpose,
                )

            finish_reason: str | None = None
            output_tokens: int | None = None
            try:
                body = response.json()
                candidates = body.get("candidates") or []
                if candidates:
                    finish_reason = candidates[0].get("finishReason")
                output_tokens = (body.get("usageMetadata") or {}).get(
                    "candidatesTokenCount"
                )
                text = _extract_text(body)
                parsed = json.loads(text)
            except (ValueError, LLMError) as exc:
                last_error = str(exc)
                details = []
                if finish_reason:
                    details.append(f"finishReason={finish_reason}")
                if output_tokens is not None:
                    details.append(f"candidatesTokenCount={output_tokens}")
                if details:
                    last_error = f"{last_error} ({', '.join(details)})"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"Gemini returned unusable JSON after {attempts} attempts: {last_error}",
                    purpose=purpose,
                ) from exc

            if not isinstance(parsed, dict):
                last_error = f"expected a JSON object, got {type(parsed).__name__}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"Gemini returned the wrong JSON shape after {attempts} attempts: {last_error}",
                    purpose=purpose,
                )
            return parsed

        raise LLMError(
            f"Gemini request failed after {attempts} attempts: {last_error or 'unknown error'}",
            purpose=purpose,
        )


def _deepseek_system_prompt(system: str, schema: dict[str, Any]) -> str:
    """Combine the caller prompt with JSON-mode instructions and the schema."""
    parts = [system.strip()] if system.strip() else []
    parts.append(
        "Respond with a single JSON object only; do not include markdown "
        "fences or commentary."
    )
    if schema:
        parts.append(
            "The JSON object must conform to this JSON Schema:\n"
            + json.dumps(schema, separators=(",", ":"))
        )
    return "\n\n".join(parts)


class DeepSeekLLMClient:
    """OpenAI-compatible DeepSeek client with JSON mode and bounded retries."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-flash",
        base_url: str = DEEPSEEK_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek provider")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.thinking_supported = True
        self._sleep = sleep
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DeepSeekLLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {
                    "role": "system",
                    "content": _deepseek_system_prompt(system, schema),
                },
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.thinking_supported:
            # Deterministic extraction does not need the default thinking pass.
            payload["thinking"] = {"type": "disabled"}
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        attempts = self.max_retries + 1
        last_error: str | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"DeepSeek request failed after {attempts} attempts: {last_error}",
                    purpose=purpose,
                ) from exc

            if response.status_code == 400 and "thinking" in payload:
                # Some deployments reject the thinking toggle outright. Drop it
                # for the rest of the session and retry before treating 400 as fatal.
                payload.pop("thinking", None)
                self.thinking_supported = False
                try:
                    response = self._client.post(url, headers=headers, json=payload)
                except httpx.TransportError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < self.max_retries:
                        self._sleep(
                            BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                        )
                        continue
                    raise LLMError(
                        f"DeepSeek request failed after {attempts} attempts: {last_error}",
                        purpose=purpose,
                    ) from exc

            if response.status_code in RETRYABLE_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"DeepSeek request failed after {attempts} attempts: {last_error}",
                    purpose=purpose,
                )

            if response.status_code >= 400:
                raise LLMError(
                    f"DeepSeek HTTP {response.status_code}: {response.text[:500]}",
                    purpose=purpose,
                )

            finish_reason: str | None = None
            output_tokens: int | None = None
            try:
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise LLMError("DeepSeek returned no choices")
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                message = choice.get("message") or {}
                text = str(message.get("content") or "").strip()
                if not text:
                    raise LLMError(
                        f"DeepSeek returned empty text (finish_reason={finish_reason})"
                    )
                output_tokens = (body.get("usage") or {}).get("completion_tokens")
                parsed = json.loads(text)
            except (ValueError, LLMError) as exc:
                last_error = str(exc)
                details = []
                if finish_reason:
                    details.append(f"finish_reason={finish_reason}")
                if output_tokens is not None:
                    details.append(f"completion_tokens={output_tokens}")
                if details:
                    last_error = f"{last_error} ({', '.join(details)})"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"DeepSeek returned unusable JSON after {attempts} attempts: {last_error}",
                    purpose=purpose,
                ) from exc

            if not isinstance(parsed, dict):
                last_error = f"expected a JSON object, got {type(parsed).__name__}"
                if attempt < self.max_retries:
                    self._sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                    continue
                raise LLMError(
                    f"DeepSeek returned the wrong JSON shape after {attempts} attempts: {last_error}",
                    purpose=purpose,
                )
            return parsed

        raise LLMError(
            f"DeepSeek request failed after {attempts} attempts: {last_error or 'unknown error'}",
            purpose=purpose,
        )


_MOCK_HYPOTHESIS_PREAMBLE = '''import numpy as np
import pandas as pd


def applies(df: pd.DataFrame) -> bool:
    """Mock hypotheses run on any non-empty frame."""
    return isinstance(df, pd.DataFrame) and len(df) > 0


'''

_MOCK_HYPOTHESIS_CODES = {
    "linear_model": _MOCK_HYPOTHESIS_PREAMBLE
    + '''PARAMETERS = {"ridge": 1e-3}


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    """Fit the first numeric column as an identity stand-in."""
    numeric = [
        name for name in df.columns
        if pd.api.types.is_numeric_dtype(df[name])
    ]
    if not numeric:
        return {"column": None, "fallback": 0.0}
    column = numeric[0]
    values = df[column]
    fallback = float(values.mean()) if values.notna().any() else 0.0
    return {"column": column, "fallback": fallback}


def _predict(model, df: pd.DataFrame):
    """Return the fitted column, or its fallback when the column is absent."""
    column = model.get("column")
    fallback = float(model.get("fallback", 0.0))
    if column is None or column not in df.columns:
        return [fallback] * len(df)
    values = pd.to_numeric(df[column], errors="coerce").fillna(fallback)
    return values.to_numpy(dtype=float)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    """Return a simple numeric stand-in so downstream plumbing can run."""
    df = df.copy()
    np.random.seed(seed)
    model = _fit(df, seed, PARAMETERS)
    return {
        "predictions": _predict(model, df),
        "target": None,
        "model_summary": {"feature": model.get("column"), "slope": 1.0},
        "notes": "Mock linear model with an identity link.",
    }
''',
    "group_comparison": _MOCK_HYPOTHESIS_PREAMBLE
    + '''PARAMETERS = {"min_group_size": 2}


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    """Fit the global mean of the first numeric column."""
    numeric = [
        name for name in df.columns
        if pd.api.types.is_numeric_dtype(df[name])
    ]
    if not numeric:
        return {"constant": 0.0}
    values = df[numeric[0]]
    constant = float(values.mean()) if values.notna().any() else 0.0
    return {"constant": constant}


def _predict(model, df: pd.DataFrame):
    """Return the fitted global fallback for every row."""
    return [float(model.get("constant", 0.0))] * len(df)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    """Predict the global mean of the first numeric column for every row."""
    df = df.copy()
    np.random.seed(seed)
    model = _fit(df, seed, PARAMETERS)
    return {
        "predictions": _predict(model, df),
        "target": None,
        "model_summary": {"group_means": [model.get("constant", 0.0)]},
        "notes": "Mock group comparison with the global fallback value.",
    }
''',
    "interaction_or_derived": _MOCK_HYPOTHESIS_PREAMBLE
    + '''PARAMETERS = {"ridge": 1e-3}


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    """Fit the first numeric column for a squared derived feature."""
    numeric = [
        name for name in df.columns
        if pd.api.types.is_numeric_dtype(df[name])
    ]
    if not numeric:
        return {"column": None, "fallback": 0.0}
    column = numeric[0]
    values = df[column]
    fallback = float(values.mean()) if values.notna().any() else 0.0
    return {"column": column, "fallback": fallback}


def _predict(model, df: pd.DataFrame):
    """Square the fitted column, or return the fallback when it is absent."""
    column = model.get("column")
    fallback = float(model.get("fallback", 0.0))
    if column is None or column not in df.columns:
        return [fallback * fallback] * len(df)
    values = pd.to_numeric(df[column], errors="coerce").fillna(fallback)
    return values.to_numpy(dtype=float) ** 2


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    """Return a stand-in derived feature so downstream plumbing can run."""
    df = df.copy()
    np.random.seed(seed)
    model = _fit(df, seed, PARAMETERS)
    return {
        "predictions": _predict(model, df),
        "target": None,
        "model_summary": {"derived_feature": f"{model.get('column')}^2"},
        "notes": "Mock derived-feature model.",
    }
''',
}


def _mock_hypothesis_fixture(purpose: str) -> dict[str, Any]:
    family = purpose.removeprefix("hypothesis_")
    code = _MOCK_HYPOTHESIS_CODES.get(
        family, _MOCK_HYPOTHESIS_CODES["linear_model"]
    )
    return {
        "title": f"Mock {family.replace('_', ' ')} hypothesis",
        "rationale": (
            "No factor applies in mock mode; this placeholder is justified from "
            "domain knowledge and exercises the hypothesis pipeline."
        ),
        "assumptions": [
            "Mock mode does not assess distributional assumptions or measurement quality."
        ],
        "required_columns": [],
        "optional_columns": [],
        "min_rows": 1,
        "claim_type": "description",
        "factors_used": [],
        "source_ids": [],
        "code": code,
    }



def _mock_fixture(purpose: str) -> dict[str, Any]:
    if purpose.startswith("classification") or purpose.startswith("refinement"):
        return {
            "action": "stop" if purpose.startswith("refinement") else "create",
            "field_id": None,
            "name": "Mock Domain",
            "description": "A deterministic placeholder field used by mock mode.",
            "reason": "mock fixture",
        }
    if purpose.startswith("target"):
        return {
            "target": None,
            "confidence": 0.0,
            "rationale": "Mock mode does not inspect the data.",
        }
    if purpose.startswith("queries"):
        return {
            "queries": [
                {
                    "source": "openalex",
                    "query": "mock determinants",
                    "rationale": "mock fixture",
                },
                {
                    "source": "openalex",
                    "query": "mock prediction",
                    "rationale": "mock fixture",
                },
                {
                    "source": "huggingface",
                    "query": "mock dataset",
                    "rationale": "mock fixture",
                },
                {
                    "source": "zenodo",
                    "query": "mock dataset",
                    "rationale": "mock fixture",
                },
                {
                    "source": "wikipedia",
                    "query": "mock topic",
                    "rationale": "mock fixture",
                },
            ]
        }
    if purpose.startswith("factors"):
        return {
            "factors": [
                {
                    "name": f"Mock factor {index}",
                    "description": "A deterministic placeholder factor.",
                    "direction": "positive",
                    "claim_type": "association",
                    "evidence_ids": ["mock:evidence-1"],
                    "candidate_columns": [],
                    "rationale": "mock fixture",
                }
                for index in range(1, 6)
            ]
        }
    if purpose.startswith("hypothesis"):
        return _mock_hypothesis_fixture(purpose)
    if purpose.startswith("critic"):
        return {
            "claim_type": "description",
            "verdict": "pass",
            "issues": [],
            "missing_information": [],
            "checks": [],
            "summary": "Mock critic result; no method gap was assessed.",
            "retrieved_chunks": ["S0-how-to-use"],
        }
    if purpose == "document_summary":
        return {
            "summary": "A deterministic mock summary of the resource.",
            "claims": ["A mock claim."],
            "methods": ["A mock method."],
            "population": None,
            "time_period": None,
            "links": [],
            "credibility_signals": {
                "peer_reviewed": True,
                "publisher": "Mock Publisher",
                "retracted": False,
            },
        }
    if purpose == "document_comparison":
        return {
            "verdict": "supports",
            "rationale": "Mock comparison result.",
            "evidence": [],
        }
    if purpose == "document_classification":
        return {
            "action": "stop",
            "field_id": None,
            "name": None,
            "description": None,
            "reason": "mock fixture",
        }
    if purpose.startswith("final_judge"):
        # Mock mode has no real evidence review.  Return an explicitly
        # unevaluated verdict so the report never shows a fake pass.
        return {
            "severity": "none",
            "finding": "",
            "evidence": "mock fixture: no final-judge review was performed",
            "suggested_fix": "",
            "falsification_test": "",
            "confidence": "low",
            "status": "suspected",
            "could_not_evaluate": True,
            "defect_class": "other",
        }
    return {"mock": True, "purpose": purpose}


class MockLLMClient:
    """Deterministic fixtures so the pipeline works before keys exist."""

    def __init__(self, fixtures: dict[str, dict[str, Any]] | None = None) -> None:
        self.fixtures = fixtures or {}
        self.calls: list[str] = []

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        purpose: str = "generic",
    ) -> dict[str, Any]:
        self.calls.append(purpose)
        fixture = self.fixtures.get(purpose)
        if fixture is None:
            fixture = _mock_fixture(purpose)
        return copy.deepcopy(fixture)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return MockLLMClient()
    if settings.llm_provider == "deepseek":
        return DeepSeekLLMClient(
            settings.deepseek_api_key or "",
            model=settings.deepseek_model,
            timeout=settings.llm_timeout,
        )
    if settings.llm_provider == "gemini":
        return GeminiLLMClient(
            settings.gemini_api_key or "",
            model=settings.gemini_model,
            timeout=settings.llm_timeout,
        )
    if settings.llm_provider == "cloudflare":
        raise LLMError(
            "Cloudflare Workers AI is configured but not implemented yet; "
            "use WHACK26_LLM_PROVIDER=deepseek, gemini, or mock"
        )
    raise LLMError(f"unknown LLM provider {settings.llm_provider!r}")
