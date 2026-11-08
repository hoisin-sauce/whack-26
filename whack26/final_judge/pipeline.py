"""Run the final judge over a selected model and its validation evidence."""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from pydantic import ValidationError

from ..common import utc_now_iso
from ..llm import LLMError
from ..model_search.types import ModelSearchResult
from .aggregate import aggregate
from .checks import CHECKS, compute_all, data_flags
from .evidence import EvidenceBundle, build_evidence_bundle
from .prompts import (
    COMMON_RULES,
    CRITICS,
    CRITIC_VERDICT_SCHEMA,
    DA_SYSTEM,
    CriticSpec,
)
from .types import FinalJudgeReport, JudgeFinding

log = logging.getLogger("whack26.final_judge")


def _empty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return True
    if isinstance(value, dict):
        return all(_empty(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_empty(item) for item in value)
    return False


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the prompt payload is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _unevaluated(critic: str, reason: str) -> JudgeFinding:
    return JudgeFinding(
        critic=critic,
        severity="none",
        finding="",
        evidence=reason,
        confidence="low",
        status="suspected",
        could_not_evaluate=True,
        defect_class="other",
    )


def _ask(
    llm: Any,
    *,
    system: str,
    user: str,
    critic: str,
    purpose: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> JudgeFinding:
    last_error: Exception | None = None
    prompt = user
    for attempt in (1, 2):
        try:
            payload = llm.generate_json(
                system=system,
                user=prompt,
                schema=CRITIC_VERDICT_SCHEMA,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=purpose,
            )
            if not isinstance(payload, dict):
                raise ValueError("final judge returned a non-object response")
            return JudgeFinding(critic=critic, **payload)
        except (LLMError, ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            if attempt == 1:
                prompt = (
                    f"{user}\n\nYour previous response was rejected: "
                    f"{str(exc)[:400]}\nReturn ONLY corrected valid JSON."
                )
    assert last_error is not None
    raise last_error


def run_critic(
    spec: CriticSpec,
    *,
    model_description: str,
    evidence: dict,
    llm: Any,
    model: str | None,
    max_tokens: int,
) -> JudgeFinding:
    supplied = evidence.get(spec.name)
    if _empty(supplied):
        return _unevaluated(spec.name, "no evidence supplied")
    computed = CHECKS[spec.name](supplied)
    user = json.dumps(
        _json_safe(
            {
                "model_description": model_description,
                "supplied": supplied,
                "computed": computed,
            }
        ),
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return _ask(
            llm,
            system=spec.system + COMMON_RULES,
            user=user,
            critic=spec.name,
            purpose=f"final_judge_{spec.name}",
            model=model,
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - a critic failure is not a run failure
        log.warning("Final judge critic %s failed: %s", spec.name, exc)
        return _unevaluated(spec.name, f"critic error: {exc}")


def devils_advocate(
    *,
    model_description: str,
    verdicts: Sequence[JudgeFinding],
    flags: Sequence[str],
    llm: Any,
    model: str | None,
    max_tokens: int,
) -> JudgeFinding:
    user = json.dumps(
        _json_safe(
            {
                "model_description": model_description,
                "code_verified_flags": list(flags),
                "prior_findings": [
                    verdict.model_dump(mode="json") for verdict in verdicts
                ],
            }
        ),
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return _ask(
            llm,
            system=DA_SYSTEM,
            user=user,
            critic="devils_advocate",
            purpose="final_judge_devils_advocate",
            model=model,
            temperature=0.4,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - the audit still returns
        log.warning("Final judge devil's advocate failed: %s", exc)
        return _unevaluated("devils_advocate", f"critic error: {exc}")


def run_final_judge(
    *,
    model_search: ModelSearchResult | None = None,
    validation_reports: Sequence[Any] = (),
    profile: Any | None = None,
    artifact: Any | None = None,
    critics: Sequence[Any] = (),
    generation_validations: Sequence[Any] = (),
    variants: Any | None = None,
    collection: Any | None = None,
    factors: Sequence[Any] = (),
    llm: Any,
    settings: Any | None = None,
    max_workers: int | None = None,
    max_tokens: int | None = None,
    min_evaluated_fraction: float | None = None,
) -> FinalJudgeReport:
    """Run the adversarial audit and return a typed, serializable report.

    The judge is advisory.  It never re-ranks candidates and never blocks the
    artifact; callers may ignore the decision or surface it in the report.
    """
    bundle: EvidenceBundle = build_evidence_bundle(
        model_search=model_search,
        validation_reports=validation_reports,
        profile=profile,
        artifact=artifact,
        critics=critics,
        generation_validations=generation_validations,
        variants=variants,
        collection=collection,
        factors=factors,
    )
    computed = compute_all(bundle.evidence)
    flags = data_flags(computed)

    workers = max_workers
    if workers is None:
        workers = int(getattr(settings, "final_judge_max_workers", 3) or 3)
    workers = max(1, int(workers))
    token_limit = max_tokens
    if token_limit is None:
        token_limit = int(getattr(settings, "final_judge_max_tokens", 2000) or 2000)
    token_limit = max(256, int(token_limit))
    fraction = min_evaluated_fraction
    if fraction is None:
        fraction = float(
            getattr(settings, "final_judge_min_evaluated_fraction", 0.5) or 0.5
        )
    model = getattr(settings, "final_judge_model", None)
    da_model = getattr(settings, "final_judge_da_model", None) or model

    def _run(spec: CriticSpec) -> JudgeFinding:
        return run_critic(
            spec,
            model_description=bundle.model_description,
            evidence=bundle.evidence,
            llm=llm,
            model=model,
            max_tokens=token_limit,
        )

    if workers > 1 and len(CRITICS) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(_run, CRITICS))
    else:
        verdicts = [_run(spec) for spec in CRITICS]

    devil = devils_advocate(
        model_description=bundle.model_description,
        verdicts=verdicts,
        flags=flags,
        llm=llm,
        model=da_model,
        max_tokens=token_limit,
    )
    aggregation = aggregate(
        verdicts,
        devil,
        flags,
        min_evaluated_fraction=fraction,
        missing_evidence=bundle.missing_evidence,
    )
    report = FinalJudgeReport(
        generated_at=utc_now_iso(),
        model_summary=bundle.model_summary,
        model_description=bundle.model_description,
        critics_run=[
            finding.critic
            for finding in verdicts
            if not finding.could_not_evaluate
        ],
        **aggregation,
    )
    if report.warnings is None:
        report.warnings = []
    return report


__all__ = [
    "devils_advocate",
    "run_critic",
    "run_final_judge",
]
