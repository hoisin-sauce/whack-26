"""Statistical-validation backend registry and external-validator seam.

Selection follows ``WHACK26_STATISTICAL_BACKEND``:

* ``auto`` — return-like series use ``stats_engine``, general data uses
  ``in_slice``, and a configured external validator takes precedence;
* ``in_slice`` — model-search metrics, permutation, stability, diagnostics;
* ``stats_engine`` — the existing four-layer engine, only for a return-like
  series, reusing :mod:`whack26.validation.pipeline`;
* ``external`` — a validator loaded by entry point or module path;
* ``none`` — no statistical validation; the result is ``insufficient``.

The engine is **never** called for binary, multiclass, count, constant, or
general regression data, and an absent external validator falls back to
``in_slice`` with a warning — never to ``stats_engine``.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import Settings
from .models import ValidationConfig, ValidationReport
from .pipeline import validate_hypothesis, validate_run
from .statistical import (
    StatisticalValidationResult,
    ValidationContext,
    error_result,
    in_slice_supported,
    insufficient_result,
    stats_engine_supported,
    validate_in_slice,
    validation_result_from_report,
)

BACKEND_NAMES = ("auto", "in_slice", "stats_engine", "external", "none")
EXTERNAL_ENTRY_POINT_GROUPS = (
    "whack26.validation",
    "whack26.statistical_validators",
)


@runtime_checkable
class StatisticalValidationBackend(Protocol):
    """One statistical-validation strategy."""

    name: str

    def supports(self, context: ValidationContext) -> bool: ...

    def validate(self, context: ValidationContext) -> StatisticalValidationResult: ...


@runtime_checkable
class ExternalStatisticalValidator(Protocol):
    """The in-progress general validation library's adapter contract."""

    def validate(self, context: ValidationContext) -> Any: ...


class InSliceBackend:
    name = "in_slice"

    def supports(self, context: ValidationContext) -> bool:
        supported, _ = in_slice_supported(context)
        return supported

    def validate(self, context: ValidationContext) -> StatisticalValidationResult:
        return validate_in_slice(context)


class StatsEngineBackend:
    name = "stats_engine"

    def supports(self, context: ValidationContext) -> bool:
        supported, _ = stats_engine_supported(context)
        return supported

    def validate(self, context: ValidationContext) -> StatisticalValidationResult:
        supported, reason = stats_engine_supported(context)
        if not supported:
            return insufficient_result(
                context.hypothesis_id,
                f"stats_engine_not_applicable: {reason}",
                backend=self.name,
                target=context.target,
                target_kind=context.resolved_target_kind,
                warnings=context.warnings,
            )
        try:
            report = self._run(context)
        except Exception as exc:  # noqa: BLE001 - validation is advisory
            return error_result(
                context.hypothesis_id,
                f"stats_engine_failed: {type(exc).__name__}: {exc}",
                backend=self.name,
                warnings=context.warnings,
            )
        result = validation_result_from_report(report, backend=self.name)
        result.target_kind = result.target_kind or context.resolved_target_kind
        result.engine_applicable = True
        result.warnings = list(dict.fromkeys([*result.warnings, *context.warnings]))
        if result.status == "skipped":
            result.reason = result.reason or "stats_engine_skipped"
        return result

    @staticmethod
    def _run(context: ValidationContext) -> ValidationReport:
        config = context.validation_config or ValidationConfig()
        if context.run_path is not None:
            run_result = validate_run(Path(context.run_path), config=config)
            for report in run_result.reports:
                if report.hypothesis_id == context.hypothesis_id:
                    return report
            if len(run_result.reports) == 1:
                return run_result.reports[0]
            raise ValueError(
                f"no report for hypothesis {context.hypothesis_id!r} in "
                f"{context.run_path}"
            )
        if (
            context.artifact is None
            or context.hypothesis is None
            or context.run_dir is None
        ):
            raise ValueError(
                "stats_engine_requires_run_artifact: a run directory, artifact, "
                "and hypothesis are required"
            )
        return validate_hypothesis(
            context.hypothesis,
            run_dir=Path(context.run_dir),
            artifact=context.artifact,
            config=config,
        )


class ExternalBackend:
    """Adapter for an externally configured validator.

    Loading uses ``module:attribute``, a plain module exposing ``validator`` /
    ``backend`` / ``VALIDATOR`` / ``validate``, or a matching entry point. A
    missing, throwing, or invalid validator falls back to ``in_slice`` with a
    warning; it never falls back to ``stats_engine``.
    """

    name = "external"

    def __init__(self, spec: str | None) -> None:
        self.spec = spec

    def supports(self, context: ValidationContext) -> bool:
        return bool(self.spec)

    def validate(self, context: ValidationContext) -> StatisticalValidationResult:
        if not self.spec:
            return self._fallback(
                context,
                "external_validator_unconfigured",
                external_status="unavailable",
            )
        try:
            validator = load_external_validator(self.spec)
        except Exception as exc:  # noqa: BLE001 - fall back to in-slice
            return self._fallback(
                context,
                "external_validator_unavailable: "
                f"{type(exc).__name__}: {exc}",
                external_status="unavailable",
            )
        try:
            raw = validator.validate(context)
        except Exception as exc:  # noqa: BLE001 - fall back to in-slice
            return self._fallback(
                context,
                f"external_validator_failed: {type(exc).__name__}: {exc}",
                external_status="error",
            )
        try:
            result = coerce_external_result(raw)
        except Exception as exc:  # noqa: BLE001 - fall back to in-slice
            return self._fallback(
                context,
                f"external_validator_invalid_result: {type(exc).__name__}: {exc}",
                external_status="invalid",
            )
        result.backend = "external"
        result.external_status = "ok"
        result.external_validator = self.spec
        if result.library_version is None:
            result.library_version = external_library_version(validator, self.spec)
        if context.warnings:
            result.warnings = list(
                dict.fromkeys([*result.warnings, *context.warnings])
            )
        return result

    def _fallback(
        self,
        context: ValidationContext,
        warning: str,
        *,
        external_status: str,
    ) -> StatisticalValidationResult:
        result = validate_in_slice(context)
        result.external_status = external_status
        result.external_validator = self.spec
        result.warnings = list(dict.fromkeys([*result.warnings, warning]))
        if result.verdict == "insufficient":
            result.reason = result.reason or "statistical_backend_unavailable"
        return result


class NoneBackend:
    name = "none"

    def supports(self, context: ValidationContext) -> bool:
        return True

    def validate(self, context: ValidationContext) -> StatisticalValidationResult:
        return insufficient_result(
            context.hypothesis_id,
            "statistical_backend_unavailable",
            backend=self.name,
            target=context.target,
            target_kind=context.resolved_target_kind,
            warnings=[
                *context.warnings,
                "statistical_backend_disabled: no statistical backend was selected",
            ],
        )


# ---------------------------------------------------------------------------
# External adapter loading
# ---------------------------------------------------------------------------


def load_external_validator(spec: str) -> ExternalStatisticalValidator:
    """Load a validator from ``module:attribute``, a module, or an entry point."""
    spec = spec.strip()
    if not spec:
        raise ValueError("the external validator spec is empty")
    module_name, _separator, attribute = spec.partition(":")
    target: Any
    if module_name:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            target = _load_entry_point(spec)
        else:
            target = module
            if attribute:
                target = getattr(module, attribute)
            else:
                for candidate in ("validator", "backend", "VALIDATOR", "validate"):
                    if hasattr(module, candidate):
                        target = getattr(module, candidate)
                        break
    else:
        target = _load_entry_point(spec)

    if isinstance(target, type):
        target = target()
    if not callable(getattr(target, "validate", None)):
        if callable(target):
            target = target()
    if not callable(getattr(target, "validate", None)):
        raise TypeError(
            f"external validator {spec!r} does not expose a callable validate()"
        )
    return target


def _load_entry_point(spec: str) -> Any:
    for group in EXTERNAL_ENTRY_POINT_GROUPS:
        try:
            entry_points = metadata.entry_points(group=group)
        except TypeError:  # pragma: no cover - older importlib.metadata
            entry_points = metadata.entry_points().get(group, [])
        for entry_point in entry_points:
            if entry_point.name == spec:
                return entry_point.load()
    raise ImportError(f"could not load external validator {spec!r}")


def external_library_version(validator: Any, spec: str) -> str | None:
    for owner in (validator, type(validator)):
        version = getattr(owner, "version", None)
        if isinstance(version, str) and version:
            return version
    module_name = spec.partition(":")[0]
    if not module_name or " " in module_name:
        return None
    top_level = module_name.split(".")[0]
    try:
        return metadata.version(top_level)
    except (metadata.PackageNotFoundError, ValueError):
        return None


def coerce_external_result(raw: Any) -> StatisticalValidationResult:
    """Validate the external library's return value."""
    if isinstance(raw, StatisticalValidationResult):
        return raw.model_copy(deep=True)
    if isinstance(raw, ValidationReport):
        return validation_result_from_report(raw, backend="external")
    if isinstance(raw, dict):
        return StatisticalValidationResult.model_validate(raw)
    raise TypeError(
        "external validator must return StatisticalValidationResult, "
        f"ValidationReport, or a dict; got {type(raw).__name__}"
    )


# ---------------------------------------------------------------------------
# Registry and selection
# ---------------------------------------------------------------------------


@dataclass
class BackendResolution:
    backend: StatisticalValidationBackend
    requested: str
    warnings: list[str] = field(default_factory=list)


def build_registry(
    *,
    external_validator: str | None = None,
) -> dict[str, StatisticalValidationBackend]:
    return {
        "in_slice": InSliceBackend(),
        "stats_engine": StatsEngineBackend(),
        "external": ExternalBackend(external_validator),
        "none": NoneBackend(),
    }


def resolve_backend(
    name: str,
    context: ValidationContext,
    *,
    external_validator: str | None = None,
) -> BackendResolution:
    requested = (name or "auto").strip().lower()
    if requested not in BACKEND_NAMES:
        return BackendResolution(
            backend=NoneBackend(),  # type: ignore[arg-type]
            requested=requested,
            warnings=[f"unknown_statistical_backend: {name!r}; using none"],
        )
    registry = build_registry(external_validator=external_validator)
    if requested != "auto":
        return BackendResolution(backend=registry[requested], requested=requested)

    if external_validator:
        return BackendResolution(backend=registry["external"], requested=requested)
    engine_supported, _ = stats_engine_supported(context)
    if engine_supported:
        return BackendResolution(
            backend=registry["stats_engine"], requested=requested
        )
    in_slice, _ = in_slice_supported(context)
    if in_slice:
        return BackendResolution(backend=registry["in_slice"], requested=requested)
    return BackendResolution(backend=registry["none"], requested=requested)


def validate_statistical(
    context: ValidationContext,
    *,
    backend: str = "auto",
    external_validator: str | None = None,
) -> StatisticalValidationResult:
    """Validate one hypothesis with the configured backend."""
    resolution = resolve_backend(
        backend, context, external_validator=external_validator
    )
    selected = resolution.backend
    result = selected.validate(context)

    if resolution.warnings:
        result.warnings = list(
            dict.fromkeys([*result.warnings, *resolution.warnings])
        )
    if resolution.requested != "auto" or selected.name != "stats_engine":
        return result

    # ``auto`` on a return series: the engine is primary, and in-slice is
    # reported alongside it when model-search data is available. The engine
    # never silently overrides model-search selection.
    supplementary: list[dict[str, Any]] = []
    in_slice_available, _ = in_slice_supported(context)
    in_slice_result = validate_in_slice(context) if in_slice_available else None
    if in_slice_result is not None:
        supplementary.append(in_slice_result.model_dump(mode="json"))
    if result.verdict == "insufficient" and in_slice_result is not None:
        in_slice_result.warnings = list(
            dict.fromkeys(
                [
                    *in_slice_result.warnings,
                    f"stats_engine_not_applicable: {result.reason or 'unavailable'}",
                ]
            )
        )
        in_slice_result.supplementary = supplementary
        return in_slice_result
    result.supplementary = supplementary
    return result


def validate_with_settings(
    context: ValidationContext,
    settings: Settings,
) -> StatisticalValidationResult:
    """Validate with ``Settings.statistical_backend`` / ``external_validator``."""
    return validate_statistical(
        context,
        backend=settings.statistical_backend,
        external_validator=settings.external_validator,
    )


__all__ = [
    "BACKEND_NAMES",
    "BackendResolution",
    "ExternalBackend",
    "ExternalStatisticalValidator",
    "InSliceBackend",
    "NoneBackend",
    "StatisticalValidationBackend",
    "StatsEngineBackend",
    "build_registry",
    "coerce_external_result",
    "external_library_version",
    "load_external_validator",
    "resolve_backend",
    "validate_statistical",
    "validate_with_settings",
]
