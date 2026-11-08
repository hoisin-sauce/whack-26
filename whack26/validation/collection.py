"""Collection-method metadata: parsing, redaction, validation, persistence.

The collection metadata records *how* a dataset was produced. It is deliberately
separate from the statistical ``ValidationReport``: this module only captures and
sanity-checks provenance. It answers questions such as "does a causal claim have
a design that could identify a treatment effect?" and never judges model
performance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..common import redact_secrets
from .generation_models import CollectionMetadata, ValidationIssue

_DESIGNS = {
    "observational",
    "experimental",
    "quasi_experimental",
    "survey",
    "administrative",
    "other",
}

_LIST_FIELDS = ("known_biases", "measurement_notes", "ethical_constraints")

# Fields that are free text and therefore may contain identifiers.
_TEXT_FIELDS = (
    "unit_of_analysis",
    "sampling",
    "time_period",
    "data_source",
    "collection_method",
    "treatment",
    "instrument",
    "running_variable",
    "identification_strategy",
)

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_IPV6_RE = re.compile(
    r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}(?![0-9a-f:])"
)
# A phone-like run of digits and separators. Dates such as ``2020-2023`` only
# contain four-digit groups and are ignored by the digit-count guard.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-.()\s]{6,}\d)(?!\d)")
_BARE_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)"
    r"\s*[:=]\s*[^\s,;\"']+"
)

_REDACTED = "[redacted]"


class CollectionMetadataError(ValueError):
    """Raised when collection metadata cannot be parsed."""


def collection_to_json(metadata: CollectionMetadata) -> str:
    """Serialize metadata deterministically for storage."""
    return json.dumps(metadata.model_dump(mode="json"), sort_keys=True)


def _read_json_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, Path):
        path = source
        text = path.read_text(encoding="utf-8")
    else:
        text = str(source)
        candidate = Path(text)
        try:
            looks_like_path = candidate.is_file()
        except OSError:
            looks_like_path = False
        if looks_like_path:
            text = candidate.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CollectionMetadataError(
            f"collection JSON is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CollectionMetadataError("collection JSON must be an object")
    unknown = sorted(set(payload) - set(CollectionMetadata.model_fields))
    if unknown:
        raise CollectionMetadataError(
            "unknown collection field(s): " + ", ".join(unknown)
        )
    return payload


def parse_collection_json(
    source: str | Path | Mapping[str, Any] | None,
) -> CollectionMetadata:
    """Parse a JSON object, a path to one, or a mapping into metadata."""
    if source is None:
        return CollectionMetadata()
    try:
        return CollectionMetadata.model_validate(_read_json_source(source))
    except ValidationError as exc:
        raise CollectionMetadataError(f"invalid collection metadata: {exc}") from exc


def _split_list(raw: str) -> list[str]:
    # Semicolons are the canonical separator; accept a pipe as a convenience.
    parts = re.split(r"[;|]", raw)
    return [part.strip() for part in parts if part.strip()]


def parse_collection_fields(
    pairs: Sequence[str] | None,
) -> dict[str, Any]:
    """Parse repeated ``key=value`` CLI fields.

    List-valued fields accept ``;``-separated values and repeated flags
    accumulate. Scalar fields use the last supplied value.
    """
    result: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise CollectionMetadataError(
                f"collection field must be key=value, got {pair!r}"
            )
        key, raw = pair.split("=", 1)
        key = key.strip().replace("-", "_")
        if key not in CollectionMetadata.model_fields:
            raise CollectionMetadataError(f"unknown collection field: {key!r}")
        value = raw.strip()
        if key in _LIST_FIELDS:
            values = _split_list(value)
            result[key] = list(result.get(key) or []) + values
        elif key == "design":
            if value not in _DESIGNS:
                raise CollectionMetadataError(
                    f"unknown design {value!r}; expected one of "
                    + ", ".join(sorted(_DESIGNS))
                )
            result[key] = value
        else:
            result[key] = value or None
    return result


def parse_collection_metadata(
    *,
    json_source: str | Path | Mapping[str, Any] | None = None,
    fields: Sequence[str] | None = None,
) -> CollectionMetadata:
    """Merge an optional JSON document with explicit CLI fields.

    Explicit fields win over JSON values, which makes it easy to start from a
    shared file and override one field on the command line.
    """
    metadata = parse_collection_json(json_source)
    updates = parse_collection_fields(fields)
    if not updates:
        return metadata
    merged = metadata.model_dump(mode="json")
    merged.update(updates)
    try:
        return CollectionMetadata.model_validate(merged)
    except ValidationError as exc:
        raise CollectionMetadataError(f"invalid collection metadata: {exc}") from exc


def _redact_phone(match: re.Match[str]) -> str:
    if sum(character.isdigit() for character in match.group(0)) < 10:
        return match.group(0)
    return _REDACTED


def redact_text(text: str) -> str:
    """Remove credentials and common identifiers from free text."""
    if not isinstance(text, str):
        return text
    value = redact_secrets(text)
    value = _BARE_SECRET_RE.sub(
        lambda match: match.group(1) + "=" + _REDACTED, value
    )
    value = _EMAIL_RE.sub(_REDACTED, value)
    value = _UUID_RE.sub(_REDACTED, value)
    value = _IPV4_RE.sub(_REDACTED, value)
    value = _IPV6_RE.sub(_REDACTED, value)
    value = _PHONE_RE.sub(_redact_phone, value)
    return value


def redact_collection(metadata: CollectionMetadata) -> CollectionMetadata:
    """Return a copy safe to place in an LLM prompt or a log."""
    payload = metadata.model_dump(mode="json")
    for field in _TEXT_FIELDS:
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = redact_text(value)
    for field in _LIST_FIELDS:
        payload[field] = [redact_text(item) for item in payload.get(field) or []]
    return CollectionMetadata.model_validate(payload)


def _issue(
    *,
    evidence: str,
    severity: str,
    finding: str,
    why: str,
    fix: str,
    chunk: str,
) -> ValidationIssue:
    return ValidationIssue(
        chunk=chunk,
        severity=severity,  # type: ignore[arg-type]
        finding=finding,
        evidence=evidence,
        why=why,
        fix=fix,
    )


def check_collection_design(
    metadata: CollectionMetadata | None,
    *,
    claim_type: str = "prediction",
    outcome: str | None = None,
) -> list[ValidationIssue]:
    """Check whether the recorded design can support the claimed inference.

    Causal claims require a design with an identification strategy, a treatment,
    an outcome, and a unit of analysis. Quasi-experimental designs additionally
    require a running variable or instrument. Missing fields become validation
    issues; this function never raises for incomplete metadata.
    """
    causal = str(claim_type).strip().lower() == "causal"
    if metadata is None:
        if not causal:
            return []
        return [
            _issue(
                evidence="claim.design",
                severity="fatal",
                finding="No collection metadata is recorded for a causal claim.",
                why=(
                    "A causal claim needs a design, treatment, outcome, and "
                    "unit of analysis before it can be assessed."
                ),
                fix=(
                    "Record the collection method with --collection-json or "
                    "--collection-field."
                ),
                chunk="S4-causal-predictive",
            )
        ]

    design = metadata.design
    issues: list[ValidationIssue] = []
    severity = "fatal" if causal else "major"

    identification = metadata.identification_strategy
    has_identification = bool(identification) or design == "experimental"
    quasi = design == "quasi_experimental"
    experimental = design == "experimental"

    if causal and design in {"observational", "survey", "administrative", "other"}:
        if not has_identification:
            issues.append(
                _issue(
                    evidence="claim.design",
                    severity="fatal",
                    finding=(
                        f"A causal claim uses an {design} design without an "
                        "identification strategy."
                    ),
                    why=(
                        "Observational, survey, and administrative designs "
                        "cannot support a causal reading on their own."
                    ),
                    fix=(
                        "Weaken the claim to prediction/description, or record "
                        "an identification strategy such as a natural "
                        "experiment, difference-in-differences, or an "
                        "instrument."
                    ),
                    chunk="S4-causal-predictive",
                )
            )

    if quasi and not identification:
        issues.append(
            _issue(
                evidence="design.identification",
                severity=severity,
                finding=(
                    "A quasi-experimental design requires an identification "
                    "strategy."
                ),
                why=(
                    "The running variable or instrument only identifies an "
                    "effect together with a stated identification argument."
                ),
                fix="Record identification_strategy in the collection metadata.",
                chunk="S4-causal-predictive",
            )
        )

    if quasi and not (metadata.running_variable or metadata.instrument):
        issues.append(
            _issue(
                evidence="design.instrument",
                severity=severity,
                finding=(
                    "A quasi-experimental design requires a running variable "
                    "or an instrument."
                ),
                why=(
                    "Without a running variable or instrument there is no "
                    "source of exogenous variation."
                ),
                fix=(
                    "Record running_variable or instrument in the collection "
                    "metadata."
                ),
                chunk="S4-causal-predictive",
            )
        )

    if (causal or experimental) and not metadata.treatment:
        issues.append(
            _issue(
                evidence="design.treatment",
                severity=severity,
                finding="No treatment is recorded for a causal or experimental design.",
                why="A treatment (or exposure) must be named to interpret an effect.",
                fix="Record treatment in the collection metadata.",
                chunk="S4-causal-predictive",
            )
        )

    if causal and not outcome:
        issues.append(
            _issue(
                evidence="claim.design",
                severity="fatal",
                finding="No outcome is recorded for a causal claim.",
                why="A causal claim must name the outcome it is about.",
                fix="Set the dataset target or name the outcome in the hypothesis.",
                chunk="S4-causal-predictive",
            )
        )

    if not metadata.unit_of_analysis:
        issues.append(
            _issue(
                evidence="unit.of_analysis",
                severity=severity,
                finding="The unit of analysis is not recorded.",
                why=(
                    "The model's row unit must match the claimed unit of "
                    "analysis; otherwise the estimate describes a different "
                    "population than the claim."
                ),
                fix="Record unit_of_analysis in the collection metadata.",
                chunk="S7-sample-size",
            )
        )

    return issues


def has_causal_design(
    metadata: CollectionMetadata | None,
    *,
    outcome: str | None = None,
) -> bool:
    """Return True when the metadata has no fatal causal-design issue."""
    issues = check_collection_design(
        metadata, claim_type="causal", outcome=outcome
    )
    return not any(issue.severity == "fatal" for issue in issues)


def persist_collection(
    store: Any,
    document_id: str,
    metadata: CollectionMetadata,
    *,
    redact: bool = True,
) -> None:
    """Persist metadata in ``datasets.collection_json`` via the store API."""
    stored = redact_collection(metadata) if redact else metadata
    store.set_collection(document_id, collection_to_json(stored))


def load_collection(store: Any, document_id: str) -> CollectionMetadata | None:
    """Load and validate persisted collection metadata, or return None."""
    payload = store.get_collection(document_id)
    if payload is None:
        return None
    try:
        return CollectionMetadata.model_validate(payload)
    except ValidationError as exc:
        raise CollectionMetadataError(
            f"stored collection metadata is invalid: {exc}"
        ) from exc


def format_collection(metadata: CollectionMetadata | None) -> str:
    """Render metadata as a compact, prompt-safe block."""
    if metadata is None:
        return "(no collection metadata recorded)"
    safe = redact_collection(metadata)
    lines = [f"design: {safe.design}"]
    for field in _TEXT_FIELDS:
        value = getattr(safe, field)
        if value:
            lines.append(f"{field}: {value}")
    for field in _LIST_FIELDS:
        values = getattr(safe, field)
        if values:
            lines.append(f"{field}: " + "; ".join(values))
    return "\n".join(lines)
