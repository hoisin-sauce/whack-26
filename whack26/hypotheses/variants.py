"""Dataset variants for agentic hypothesis generation.

The generator must not merely fit the observed data; it must also show that a
model is not exploiting personally identifiable information (PII) and is not
"unscrambling" labels that were deliberately jumbled.  This module builds the
three variants described in the plan: the original frame, an anonymized frame
with direct identifiers removed/hashed, and a jumbled-label frame that
preserves the target's marginal distribution.

Everything here is pure: input frames and profiles are copied, never mutated,
and every random operation is seeded so repeated runs produce identical
variants.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..common import redact_secrets
from ..profile import DatasetProfile, _name_looks_sensitive

__all__ = [
    "DatasetVariant",
    "JumbledTarget",
    "VariantReport",
    "VariantScore",
    "VariantSet",
    "aggregate_variant_reports",
    "anonymize_frame",
    "build_variants",
    "evaluate_hypothesis_variants",
    "evaluate_variants",
    "jumble_target",
    "redact_text",
    "write_variant_report",
]

VariantKind = Literal["original", "anonymized", "jumbled"]
JumbleMethod = Literal[
    "iid_permutation",
    "block_permutation",
    "group_permutation",
    "identity",
]

_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_IPV6_RE = re.compile(
    r"(?i)(?<![0-9a-f:])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}(?![0-9a-f:])"
)
# At least eight digits, optionally grouped, so ordinary years and short codes
# are not mistaken for phone numbers.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-.()\s]{6,}\d)(?!\d)")
_IP_NAME_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:ip|ipv4|ipv6|ipaddr|ipaddress|ip_address)(?:$|[^a-z0-9])"
)
_BARE_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret|password)"
    r"\s*[:=]\s*[^\s,;\"']+"
)

_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    _EMAIL_RE,
    _UUID_RE,
    _IPV4_RE,
    _IPV6_RE,
    _PHONE_RE,
)

_REDACTED = "[redacted]"

# Score thresholds above which a jumbled-label model is suspicious.  These are
# deliberately conservative; a jumbled model should be at chance.
_DEFAULT_HIGHER_LEAK_THRESHOLD: dict[str, float] = {
    "r2": 0.05,
    "auc": 0.55,
    "accuracy": 0.55,
    "balanced_accuracy": 0.55,
}
_LOWER_IS_BETTER_METRICS = frozenset(
    {"rmse", "mae", "log_loss", "poisson_deviance", "aic", "bic"}
)


class DatasetVariant(BaseModel):
    kind: VariantKind
    frame_sha256: str
    target_column: str | None
    rows: int
    seed: int
    dropped_columns: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VariantSet(BaseModel):
    original: DatasetVariant
    anonymized: DatasetVariant
    jumbled: DatasetVariant


@dataclass
class JumbledTarget:
    """Result of :func:`jumble_target`; the frame index is preserved."""

    series: pd.Series
    method: JumbleMethod
    seed: int
    group_column: str | None = None
    block_size: int | None = None
    notes: list[str] | None = None


class VariantScore(BaseModel):
    kind: VariantKind
    metric: str
    score: float | None = None
    baseline: float | None = None
    permutation_p: float | None = None
    notes: list[str] = Field(default_factory=list)


class VariantReport(BaseModel):
    """Variant metadata plus the leak/PII gates for one generated model."""

    variants: VariantSet
    hypothesis_id: str | None = None
    seed: int = 0
    target_column: str | None = None
    metric: str | None = None
    higher_is_better: bool = True
    scores: list[VariantScore] = Field(default_factory=list)
    jumbled_label_leak: bool = False
    pii_reliance: bool = False
    verdict: Literal["pass", "reject", "insufficient"] = "insufficient"
    reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_artifact(self) -> dict:
        return self.model_dump(mode="json")

    def write(self, path) -> None:
        """Persist this report as ``variants.json``."""
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Text sanitizers
# ---------------------------------------------------------------------------


def redact_text(text: str) -> str:
    """Remove credentials and common PII tokens from free text."""
    if not isinstance(text, str):
        return text
    redacted = redact_secrets(text)
    redacted = _BARE_SECRET_RE.sub(
        lambda match: match.group(1) + "=" + _REDACTED, redacted
    )
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def _is_text_series(series: pd.Series) -> bool:
    try:
        if pd.api.types.is_object_dtype(series.dtype):
            return True
        return bool(pd.api.types.is_string_dtype(series.dtype))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _pii_ratio(series: pd.Series, pattern: re.Pattern[str]) -> float:
    values = [value for value in series.dropna().astype(str).head(200)]
    if not values:
        return 0.0
    matched = sum(1 for value in values if pattern.search(value) is not None)
    return matched / len(values)


def _identifier_name(name: str) -> bool:
    """True for a name that directly identifies a person or device."""
    return bool(_name_looks_sensitive(name)) or bool(_IP_NAME_RE.search(name))


def _hash_series(series: pd.Series, salt: str) -> pd.Series:
    def hashed(value: object) -> object:
        try:
            if pd.isna(value):
                return value
        except (TypeError, ValueError):
            pass
        digest = hashlib.sha256(
            f"{salt}\x1f{value}".encode("utf-8", errors="replace")
        ).hexdigest()
        return f"id_{digest[:16]}"

    return series.map(hashed)


def _redact_series(series: pd.Series) -> pd.Series:
    def cleaned(value: object) -> object:
        if isinstance(value, str):
            return redact_text(value)
        return value

    redacted = series.map(cleaned)
    try:
        return redacted.astype(series.dtype)
    except (TypeError, ValueError):  # pragma: no cover - dtype may be exotic
        return redacted


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


def anonymize_frame(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    hash_ids: bool = True,
    hash_salt: str = "",
    blind: bool = False,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str], list[str]]:
    """Return an anonymized copy of ``frame``.

    Direct identifiers are dropped.  High-cardinality non-sensitive identifier
    columns are hashed (or dropped when ``hash_ids`` is false).  PII tokens
    inside kept text columns are redacted.  Remaining columns keep their names
    and dtypes unless ``blind`` renames non-target columns.

    Pass ``return_details=True`` to also receive the dropped column names and
    human-readable notes (``(frame, dropped, notes)``).
    """
    result = _anonymize_frame(
        profile,
        frame,
        hash_ids=hash_ids,
        hash_salt=hash_salt,
        blind=blind,
    )
    return result if return_details else result[0]


def _anonymize_frame(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    hash_ids: bool = True,
    hash_salt: str = "",
    blind: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    working = frame.copy(deep=True)
    profiles = {column.name: column for column in profile.columns}
    target = profile.target.name if profile.target is not None else None
    dropped: list[str] = []
    notes: list[str] = []

    for name in list(working.columns):
        column_name = str(name)
        series = working[name]
        column_profile = profiles.get(column_name)

        if target is not None and column_name == target:
            if _identifier_name(column_name):
                working = working.drop(columns=[name])
                dropped.append(column_name)
                notes.append(
                    f"dropped target column {column_name!r}: it looks like a "
                    "direct identifier"
                )
            continue

        if _identifier_name(column_name):
            working = working.drop(columns=[name])
            dropped.append(column_name)
            notes.append(
                f"dropped direct identifier column {column_name!r}"
            )
            continue

        if column_profile is not None and column_profile.role == "id":
            if hash_ids:
                working[name] = _hash_series(series, hash_salt)
                notes.append(
                    f"hashed high-cardinality identifier column {column_name!r}"
                )
            else:
                working = working.drop(columns=[name])
                dropped.append(column_name)
                notes.append(
                    f"dropped high-cardinality identifier column {column_name!r}"
                )
            continue

        if not _is_text_series(series):
            continue

        ratios = {
            label: _pii_ratio(series, pattern)
            for label, pattern in (
                ("email", _EMAIL_RE),
                ("uuid", _UUID_RE),
                ("ip", _IPV4_RE),
                ("ipv6", _IPV6_RE),
                ("phone", _PHONE_RE),
            )
        }
        label, ratio = max(ratios.items(), key=lambda item: item[1])
        if ratio >= 0.5:
            working = working.drop(columns=[name])
            dropped.append(column_name)
            notes.append(
                f"dropped column {column_name!r}: values look like {label} "
                "identifiers"
            )
            continue

        redacted = _redact_series(series)
        if not redacted.equals(series):
            working[name] = redacted
            notes.append(f"redacted PII tokens in column {column_name!r}")

    if blind:
        renames = {}
        index = 0
        for name in working.columns:
            if target is not None and str(name) == target:
                continue
            index += 1
            renames[name] = f"column_{index}"
        if renames:
            working = working.rename(columns=renames)
            notes.append(f"blind variant renamed {len(renames)} non-target columns")

    return working, dropped, notes


# ---------------------------------------------------------------------------
# Jumbling
# ---------------------------------------------------------------------------


def _iid_permutation(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    positions = rng.permutation(len(series))
    if np.array_equal(positions, np.arange(len(series))):
        # A tiny frame can randomly draw the identity permutation.  Rotate so
        # the jumbled variant is visibly different, deterministically.
        positions = np.roll(positions, 1)
    values = series.to_numpy(copy=True)[positions]
    return pd.Series(values, index=series.index, name=series.name, dtype=series.dtype)


def _block_permutation(
    series: pd.Series,
    rng: np.random.Generator,
    *,
    block_size: int,
) -> tuple[pd.Series, int]:
    n = len(series)
    if n < 2:
        return series.copy(deep=True), max(1, n)
    size = max(1, min(int(block_size), n))
    if size == n:
        size = max(1, n - 1)
    blocks = [np.arange(start, min(start + size, n)) for start in range(0, n, size)]
    order = rng.permutation(len(blocks))
    if len(blocks) > 1 and np.array_equal(order, np.arange(len(blocks))):
        order = np.roll(order, 1)
    positions = np.concatenate([blocks[index] for index in order])
    values = series.to_numpy(copy=True)[positions]
    return pd.Series(values, index=series.index, name=series.name, dtype=series.dtype), size


def _group_permutation(
    series: pd.Series,
    groups: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    values = series.to_numpy(copy=True)
    group_values = groups.to_numpy(copy=True)
    result = values.copy()
    labels = [label for label in pd.unique(group_values) if not _is_missing(label)]
    labels.sort(key=lambda label: str(label))
    for label in labels:
        positions = np.flatnonzero(group_values == label)
        if len(positions) < 2:
            continue
        order = rng.permutation(len(positions))
        if np.array_equal(order, np.arange(len(positions))):
            order = np.roll(order, 1)
        result[positions] = values[positions][order]
    return pd.Series(result, index=series.index, name=series.name, dtype=series.dtype)


def _is_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def jumble_target(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    group_column: str | None = None,
    block_size: int | None = None,
    return_details: bool = False,
) -> pd.Series | JumbledTarget:
    """Return a jumbled target series with the same dtype and distribution.

    * IID data: a seeded random permutation.
    * Time series (``profile.time_column``): a contiguous block permutation so
      local autocorrelation survives while the target/feature alignment breaks.
    * Grouped data (``group_column``): permutation within each group.

    Pass ``return_details=True`` to receive a :class:`JumbledTarget` carrying
    the method, seed, block/group metadata, and notes.
    """
    result = _jumble_target(
        profile,
        frame,
        seed=seed,
        group_column=group_column,
        block_size=block_size,
    )
    return result if return_details else result.series


def _jumble_target(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    group_column: str | None = None,
    block_size: int | None = None,
) -> JumbledTarget:
    target = profile.target.name if profile.target is not None else None
    if target is None or target not in frame.columns:
        raise ValueError("jumble_target requires a target column present in the frame")

    series = frame[target].copy(deep=True)
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    if group_column is not None and group_column in frame.columns:
        groups = frame[group_column]
        jumbled = _group_permutation(series, groups, rng)
        notes.append(f"grouped permutation within {group_column!r}")
        return JumbledTarget(
            series=jumbled,
            method="group_permutation",
            seed=seed,
            group_column=group_column,
            notes=notes,
        )

    time_column = profile.time_column
    if time_column is not None and time_column in frame.columns:
        n = len(series)
        chosen = (
            int(block_size)
            if block_size is not None and int(block_size) > 0
            else max(2, int(round(math.sqrt(max(n, 2)))))
        )
        jumbled, size = _block_permutation(series, rng, block_size=chosen)
        notes.append(
            f"block permutation with block_size={size} along {time_column!r}"
        )
        return JumbledTarget(
            series=jumbled,
            method="block_permutation",
            seed=seed,
            block_size=size,
            notes=notes,
        )

    jumbled = _iid_permutation(series, rng)
    notes.append("IID random permutation")
    return JumbledTarget(
        series=jumbled,
        method="iid_permutation",
        seed=seed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(repr([str(column) for column in frame.columns]).encode("utf-8"))
    digest.update(repr([str(dtype) for dtype in frame.dtypes]).encode("utf-8"))
    try:
        hashed = pd.util.hash_pandas_object(frame, index=False)
        digest.update(hashed.to_numpy(dtype=np.uint64).tobytes())
    except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
        digest.update(frame.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def _variant(
    kind: VariantKind,
    frame: pd.DataFrame,
    *,
    target_column: str | None,
    seed: int,
    dropped_columns: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> DatasetVariant:
    return DatasetVariant(
        kind=kind,
        frame_sha256=_frame_sha256(frame),
        target_column=target_column,
        rows=len(frame),
        seed=seed,
        dropped_columns=list(dropped_columns),
        notes=list(notes),
    )


def build_variants(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    group_column: str | None = None,
    block_size: int | None = None,
    hash_ids: bool = True,
    hash_salt: str = "",
    blind: bool = False,
) -> tuple[VariantSet, dict[str, pd.DataFrame]]:
    """Build the original, anonymized, and jumbled frames.

    The input frame is never mutated.  The returned mapping always contains
    ``original``, ``anonymized``, and ``jumbled`` copies.
    """
    target = profile.target.name if profile.target is not None else None
    original_frame = frame.copy(deep=True)

    anonymized_frame, dropped, anonymized_notes = anonymize_frame(
        profile,
        original_frame,
        hash_ids=hash_ids,
        hash_salt=hash_salt,
        blind=blind,
        return_details=True,
    )
    anonymized_target = (
        target if target is not None and target in anonymized_frame.columns else None
    )

    jumbled_frame = original_frame.copy(deep=True)
    jumbled_notes: list[str] = []
    if target is not None and target in jumbled_frame.columns:
        jumbled = jumble_target(
            profile,
            original_frame,
            seed=seed,
            group_column=group_column,
            block_size=block_size,
            return_details=True,
        )
        jumbled_frame[target] = jumbled.series
        jumbled_notes.extend(jumbled.notes or [])
        jumbled_notes.append(f"method={jumbled.method}")
        jumbled_notes.append(f"seed={seed}")
    else:
        jumbled_notes.append("no target column; jumbled frame equals the original")

    variants = VariantSet(
        original=_variant(
            "original",
            original_frame,
            target_column=target,
            seed=seed,
        ),
        anonymized=_variant(
            "anonymized",
            anonymized_frame,
            target_column=anonymized_target,
            seed=seed,
            dropped_columns=dropped,
            notes=anonymized_notes,
        ),
        jumbled=_variant(
            "jumbled",
            jumbled_frame,
            target_column=target,
            seed=seed,
            notes=jumbled_notes,
        ),
    )
    frames = {
        "original": original_frame,
        "anonymized": anonymized_frame,
        "jumbled": jumbled_frame,
    }
    return variants, frames


# ---------------------------------------------------------------------------
# Variant evaluation
# ---------------------------------------------------------------------------


def _metric_direction(metric: str, higher_is_better: bool | None) -> bool:
    if higher_is_better is not None:
        return bool(higher_is_better)
    return metric not in _LOWER_IS_BETTER_METRICS


def _default_leak_threshold(metric: str, higher_is_better: bool) -> float:
    if higher_is_better:
        return _DEFAULT_HIGHER_LEAK_THRESHOLD.get(metric, 0.05)
    return 0.05


def _relative_degradation(
    original: float,
    comparison: float,
    *,
    higher_is_better: bool,
) -> float:
    denominator = max(abs(original), 1e-9)
    if higher_is_better:
        improvement = original - comparison
    else:
        improvement = comparison - original
    return improvement / denominator


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_variants(
    variants: VariantSet,
    *,
    metric: str = "r2",
    original: float | None = None,
    anonymized: float | None = None,
    jumbled: float | None = None,
    jumbled_baseline: float | None = None,
    permutation_p: float | None = None,
    pii_columns_used: Sequence[str] = (),
    higher_is_better: bool | None = None,
    leak_threshold: float | None = None,
    permutation_alpha: float = 0.05,
    performance_drop_threshold: float = 0.5,
    original_score: float | None = None,
    anonymized_score: float | None = None,
    jumbled_score: float | None = None,
    pii_columns: Sequence[str] | None = None,
) -> VariantReport:
    """Evaluate original/anonymized/jumbled scores and set the leak gates.

    The jumbled gate rejects a model when its jumbled score clears the
    metric's chance threshold and the permutation p-value is below
    ``permutation_alpha``.  The PII gate rejects a model that uses a dropped
    column, or whose performance collapses after anonymization in a way
    attributable to PII columns.

    ``original_score``/``anonymized_score``/``jumbled_score`` and
    ``pii_columns`` are accepted as explicit aliases.
    """
    if original_score is not None:
        original = original_score
    if anonymized_score is not None:
        anonymized = anonymized_score
    if jumbled_score is not None:
        jumbled = jumbled_score
    if pii_columns is not None:
        pii_columns_used = pii_columns
    higher = _metric_direction(metric, higher_is_better)
    threshold = (
        float(leak_threshold)
        if leak_threshold is not None
        else _default_leak_threshold(metric, higher)
    )

    original_score = _finite_or_none(original)
    anonymized_score = _finite_or_none(anonymized)
    jumbled_score = _finite_or_none(jumbled)
    baseline_score = _finite_or_none(jumbled_baseline)
    p_value = _finite_or_none(permutation_p)

    scores = [
        VariantScore(kind="original", metric=metric, score=original_score),
        VariantScore(kind="anonymized", metric=metric, score=anonymized_score),
        VariantScore(
            kind="jumbled",
            metric=metric,
            score=jumbled_score,
            baseline=baseline_score,
            permutation_p=p_value,
        ),
    ]

    reasons: list[str] = []
    notes: list[str] = []

    if higher:
        crosses = jumbled_score is not None and jumbled_score >= threshold
    else:
        crosses = (
            jumbled_score is not None
            and baseline_score is not None
            and jumbled_score <= baseline_score * (1.0 - threshold)
        )
        if jumbled_score is not None and baseline_score is None:
            notes.append(
                "jumbled baseline is required for lower-is-better metrics; "
                "leak gate cannot fire"
            )

    if crosses and (p_value is None or p_value < permutation_alpha):
        jumbled_label_leak = True
        reasons.append(
            f"jumbled_label_leak: {metric}={jumbled_score!r} crosses threshold "
            f"{threshold!r} (permutation p={p_value!r})"
        )
        if p_value is None:
            notes.append("permutation p-value was not supplied")
    else:
        jumbled_label_leak = False

    dropped = set(variants.anonymized.dropped_columns)
    used_dropped = sorted(dropped.intersection(str(column) for column in pii_columns_used))
    pii_reliance = bool(used_dropped)
    if used_dropped:
        reasons.append(
            "pii_reliance: model uses dropped anonymized column(s): "
            + ", ".join(used_dropped)
        )

    if (
        not pii_reliance
        and original_score is not None
        and anonymized_score is not None
        and dropped
    ):
        degradation = _relative_degradation(
            original_score, anonymized_score, higher_is_better=higher
        )
        if degradation >= performance_drop_threshold:
            if pii_columns_used:
                pii_reliance = True
                reasons.append(
                    "pii_reliance: performance dropped "
                    f"{degradation:.0%} after anonymization and the model used "
                    "PII-adjacent columns"
                )
            else:
                notes.append(
                    "performance dropped "
                    f"{degradation:.0%} after anonymization, but no dropped column "
                    "was used; not attributed to PII"
                )

    if jumbled_label_leak or pii_reliance:
        verdict: Literal["pass", "reject", "insufficient"] = "reject"
    elif original_score is None:
        verdict = "insufficient"
        if not reasons:
            reasons.append("original score is missing")
    elif jumbled_score is None:
        verdict = "insufficient"
        if not reasons:
            reasons.append("jumbled score is missing")
    else:
        verdict = "pass"

    return VariantReport(
        variants=variants,
        seed=variants.original.seed,
        target_column=variants.original.target_column,
        metric=metric,
        higher_is_better=higher,
        scores=scores,
        jumbled_label_leak=jumbled_label_leak,
        pii_reliance=pii_reliance,
        verdict=verdict,
        reasons=reasons,
        notes=notes,
    )


def write_variant_report(report: VariantReport, path) -> None:
    """Write ``variants.json`` deterministically."""
    report.write(path)


# ---------------------------------------------------------------------------
# Running one generated hypothesis against the three variants
# ---------------------------------------------------------------------------


def _metric_for_target(target_kind: str | None) -> tuple[str, bool]:
    kind = (target_kind or "continuous").strip().lower()
    if kind in {"binary", "multiclass"}:
        return "accuracy", True
    if kind == "count":
        return "poisson_deviance", False
    return "r2", True


def _score_predictions(
    frame: pd.DataFrame,
    predictions_path: Path,
    target: str | None,
    metric: str,
) -> float | None:
    """Align a saved predictions CSV with a variant frame and score it."""
    if target is None or target not in frame.columns:
        return None
    if not predictions_path.is_file():
        return None
    try:
        predictions = pd.read_csv(predictions_path)
    except Exception:  # noqa: BLE001 - a broken predictions file is a missing score
        return None
    if not {"row_index", "prediction"}.issubset(predictions.columns):
        return None

    try:
        aligned = predictions.set_index("row_index")["prediction"]
        actual = frame[target].reindex(aligned.index)
    except Exception:  # noqa: BLE001 - misaligned rows are a missing score
        return None
    paired = pd.DataFrame({"actual": actual, "prediction": aligned}).dropna()
    if paired.empty:
        return None

    y_true = paired["actual"]
    y_pred = paired["prediction"]
    if metric == "accuracy":
        if pd.api.types.is_numeric_dtype(y_true) and pd.api.types.is_numeric_dtype(
            y_pred
        ):
            unique = set(pd.unique(y_true).tolist())
            if unique.issubset({0, 1, 0.0, 1.0}):
                y_pred = (pd.to_numeric(y_pred, errors="coerce") >= 0.5).astype(
                    float
                )
        return float((y_true.astype(str) == y_pred.astype(str)).mean())

    true_values = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    true_values = true_values[finite]
    predicted_values = predicted_values[finite]
    if true_values.size == 0:
        return None

    if metric == "poisson_deviance":
        observed = np.clip(true_values, 0.0, None)
        expected = np.clip(predicted_values, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                observed > 0.0,
                observed * np.log(observed / expected) - (observed - expected),
                0.0 - (observed - expected),
            )
        value = float(2.0 * np.mean(terms))
        return value if np.isfinite(value) else None

    residual = float(np.sum((true_values - predicted_values) ** 2))
    total = float(np.sum((true_values - np.mean(true_values)) ** 2))
    if total <= 0.0:
        return None
    value = 1.0 - residual / total
    return value if np.isfinite(value) else None


def _baseline_score(
    frame: pd.DataFrame,
    target: str | None,
    metric: str,
) -> float | None:
    if target is None or target not in frame.columns:
        return None
    values = frame[target]
    if values.empty:
        return None
    if metric == "accuracy":
        mode = values.mode(dropna=True)
        if mode.empty:
            return None
        return float((values.astype(str) == str(mode.iloc[0])).mean())
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    mean = float(numeric.mean())
    if metric == "poisson_deviance":
        observed = np.clip(numeric.to_numpy(dtype=float), 0.0, None)
        expected = np.clip(np.full_like(observed, mean), 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                observed > 0.0,
                observed * np.log(observed / expected) - (observed - expected),
                0.0 - (observed - expected),
            )
        value = float(2.0 * np.mean(terms))
        return value if np.isfinite(value) else None
    # A constant prediction has zero explained variance by definition.
    return 0.0


def evaluate_hypothesis_variants(
    *,
    hypothesis_id: str,
    module_path: str | Path,
    frames: Mapping[str, pd.DataFrame],
    variants: VariantSet,
    target_column: str | None,
    target_kind: str | None,
    run_dir: str | Path,
    required_columns: Sequence[str] = (),
    optional_columns: Sequence[str] = (),
    seed: int = 0,
    timeout: float = 60.0,
) -> VariantReport:
    """Run one generated module on original/anonymized/jumbled frames.

    Each run happens in its own child-process run directory so the original
    predictions are never overwritten. A missing variant or a failed execution
    leaves that score absent; :func:`evaluate_variants` then reports
    ``insufficient`` rather than inventing a pass.
    """
    from .execute import run_module

    run_dir = Path(run_dir)
    source_module = Path(module_path)
    if not source_module.is_absolute():
        source_module = run_dir / source_module
    metric, higher_is_better = _metric_for_target(target_kind)
    warnings: list[str] = []
    scores: dict[str, float | None] = {}
    baselines: dict[str, float | None] = {}

    for kind in ("original", "anonymized", "jumbled"):
        frame = frames.get(kind)
        if frame is None:
            warnings.append(f"variant_missing:{kind}")
            scores[kind] = None
            baselines[kind] = None
            continue

        variant_dir = run_dir / "variants" / kind
        hypotheses_dir = variant_dir / "hypotheses"
        hypotheses_dir.mkdir(parents=True, exist_ok=True)
        copied = hypotheses_dir / source_module.name
        copied.write_bytes(source_module.read_bytes())
        dataset_path = variant_dir / "dataset.csv"
        # The runner reads a plain CSV and reports a positional row index, so
        # reset the index and never inject an extra feature column.
        variant_frame = frame.reset_index(drop=True)
        variant_frame.to_csv(dataset_path, index=False)
        frame = variant_frame

        try:
            execution = run_module(
                copied,
                dataset_path,
                seed,
                run_dir=variant_dir,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - one variant must not abort the gate
            warnings.append(f"variant_execution_failed:{kind}:{type(exc).__name__}")
            scores[kind] = None
            baselines[kind] = _baseline_score(frame, target_column, metric)
            continue

        if execution.status != "ok" or not execution.predictions_path:
            warnings.append(
                f"variant_execution_{execution.status}:{kind}"
                + (f":{execution.error}" if execution.error else "")
            )
            scores[kind] = None
        else:
            scores[kind] = _score_predictions(
                frame,
                variant_dir / execution.predictions_path,
                target_column,
                metric,
            )
            if scores[kind] is None:
                warnings.append(f"variant_score_missing:{kind}")
        baselines[kind] = _baseline_score(frame, target_column, metric)

    pii_columns_used = [
        column
        for column in dict.fromkeys([*required_columns, *optional_columns])
        if column
    ]
    report = evaluate_variants(
        variants,
        metric=metric,
        original=scores.get("original"),
        anonymized=scores.get("anonymized"),
        jumbled=scores.get("jumbled"),
        jumbled_baseline=baselines.get("jumbled"),
        pii_columns_used=pii_columns_used,
        higher_is_better=higher_is_better,
    )
    report.hypothesis_id = hypothesis_id
    report.warnings = list(dict.fromkeys([*report.warnings, *warnings]))
    return report


def aggregate_variant_reports(
    variants: VariantSet,
    reports: Sequence[VariantReport],
) -> VariantReport:
    """Merge per-hypothesis variant reports into the run-level artifact report."""
    scores: list[VariantScore] = []
    reasons: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    any_leak = False
    any_pii = False
    all_pass = bool(reports)

    for report in reports:
        label = report.hypothesis_id or "(unknown)"
        any_leak = any_leak or report.jumbled_label_leak
        any_pii = any_pii or report.pii_reliance
        all_pass = all_pass and report.verdict == "pass"
        for score in report.scores:
            scores.append(
                score.model_copy(
                    update={
                        "notes": [
                            *score.notes,
                            f"hypothesis={label}",
                        ]
                    }
                )
            )
        for reason in report.reasons:
            reasons.append(f"{label}: {reason}")
            warnings.append(f"{label}: {reason}")
        for warning in report.warnings:
            warnings.append(f"{label}: {warning}")
        notes.extend(f"{label}: {note}" for note in report.notes)

    if any_leak or any_pii:
        verdict: Literal["pass", "reject", "insufficient"] = "reject"
    elif all_pass:
        verdict = "pass"
    else:
        verdict = "insufficient"

    first = reports[0] if reports else None
    return VariantReport(
        variants=variants,
        seed=variants.original.seed,
        target_column=first.target_column if first is not None else None,
        metric=first.metric if first is not None else None,
        higher_is_better=(
            first.higher_is_better if first is not None else True
        ),
        scores=scores,
        jumbled_label_leak=any_leak,
        pii_reliance=any_pii,
        verdict=verdict,
        reasons=list(dict.fromkeys(reasons)),
        notes=list(dict.fromkeys(notes)),
        warnings=list(dict.fromkeys(warnings)),
    )
