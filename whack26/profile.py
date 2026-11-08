"""Dataset profiling, target inference, and compact dataset summaries."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .io import LoadedDataset

_SUMMARY_LIMIT = 6000
_SUMMARY_TRUNCATION = "\n... (truncated)"
_MAX_SAMPLE_COLUMNS = 12
_MAX_SAMPLE_ROWS = 3
_SAMPLE_VALUE_LIMIT = 80
_MIN_TARGET_SCORE = 0.5

_TARGET_STRONG_TOKENS = {"target", "label", "outcome", "y"}
_TARGET_MEDIUM_TOKENS = {"price", "value", "class", "result"}

_ID_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:id|ids|identifier|index|key|keys|uuid|guid)(?:$|[^a-z0-9])"
)
_TIME_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:date|datetime|time|timestamp|year|month)(?:$|[^a-z0-9])"
)
_LEAKAGE_NAME_RE = re.compile(r"outcome|result|target|future|post|after")
_UUID_VALUE_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_EMAIL_VALUE_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_DATE_VALUE_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")

# Roles that can never be a prediction target. ``_column_role`` applies these
# before ``target`` so a column keeps exactly one role, and target selection
# rejects any column whose target-free role is in this set.
_REJECTED_TARGET_ROLES = frozenset({"id", "time", "leakage_risk"})

_SENSITIVE_SAMPLE_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:"
    r"id|ids|uuid|guid|key|keys|index|"
    r"email|emails|e[-_ ]?mail|mail|"
    r"name|names|firstname|lastname|surname|forename|fullname|username|"
    r"address|addr|street|postal|postcode|zip|zipcode|"
    r"phone|telephone|tel|mobile|fax|contact|"
    r"ticket|tickets|booking|bookings|reservation|reservations|pnr|seat|seats|cabin|cabins|"
    r"account|accounts|iban|swift|ssn|passport|"
    r"url|uri|link|website|href|"
    r"desc|description|text|comment|comments|note|notes|message|messages|"
    r"summary|review|reviews|bio|content|body|abstract"
    r")(?:$|[^a-z0-9])"
)

# Compound identifier names that ``_name_tokens`` keeps as one token (for
# example ``userid``). Matching is exact so ordinary words that merely end in
# "id" (``grid``, ``valid``, ``solid``) stay non-sensitive.
_SENSITIVE_COMPOUND_ID_TOKENS = frozenset(
    {
        "userid",
        "userids",
        "accountid",
        "accountids",
        "customerid",
        "customerids",
        "patientid",
        "memberid",
        "employeeid",
        "studentid",
        "staffid",
        "orderid",
        "transactionid",
        "recordid",
        "rowid",
        "sessionid",
        "deviceid",
        "clientid",
        "invoiceid",
        "ticketid",
        "bookingid",
        "reservationid",
        "policyid",
        "claimid",
        "productid",
        "itemid",
        "listingid",
        "propertyid",
    }
)

_TARGET_SYSTEM_PROMPT = (
    "You choose the prediction target for a tabular dataset. Pick one real column "
    "that a model could predict from the other columns. Do not pick an identifier, "
    "timestamp, or leakage column. Return only the requested JSON object."
)
_TARGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "nullable": True,
            "description": "A real column name, or null when no target is plausible.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the selected target.",
        },
        "rationale": {
            "type": "string",
            "description": "One short reason for the choice.",
        },
    },
    "required": ["target", "confidence", "rationale"],
}


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: Literal["feature", "target", "id", "time", "leakage_risk", "unknown"]
    missing_pct: float
    cardinality: int
    sample_values: list[str]
    numeric: dict | None
    top_values: list[tuple[str, int]] | None
    datetime: bool


class TargetChoice(BaseModel):
    name: str
    source: Literal["user", "inferred"]
    confidence: float


class DatasetProfile(BaseModel):
    dataset_id: str
    source_path: str
    sha256: str
    format: str
    rows: int
    columns: list[ColumnProfile]
    description: str | None
    target: TargetChoice | None
    time_column: str | None
    warnings: list[str]


@dataclass
class _TargetCandidate:
    name: str
    score: float
    position: int = 0


def _display_name(source_path: str) -> str:
    parts = [part for part in re.split(r"[\\/]", source_path) if part]
    return parts[-1] if parts else "dataset"


def _clean_text(value: Any, *, limit: int = _SAMPLE_VALUE_LIMIT) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def _name_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]


def _is_datetime_series(series: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(series.dtype))


def _is_numeric_series(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype)
    )


def _to_number(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _missing_pct(series: pd.Series, rows: int) -> float:
    if rows <= 0:
        return 0.0
    return round(float(series.isna().sum()) / rows * 100.0, 2)


def _sample_values(series: pd.Series) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for value in series.dropna():
        text = _clean_text(value)
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
        if len(values) >= 5:
            break
    return values


def _top_values(series: pd.Series, cardinality: int) -> list[tuple[str, int]] | None:
    if cardinality > 50:
        return None
    counts = series.value_counts(dropna=True)
    return [
        (_clean_text(value), int(count)) for value, count in counts.head(5).items()
    ]


def _numeric_stats(series: pd.Series) -> dict[str, int | float | None]:
    values = series.dropna()
    if values.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "zeros": 0,
            "negatives": 0,
        }
    return {
        "min": _to_number(values.min()),
        "max": _to_number(values.max()),
        "mean": _to_number(values.mean()),
        "std": _to_number(values.std()),
        "zeros": int((values == 0).sum()),
        "negatives": int((values < 0).sum()),
    }


def _looks_like_id(
    name: str, series: pd.Series, cardinality: int, rows: int
) -> bool:
    """Return True only for a strong identifier signal.

    A near-unique column is not enough on its own: an integer or string dtype
    would otherwise make every high-cardinality measure (price, count, free
    text) look like an identifier. We require a name hit or one of these
    structural signals:

    - a UUID or email value pattern;
    - an exact, gapless integer sequence (``1, 2, 3, ...``);
    - exact uniqueness with no repeats for short, space-free string codes.
    """
    if rows <= 0 or cardinality <= 0:
        return False
    if _ID_NAME_RE.search(name.lower()) is not None:
        return True
    if cardinality / rows < 0.98:
        return False

    values = series.dropna()
    if values.empty:
        return False

    if _is_numeric_series(series):
        if not pd.api.types.is_integer_dtype(series.dtype):
            # Continuous measures are frequently unique; that is not an id.
            return False
        if len(values) != cardinality or cardinality != rows:
            return False
        differences = values.diff().dropna()
        if differences.empty:
            return False
        return bool(
            (differences == 1).all() or (differences == -1).all()
        )

    if _is_datetime_series(series):
        return False

    sample = values.head(20).astype(str)
    if sample.map(lambda value: _UUID_VALUE_RE.fullmatch(value) is not None).all():
        return True
    if sample.map(lambda value: _EMAIL_VALUE_RE.fullmatch(value) is not None).all():
        return True
    if cardinality == len(values) == rows:
        # Date-like strings are time values, not identifiers, even when a CSV
        # reader left them as text.
        if sample.str.contains(_DATE_VALUE_RE, regex=True).any():
            return False
        lengths = sample.map(len)
        has_space = sample.str.contains(r"\s", regex=True).any()
        if int(lengths.max()) <= 40 and not bool(has_space):
            return True
    return False


def _looks_like_time(name: str, series: pd.Series) -> bool:
    return _is_datetime_series(series) or _TIME_NAME_RE.search(name.lower()) is not None


def _column_role(
    name: str,
    series: pd.Series,
    *,
    cardinality: int,
    rows: int,
    target: str | None,
) -> Literal["feature", "target", "id", "time", "leakage_risk", "unknown"]:
    """Assign exactly one role, checking non-target roles first.

    Structural roles (id/time) and leakage take precedence over ``target`` so
    an ``id_target``/``date_target`` column cannot silently become the target.
    Target selection rejects columns whose role is in
    :data:`_REJECTED_TARGET_ROLES`, so a ``target`` role is only ever assigned
    to a plausible feature column.
    """
    if _looks_like_time(name, series):
        return "time"
    if _looks_like_id(name, series, cardinality, rows):
        return "id"
    if _LEAKAGE_NAME_RE.search(name.lower()) is not None:
        return "leakage_risk"
    if target is not None and name == target:
        return "target"
    return "feature"


def _target_free_role(
    name: str, series: pd.Series, *, rows: int
) -> Literal["feature", "target", "id", "time", "leakage_risk", "unknown"]:
    """Role of a column ignoring any target designation."""
    return _column_role(
        name,
        series,
        cardinality=int(series.nunique(dropna=True)),
        rows=rows,
        target=None,
    )


def _profile_column(
    name: str, series: pd.Series, *, rows: int, target: str | None
) -> ColumnProfile:
    cardinality = int(series.nunique(dropna=True))
    numeric = _numeric_stats(series) if _is_numeric_series(series) else None
    datetime = _is_datetime_series(series)
    role = _column_role(
        name, series, cardinality=cardinality, rows=rows, target=target
    )
    sensitive = _name_looks_sensitive(name)
    top_values = (
        None
        if numeric is not None or sensitive
        else _top_values(series, cardinality)
    )
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        missing_pct=_missing_pct(series, rows),
        cardinality=cardinality,
        sample_values=[] if sensitive else _sample_values(series),
        numeric=numeric,
        top_values=top_values,
        datetime=datetime,
    )


def _target_score(name: str, series: pd.Series, rows: int) -> float:
    tokens = set(_name_tokens(name))
    name_is_target_like = bool(
        tokens & (_TARGET_STRONG_TOKENS | _TARGET_MEDIUM_TOKENS)
    )
    score = 0.0
    if tokens & _TARGET_STRONG_TOKENS:
        score += 0.55
    if tokens & _TARGET_MEDIUM_TOKENS:
        score += 0.40

    if pd.api.types.is_bool_dtype(series.dtype):
        score += 0.15
    elif _is_numeric_series(series):
        score += 0.25
    elif _is_datetime_series(series):
        score -= 0.35
    else:
        score += 0.10

    cardinality = int(series.nunique(dropna=True))
    if cardinality <= 1:
        score -= 1.0
    elif rows > 5:
        ratio = cardinality / rows
        if name_is_target_like:
            # A named target can legitimately be continuous and near-unique.
            if cardinality <= 20:
                score += 0.05
        elif ratio >= 0.98:
            score -= 0.55
        elif ratio >= 0.9:
            score -= 0.25
        elif cardinality <= 20:
            score += 0.05
    elif cardinality <= 20:
        score += 0.05

    if rows > 0 and float(series.isna().sum()) / rows > 0.5:
        score -= 0.10
    return score


def _rank_target_candidates(frame: pd.DataFrame) -> list[_TargetCandidate]:
    rows = len(frame)
    scored: list[tuple[int, _TargetCandidate]] = []
    for position, column in enumerate(frame.columns):
        name = str(column)
        series = frame.iloc[:, position]
        scored.append(
            (
                position,
                _TargetCandidate(
                    name=name,
                    score=_target_score(name, series, rows),
                    position=position,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[1].score, item[0]))
    return [candidate for _, candidate in scored]


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(number):
        return 0.5
    return round(min(0.99, max(0.0, number)), 2)


def _llm_target_choice(
    frame: pd.DataFrame,
    candidates: list[_TargetCandidate],
    *,
    display_name: str,
    description: str | None,
    llm: Any,
    warnings: list[str],
) -> TargetChoice | None:
    lines = [
        f"Dataset: {display_name}",
        f"Rows: {len(frame)}; Columns: {len(frame.columns)}",
    ]
    if description:
        lines.append(f"Description: {_clean_text(description, limit=500)}")
    lines.append("Top target candidates (name, dtype, unique values, missing %):")
    for candidate in candidates[:5]:
        series = frame.iloc[:, candidate.position]
        cardinality = int(series.nunique(dropna=True))
        missing = round(float(series.isna().sum()) / max(len(frame), 1) * 100.0, 2)
        lines.append(
            f"- {candidate.name} ({series.dtype}, {cardinality} unique, {missing}% missing)"
        )
    lines.append(
        "Choose the column a model should predict, or null when no column is plausible."
    )

    try:
        response = llm.generate_json(
            system=_TARGET_SYSTEM_PROMPT,
            user="\n".join(lines),
            schema=_TARGET_SCHEMA,
            purpose="target",
        )
    except Exception as exc:
        warnings.append(
            "target selection LLM failed "
            f"({type(exc).__name__}: {exc}); falling back to the heuristic"
        )
        return None

    if not isinstance(response, dict):
        warnings.append(
            "target selection LLM returned a non-object response; "
            "falling back to the heuristic"
        )
        return None

    selected = response.get("target")
    positions = {
        str(column): position for position, column in enumerate(frame.columns)
    }
    if isinstance(selected, str) and selected in positions:
        series = frame.iloc[:, positions[selected]]
        role = _target_free_role(selected, series, rows=len(frame))
        if role in _REJECTED_TARGET_ROLES:
            warnings.append(
                f"target selection LLM chose {selected!r}, which is a {role} "
                "column; rejecting it and falling back to the heuristic"
            )
            return None
        return TargetChoice(
            name=selected,
            source="inferred",
            confidence=_clamp_confidence(response.get("confidence")),
        )

    warnings.append(
        f"target selection LLM chose {selected!r}, which is not a column; "
        "falling back to the heuristic"
    )
    return None


def _select_target(
    frame: pd.DataFrame,
    *,
    explicit_target: str | None,
    display_name: str,
    description: str | None,
    llm: Any,
    warnings: list[str],
) -> TargetChoice | None:
    if explicit_target is not None:
        return TargetChoice(name=explicit_target, source="user", confidence=1.0)

    candidates = _rank_target_candidates(frame)
    if llm is not None:
        choice = _llm_target_choice(
            frame,
            candidates,
            display_name=display_name,
            description=description,
            llm=llm,
            warnings=warnings,
        )
        if choice is not None:
            return choice

    eligible = [
        candidate
        for candidate in candidates
        if _target_free_role(
            candidate.name,
            frame.iloc[:, candidate.position],
            rows=len(frame),
        )
        not in _REJECTED_TARGET_ROLES
    ]
    if eligible and eligible[0].score >= _MIN_TARGET_SCORE:
        best = eligible[0]
        return TargetChoice(
            name=best.name,
            source="inferred",
            confidence=round(min(0.95, max(0.05, best.score)), 2),
        )

    warnings.append(
        "no plausible target column was found; continuing without a target"
    )
    return None


def _hash_dataset(
    loaded: LoadedDataset, source_path: Path, warnings: list[str]
) -> str:
    if source_path.is_file():
        try:
            return hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            warnings.append(
                f"could not hash {source_path.name!r} from disk "
                f"({type(exc).__name__}: {exc}); hashing a CSV representation instead"
            )
    csv_bytes = loaded.frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def build_profile(
    loaded: LoadedDataset,
    *,
    description: str | None = None,
    target: str | None = None,
    llm: Any = None,
) -> DatasetProfile:
    """Build a typed profile, inferring a target when one is not supplied."""
    frame = loaded.frame
    positions = {
        str(column): position for position, column in enumerate(frame.columns)
    }
    if target is not None and target not in positions:
        available = ", ".join(str(name) for name in frame.columns)
        raise ValueError(
            f"target column {target!r} is not in the dataset; "
            f"available columns: {available}"
        )
    if target is not None:
        series = frame.iloc[:, positions[target]]
        role = _target_free_role(target, series, rows=len(frame))
        if role in _REJECTED_TARGET_ROLES:
            raise ValueError(
                f"target column {target!r} is a {role} column; "
                "choose a feature column instead"
            )

    warnings = list(loaded.warnings)
    source_path = Path(loaded.source_path)
    display_name = _display_name(str(loaded.source_path))
    sha256 = _hash_dataset(loaded, source_path, warnings)

    target_choice = _select_target(
        frame,
        explicit_target=target,
        display_name=display_name,
        description=description,
        llm=llm,
        warnings=warnings,
    )
    target_name = target_choice.name if target_choice is not None else None

    columns = [
        _profile_column(str(column), frame.iloc[:, position], rows=len(frame), target=target_name)
        for position, column in enumerate(frame.columns)
    ]
    time_column = next(
        (column.name for column in columns if column.role == "time"), None
    )

    return DatasetProfile(
        dataset_id="ds_" + sha256[:12],
        source_path=str(loaded.source_path),
        sha256=sha256,
        format=loaded.format,
        rows=len(frame),
        columns=columns,
        description=description,
        target=target_choice,
        time_column=time_column,
        warnings=warnings,
    )


def _format_pct(value: float) -> str:
    return str(round(float(value), 2))


def _format_stat(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.4g}"
    return str(value)


def _first_clean_values(series: pd.Series, limit: int = 20) -> list[str]:
    values: list[str] = []
    for value in series.dropna():
        values.append(_clean_text(value))
        if len(values) >= limit:
            break
    return values


def _name_looks_sensitive(name: str) -> bool:
    lower = name.lower()
    if _SENSITIVE_SAMPLE_NAME_RE.search(lower) is not None:
        return True
    tokens = _name_tokens(lower)
    for token in tokens:
        if token.endswith("name") or token.endswith("email") or token.endswith("mail"):
            return True
        if token in _SENSITIVE_COMPOUND_ID_TOKENS:
            return True
    return False


def _looks_like_free_text(series: pd.Series, profile: ColumnProfile | None) -> bool:
    if profile is not None and (profile.numeric is not None or profile.datetime):
        return False
    values = _first_clean_values(series)
    if not values:
        return False
    average_length = sum(len(value) for value in values) / len(values)
    maximum_length = max(len(value) for value in values)
    cardinality = (
        profile.cardinality if profile is not None else int(series.nunique(dropna=True))
    )
    ratio = cardinality / len(series) if len(series) else 0.0
    return average_length > 60 or (maximum_length >= 80 and ratio >= 0.5)


def _omit_from_samples(
    name: str, series: pd.Series, profile: ColumnProfile | None
) -> bool:
    if _name_looks_sensitive(name):
        return True
    return _looks_like_free_text(series, profile)


def _sample_positions(profile: DatasetProfile, frame: pd.DataFrame) -> list[int]:
    by_name = {column.name: column for column in profile.columns}
    selected: list[int] = []
    for position, column in enumerate(frame.columns):
        name = str(column)
        if _omit_from_samples(name, frame.iloc[:, position], by_name.get(name)):
            continue
        selected.append(position)
        if len(selected) >= _MAX_SAMPLE_COLUMNS:
            break
    if len(selected) < 2:
        fallback: list[int] = []
        for position, column in enumerate(frame.columns):
            name = str(column)
            if _omit_from_samples(name, frame.iloc[:, position], by_name.get(name)):
                continue
            fallback.append(position)
            if len(fallback) >= 3:
                break
        return fallback
    return selected


def _sample_cell(value: Any) -> str:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
    except (TypeError, ValueError):
        pass
    return _clean_text(value).replace("|", "/")


def _sample_table(frame: pd.DataFrame, positions: list[int]) -> list[str]:
    if not positions:
        return ["(no columns)"]
    lines = [
        " | ".join(
            _clean_text(frame.columns[position]).replace("|", "/")
            for position in positions
        )
    ]
    for row_position in range(min(_MAX_SAMPLE_ROWS, frame.shape[0])):
        lines.append(
            " | ".join(
                _sample_cell(frame.iloc[row_position, column_position])
                for column_position in positions
            )
        )
    return lines


def build_summary(profile: DatasetProfile, frame: pd.DataFrame) -> str:
    """Render the compact text summary used by the research prompts."""
    lines = [f"Dataset: {_display_name(profile.source_path)}"]
    if profile.description:
        lines.append(f"Description: {_clean_text(profile.description, limit=500)}")
    lines.append(f"Rows: {profile.rows}; Columns: {len(profile.columns)}")

    if profile.target is None:
        lines.append("Target: none")
    else:
        target_column = next(
            (column for column in profile.columns if column.name == profile.target.name),
            None,
        )
        if target_column is None:
            lines.append(f"Target: {profile.target.name}")
        else:
            lines.append(
                f"Target: {target_column.name} ({target_column.dtype}, "
                f"{target_column.cardinality} unique, "
                f"{_format_pct(target_column.missing_pct)}%)"
            )
    lines.append(f"Time column: {profile.time_column or 'none'}")

    lines.append("Columns:")
    for column in profile.columns:
        details = [column.role]
        if column.numeric is not None:
            stats = column.numeric
            details.append(
                "numeric min={min}, max={max}, mean={mean}, std={std}, "
                "zeros={zeros}, negatives={negatives}".format(
                    min=_format_stat(stats.get("min")),
                    max=_format_stat(stats.get("max")),
                    mean=_format_stat(stats.get("mean")),
                    std=_format_stat(stats.get("std")),
                    zeros=_format_stat(stats.get("zeros")),
                    negatives=_format_stat(stats.get("negatives")),
                )
            )
        if column.top_values and not _name_looks_sensitive(column.name):
            top = ", ".join(
                f"{value} ({count})" for value, count in column.top_values
            )
            details.append(f"top: {top}")
        lines.append(
            f"- {column.name} ({column.dtype}, "
            f"missing {_format_pct(column.missing_pct)}%, "
            f"{column.cardinality} unique): " + "; ".join(details)
        )

    lines.append("Sample rows (first 3, values truncated):")
    lines.extend(_sample_table(frame, _sample_positions(profile, frame)))

    summary = "\n".join(lines)
    if len(summary) > _SUMMARY_LIMIT:
        summary = (
            summary[: _SUMMARY_LIMIT - len(_SUMMARY_TRUNCATION)]
            + _SUMMARY_TRUNCATION
        )
    return summary
