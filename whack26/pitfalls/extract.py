"""Turn field-scoped research records into cited pitfall evidence.

The extractor is intentionally conservative.  A record is kept only when its
title or abstract contains a recognised pitfall signal.  The resulting record
does not claim the dataset is affected; :func:`applicability` records whether
the profile shows the precondition, and the critic is told to treat ``unknown``
as background rather than as a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..common import redact_secrets
from ..profile import DatasetProfile
from ..research.base import ResearchRecord
from .queries import field_id, field_level, field_name
from .types import FieldPitfall, PitfallCategory

_CATEGORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "censoring": re.compile(
        r"\bcensor(?:ed|ing)?\b|\btruncat(?:ed|ion)\b|top[- ]?cod(?:ed|ing)?|"
        r"\bceiling effect\b|\bfloor effect\b|\bthreshold effect\b|"
        r"\bboundary (?:value|observation)s?\b",
        re.IGNORECASE,
    ),
    "selection": re.compile(
        r"\bselection bias\b|\bsurvivorship bias\b|\bself[- ]?selection\b|"
        r"\bsample selection\b|\bnon[- ]?response bias\b|\bsampling bias\b|"
        r"\bselected sample\b|\btruncation by selection\b",
        re.IGNORECASE,
    ),
    "aggregation": re.compile(
        r"\baggregat(?:ed|ion|ing)\b|\becological fallacy\b|"
        r"\bmodifiable areal unit\b|\bunit of analysis\b|\bgroup[- ]level\b|"
        r"\bblock[- ]group\b|\bmedian of medians\b|\bdistrict[- ]level\b",
        re.IGNORECASE,
    ),
    "measurement_error": re.compile(
        r"\bmeasurement error\b|\berrors?[- ]in[- ]variables\b|"
        r"\battenuat(?:ed|ion)\b|\bregression dilution\b|\bmismeasur(?:ed|ement)\b|"
        r"\bproxy (?:variable|measure)\b|\bnoisy (?:measure|proxy)\b",
        re.IGNORECASE,
    ),
    "missing_data": re.compile(
        r"\bmissing (?:data|values?|ness)\b|\bimput(?:e|ed|ation)\b|"
        r"\bcomplete[- ]case\b|\bnon[- ]?ignorable missingness\b|"
        r"\bdrop(?:ped|ping)? (?:rows|observations)\b",
        re.IGNORECASE,
    ),
    "spatial_dependence": re.compile(
        r"\bspatial (?:autocorrelation|dependence|correlation|spillover|error)\b|"
        r"\bgeograph(?:ic|ical) (?:dependence|clustering|autocorrelation)\b|"
        r"\bnearby (?:properties|observations|units)\b|\bneighbou?ring (?:areas|units)\b|"
        r"\bspatial cross[- ]validation\b",
        re.IGNORECASE,
    ),
    "temporal_dependence": re.compile(
        r"\btime[- ]series\b|\bserial correlation\b|\bautocorrelat(?:ed|ion)\b|"
        r"\bnon[- ]?stationar(?:y|ity)\b|\btemporal dependence\b|"
        r"\blook[- ]?ahead bias\b|\bfuture information\b",
        re.IGNORECASE,
    ),
    "confounding": re.compile(
        r"\bconfound(?:ed|ing|er)?\b|\bomitted variable\b|\bendogen(?:eity|ous)\b|"
        r"\bunobserved heterogeneity\b|\breverse caus(?:ation|ality)\b|"
        r"\bsimultaneity\b|\bomitted[- ]variable bias\b",
        re.IGNORECASE,
    ),
    "multiplicity": re.compile(
        r"\bmultiple (?:comparisons?|testing)\b|\bp[- ]?hack(?:ing)?\b|"
        r"\bdata dredg(?:ing)?\b|\bwinner'?s curse\b|"
        r"\bselection of the best\b|\bfalse discovery\b",
        re.IGNORECASE,
    ),
    "rare_group": re.compile(
        r"\brare (?:group|category|event)s?\b|\bsmall (?:group|sample)s?\b|"
        r"\bsparse (?:group|category|data)\b|\bfew observations\b",
        re.IGNORECASE,
    ),
}

_FALLBACK_PATTERN = re.compile(
    r"\blimitation(?:s)?\b|\bcaveat(?:s)?\b|\bpitfall(?:s)?\b|"
    r"\bthreat(?:s)? to validity\b|\bcaution\b|\bconcern(?:s)?\b",
    re.IGNORECASE,
)

_CHECK_TEXT: dict[str, str] = {
    "censoring": (
        "Check the target's minimum and maximum and the share of rows at a "
        "boundary; compare OLS with a censored model or a boundary indicator."
    ),
    "selection": (
        "Compare the sample frame with the population of interest, report who "
        "is missing, and run a sensitivity check on the selection rule."
    ),
    "aggregation": (
        "Confirm the unit of analysis; re-run with per-unit ratios and, where "
        "possible, with data at the unit the claim is about."
    ),
    "measurement_error": (
        "Report measurement error for the predictor and consider attenuation; "
        "check whether the column is a proxy rather than the construct."
    ),
    "missing_data": (
        "Report the missingness mechanism and compare complete-case with an "
        "explicit imputation."
    ),
    "spatial_dependence": (
        "Use spatial or grouped validation and test the residuals for spatial "
        "autocorrelation before treating rows as independent."
    ),
    "temporal_dependence": (
        "Use a time-ordered split, test residual autocorrelation, and never "
        "train on rows dated after the evaluation rows."
    ),
    "confounding": (
        "List the omitted variables and the identification strategy; do not "
        "make a causal claim without one."
    ),
    "multiplicity": (
        "Pre-specify one model or report a correction for the search."
    ),
    "rare_group": (
        "Report group sizes and uncertainty; avoid stable claims about rare "
        "categories."
    ),
    "other": (
        "Read the cited source and decide whether the caveat applies to this "
        "dataset before using it as a finding."
    ),
}

_CATEGORY_PRIORITY: tuple[str, ...] = (
    "censoring",
    "selection",
    "aggregation",
    "measurement_error",
    "missing_data",
    "spatial_dependence",
    "temporal_dependence",
    "confounding",
    "multiplicity",
    "rare_group",
    "other",
)

_AGGREGATE_DESCRIPTION_RE = re.compile(
    r"\b(?:census|block[- ]group|district|county|tract|aggregat(?:e|ed|ion)|"
    r"survey|administrative|regional|area[- ]level)\b",
    re.IGNORECASE,
)
_GEO_NAMES = frozenset(
    {"latitude", "longitude", "lat", "lon", "lng", "geo", "geometry"}
)
_DERIVED_NAME_RE = re.compile(
    r"(?:median|mean|avg|average|rate|ratio|index|score|estimate|percent|pct|share)",
    re.IGNORECASE,
)
_AGGREGATE_COLUMN_RE = re.compile(
    r"^(?:total_|count|population|households|num_|number_)", re.IGNORECASE
)

_SCIENTIFIC_NAME_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class DatasetSignals:
    """Profile facts used to decide whether a pitfall could apply."""

    rows: int = 0
    target: str | None = None
    has_time: bool = False
    geo_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    derived_columns: list[str] = field(default_factory=list)
    aggregate_columns: list[str] = field(default_factory=list)
    aggregate_description: bool = False
    boundary_columns: list[tuple[str, int, float]] = field(default_factory=list)
    rare_columns: list[tuple[str, int, int]] = field(default_factory=list)


def _slug(value: str) -> str:
    text = _SCIENTIFIC_NAME_RE.sub("-", value.lower()).strip("-")
    return text[:40] or "field"


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _matched_terms(text: str) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for category in _CATEGORY_PRIORITY:
        pattern = _CATEGORY_PATTERNS.get(category)
        if pattern is None:
            continue
        terms: list[str] = []
        for match in pattern.finditer(text):
            term = " ".join(match.group(0).lower().split())
            if term and term not in terms:
                terms.append(term)
        if terms:
            matches[category] = terms
    if not matches:
        fallback = [match.group(0).lower() for match in _FALLBACK_PATTERN.finditer(text)]
        if fallback:
            matches["other"] = sorted(set(fallback))
    return matches


def _best_excerpt(text: str, terms: Sequence[str]) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text[:500]
    lowered_terms = [term.lower() for term in terms]

    def score(sentence: str) -> tuple[int, int]:
        lowered = sentence.lower()
        count = sum(1 for term in lowered_terms if term in lowered)
        return (count, -len(sentence))

    best = max(sentences, key=score)
    return best[:500]


def _primary_category(matches: dict[str, list[str]]) -> tuple[str, list[str]]:
    category = max(
        matches,
        key=lambda item: (
            len(matches[item]),
            -_CATEGORY_PRIORITY.index(item)
            if item in _CATEGORY_PRIORITY
            else -len(_CATEGORY_PRIORITY),
        ),
    )
    return category, matches[category]


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _record_text(record: ResearchRecord) -> str:
    title = str(_record_value(record, "title", "") or "")
    abstract = str(_record_value(record, "abstract", "") or "")
    return f"{title}. {abstract}".strip()


def _numeric_boundary(
    frame: Any, column_name: str
) -> tuple[int, float] | None:
    try:
        series = frame[column_name]
    except (KeyError, TypeError):
        return None
    try:
        values = series.dropna()
    except AttributeError:
        return None
    if len(values) == 0:
        return None
    try:
        maximum = values.max()
        minimum = values.min()
        count = int((values == maximum).sum())
    except (TypeError, ValueError):
        return None
    if count <= 0 or maximum == minimum:
        return None
    pct = count / len(values) * 100.0
    return count, pct


def build_signals(
    profile: DatasetProfile,
    *,
    frame: Any | None = None,
) -> DatasetSignals:
    """Read the profile and optional frame into applicability facts."""
    signals = DatasetSignals(
        rows=int(getattr(profile, "rows", 0) or 0),
        target=(
            str(profile.target.name)
            if getattr(profile, "target", None) is not None
            else None
        ),
        has_time=bool(getattr(profile, "time_column", None)),
        aggregate_description=bool(
            _AGGREGATE_DESCRIPTION_RE.search(str(getattr(profile, "description", "") or ""))
        ),
    )
    columns = list(getattr(profile, "columns", []) or [])
    for column in columns:
        name = str(getattr(column, "name", "") or "")
        lowered = name.lower()
        if lowered in _GEO_NAMES:
            signals.geo_columns.append(name)
        if float(getattr(column, "missing_pct", 0.0) or 0.0) > 0.0:
            signals.missing_columns.append(name)
        if _DERIVED_NAME_RE.search(name):
            signals.derived_columns.append(name)
        if _AGGREGATE_COLUMN_RE.search(name):
            signals.aggregate_columns.append(name)

        top_values = list(getattr(column, "top_values", []) or [])
        if top_values and signals.rows > 0:
            threshold = max(10, int(signals.rows * 0.01))
            counts = [
                int(count)
                for _, count in top_values
                if isinstance(count, (int, float))
            ]
            if counts and min(counts) < threshold:
                signals.rare_columns.append((name, min(counts), threshold))

    if frame is not None:
        for column in columns:
            name = str(getattr(column, "name", "") or "")
            numeric = getattr(column, "numeric", None)
            if not isinstance(numeric, dict):
                continue
            boundary = _numeric_boundary(frame, name)
            if boundary is None:
                continue
            count, pct = boundary
            if pct >= 1.0:
                signals.boundary_columns.append((name, count, pct))
    return signals


def applicability(
    category: str,
    signals: DatasetSignals,
) -> tuple[str, str]:
    """Return ``(applies, reason)`` for one category and profile."""
    if category == "censoring":
        if signals.boundary_columns:
            name, count, pct = signals.boundary_columns[0]
            return (
                "yes",
                f"{name} has {count} row(s) ({pct:.1f}%) at its maximum; "
                "a boundary effect is plausible.",
            )
        return (
            "unknown",
            "No large boundary concentration was detected, but censoring "
            "cannot be ruled out from the profile alone.",
        )
    if category == "missing_data":
        if signals.missing_columns:
            return (
                "yes",
                "Columns with missing values: "
                + ", ".join(signals.missing_columns[:6])
                + ".",
            )
        return "no", "The profile reports no missing values."
    if category == "spatial_dependence":
        if signals.geo_columns:
            return (
                "yes",
                "Geographic columns are present: "
                + ", ".join(signals.geo_columns[:4])
                + ".",
            )
        return (
            "unknown",
            "No explicit geographic columns were found; spatial dependence "
            "may still be present through unobserved location.",
        )
    if category == "temporal_dependence":
        if signals.has_time:
            return "yes", "A time column is present."
        return "no", "No time column was detected."
    if category == "aggregation":
        if signals.aggregate_description:
            return (
                "yes",
                "The description indicates aggregate or area-level data.",
            )
        if signals.aggregate_columns:
            return (
                "unknown",
                "Aggregate-looking columns are present: "
                + ", ".join(signals.aggregate_columns[:4])
                + "; confirm the unit of analysis.",
            )
        return (
            "unknown",
            "The unit of analysis is not established by the profile.",
        )
    if category == "measurement_error":
        if signals.derived_columns:
            return (
                "yes",
                "Derived or summary columns are present: "
                + ", ".join(signals.derived_columns[:4])
                + ".",
            )
        return (
            "unknown",
            "Measurement error cannot be assessed from the profile alone.",
        )
    if category == "rare_group":
        if signals.rare_columns:
            name, count, threshold = signals.rare_columns[0]
            return (
                "yes",
                f"{name} has a category with {count} row(s), below the "
                f"{threshold}-row screening threshold.",
            )
        return (
            "unknown",
            "Rare-group support was not detected in the stored top values.",
        )
    if category == "selection":
        return (
            "unknown",
            "The profile does not describe the sampling frame or selection rule.",
        )
    if category == "confounding":
        return (
            "unknown",
            "Confounding cannot be established without a causal design or "
            "collection metadata.",
        )
    if category == "multiplicity":
        return (
            "unknown",
            "The number of models and looks is not a property of the dataset "
            "profile.",
        )
    return "unknown", "The profile does not establish applicability."


def _source_prior(source: str) -> float:
    values = {
        "openalex": 0.80,
        "semantic_scholar": 0.75,
        "arxiv": 0.70,
        "websearch": 0.40,
        "catalog": 0.60,
    }
    return values.get(str(source).lower(), 0.50)


def _confidence(
    *,
    terms: Sequence[str],
    relevance: float,
    source: str,
    applies: str,
) -> float:
    value = 0.40 + 0.08 * min(len(terms), 4) + 0.15 * max(0.0, min(1.0, relevance))
    value += 0.10 * _source_prior(source)
    if applies == "yes":
        value += 0.10
    return round(max(0.0, min(0.95, value)), 3)


def extract_field_pitfalls(
    field: Any,
    records: Sequence[ResearchRecord],
    profile: DatasetProfile,
    *,
    frame: Any | None = None,
    signals: DatasetSignals | None = None,
    max_items: int = 3,
    scope: str = "ancestor",
) -> list[FieldPitfall]:
    """Extract up to ``max_items`` cited pitfalls for one field."""
    if max_items <= 0 or not records:
        return []
    signals = signals or build_signals(profile, frame=frame)
    name = field_name(field)
    identifier = field_id(field) or f"field-{_slug(name)}"
    level = field_level(field)

    candidates: list[
        tuple[int, float, ResearchRecord, str, list[str], str, float, int, str]
    ] = []
    for record in records:
        text = _record_text(record)
        if not text:
            continue
        matches = _matched_terms(text)
        if not matches:
            continue
        category, terms = _primary_category(matches)
        try:
            relevance = float(_record_value(record, "relevance", 0.0) or 0.0)
        except (TypeError, ValueError):
            relevance = 0.0
        source = str(_record_value(record, "source", "") or "")
        try:
            year = int(_record_value(record, "year", 0) or 0)
        except (TypeError, ValueError):
            year = 0
        score = (
            len(terms)
            + 0.20 * max(0.0, min(1.0, relevance))
            + 0.10 * _source_prior(source)
        )
        # Specific categories outrank the generic ``other`` fallback even when
        # a generic paper repeats more limitation words.
        specific = 0 if category != "other" else 1
        candidates.append(
            (specific, score, record, category, terms, text, relevance, year, source)
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            -item[1],
            -item[7],
            str(_record_value(item[2], "id", "") or ""),
        )
    )

    pitfalls: list[FieldPitfall] = []
    seen_records: set[str] = set()
    seen_titles: set[str] = set()
    other_count = 0
    for _, _, record, category, terms, text, relevance, year, source in candidates:
        record_id = str(_record_value(record, "id", "") or "")
        if not record_id or record_id in seen_records:
            continue
        title = str(_record_value(record, "title", "") or "")
        title_key = " ".join(title.lower().split())
        if title_key in seen_titles:
            continue
        if category == "other" and other_count >= 1:
            continue
        seen_records.add(record_id)
        seen_titles.add(title_key)
        if category == "other":
            other_count += 1
        applies, reason = applicability(category, signals)
        index = len(pitfalls) + 1
        pitfalls.append(
            FieldPitfall(
                id=f"F:{identifier}:{index}",
                field_id=field_id(field),
                field_name=name,
                field_level=level,
                scope="leaf" if scope == "leaf" else "ancestor",
                title=redact_secrets(title),
                pitfall=redact_secrets(_best_excerpt(text, terms)),
                category=category,  # type: ignore[arg-type]
                applies_to_dataset=applies,  # type: ignore[arg-type]
                applicability_reason=redact_secrets(reason),
                check=redact_secrets(_CHECK_TEXT.get(category, _CHECK_TEXT["other"])),
                evidence_id=record_id,
                source=source,
                url=redact_secrets(str(_record_value(record, "url", "") or "")),
                year=year or None,
                relevance=round(relevance, 6),
                confidence=_confidence(
                    terms=terms,
                    relevance=relevance,
                    source=source,
                    applies=applies,
                ),
            )
        )
        if len(pitfalls) >= max_items:
            break
    return pitfalls


__all__ = [
    "DatasetSignals",
    "applicability",
    "build_signals",
    "extract_field_pitfalls",
]
