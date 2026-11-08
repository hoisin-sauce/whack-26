"""Typed run artifact, stable IDs, writers, and handoff validation."""

from __future__ import annotations

import json
import platform
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field as PField, ValidationError

from .classify import FieldPath
from .critic.models import CriticResult
from .factors import Factor
from .hypotheses.generate import GeneratedHypothesis
from .ids import factor_id
from .profile import DatasetProfile
from .research.pipeline import ResearchResult
from .screen import ScreenSummary

SCHEMA_VERSION = "1.0.0"
_ABSOLUTE_MARKERS = ("/", "\\")
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{20,}")),
    ("huggingface_token", re.compile(r"hf_[0-9A-Za-z]{20,}")),
    ("groq_token", re.compile(r"gsk_[0-9A-Za-z]{20,}")),
    ("openai_token", re.compile(r"sk-[0-9A-Za-z]{20,}")),
    ("bearer_token", re.compile(r"Bearer\s+[0-9A-Za-z._\-]{20,}")),
    ("goog_api_key_header", re.compile(r"x-goog-api-key\s*[:=]", re.IGNORECASE)),
)


class RunInfo(BaseModel):
    run_id: str
    started_at: str
    finished_at: str
    mode: Literal["live", "mock"]
    seed: int
    tool_version: str
    models: dict[str, str]


class DatasetColumn(BaseModel):
    name: str
    dtype: str
    role: str
    missing_pct: float
    cardinality: int


class TargetSection(BaseModel):
    name: str
    source: str
    confidence: float


class DatasetSection(BaseModel):
    id: str
    source_path: str
    stored_path: str = "dataset.csv"
    sha256: str
    format: str
    rows: int
    description: str | None = None
    target: TargetSection | None = None
    time_column: str | None = None
    columns: list[DatasetColumn] = PField(default_factory=list)


class FieldSection(BaseModel):
    field_id: str
    name: str
    level: int
    description: str


class ClassificationSection(BaseModel):
    path: list[FieldSection] = PField(default_factory=list)
    confidence: float = 0.0
    created_fields: list[str] = PField(default_factory=list)
    refined_after_research: bool = False
    refinement_reason: str | None = None


class QuerySection(BaseModel):
    source: str
    query: str
    rationale: str = ""


class ResearchRecordSection(BaseModel):
    id: str
    source: str
    source_id: str
    title: str
    abstract: str | None = None
    authors: list[str] = PField(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str
    license: str | None = None
    retrieved_at: str
    query: str
    relevance: float = 0.0
    also_seen_in: list[str] = PField(default_factory=list)


class ResearchSection(BaseModel):
    queries: list[QuerySection] = PField(default_factory=list)
    records: list[ResearchRecordSection] = PField(default_factory=list)
    cache: dict[str, Any] = PField(default_factory=dict)
    limited_mode: bool = False
    offline: bool = False
    warnings: list[str] = PField(default_factory=list)


class FactorMapping(BaseModel):
    column: str | None = None
    confidence: float | None = None
    in_dataset: bool = False
    rationale: str = ""


class FactorSection(BaseModel):
    id: str
    name: str
    description: str
    direction: str
    claim_type: str
    evidence_ids: list[str] = PField(default_factory=list)
    mapping: FactorMapping = PField(default_factory=FactorMapping)
    rationale: str = ""


class CorrelationSection(BaseModel):
    a: str
    b: str
    pearson: float | None = None
    spearman: float | None = None


class ScreenSection(BaseModel):
    target_summary: dict[str, Any] = PField(default_factory=dict)
    missingness: dict[str, float] = PField(default_factory=dict)
    top_correlations: list[CorrelationSection] = PField(default_factory=list)
    time_trend: dict[str, Any] | None = None


class ExecutionSection(BaseModel):
    status: str
    runtime_ms: int = 0
    n_predictions: int | None = None
    predictions_path: str | None = None
    error: str | None = None


class CriticIssueSection(BaseModel):
    chunk: str
    severity: str
    finding: str
    evidence: str = ""
    why: str = ""
    fix: str = ""


class CriticCheckSection(BaseModel):
    id: str
    status: str
    severity: str = "none"
    reason: str = ""
    citations: list[str] = PField(default_factory=list)


class CriticSection(BaseModel):
    claim_type: str
    verdict: str
    issues: list[CriticIssueSection] = PField(default_factory=list)
    missing_information: list[str] = PField(default_factory=list)
    checks: list[CriticCheckSection] = PField(default_factory=list)
    summary: str = ""
    retrieved_chunks: list[str] = PField(default_factory=list)
    model: str = "unknown"
    corpus_version: str = "unknown"


class HypothesisSection(BaseModel):
    id: str
    title: str
    method_family: str
    claim_type: str
    target: str | None = None
    required_columns: list[str] = PField(default_factory=list)
    optional_columns: list[str] = PField(default_factory=list)
    min_rows: int = 1
    assumptions: list[str] = PField(default_factory=list)
    factors_used: list[str] = PField(default_factory=list)
    source_ids: list[str] = PField(default_factory=list)
    rationale: str = ""
    code_path: str
    code_sha256: str
    generation_history: list[dict[str, Any]] = PField(default_factory=list)
    execution: ExecutionSection
    critic: CriticSection | None = None
    warnings: list[str] = PField(default_factory=list)


class CriticSummary(BaseModel):
    verdicts: dict[str, int] = PField(default_factory=dict)
    degraded: bool = False


class Provenance(BaseModel):
    python: str
    packages: dict[str, str] = PField(default_factory=dict)
    corpus_version: str | None = None
    git_commit: str | None = None


class Artifact(BaseModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    run: RunInfo
    dataset: DatasetSection
    classification: ClassificationSection
    research: ResearchSection
    factors: list[FactorSection] = PField(default_factory=list)
    screen: ScreenSection
    hypotheses: list[HypothesisSection] = PField(default_factory=list)
    critic_summary: CriticSummary
    warnings: list[str] = PField(default_factory=list)
    provenance: Provenance


def _dataset_section(profile: DatasetProfile) -> DatasetSection:
    return DatasetSection(
        id=profile.dataset_id,
        source_path=profile.source_path,
        stored_path="dataset.csv",
        sha256=profile.sha256,
        format=profile.format,
        rows=profile.rows,
        description=profile.description,
        target=(
            TargetSection(
                name=profile.target.name,
                source=profile.target.source,
                confidence=profile.target.confidence,
            )
            if profile.target
            else None
        ),
        time_column=profile.time_column,
        columns=[
            DatasetColumn(
                name=column.name,
                dtype=column.dtype,
                role=column.role,
                missing_pct=column.missing_pct,
                cardinality=column.cardinality,
            )
            for column in profile.columns
        ],
    )


def _classification_section(path: FieldPath) -> ClassificationSection:
    return ClassificationSection(
        path=[
            FieldSection(
                field_id=field.id,
                name=field.name,
                level=field.level,
                description=field.description,
            )
            for field in path.fields
        ],
        confidence=path.confidence,
        created_fields=path.created_fields,
        refined_after_research=path.refined_after_research,
        refinement_reason=path.refinement_reason,
    )


def _research_section(research: ResearchResult) -> ResearchSection:
    return ResearchSection(
        queries=[
            QuerySection(
                source=query.source,
                query=query.query,
                rationale=query.rationale,
            )
            for query in research.queries
        ],
        records=[
            ResearchRecordSection.model_validate(record.model_dump(mode="json"))
            for record in research.records
        ],
        cache=dict(research.cache),
        limited_mode=research.limited_mode,
        offline=research.offline,
        warnings=list(research.warnings),
    )


def _factor_sections(
    factors: Sequence[Factor], profile: DatasetProfile
) -> list[FactorSection]:
    sections: list[FactorSection] = []
    for factor in factors:
        sections.append(
            FactorSection(
                id=factor.id or factor_id(profile.dataset_id, factor.name),
                name=factor.name,
                description=factor.description,
                direction=factor.direction,
                claim_type=factor.claim_type,
                evidence_ids=list(factor.evidence_ids),
                mapping=FactorMapping(
                    column=factor.mapped_column,
                    confidence=factor.mapping_confidence,
                    in_dataset=factor.in_dataset,
                    rationale=factor.rationale,
                ),
                rationale=factor.rationale,
            )
        )
    return sections


def _screen_section(screen: ScreenSummary) -> ScreenSection:
    return ScreenSection(
        target_summary=screen.target.model_dump(mode="json"),
        missingness=dict(screen.missingness),
        top_correlations=[
            CorrelationSection.model_validate(pair.model_dump(mode="json"))
            for pair in screen.top_correlations
        ],
        time_trend=(
            screen.time_trend.model_dump(mode="json")
            if screen.time_trend is not None
            else None
        ),
    )


def _hypothesis_section(
    hypothesis: GeneratedHypothesis,
    critic: CriticResult | None,
) -> HypothesisSection:
    payload = hypothesis.to_artifact()
    payload["execution"] = ExecutionSection.model_validate(payload["execution"])
    if critic is not None:
        payload["critic"] = CriticSection.model_validate(
            critic.model_dump(mode="json")
        )
    payload["warnings"] = list(hypothesis.validation_warnings)
    return HypothesisSection.model_validate(payload)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("pandas", "numpy", "scipy", "httpx", "pydantic"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit(base_dir: Path) -> str | None:
    head = base_dir / ".git" / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if content.startswith("ref: "):
        ref_path = base_dir / ".git" / content[len("ref: ") :].strip()
        try:
            return ref_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    return content or None


def build_artifact(
    *,
    run: RunInfo,
    profile: DatasetProfile,
    classification: FieldPath,
    research: ResearchResult,
    factors: Sequence[Factor],
    screen: ScreenSummary,
    hypotheses: Sequence[GeneratedHypothesis],
    critics: Sequence[CriticResult],
    warnings: Sequence[str],
    corpus_version: str | None,
    base_dir: Path,
) -> Artifact:
    """Assemble and validate the artifact from completed stage outputs."""
    by_id = {critic.hypothesis_id: critic for critic in critics}
    verdicts: dict[str, int] = {
        "pass": 0,
        "refine": 0,
        "reject": 0,
        "insufficient": 0,
    }
    degraded = False
    for critic in critics:
        verdicts[critic.verdict] = verdicts.get(critic.verdict, 0) + 1
        degraded = degraded or critic.degraded

    all_warnings: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        text = str(warning).strip()
        if text and text not in seen:
            seen.add(text)
            all_warnings.append(text)

    return Artifact(
        run=run,
        dataset=_dataset_section(profile),
        classification=_classification_section(classification),
        research=_research_section(research),
        factors=_factor_sections(factors, profile),
        screen=_screen_section(screen),
        hypotheses=[
            _hypothesis_section(hypothesis, by_id.get(hypothesis.id))
            for hypothesis in hypotheses
        ],
        critic_summary=CriticSummary(verdicts=verdicts, degraded=degraded),
        warnings=all_warnings,
        provenance=Provenance(
            python=platform.python_version(),
            packages=_package_versions(),
            corpus_version=corpus_version,
            git_commit=_git_commit(base_dir),
        ),
    )


def write_artifact(path: Path, artifact: Artifact) -> None:
    path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def artifact_json_schema() -> dict[str, Any]:
    return Artifact.model_json_schema()


def _relative_path_problem(value: str | None) -> str | None:
    if value is None:
        return None
    if Path(value).is_absolute():
        return f"path must be relative: {value}"
    if value.startswith(("/", "\\")) or ".." in Path(value).parts:
        return f"path escapes the run directory: {value}"
    return None


def validate_artifact(
    path: Path, *, check_files: bool = True
) -> tuple[Artifact | None, list[str]]:
    """Validate the artifact schema and handoff invariants."""
    errors: list[str] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [f"could not read artifact: {exc}"]
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, [f"artifact is not valid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["artifact must be a JSON object"]
    major = str(payload.get("schema_version", "")).split(".")[0]
    if major not in {"1"}:
        errors.append(
            f"unknown schema major {major!r}; this tool supports schema 1.x"
        )
    try:
        artifact = Artifact.model_validate(payload)
    except ValidationError as exc:
        errors.append(f"schema validation failed: {exc}")
        return None, errors

    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(raw_text):
            errors.append(f"artifact appears to contain a credential ({label})")

    base = path.parent
    dataset_path = artifact.dataset.stored_path
    problems = [_relative_path_problem(dataset_path)]
    for hypothesis in artifact.hypotheses:
        problems.extend(
            [
                _relative_path_problem(hypothesis.code_path),
                _relative_path_problem(
                    hypothesis.execution.predictions_path
                ),
            ]
        )
    for problem in problems:
        if problem:
            errors.append(problem)

    if check_files:
        if not (base / dataset_path).is_file():
            errors.append(f"stored dataset is missing: {dataset_path}")
        for hypothesis in artifact.hypotheses:
            code = base / hypothesis.code_path
            if not code.is_file():
                errors.append(f"hypothesis code is missing: {hypothesis.code_path}")
            predictions = hypothesis.execution.predictions_path
            if predictions and not (base / predictions).is_file():
                errors.append(f"predictions are missing: {predictions}")
            if hypothesis.execution.status == "ok":
                if hypothesis.execution.n_predictions is None:
                    errors.append(
                        f"{hypothesis.id}: successful execution has no prediction count"
                    )
            if hypothesis.critic is None:
                errors.append(f"{hypothesis.id}: critic result is missing")
    if len(artifact.hypotheses) != 3:
        errors.append(
            f"artifact must contain exactly three hypotheses, got {len(artifact.hypotheses)}"
        )
    families = [hypothesis.method_family for hypothesis in artifact.hypotheses]
    if len(set(families)) != len(families):
        errors.append("hypothesis method families are not unique")
    if any("predictions" in record.model_dump() for record in artifact.research.records):
        errors.append("research records must not embed prediction arrays")
    return artifact, errors
