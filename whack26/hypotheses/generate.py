"""Hypothesis generation, validation, repair, and execution orchestration."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..config import Settings
from ..embeddings import cosine
from ..factors import Factor
from ..ids import hypothesis_id
from ..llm import LLMError
from ..profile import DatasetProfile
from ..research.base import ResearchRecord
from ..screen import ScreenSummary, format_screen
from . import repair
from .execute import ExecutionResult, run_module
from .validate import MetadataValidation, validate_hypothesis

MethodFamily = Literal["linear_model", "group_comparison", "interaction_or_derived"]
ClaimType = Literal["description", "prediction", "causal", "explanatory"]

FAMILIES: tuple[MethodFamily, ...] = (
    "linear_model",
    "group_comparison",
    "interaction_or_derived",
)

FAMILY_DESCRIPTIONS: dict[MethodFamily, str] = {
    "linear_model": (
        "Fit one linear model. Use ordinary least squares via "
        "`numpy.linalg.lstsq` for a numeric target, or a logistic link via "
        "`scipy.optimize.minimize` for a binary target. Optionally transform a "
        "feature with log or sqrt. Expose one positive numeric hyperparameter "
        "as `PARAMETERS = {\"ridge\": 1e-3}` and apply it as a small ridge "
        "penalty. Do not use sklearn or statsmodels."
    ),
    "group_comparison": (
        "Compare a group statistic. Choose one categorical column, or bin one "
        "numeric factor into quantiles, and predict the target group mean (or "
        "binary rate). Every row gets its group's statistic; unseen groups fall "
        "back to the global mean or rate. Expose one integer hyperparameter as "
        "`PARAMETERS = {\"min_group_size\": 2}`; clamp it to at least 1."
    ),
    "interaction_or_derived": (
        "Fit one linear model with either one interaction between two factors or "
        "one ratio/derived feature. Use `numpy.linalg.lstsq` for a numeric target "
        "or a logistic link via `scipy.optimize.minimize` for a binary target. "
        "Expose one positive numeric hyperparameter as "
        "`PARAMETERS = {\"ridge\": 1e-3}` and apply it as a small ridge penalty. "
        "State how the interaction or derived feature was constructed."
    ),
}

CLAIM_TYPE_ALIASES = {
    "predictive": "prediction",
    "association": "description",
    "associational": "description",
    "descriptive": "description",
    "explanatory": "explanatory",
    "inferential": "explanatory",
    "causal": "causal",
    "prediction": "prediction",
    "description": "description",
}

SYSTEM_PROMPT = (
    "You write one runnable statistical analysis hypothesis as a Python module "
    "for a tabular dataset. The code is executed in a sandboxed child process. "
    "Use only the supplied dataset facts and research records; never invent "
    "columns, statistics, or citations. Use predictive or associative language "
    "unless the data description explicitly says the data came from a randomized "
    "or quasi-experimental design. Return only the requested JSON."
)

HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 200},
        "rationale": {"type": "string", "maxLength": 1200},
        "assumptions": {
            "type": "array",
            "items": {"type": "string", "maxLength": 400},
            "minItems": 1,
        },
        "required_columns": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
        },
        "optional_columns": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
        },
        "min_rows": {"type": "integer"},
        "claim_type": {
            "type": "string",
            "enum": ["description", "prediction", "causal", "explanatory"],
        },
        "factors_used": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
        },
        "source_ids": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
        },
        "code": {"type": "string"},
    },
    "required": [
        "title",
        "rationale",
        "assumptions",
        "required_columns",
        "optional_columns",
        "min_rows",
        "claim_type",
        "factors_used",
        "source_ids",
        "code",
    ],
    "propertyOrdering": [
        "title",
        "rationale",
        "assumptions",
        "required_columns",
        "optional_columns",
        "min_rows",
        "claim_type",
        "factors_used",
        "source_ids",
        "code",
    ],
}


class HypothesisGenerationError(RuntimeError):
    """Raised when fewer than three hypotheses survive."""


class HypothesisDraft(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    rationale: str = ""
    assumptions: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    optional_columns: list[str] = Field(default_factory=list)
    min_rows: int = 1
    claim_type: ClaimType = "prediction"
    factors_used: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    code: str = ""
    target: str | None = None

    @field_validator("claim_type", mode="before")
    @classmethod
    def _normalize_claim_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return CLAIM_TYPE_ALIASES.get(value.strip().lower(), value.strip().lower())
        return value

    @field_validator(
        "assumptions",
        "required_columns",
        "optional_columns",
        "factors_used",
        "source_ids",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


def _coerce_draft(payload: dict[str, Any]) -> HypothesisDraft:
    data = dict(payload)
    data.pop("method_family", None)
    for key in (
        "title",
        "rationale",
        "assumptions",
        "required_columns",
        "optional_columns",
        "factors_used",
        "source_ids",
        "code",
    ):
        if data.get(key) is None:
            data[key] = "" if key in {"title", "rationale", "code"} else []
    if data.get("min_rows") in (None, ""):
        data["min_rows"] = 1
    return HypothesisDraft.model_validate(data)


@dataclass
class HypothesisContext:
    profile: DatasetProfile
    summary: str
    path_names: Sequence[str]
    factors: Sequence[Factor]
    records: Sequence[ResearchRecord]
    screen: ScreenSummary
    model_search: Any | None = None
    research_models: Sequence[Any] = field(default_factory=tuple)
    variant_report: Any | None = None
    context_pack: Any | None = None
    collection: Any | None = None

    @property
    def dataset_columns(self) -> list[str]:
        return [column.name for column in self.profile.columns]

    @property
    def factor_ids(self) -> list[str]:
        return [factor.id for factor in self.factors if factor.id]

    @property
    def source_ids(self) -> list[str]:
        return [record.id for record in self.records]

    @property
    def target(self) -> str | None:
        return self.profile.target.name if self.profile.target else None


@dataclass
class GeneratedHypothesis:
    id: str
    method_family: MethodFamily
    draft: HypothesisDraft
    code_sha256: str
    code_path: str
    execution: ExecutionResult
    generation_history: list[dict[str, Any]]
    validation_warnings: list[str] = field(default_factory=list)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.draft.title,
            "method_family": self.method_family,
            "claim_type": self.draft.claim_type,
            "target": self.draft.target,
            "required_columns": self.draft.required_columns,
            "optional_columns": self.draft.optional_columns,
            "min_rows": self.draft.min_rows,
            "assumptions": self.draft.assumptions,
            "factors_used": self.draft.factors_used,
            "source_ids": self.draft.source_ids,
            "rationale": self.draft.rationale,
            "code_path": self.code_path,
            "code_sha256": self.code_sha256,
            "generation_history": self.generation_history,
            "execution": self.execution.to_artifact(),
        }


@dataclass
class HypothesisSet:
    hypotheses: list[GeneratedHypothesis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _records_text(records: Sequence[ResearchRecord], limit: int = 5) -> str:
    lines: list[str] = []
    for record in records[:limit]:
        abstract = (record.abstract or "").strip()
        if len(abstract) > 1200:
            abstract = abstract[:1197] + "..."
        lines.append(
            f"- id={record.id} | source={record.source} | year={record.year} | "
            f"title={record.title}"
            + (f"\n  abstract: {abstract}" if abstract else "")
        )
    return "\n".join(lines) or "(none)"


def _factors_text(factors: Sequence[Factor]) -> str:
    lines: list[str] = []
    for factor in factors:
        lines.append(
            f"- id={factor.id or '(none)'} | name={factor.name} | "
            f"direction={factor.direction} | claim={factor.claim_type} | "
            f"mapped_column={factor.mapped_column or '(unmapped)'} | "
            f"evidence={', '.join(factor.evidence_ids) or '(none)'}\n"
            f"  {factor.description}"
        )
    return "\n".join(lines) or (
        "(no factors were extracted; explain in the rationale that no factor "
        "applies and justify the hypothesis from the dataset description)"
    )


def _columns_text(context: HypothesisContext) -> str:
    lines: list[str] = []
    target = context.target
    for column in context.profile.columns:
        marker = " [TARGET - never a feature]" if column.name == target else ""
        lines.append(
            f"- {column.name} ({column.dtype}, role={column.role}, "
            f"missing={column.missing_pct}%, {column.cardinality} unique){marker}"
        )
    return "\n".join(lines)


def _target_text(context: HypothesisContext) -> str:
    target = context.target
    if target is None:
        return (
            "No target column is defined. Write a descriptive hypothesis that "
            "still returns one prediction per row; set `target` to None."
        )
    screen_target = context.screen.target
    if screen_target.kind == "numeric":
        link = "identity (numeric target)"
    elif screen_target.kind == "categorical" and len(screen_target.top_values or []) == 2:
        link = "logit (binary target)"
    else:
        link = "descriptive (multi-class target)"
    return f"Target: {target} | link suggestion: {link}"


def _model_search_text(value: Any) -> str:
    if value is None:
        return "(none)"
    candidates = list(getattr(value, "candidates", []) or [])
    if not candidates:
        return "(no candidates)"
    lines: list[str] = []
    selected = getattr(value, "selected", None)
    if selected is not None:
        lines.append(f"Selected family: {getattr(getattr(selected, 'spec', None), 'family', '?')}")
    for candidate in candidates[:5]:
        spec = getattr(candidate, "spec", None)
        family = getattr(spec, "family", "?")
        features = ", ".join(getattr(spec, "features", []) or [])
        metrics = getattr(candidate, "metrics", {}) or {}
        rendered = ", ".join(
            f"{name}={float(score):.3f}"
            for name, score in metrics.items()
            if isinstance(score, (int, float))
        )
        diagnostics = [
            f"{getattr(item, 'name', '?')}:{getattr(item, 'status', '?')}"
            for item in (getattr(candidate, "diagnostics", []) or [])
        ]
        lines.append(
            f"- {family}: features=[{features}] metrics=[{rendered}]"
            + (f" diagnostics=[{', '.join(diagnostics)}]" if diagnostics else "")
        )
    return "\n".join(lines)


def _research_models_text(values: Sequence[Any]) -> str:
    if not values:
        return "(none)"
    lines: list[str] = []
    for value in values[:5]:
        family = getattr(value, "family", "?")
        outcome = getattr(value, "outcome", "?")
        predictors = ", ".join(getattr(value, "mapped_columns", []) or [])
        confidence = getattr(value, "confidence", None)
        evidence = str(getattr(value, "evidence", ""))[:240]
        lines.append(
            f"- {family} for {outcome}; columns=[{predictors}]; "
            f"confidence={confidence}; evidence={evidence}"
        )
    return "\n".join(lines)


def _variant_text(value: Any) -> str:
    if value is None:
        return "(not evaluated)"
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    elif isinstance(value, dict):
        payload = value
    else:
        payload = {"value": str(value)}
    keys = (
        "jumbled_label_leak",
        "pii_reliance",
        "anonymized",
        "jumbled",
        "warnings",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    return str(summary)


def _base_prompt(
    context: HypothesisContext,
    family: MethodFamily,
    *,
    feedback: str | None = None,
    simpler: bool = False,
) -> str:
    target = context.target
    if target is not None:
        target_rules = (
            f"- Never use the target column ({target}) as a feature.\n"
            "- `required_columns` and `optional_columns` list feature columns only. "
            f"Never put the target column ({target}) in either list; the runner "
            "loads the target separately and passes it to the module only through "
            "the frame.\n"
        )
    else:
        target_rules = (
            "- No target column is defined; treat the columns as descriptive "
            "features and set `target` to None in the returned dict.\n"
        )
    parts = [
        "Dataset summary:\n" + context.summary,
        "Field path: " + (" > ".join(context.path_names) or "(none)"),
        _target_text(context),
        "Cheap descriptive screen (no hypothesis tests; context only):\n"
        + format_screen(context.screen, max_chars=2500),
        "Factors from research (use the fac_ ids in `factors_used`):\n"
        + _factors_text(context.factors),
        "Top research records (use their ids in `source_ids`):\n"
        + _records_text(context.records),
    ]
    if context.context_pack is not None:
        from .context import render_context_pack

        parts.append(
            "Ranked context pack (field path, ancestor documents, statistics):\n"
            + render_context_pack(context.context_pack)
        )
    if context.model_search is not None:
        parts.append(
            "Automated model search (prefer these specifications when they fit):\n"
            + _model_search_text(context.model_search)
        )
    if context.research_models:
        parts.append(
            "Applicable models extracted from research (cite the source and map "
            "predictors to these columns):\n"
            + _research_models_text(context.research_models)
        )
    if context.variant_report is not None:
        parts.append(
            "Dataset variant checks (jumbled labels and anonymization):\n"
            + _variant_text(context.variant_report)
        )
    parts.extend(
        [
            "Dataset columns:\n" + _columns_text(context),
        "Write one Python module with exactly these two public functions and the "
        "private validation helpers:\n"
        "```python\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "PARAMETERS = {\"ridge\": 1e-3}  # at least one real numeric hyperparameter\n\n"
        "def applies(df: pd.DataFrame) -> bool:\n"
        "    \"\"\"Return True when the hypothesis can run on df. Must not raise.\"\"\"\n\n"
        "def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):\n"
        "    \"\"\"Fit on df and return a model object. The target is available here.\"\"\"\n"
        "    ...\n\n"
        "def _predict(model, df: pd.DataFrame):\n"
        "    \"\"\"Predict one value per row. df has features only; never read the target.\"\"\"\n"
        "    ...\n\n"
        "def run(df: pd.DataFrame, seed: int = 0) -> dict:\n"
        "    \"\"\"Fit the model and return predictions. Must not mutate df.\"\"\"\n"
        "    model = _fit(df, seed, PARAMETERS)\n"
        "    predict_frame = df.drop(columns=[TARGET]) if TARGET in df.columns else df\n"
        "    return {\"predictions\": _predict(model, predict_frame), \"target\": TARGET}\n"
        "```\n"
        "`run` must return a JSON-serializable dict with:\n"
        "- `predictions`: one value per row (list, pandas Series, or numpy array);\n"
        "- `target`: the target column name, or None for a descriptive hypothesis;\n"
        "- optional `row_index` aligned with predictions when rows are dropped;\n"
        "- optional `model_summary` dict and `notes` string.\n",
        "Rules:\n"
        "- Start `run` by copying the frame (`df = df.copy()`) and never mutate input.\n"
        "- Implement `_fit` and `_predict` so the statistical validator can fit on "
        "a training split and predict held-out rows. `_predict` must return one "
        "prediction per input row and must never read the target column.\n"
        "- `PARAMETERS` must be a module-level dict with at least one positive "
        "numeric hyperparameter that `_fit` actually uses. The validator perturbs it.\n"
        f"- Replace `TARGET` in the example with the actual target column name: "
        f"{target!r}. If there is no target, use `None` and do not drop a column.\n"
        + target_rules
        + "- Handle missing values explicitly and say what you did in `notes`.\n"
        "- Call `np.random.seed(seed)` and `random.seed(seed)` if you sample.\n"
        "- Use predictive or associative language. Do not claim causation.\n"
        "- Allowed imports only: pandas, numpy, scipy, math, statistics, datetime, "
        "itertools, functools, collections, re, warnings.\n"
        "- Never import sklearn, statsmodels, torch, tensorflow, or any other package.\n"
        "- Never read files, environment variables, or the network.\n"
        "- Do not call dropna() on a large fraction of the data without a note.\n"
        "- At least one specific assumption, a non-empty rationale, and code <= 20 KB.\n"
        "- If you use factors, cite at least one factor id and at least one source id. "
        "If no factor applies, say so explicitly in the rationale.\n"
        f"- Required columns must exist in this dataset: {', '.join(context.dataset_columns)}.\n",
        f"Method family: {family}\n{FAMILY_DESCRIPTIONS[family]}",
        "Return only the JSON object requested by the schema. The `code` value must "
        "be the complete module source, not a diff or a fragment.",
        ]
    )
    if feedback:
        parts.append("Previous attempt feedback:\n" + feedback)
    if simpler:
        parts.append(repair.simpler_instruction(family))
    return "\n\n".join(parts)


def _normalized_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = "_"
            node.annotation = None
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(
                ast.Constant(value=type(node.value).__name__), node
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.returns = None
            self.generic_visit(node)
            return node

    return ast.dump(Normalize().visit(tree), include_attributes=False)


class HypothesisGenerator:
    """Generate three distinct hypotheses, repairing failures in place."""

    def __init__(
        self,
        *,
        context: HypothesisContext,
        llm: Any,
        embeddings: Any,
        run_dir: Path,
        dataset_path: Path,
        seed: int = 0,
        settings: Settings | None = None,
    ) -> None:
        self.context = context
        self.llm = llm
        self.embeddings = embeddings
        self.run_dir = run_dir
        self.dataset_path = dataset_path
        self.seed = seed
        self.settings = settings
        self._accepted: list[GeneratedHypothesis] = []
        self._accepted_vectors: list[list[float] | None] = []
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self.warnings: list[str] = []

    def _remember(
        self,
        family: MethodFamily,
        history: list[dict[str, Any]],
        entry: dict[str, Any],
    ) -> None:
        """Append one attempt and persist the history for failed runs."""
        history.append(entry)
        self._histories[family] = list(history)
        path = self.run_dir / "logs" / "generation.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._histories, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Public API

    def generate(self) -> HypothesisSet:
        for family in FAMILIES:
            result = self._generate_family(family)
            if result is None:
                history = self._histories.get(family, [])
                detail = ""
                if history:
                    last = history[-1]
                    detail = (
                        f" last attempt: {last.get('status')}: "
                        f"{str(last.get('error') or '(no error)')[:600]}"
                    )
                raise HypothesisGenerationError(
                    f"family {family!r} did not produce a runnable hypothesis;"
                    f"{detail} see {self.run_dir / 'logs' / 'generation.json'}"
                )
        return HypothesisSet(hypotheses=self._accepted, warnings=self.warnings)

    # ------------------------------------------------------------------
    # One family

    def _ask(
        self,
        family: MethodFamily,
        *,
        feedback: str | None,
        simpler: bool,
    ) -> HypothesisDraft:
        payload = self.llm.generate_json(
            system=SYSTEM_PROMPT,
            user=_base_prompt(
                self.context, family, feedback=feedback, simpler=simpler
            ),
            schema=HYPOTHESIS_SCHEMA,
            max_tokens=4096,
            purpose=f"hypothesis_{family}",
        )
        if not isinstance(payload, dict):
            raise LLMError(
                f"hypothesis generator returned {type(payload).__name__}, not an object",
                purpose=f"hypothesis_{family}",
            )
        try:
            return _coerce_draft(payload)
        except ValidationError as exc:
            raise LLMError(
                f"hypothesis metadata failed schema validation: {exc}",
                purpose=f"hypothesis_{family}",
            ) from exc

    def _generate_family(self, family: MethodFamily) -> GeneratedHypothesis | None:
        history: list[dict[str, Any]] = []
        failures: Counter[str] = Counter()
        feedback: str | None = None
        simpler = False
        calls = 0

        while calls < 6:
            calls += 1
            try:
                draft = self._ask(family, feedback=feedback, simpler=simpler)
            except LLMError as exc:
                failures["llm_error"] += 1
                self._remember(family, history,
                    {
                        "attempt": calls,
                        "status": "llm_error",
                        "error": str(exc)[:1000],
                    }
                )
                feedback = repair.contract_feedback(str(exc))
                if failures["llm_error"] > 1:
                    break
                continue

            validation = validate_hypothesis(
                draft,
                target=self.context.target,
                dataset_columns=self.context.dataset_columns,
                known_factor_ids=self.context.factor_ids,
                known_source_ids=self.context.source_ids,
            )
            if not validation.ok:
                failures["static"] += 1
                self._remember(family, history,
                    {
                        "attempt": calls,
                        "status": "static_invalid",
                        "error": "; ".join(validation.errors)[:1500],
                    }
                )
                feedback = repair.static_feedback(validation.errors)
                if failures["static"] > 1:
                    break
                continue

            result = self._execute_draft(family, draft, validation, history, calls)
            if isinstance(result, GeneratedHypothesis):
                if not self._distinct(draft.code):
                    failures["duplicate"] += 1
                    self._remember(family, history,
                        {
                            "attempt": calls,
                            "status": "duplicate",
                            "error": "code and normalized AST are too similar",
                        }
                    )
                    feedback = repair.duplicate_feedback()
                    if failures["duplicate"] > 1:
                        break
                    continue
                self._remember(family, history, {"attempt": calls, "status": "ok", "error": None})
                result.generation_history = history
                result.validation_warnings = list(validation.warnings)
                self.warnings.extend(validation.warnings)
                self._accepted.append(result)
                self._remember_vector(draft.code)
                return result

            execution = result
            self._remember(
                family,
                history,
                {
                    "attempt": calls,
                    "status": execution.status,
                    "error": (execution.error or "")[:1500],
                },
            )
            failures[execution.status] += 1
            if execution.status in {"applies_false", "timeout"}:
                feedback = repair.contract_feedback(
                    execution.error or f"applies returned {execution.status}"
                )
                if failures[execution.status] > 1:
                    break
            elif execution.status == "invalid_output":
                feedback = repair.execution_feedback(
                    execution.status, execution.error, execution.traceback
                )
                if failures[execution.status] > 1:
                    break
            else:
                feedback = repair.execution_feedback(
                    execution.status, execution.error, execution.traceback
                )
                if failures[execution.status] > 2:
                    break
            if calls >= 5 and not simpler:
                simpler = True

        # One last replacement with the simplest template.
        for extra in (1, 2):
            calls += 1
            try:
                draft = self._ask(family, feedback=feedback, simpler=True)
            except LLMError as exc:
                self._remember(family, history,
                    {
                        "attempt": calls,
                        "status": "llm_error",
                        "error": str(exc)[:1000],
                    }
                )
                continue
            validation = validate_hypothesis(
                draft,
                target=self.context.target,
                dataset_columns=self.context.dataset_columns,
                known_factor_ids=self.context.factor_ids,
                known_source_ids=self.context.source_ids,
            )
            if not validation.ok:
                self._remember(family, history,
                    {
                        "attempt": calls,
                        "status": "static_invalid",
                        "error": "; ".join(validation.errors)[:1500],
                    }
                )
                feedback = repair.static_feedback(validation.errors)
                continue
            result = self._execute_draft(family, draft, validation, history, calls)
            if isinstance(result, GeneratedHypothesis):
                if not self._distinct(draft.code):
                    self._remember(family, history,
                        {
                            "attempt": calls,
                            "status": "duplicate",
                            "error": "code and normalized AST are too similar",
                        }
                    )
                    continue
                self._remember(family, history, {"attempt": calls, "status": "ok", "error": None})
                result.generation_history = history
                result.validation_warnings = list(validation.warnings)
                self.warnings.extend(validation.warnings)
                self._accepted.append(result)
                self._remember_vector(draft.code)
                return result
            self._remember(
                family,
                history,
                {
                    "attempt": calls,
                    "status": result.status,
                    "error": (result.error or "")[:1500],
                },
            )
            feedback = repair.execution_feedback(
                result.status, result.error, result.traceback
            )

        self.warnings.append(
            f"hypothesis_family_failed: {family} after {calls} attempt(s)"
        )
        return None

    def _execute_draft(
        self,
        family: MethodFamily,
        draft: HypothesisDraft,
        validation: MetadataValidation,
        history: list[dict[str, Any]],
        attempt: int,
    ) -> GeneratedHypothesis | ExecutionResult:
        code_sha256 = hashlib.sha256(draft.code.encode("utf-8")).hexdigest()
        hyp_id = hypothesis_id(draft.code, draft.factors_used)
        draft.target = self.context.target
        hypothesis_dir = self.run_dir / "hypotheses"
        hypothesis_dir.mkdir(parents=True, exist_ok=True)
        module_path = hypothesis_dir / f"{hyp_id}.py"
        module_path.write_text(draft.code, encoding="utf-8")

        execution = run_module(
            module_path,
            self.dataset_path,
            self.seed,
            run_dir=self.run_dir,
        )
        if not execution.ok:
            return execution
        return GeneratedHypothesis(
            id=hyp_id,
            method_family=family,
            draft=draft,
            code_sha256=code_sha256,
            code_path=f"hypotheses/{hyp_id}.py",
            execution=execution,
            generation_history=history,
            validation_warnings=list(validation.warnings),
        )

    def _remember_vector(self, code: str) -> None:
        vector: list[float] | None = None
        try:
            vectors = self.embeddings.embed([code])
            if vectors:
                vector = vectors[0]
        except Exception:  # noqa: BLE001 - distinctness falls back to the AST
            vector = None
        self._accepted_vectors.append(vector)

    def _distinct(self, code: str) -> bool:
        signature = _normalized_signature(code)
        vector: list[float] | None = None
        try:
            vectors = self.embeddings.embed([code])
            if vectors:
                vector = vectors[0]
        except Exception:  # noqa: BLE001
            vector = None
        for existing, existing_vector in zip(self._accepted, self._accepted_vectors):
            if _normalized_signature(existing.draft.code) != signature:
                continue
            if vector is None or existing_vector is None:
                # Without both vectors the structural match is enough.
                return False
            try:
                similarity = cosine(vector, existing_vector)
            except ValueError:
                similarity = 0.0
            if similarity >= 0.95:
                return False
        return True
