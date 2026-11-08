"""Static validation for generated hypothesis modules.

The validator is deliberately conservative: generated code may use only the
pinned numerical stack and a small standard-library allowlist, and it must not
read files, reach the network, or touch the target column on the feature side.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

ALLOWED_IMPORTS = {
    "__future__",
    "collections",
    "datetime",
    "functools",
    "itertools",
    "math",
    "numpy",
    "pandas",
    "re",
    "scipy",
    "statistics",
    "warnings",
}

BANNED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

BANNED_METHODS = {
    "fromfile",
    "load",
    "loadmat",
    "loadtxt",
    "memmap",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_json",
    "read_parquet",
    "read_pickle",
    "read_sql",
    "read_table",
    "save",
    "savetxt",
    "savez",
    "savez_compressed",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
    "tofile",
}

MAX_CODE_BYTES = 20 * 1024

_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_SOCKET_RE = re.compile(r"\bsocket\b", re.IGNORECASE)
_PATH_RE = re.compile(
    r"(?:^[A-Za-z]:[\\/]|^\\\\|^/|^\.\.?[\\/]|\.(?:csv|tsv|parquet|json|pkl|pickle|xlsx?|py)$)",
    re.IGNORECASE,
)

_TARGET_VARIABLE_NAMES = {
    "y",
    "y_true",
    "y_actual",
    "target",
    "targets",
    "actual",
    "actuals",
    "outcome",
    "response",
    "label",
    "labels",
    "truth",
    "observed",
    "events",
}

_STACKING_CALLS = {
    "array",
    "asarray",
    "column_stack",
    "concat",
    "concatenate",
    "hstack",
    "stack",
    "vstack",
}


@dataclass
class StaticIssue:
    rule: str
    message: str
    line: int | None = None


@dataclass
class StaticValidation:
    ok: bool = True
    issues: list[StaticIssue] = field(default_factory=list)
    code_bytes: int = 0

    @property
    def messages(self) -> list[str]:
        return [
            f"{issue.rule}: {issue.message}"
            + (f" (line {issue.line})" if issue.line is not None else "")
            for issue in self.issues
        ]


class _Validator(ast.NodeVisitor):
    def __init__(self, code: str, *, target: str | None) -> None:
        self.code = code
        self.target = target
        self.issues: list[StaticIssue] = []
        self._functions: dict[str, ast.FunctionDef] = {}

    def _add(self, rule: str, message: str, node: ast.AST | None = None) -> None:
        line = getattr(node, "lineno", None) if node is not None else None
        self.issues.append(StaticIssue(rule=rule, message=message, line=line))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self._add(
                    "import",
                    f"import {alias.name!r} is outside the allowlist",
                    node,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._add("import", "relative imports are not allowed", node)
            return
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS:
            self._add(
                "import",
                f"from {node.module!r} import is outside the allowlist",
                node,
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in BANNED_CALLS:
            self._add("call", f"call to {name!r} is not allowed", node)
        if name in BANNED_METHODS:
            self._add("io", f"file access through {name!r} is not allowed", node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self._add(
                "attribute",
                f"dunder attribute {node.attr!r} is not allowed",
                node,
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            if _URL_RE.search(node.value):
                self._add("literal", "URL string literals are not allowed", node)
            if _SOCKET_RE.search(node.value):
                self._add(
                    "literal", "network-related string literals are not allowed", node
                )
            if _PATH_RE.search(node.value.strip()):
                self._add("literal", "file path string literals are not allowed", node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._functions[node.name] = node
        if not node.name.startswith("_") and node.name not in {"applies", "run"}:
            self._add(
                "function",
                f"public helper {node.name!r} is not part of the contract",
                node,
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add("function", "async functions are not allowed", node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add("statement", "class definitions are not allowed", node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.generic_visit(node)


def _module_statements_allowed(tree: ast.Module) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    for index, node in enumerate(tree.body):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
            ),
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if index == 0 and isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None:
                continue
            if isinstance(
                value,
                (
                    ast.Constant,
                    ast.Tuple,
                    ast.List,
                    ast.Dict,
                    ast.Set,
                    ast.UnaryOp,
                    ast.BinOp,
                    ast.Name,
                    ast.Attribute,
                ),
            ):
                continue
            issues.append(
                StaticIssue(
                    "statement",
                    "module-level assignments may only use constants or aliases",
                    getattr(node, "lineno", None),
                )
            )
            continue
        issues.append(
            StaticIssue(
                "statement",
                f"module-level {type(node).__name__} is not allowed",
                getattr(node, "lineno", None),
            )
        )
    return issues


def _argument_names(node: ast.FunctionDef) -> list[str]:
    return [argument.arg for argument in node.args.posonlyargs + node.args.args]


def _check_signatures(tree: ast.Module) -> list[StaticIssue]:
    issues: list[StaticIssue] = []
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    applies = functions.get("applies")
    run = functions.get("run")
    if applies is None:
        issues.append(StaticIssue("contract", "missing `applies(df)`", None))
    else:
        names = _argument_names(applies)
        if names != ["df"] or applies.args.vararg or applies.args.kwarg:
            issues.append(
                StaticIssue(
                    "contract",
                    "`applies` must have the signature `applies(df)`",
                    applies.lineno,
                )
            )
    if run is None:
        issues.append(StaticIssue("contract", "missing `run(df, seed=0)`", None))
    else:
        names = _argument_names(run)
        if len(names) < 2 or names[0] != "df" or names[1] != "seed":
            issues.append(
                StaticIssue(
                    "contract",
                    "`run` must have the signature `run(df, seed=0)`",
                    run.lineno,
                )
            )
        elif run.args.defaults:
            defaults = run.args.defaults
            last = defaults[-1]
            if not (
                isinstance(last, ast.Constant) and last.value == 0
            ):
                issues.append(
                    StaticIssue(
                        "contract",
                        "`run` must give `seed` a default value of 0",
                        run.lineno,
                    )
                )
        elif len(names) == 2:
            issues.append(
                StaticIssue(
                    "contract",
                    "`run` must give `seed` a default value of 0",
                    run.lineno,
                )
            )
    return issues


def validate_code(code: str, *, target: str | None = None) -> StaticValidation:
    """Parse and statically validate one generated module."""
    encoded = code.encode("utf-8")
    issues: list[StaticIssue] = []
    if not code.strip():
        return StaticValidation(
            ok=False,
            issues=[StaticIssue("code", "code is empty")],
            code_bytes=0,
        )
    if len(encoded) > MAX_CODE_BYTES:
        issues.append(
            StaticIssue(
                "code",
                f"code is {len(encoded)} bytes; the limit is {MAX_CODE_BYTES}",
            )
        )
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticValidation(
            ok=False,
            issues=issues
            + [
                StaticIssue(
                    "syntax",
                    f"{type(exc).__name__}: {exc.msg}",
                    exc.lineno,
                )
            ],
            code_bytes=len(encoded),
        )
    validator = _Validator(code, target=target)
    validator.visit(tree)
    issues.extend(validator.issues)
    issues.extend(_module_statements_allowed(tree))
    issues.extend(_check_signatures(tree))
    try:
        compile(code, "<hypothesis>", "exec")
    except SyntaxError as exc:
        issues.append(
            StaticIssue("syntax", f"{type(exc).__name__}: {exc.msg}", exc.lineno)
        )
    return StaticValidation(
        ok=not issues,
        issues=issues,
        code_bytes=len(encoded),
    )


def _is_df_target(node: ast.AST, target: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "df"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == target
    )


def _contains_df_target(node: ast.AST, target: str) -> bool:
    return any(_is_df_target(child, target) for child in ast.walk(node))


def target_feature_references(code: str, target: str) -> list[int]:
    """Return line numbers where the target is plausibly used as a feature.

    The check is intentionally heuristic. It catches the common generated-code
    mistakes: assigning the target to a feature variable, stacking it into a
    design matrix, and selecting it in a multi-column feature frame.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    suspicious: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if _contains_df_target(node.value, target):
                for assigned in node.targets:
                    if isinstance(assigned, ast.Name) and (
                        assigned.id.lower() not in _TARGET_VARIABLE_NAMES
                    ):
                        suspicious.add(getattr(node, "lineno", 0))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _STACKING_CALLS:
                if any(_contains_df_target(argument, target) for argument in node.args):
                    suspicious.add(getattr(node, "lineno", 0))
            if node.func.attr == "fit" and node.args:
                if any(_contains_df_target(argument, target) for argument in node.args):
                    suspicious.add(getattr(node, "lineno", 0))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.List):
            if any(
                isinstance(item, ast.Constant) and item.value == target
                for item in node.slice.elts
            ):
                suspicious.add(getattr(node, "lineno", 0))
    return sorted(suspicious)


def _parameter_names(node: ast.FunctionDef) -> list[str]:
    return [argument.arg for argument in node.args.posonlyargs + node.args.args]


def _validation_surface_warnings(
    code: str, target: str | None
) -> list[str]:
    """Check the optional fit/predict surface used by statistical validation.

    Missing helpers are a warning, not a contract error: legacy modules still
    execute. The validator then reports partial or in-sample results.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    warnings: list[str] = []
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    fit = functions.get("_fit")
    predict = functions.get("_predict")
    if fit is None:
        warnings.append(
            "validation_contract: missing `_fit(df, seed=0, params=None)`; "
            "out-of-sample and sensitivity layers will be skipped"
        )
    else:
        names = _parameter_names(fit)
        if not names or names[0] != "df":
            warnings.append(
                "validation_contract: `_fit` must take the frame as its first argument"
            )
    if predict is None:
        warnings.append(
            "validation_contract: missing `_predict(model, df)`; "
            "out-of-sample and sensitivity layers will be skipped"
        )
    else:
        names = _parameter_names(predict)
        if len(names) < 2 or names[1] != "df":
            warnings.append(
                "validation_contract: `_predict` must have the signature "
                "`_predict(model, df)`"
            )
    parameters_node = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(item, ast.Name) and item.id == "PARAMETERS"
                for item in targets
            ):
                parameters_node = node.value
                break
    if parameters_node is None:
        warnings.append(
            "validation_contract: missing module-level `PARAMETERS` dict; "
            "the parameter-sensitivity layer will be skipped"
        )
    elif not isinstance(parameters_node, ast.Dict) or not parameters_node.keys:
        warnings.append(
            "validation_contract: `PARAMETERS` must be a non-empty dict of "
            "numeric model hyperparameters"
        )
    elif not any(
        isinstance(value, ast.Constant)
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and value.value > 0
        for value in parameters_node.values
    ):
        warnings.append(
            "validation_contract: `PARAMETERS` needs at least one positive "
            "numeric value so the sensitivity layer is non-degenerate"
        )
    if predict is not None and target:
        for child in ast.walk(predict):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.slice, ast.Constant)
                and child.slice.value == target
                and isinstance(child.value, ast.Name)
            ):
                warnings.append(
                    "validation_contract: `_predict` references the target column "
                    f"{target!r}; prediction-time frames omit it"
                )
                break
    return warnings


@dataclass
class MetadataValidation:
    ok: bool
    static: StaticValidation
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_hypothesis(
    draft: Any,
    *,
    target: str | None,
    dataset_columns: Sequence[str],
    known_factor_ids: Iterable[str] = (),
    known_source_ids: Iterable[str] = (),
) -> MetadataValidation:
    """Validate a generated draft's code and metadata before execution."""
    errors: list[str] = []
    warnings: list[str] = []
    static = validate_code(draft.code, target=target)
    if not static.ok:
        errors.extend(static.messages)
    warnings.extend(_validation_surface_warnings(draft.code, target))

    if not str(getattr(draft, "rationale", "")).strip():
        errors.append("rationale must not be empty")
    assumptions = [
        str(item).strip()
        for item in getattr(draft, "assumptions", [])
        if str(item).strip()
    ]
    if not assumptions:
        errors.append("at least one assumption is required")
    elif any(
        item.lower() in {"all assumptions met", "none", "n/a"}
        or len(item.split()) < 3
        for item in assumptions
    ):
        warnings.append("assumptions are generic or one-word")

    columns = set(str(column) for column in dataset_columns)
    required = [str(column) for column in getattr(draft, "required_columns", [])]
    optional = [str(column) for column in getattr(draft, "optional_columns", [])]
    if target is not None and target in required + optional:
        errors.append(
            f"required_columns/optional_columns are feature columns and must not "
            f"include the target column {target!r}; the target is supplied separately"
        )
    missing_required = [column for column in required if column not in columns]
    if missing_required:
        errors.append(
            "required columns are not in the dataset: " + ", ".join(missing_required)
        )
    unknown_optional = [column for column in optional if column not in columns]
    if unknown_optional:
        warnings.append(
            "optional columns are not in the dataset: " + ", ".join(unknown_optional)
        )

    known_factors = set(known_factor_ids)
    factors_used = [str(item) for item in getattr(draft, "factors_used", [])]
    unknown_factors = [item for item in factors_used if item not in known_factors]
    if unknown_factors:
        errors.append(
            "factors_used contains ids outside this run: "
            + ", ".join(unknown_factors)
        )
    if not factors_used:
        rationale = str(getattr(draft, "rationale", "")).lower()
        if not any(
            phrase in rationale
            for phrase in (
                "no factor",
                "no applicable factor",
                "none of the factors",
                "domain knowledge",
                "no mapped factor",
            )
        ):
            errors.append(
                "when no factor is used, the rationale must say so explicitly"
            )

    known_sources = set(known_source_ids)
    sources = [str(item) for item in getattr(draft, "source_ids", [])]
    if factors_used and not sources:
        errors.append("at least one source id is required when factors are used")
    unknown_sources = [item for item in sources if item not in known_sources]
    if unknown_sources:
        errors.append(
            "source_ids contains ids outside this run: " + ", ".join(unknown_sources)
        )

    min_rows = int(getattr(draft, "min_rows", 1) or 1)
    if min_rows <= 0:
        errors.append("min_rows must be positive")

    return MetadataValidation(
        ok=not errors,
        static=static,
        errors=errors,
        warnings=warnings,
    )
