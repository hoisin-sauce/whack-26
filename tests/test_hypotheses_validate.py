from __future__ import annotations

from types import SimpleNamespace

from whack26.hypotheses.validate import (
    target_feature_references,
    validate_code,
    validate_hypothesis,
)

VALID_CODE = '''import numpy as np
import pandas as pd

MIN_ROWS = 10
PARAMETERS = {"ridge": 1e-3}


def applies(df: pd.DataFrame) -> bool:
    """Run on any frame with the two required columns."""
    return {"x", "y"}.issubset(set(df.columns))


def _fit(df: pd.DataFrame, seed: int = 0, params: dict | None = None):
    """Fit a simple least-squares line without mutating the input."""
    np.random.seed(seed)
    df = df.copy()
    x = df["x"].fillna(df["x"].mean()).to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"coefficients": coefficients.tolist()}


def _predict(model, df: pd.DataFrame):
    """Predict from feature columns only."""
    x = df["x"].fillna(0.0).to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    return design @ np.asarray(model["coefficients"], dtype=float)


def run(df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit the model and return predictions without mutating the input."""
    model = _fit(df, seed, PARAMETERS)
    return {
        "predictions": _predict(model, df.drop(columns=["y"])),
        "target": "y",
        "model_summary": {"coefficients": model["coefficients"]},
        "notes": "Missing x values were mean-imputed.",
    }
'''


def make_draft(**overrides):
    values = {
        "title": "Income predicts price",
        "rationale": "Higher income is associated with higher prices.",
        "assumptions": ["The relationship is approximately linear."],
        "required_columns": ["x"],
        "optional_columns": [],
        "min_rows": 10,
        "claim_type": "prediction",
        "factors_used": ["fac_1"],
        "source_ids": ["openalex:W1"],
        "code": VALID_CODE,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_valid_code_passes() -> None:
    result = validate_code(VALID_CODE, target="y")
    assert result.ok, result.messages


def test_banned_import_is_rejected() -> None:
    code = "import sklearn\n\n" + VALID_CODE
    result = validate_code(code, target="y")
    assert not result.ok
    assert any("sklearn" in message for message in result.messages)


def test_banned_call_is_rejected() -> None:
    code = VALID_CODE.replace(
        "    np.random.seed(seed)", "    eval('1 + 1')\n    np.random.seed(seed)"
    )
    result = validate_code(code)
    assert not result.ok
    assert any("eval" in message for message in result.messages)


def test_file_path_literal_is_rejected() -> None:
    code = VALID_CODE.replace(
        '    df = df.copy()', '    df = df.copy()\n    name = "data/secret.csv"'
    )
    result = validate_code(code)
    assert not result.ok
    assert any("file path" in message for message in result.messages)


def test_missing_contract_function_is_rejected() -> None:
    code = VALID_CODE.replace("def applies(", "def not_applies(")
    result = validate_code(code)
    assert not result.ok
    assert any("applies" in message for message in result.messages)


def test_module_level_statement_is_rejected() -> None:
    code = VALID_CODE.replace("MIN_ROWS = 10", "MIN_ROWS = 10\nprint(MIN_ROWS)")
    result = validate_code(code)
    assert not result.ok
    assert any("module-level" in message for message in result.messages)


def test_target_feature_reference_detects_stacked_target() -> None:
    code = VALID_CODE.replace(
        'design = np.column_stack([np.ones(len(x)), x])',
        'design = np.column_stack([np.ones(len(x)), x, df["y"]])',
    )
    assert target_feature_references(code, "y")


def test_target_feature_reference_allows_target_outcome_variable() -> None:
    assert target_feature_references(VALID_CODE, "y") == []


def test_metadata_accepts_cited_factor_and_source() -> None:
    result = validate_hypothesis(
        make_draft(),
        target="y",
        dataset_columns=["x", "y"],
        known_factor_ids=["fac_1"],
        known_source_ids=["openalex:W1"],
    )
    assert result.ok, result.errors
    assert result.warnings == []


def test_validation_surface_warns_when_predict_reads_target() -> None:
    code = VALID_CODE.replace(
        'x = df["x"].fillna(0.0)', 'x = df["y"].fillna(0.0)'
    )
    result = validate_hypothesis(
        make_draft(code=code),
        target="y",
        dataset_columns=["x", "y"],
        known_factor_ids=["fac_1"],
        known_source_ids=["openalex:W1"],
    )

    assert result.ok, result.errors
    assert any("references the target" in item for item in result.warnings)


def test_missing_validation_surface_is_a_warning_not_an_error() -> None:
    code = VALID_CODE.replace("PARAMETERS = {\"ridge\": 1e-3}\n", "")

    result = validate_hypothesis(
        make_draft(code=code),
        target="y",
        dataset_columns=["x", "y"],
        known_factor_ids=["fac_1"],
        known_source_ids=["openalex:W1"],
    )

    assert result.ok, result.errors
    assert any("PARAMETERS" in item for item in result.warnings)


def test_metadata_rejects_target_in_required_columns() -> None:
    result = validate_hypothesis(
        make_draft(required_columns=["x", "y"]),
        target="y",
        dataset_columns=["x", "y"],
        known_factor_ids=["fac_1"],
        known_source_ids=["openalex:W1"],
    )
    assert not result.ok
    assert any("target column" in error for error in result.errors)


def test_metadata_rejects_unknown_factor_id() -> None:
    result = validate_hypothesis(
        make_draft(factors_used=["fac_unknown"]),
        target="y",
        dataset_columns=["x", "y"],
        known_factor_ids=["fac_1"],
        known_source_ids=["openalex:W1"],
    )
    assert not result.ok
    assert any("outside this run" in error for error in result.errors)


def test_metadata_requires_factor_justification_when_none_used() -> None:
    result = validate_hypothesis(
        make_draft(factors_used=[], source_ids=[], rationale="A simple baseline."),
        target="y",
        dataset_columns=["x", "y"],
    )
    assert not result.ok
    assert any("no factor" in error.lower() for error in result.errors)


def test_metadata_allows_explicit_no_factor_justification() -> None:
    result = validate_hypothesis(
        make_draft(
            factors_used=[],
            source_ids=[],
            rationale="No factor applies; the baseline is justified from domain knowledge.",
        ),
        target="y",
        dataset_columns=["x", "y"],
    )
    assert result.ok, result.errors
