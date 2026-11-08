"""Isolated child-process runner for one hypothesis's statistical validation.

Invocation:

    python -I runner.py <config.json> <report.json>

The runner imports the generated module, reconstructs a numeric feature
matrix, and calls the four-layer engine in ``stats_engine.py``. It writes one
JSON report to ``report.json`` and never mixes generated-code stdout into the
report.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import math
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# ``python -I`` removes the script directory from sys.path. The validation
# child needs the repository root for ``stats_engine`` and the package root for
# whack26 modules, so add it explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats_engine import (  # noqa: E402
    ENGINE_PARAM_KEYS,
    dsr_dynamic,
    monte_carlo,
    oos_decay,
    pss,
    r2_score,
    trial_sharpes,
    validation_engine,
)

from whack26.validation import engine_bridge  # noqa: E402
from whack26.validation.adapter import (  # noqa: E402
    AdapterSkip,
    FeatureMatrix,
    align_predictions,
    build_feature_matrix,
)
from whack26.validation.models import (  # noqa: E402
    ValidationConfig,
    ValidationLayerReport,
    ValidationReport,
)

sys.dont_write_bytecode = True


def _block_network() -> None:
    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is blocked in the validation runner")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _normalise_predictions(result: Any, expected: int) -> np.ndarray:
    if isinstance(result, dict):
        result = result.get("predictions")
    if isinstance(result, pd.Series):
        values = result.to_numpy(dtype=float)
    elif isinstance(result, np.ndarray):
        values = result.astype(float).ravel()
    elif isinstance(result, (list, tuple)):
        values = np.asarray(result, dtype=float).ravel()
    else:
        raise TypeError(
            "`_predict` must return a list, pandas Series, or numpy array; "
            f"got {type(result).__name__}"
        )
    if values.shape[0] != expected:
        raise ValueError(
            f"`_predict` returned {values.shape[0]} predictions for {expected} rows"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("`_predict` returned non-finite predictions")
    return values


def _call_fit(
    module: Any,
    frame: pd.DataFrame,
    seed: int,
    params: dict[str, Any],
) -> Any:
    fit = module._fit
    signature = inspect.signature(fit)
    parameters = signature.parameters
    has_var_kwargs = any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "seed" in parameters or has_var_kwargs:
        kwargs["seed"] = seed
    if "params" in parameters or has_var_kwargs:
        kwargs["params"] = params
    if kwargs:
        return fit(frame, **kwargs)
    positional = [
        item
        for item in parameters.values()
        if item.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) <= 1:
        return fit(frame)
    if len(positional) == 2:
        return fit(frame, seed)
    return fit(frame, seed, params)


def _call_predict(module: Any, model: Any, frame: pd.DataFrame) -> np.ndarray:
    predict = module._predict
    signature = inspect.signature(predict)
    parameters = signature.parameters
    if len(parameters) >= 2 or any(
        item.kind is inspect.Parameter.VAR_POSITIONAL
        for item in parameters.values()
    ):
        result = predict(model, frame)
    else:  # pragma: no cover - the contract requires two arguments
        result = predict(frame)
    return _normalise_predictions(result, len(frame))


def _module_parameters(module: Any) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    raw = getattr(module, "PARAMETERS", {})
    if not isinstance(raw, dict):
        warnings.append("PARAMETERS is not a dict; no model hyperparameters were used")
        return {}, warnings
    parameters: dict[str, Any] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            warnings.append(f"PARAMETERS[{name!r}] is not numeric and was ignored")
            continue
        if not _finite(value):
            warnings.append(f"PARAMETERS[{name!r}] is non-finite and was ignored")
            continue
        parameters[str(name)] = value
    return parameters, warnings


def _perturbable(parameters: dict[str, Any]) -> dict[str, Any]:
    """Drop zero-valued floats that ``pss`` would perturb to themselves."""
    return {
        name: value
        for name, value in parameters.items()
        if not (isinstance(value, float) and value == 0.0)
    }


def _layer(
    name: str,
    status: str,
    reason: str | None = None,
    **metrics: Any,
) -> ValidationLayerReport:
    return ValidationLayerReport(
        layer=name,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        metrics=_json_safe(metrics),
    )


def _finish(report: ValidationReport) -> ValidationReport:
    statuses = [layer.status for layer in report.layers]
    if not statuses:
        return report
    if "error" in statuses:
        report.status = "error"
        report.verdict = "inconclusive"
    elif "failed" in statuses:
        report.status = "failed"
        report.verdict = "fail"
    elif statuses and all(status == "passed" for status in statuses):
        report.status = "passed"
        report.verdict = "pass"
    elif any(status == "passed" for status in statuses):
        report.status = "partial"
        report.verdict = "inconclusive"
    else:
        report.status = "skipped"
        report.verdict = "inconclusive"
    return report


def _copy_layer_metrics(report: ValidationReport) -> None:
    for layer in report.layers:
        if report.dsr_score is None and layer.metrics.get("dsr_score") is not None:
            report.dsr_score = _json_safe(layer.metrics["dsr_score"])
        if report.p_value is None and layer.metrics.get("p_value") is not None:
            report.p_value = _json_safe(layer.metrics["p_value"])
        if report.oos_score is None and layer.metrics.get("oos_score") is not None:
            report.oos_score = _json_safe(layer.metrics["oos_score"])
        if report.decay_rate is None and layer.metrics.get("decay_rate") is not None:
            report.decay_rate = _json_safe(layer.metrics["decay_rate"])
        if report.tolerance is None and layer.metrics.get("tolerance") is not None:
            report.tolerance = _json_safe(layer.metrics["tolerance"])
        if layer.layer.startswith("Layer 4") and layer.metrics.get("message"):
            report.pss_message = str(layer.metrics["message"])


def _engine_layers(engine: dict[str, Any]) -> list[ValidationLayerReport]:
    names = [
        "Layer 1 (DSR)",
        "Layer 2 (Monte Carlo)",
        "Layer 3 (OOS Decay)",
        "Layer 4 (Parameter Sensitivity)",
    ]
    metrics = [
        {"dsr_score": engine.get("dsr_score")},
        {"p_value": engine.get("p_value")},
        {
            "oos_score": engine.get("oos_score"),
            "decay_rate": engine.get("decay_rate"),
            "tolerance": engine.get("tolerance"),
        },
        {"message": engine.get("message")} if engine.get("passed") else {},
    ]
    metrics = [
        {key: value for key, value in item.items() if value is not None}
        for item in metrics
    ]
    if engine.get("passed"):
        return [
            _layer(name, "passed", **metrics[index])
            for index, name in enumerate(names)
        ]
    failed = str(engine.get("failed_layer") or "")
    layers: list[ValidationLayerReport] = []
    stopped = False
    for index, name in enumerate(names):
        if stopped:
            layers.append(_layer(name, "skipped", "engine stopped at an earlier layer"))
        elif name == failed:
            layers.append(
                _layer(
                    name,
                    "failed",
                    str(engine.get("message") or "layer failed"),
                    **metrics[index],
                )
            )
            stopped = True
        else:
            layers.append(_layer(name, "passed"))
    return layers


_BRIDGE_TARGET_KINDS = frozenset({"binary", "multiclass", "count"})
_LEGACY_TARGET_KINDS = frozenset({"continuous", "time_series"})


def _bridge_engine(
    *,
    matrix: FeatureMatrix,
    config: ValidationConfig,
    description: str | None,
    base_params: dict[str, Any],
) -> dict[str, Any]:
    """Validate a non-continuous matrix with the general ``stats_engine`` API.

    The bridge returns project layer reports already mapped, so they are kept
    under a ``layers`` list (the legacy engine result keeps ``layers`` as a
    dict, which the caller translates with :func:`_engine_layers`). Bridge
    failures are reported as an error layer rather than raised: validation is
    advisory and the parent process must still receive a report.
    """
    try:
        result = engine_bridge.run_engine(
            matrix=matrix,
            config=config,
            description=description,
            base_params=base_params,
        )
        layers = engine_bridge.engine_result_to_layers(result)
    except Exception as exc:  # noqa: BLE001 - bridge failures become error reports
        message = f"{type(exc).__name__}: {exc}"
        return {
            "passed": False,
            "error": message,
            "message": message,
            "layers": [_layer("Validation", "error", message)],
        }
    outcome = {
        key: result.get(key)
        for key in (
            "passed",
            "message",
            "dsr_score",
            "p_value",
            "oos_score",
            "decay_rate",
            "tolerance",
        )
    }
    outcome["layers"] = layers
    return outcome


def _apply_engine_outcome(
    report: ValidationReport, engine: dict[str, Any]
) -> ValidationReport:
    """Copy a full-engine outcome onto ``report``.

    Legacy outcomes carry the engine's result dict shape and need
    :func:`_engine_layers`. Bridge outcomes already carry a list of project
    layer reports, identified by ``layers`` being a list instead of a dict.
    """
    layers = engine.get("layers")
    if isinstance(layers, list):
        report.layers = layers
        report.pss_message = str(engine.get("message") or "")
        if engine.get("error"):
            report.error = str(engine["error"])
    else:
        report.layers = _engine_layers(engine)
        report.pss_message = (
            "All 4 validation layers cleared successfully."
            if engine.get("passed")
            else str(engine.get("message") or "")
        )
    report.dsr_score = _json_safe(engine.get("dsr_score"))
    report.p_value = _json_safe(engine.get("p_value"))
    report.oos_score = _json_safe(engine.get("oos_score"))
    report.decay_rate = _json_safe(engine.get("decay_rate"))
    report.tolerance = _json_safe(engine.get("tolerance"))
    return report


def _full_engine(
    *,
    builder: Callable[[dict[str, Any], np.ndarray, np.ndarray], Callable],
    matrix: FeatureMatrix,
    base_params: dict[str, Any],
    config: ValidationConfig,
    description: str | None,
) -> dict[str, Any]:
    """Run the four-layer engine for this matrix's target kind.

    ``binary``, ``multiclass``, and ``count`` targets are routed to the
    general engine through :mod:`whack26.validation.engine_bridge`; the
    legacy ``validation_engine`` only understands continuous return-like
    series. Unsupported kinds are skipped rather than forced through either
    engine.
    """
    kind = str(getattr(matrix, "target_kind", "continuous") or "continuous")
    kind = kind.strip().lower()
    if kind in _BRIDGE_TARGET_KINDS:
        return _bridge_engine(
            matrix=matrix,
            config=config,
            description=description,
            base_params=base_params,
        )
    if kind not in _LEGACY_TARGET_KINDS:
        message = f"unsupported target kind {kind!r}"
        return {
            "passed": False,
            "message": message,
            "layers": [_layer("Target contract", "skipped", message)],
        }

    split = int(matrix.rows * config.train_split_ratio)
    if split <= 0 or split >= matrix.rows:
        raise ValueError("train_split_ratio produced an empty train or test split")
    model = builder(base_params, matrix.X[:split], matrix.y[:split])
    train_predictions = model(matrix.X[:split])
    test_predictions = model(matrix.X[split:])
    sharpes, winning_returns = trial_sharpes(
        matrix.X[:split],
        matrix.y[:split],
        matrix.periods_per_year,
        model_predict=model,
    )
    if len(sharpes) < 2 or winning_returns.size == 0:
        raise ValueError("fewer than two real trials were available for Layer 1")
    return validation_engine(
        winning_returns=winning_returns,
        all_trial_sharpes=sharpes,
        predictions=test_predictions,
        actuals=matrix.y[split:],
        features=matrix.X,
        targets=matrix.y,
        base_model_builder=builder,
        base_params=base_params,
        periods_per_year=matrix.periods_per_year,
        metric_func=r2_score,
        higher_is_better=True,
        data_description=description,
    )


def _manual_layers(
    *,
    matrix: FeatureMatrix,
    config: ValidationConfig,
    predictions: np.ndarray,
    builder: Callable[[dict[str, Any], np.ndarray, np.ndarray], Callable] | None,
    base_params: dict[str, Any],
    has_perturbable: bool,
    in_sample: bool,
    warnings: list[str],
) -> list[ValidationLayerReport]:
    layers: list[ValidationLayerReport] = []
    split = int(matrix.rows * config.train_split_ratio)

    # Layer 1: real trial Sharpes and the DSR multiple-testing penalty.
    try:
        if in_sample:
            sharpes, winning_returns = trial_sharpes(
                matrix.X,
                matrix.y,
                matrix.periods_per_year,
                model_predict=lambda _X: predictions,
            )
        else:
            assert builder is not None
            model = builder(base_params, matrix.X[:split], matrix.y[:split])
            sharpes, winning_returns = trial_sharpes(
                matrix.X[:split],
                matrix.y[:split],
                matrix.periods_per_year,
                model_predict=model,
            )
        if len(sharpes) < 2 or winning_returns.size == 0:
            layers.append(
                _layer(
                    "Layer 1 (DSR)",
                    "skipped",
                    "at least two real trials are required",
                )
            )
        else:
            score = dsr_dynamic(
                winning_returns, sharpes, matrix.periods_per_year
            )
            passed = _finite(score) and score >= config.min_dsr
            layers.append(
                _layer(
                    "Layer 1 (DSR)",
                    "passed" if passed else "failed",
                    None if passed else f"DSR {score:.4f} < {config.min_dsr}",
                    dsr_score=score,
                    trials=len(sharpes),
                )
            )
    except Exception as exc:  # noqa: BLE001
        layers.append(_layer("Layer 1 (DSR)", "error", f"{type(exc).__name__}: {exc}"))

    # Layer 2: permutation test.
    try:
        if in_sample:
            actuals = matrix.y
            scored_predictions = predictions
        else:
            actuals = matrix.y[split:]
            assert builder is not None
            model = builder(base_params, matrix.X[:split], matrix.y[:split])
            scored_predictions = model(matrix.X[split:])
        p_value = monte_carlo(
            scored_predictions,
            actuals,
            config.num_permutations,
            metric_func=r2_score,
            higher_is_better=True,
        )
        passed = _finite(p_value) and p_value <= config.max_p
        layers.append(
            _layer(
                "Layer 2 (Monte Carlo)",
                "passed" if passed else "failed",
                None if passed else f"p-value {p_value:.4f} > {config.max_p}",
                p_value=p_value,
                in_sample=in_sample,
            )
        )
        if in_sample:
            warnings.append(
                "layer2_in_sample: saved predictions were fitted on the full dataset"
            )
    except Exception as exc:  # noqa: BLE001
        layers.append(
            _layer("Layer 2 (Monte Carlo)", "error", f"{type(exc).__name__}: {exc}")
        )

    # Layer 3: chronological OOS decay.
    if builder is None or in_sample:
        layers.append(
            _layer(
                "Layer 3 (OOS Decay)",
                "skipped",
                "fit/predict helpers are required for out-of-sample evaluation",
            )
        )
    else:
        try:
            result = oos_decay(
                builder,
                matrix.X,
                matrix.y,
                base_params,
                train_split_ratio=config.train_split_ratio,
                metric_func=r2_score,
                higher_is_better=True,
            )
            if not result.get("success"):
                layers.append(
                    _layer(
                        "Layer 3 (OOS Decay)",
                        "failed",
                        str(result.get("error") or "OOS decay failed"),
                        **{
                            key: result.get(key)
                            for key in ("oos_score", "decay_rate", "tolerance")
                            if result.get(key) is not None
                        },
                    )
                )
            else:
                layers.append(
                    _layer(
                        "Layer 3 (OOS Decay)",
                        "passed",
                        **{
                            key: result.get(key)
                            for key in ("oos_score", "decay_rate", "tolerance")
                            if result.get(key) is not None
                        },
                    )
                )
        except Exception as exc:  # noqa: BLE001
            layers.append(
                _layer("Layer 3 (OOS Decay)", "error", f"{type(exc).__name__}: {exc}")
            )

    # Layer 4: parameter sensitivity.
    if builder is None or in_sample or not has_perturbable:
        reason = (
            "fit/predict helpers are required"
            if builder is None or in_sample
            else "no non-degenerate numeric hyperparameter was declared"
        )
        layers.append(_layer("Layer 4 (Parameter Sensitivity)", "skipped", reason))
    else:
        try:
            passed, message = pss(
                builder,
                matrix.X,
                matrix.y,
                base_params,
                sensitivity_range=config.sensitivity_range,
                max_performance_drop=config.max_performance_drop,
                metric_func=r2_score,
                higher_is_better=True,
            )
            layers.append(
                _layer(
                    "Layer 4 (Parameter Sensitivity)",
                    "passed" if passed else "failed",
                    None if passed else message,
                    message=message,
                )
            )
        except Exception as exc:  # noqa: BLE001
            layers.append(
                _layer(
                    "Layer 4 (Parameter Sensitivity)",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return layers


def _load_module(path: Path) -> Any:
    module_name = f"_whack26_validation_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load a module from {path.name!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate(payload: dict[str, Any]) -> ValidationReport:
    module_path = Path(payload["module_path"])
    dataset_path = Path(payload["dataset_path"])
    predictions_path = payload.get("predictions_path")
    target = payload.get("target")
    config = ValidationConfig(**payload.get("config", {}))
    hypothesis_id = str(payload.get("hypothesis_id") or module_path.stem)

    report = ValidationReport(
        hypothesis_id=hypothesis_id,
        target=target,
        config=config.as_dict(),
        periods_per_year=config.periods_per_year or 1.0,
    )

    frame = pd.read_csv(dataset_path)
    module = _load_module(module_path)
    applies = getattr(module, "applies", None)
    if not callable(applies):
        return _finish(
            report.model_copy(
                update={
                    "status": "error",
                    "error": "the module does not define callable applies(df)",
                }
            )
        )
    if not bool(applies(frame.copy())):
        return _finish(
            report.model_copy(
                update={
                    "status": "skipped",
                    "layers": [
                        _layer("Applicability", "skipped", "applies(df) returned False")
                    ],
                }
            )
        )

    if target is None:
        return _finish(
            report.model_copy(
                update={
                    "status": "skipped",
                    "layers": [
                        _layer(
                            "Target contract",
                            "skipped",
                            "no target column was defined for this hypothesis",
                        )
                    ],
                }
            )
        )

    try:
        matrix = build_feature_matrix(
            frame,
            target=target,
            required_columns=payload.get("required_columns", []),
            optional_columns=payload.get("optional_columns", []),
            time_column=payload.get("time_column"),
            period_override=config.periods_per_year,
            max_features=config.max_features,
            max_categories=config.max_categories,
            target_kind=payload.get("target_kind"),
        )
    except AdapterSkip as exc:
        return _finish(
            report.model_copy(
                update={
                    "status": "skipped",
                    "target": target,
                    "layers": [
                        _layer(
                            "Target contract"
                            if exc.reason.startswith("target_contract")
                            else "Validation",
                            "skipped",
                            f"{exc.reason}: {exc.message}",
                        )
                    ],
                }
            )
        )

    report = report.model_copy(
        update={
            "target": target,
            "rows_used": matrix.rows,
            "dropped_rows": matrix.dropped_rows,
            "feature_columns": matrix.feature_columns,
            "periods_per_year": matrix.periods_per_year,
            "periods_per_year_source": matrix.periods_per_year_source,
        }
    )

    parameters, parameter_warnings = _module_parameters(module)
    report.warnings.extend(parameter_warnings)
    engine_params = config.engine_params(_perturbable(parameters))
    has_fit = callable(getattr(module, "_fit", None))
    has_predict = callable(getattr(module, "_predict", None))
    has_fit_predict = has_fit and has_predict
    report.has_fit_predict = has_fit_predict
    if not has_fit_predict:
        report.in_sample_fallback = True
        report.warnings.append(
            "missing_fit_predict: statistical validation used saved in-sample "
            "predictions and skipped layers 3-4"
        )

    builder: Callable[[dict[str, Any], np.ndarray, np.ndarray], Callable] | None
    if has_fit_predict:

        def builder(
            params: dict[str, Any],
            X: np.ndarray,
            y: np.ndarray,
        ) -> Callable[[np.ndarray], np.ndarray]:
            model_params = {
                key: value
                for key, value in params.items()
                if key not in ENGINE_PARAM_KEYS
            }
            train = matrix.frame_from_matrix(X, y)
            model = _call_fit(module, train, config.seed, model_params)

            def predict(new_X: np.ndarray) -> np.ndarray:
                prediction_frame = matrix.frame_from_matrix(new_X)
                return _call_predict(module, model, prediction_frame)

            return predict

        # Full orchestration when the parameter set is non-degenerate. Layers
        # 1-3 stay available when Layer 4 has nothing real to perturb.
        perturbers = _perturbable(parameters)
        if perturbers:
            try:
                engine = _full_engine(
                    builder=builder,
                    matrix=matrix,
                    base_params=engine_params,
                    config=config,
                    description=payload.get("description"),
                )
                return _finish(_apply_engine_outcome(report, engine))
            except Exception as exc:  # noqa: BLE001 - fall back to explicit layers
                report.warnings.append(
                    "engine_fallback: "
                    f"{type(exc).__name__}: {exc}; ran layers explicitly"
                )

        report.layers = _manual_layers(
            matrix=matrix,
            config=config,
            predictions=np.empty(0),
            builder=builder,
            base_params=engine_params,
            has_perturbable=bool(perturbers),
            in_sample=False,
            warnings=report.warnings,
        )
        _copy_layer_metrics(report)
        return _finish(report)

    if not predictions_path:
        report.layers = [
            _layer(
                "Predictions",
                "skipped",
                "the module has no fit/predict helpers and no saved predictions",
            )
        ]
        return _finish(report)
    predictions = pd.read_csv(Path(predictions_path))
    aligned = align_predictions(predictions, matrix.retained_index)
    report.layers = _manual_layers(
        matrix=matrix,
        config=config,
        predictions=aligned,
        builder=None,
        base_params=engine_params,
        has_perturbable=False,
        in_sample=True,
        warnings=report.warnings,
    )
    _copy_layer_metrics(report)
    return _finish(report)


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: runner.py <config.json> <report.json>", file=sys.stderr)
        return 2
    config_path = Path(argv[1])
    report_path = Path(argv[2])
    _block_network()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    hypothesis_id = str(payload.get("hypothesis_id") or "unknown")
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            report = _validate(payload)
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        traceback.print_exc(file=sys.stderr)
        report = ValidationReport(
            hypothesis_id=hypothesis_id,
            status="error",
            verdict="inconclusive",
            error=f"{type(exc).__name__}: {exc}",
            layers=[
                _layer(
                    "Validation",
                    "error",
                    f"{type(exc).__name__}: {exc}",
                )
            ],
        )
    try:
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"could not write validation report: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
