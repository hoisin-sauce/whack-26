# Statistical validation: four-layer acceptance

Status: implemented and mock-tested; live two-dataset run pending provider quota.

The validator consumes a generated hypothesis after execution and runs the
unmodified engine in `stats_engine.py`. It is a separate stage from the method
critic: the critic audits the derivation, the validator measures the fitted
model.

## Contract

Generated modules expose:

```python
PARAMETERS = {"ridge": 1e-3}

def _fit(df, seed=0, params=None): ...
def _predict(model, df): ...
def run(df, seed=0): ...
```

`_fit` receives the target column. `_predict` receives feature columns only, so
it cannot read the target at prediction time. `run` uses the same path and
remains the public execution contract.

Modules without `_fit`/`_predict` still validate from their saved
`predictions/<hyp_id>.csv`. Those reports are marked
`in_sample_fallback: true` and Layers 3-4 are `skipped`.

## Target contract

The validator selects a path by target kind:

| Target kind | Engine | Primary metric |
|---|---|---|
| `continuous`, `time_series` | four-layer `validation_engine` (return-like series) | R² |
| `binary` | `stats_engine.validate_model` | log loss (AUC and accuracy also reported) |
| `multiclass` | `stats_engine.validate_model` | log loss (macro F1 and one-vs-rest AUC also reported) |
| `count` | `stats_engine.validate_model` | Poisson deviance (Pearson dispersion also reported) |

The adapter accepts numeric, boolean, and string class labels and encodes them
to stable integer codes; the report maps those codes back to the original
labels. Count targets must be non-negative integers. Constant targets, fewer
than 20 finite rows, or fewer than two classes become
`skipped: target_contract`.

Kind inference is deliberately simple: two distinct values are binary,
3–10 non-numeric values are multiclass, and non-negative integers are count
when they have 11+ levels or a count-like name. Otherwise the column is
continuous, where the existing small-integer-category guard may still skip it.
Continuous targets drop rows with missing target values; classification and
count targets require a complete target column.

The generated module's positive `PARAMETERS` are passed to the engine, so
Layer 4 can perturb real hyperparameters on the non-continuous path too.
Classification and count runs therefore report **`partial`** rather than
`passed`: Layer 1 (DSR) is return-series-only and is reported as
`not_applicable`, which the status mapping treats like a skip. This mirrors the
existing zero-parameter continuous behavior.

The `stats_engine` statistical backend remains reserved for continuous,
return-like series. General regression, classification, and count targets use
the in-slice backend or the new engine path; they never fall back to the
strategy engine.

## Layer mapping

| Report layer | Continuous return path | Classification / count path |
|---|---|---|
| Layer 1 (DSR) | `trial_sharpes` + `dsr_dynamic` | `not_applicable` |
| Layer 2 (Monte Carlo) | `monte_carlo` on held-out predictions | label-permutation test with the primary metric |
| Layer 3 (OOS Decay) | `oos_decay` on a chronological split | chronological out-of-sample decay |
| Layer 4 (Parameter Sensitivity) | `pss` over `PARAMETERS` | `pss` over `PARAMETERS`; `not_applicable` when none |

The full continuous path calls `validation_engine`. When `PARAMETERS` has no
non-degenerate numeric value, Layers 1-3 run explicitly and Layer 4 is
`skipped: degenerate_hyperparameter`. Zero-valued floats are removed before the
engine because `pss` perturbs them to themselves and reports a vacuous pass.

Classification calibration (`max_ece`) and count dispersion (`max_dispersion`)
are optional gates. Both values are reported in the layer diagnostics even when
no gate is configured; an unavailable diagnostic never fails a run by itself.

## Report statuses

| Status | Meaning |
|---|---|
| `passed` | All four layers passed |
| `failed` | At least one layer failed |
| `partial` | Some layers passed; others were skipped, including Layer 1 for classification/count, in-sample fallback, or no perturbable parameter |
| `skipped` | No layer could run (target contract, no target, no predictions) |
| `error` | Runner, configuration, or engine error |

`validation_summary.counts` aggregates these statuses. The engine's `passed`
boolean is only one input into the mapping.

## Runtime and isolation

- Each hypothesis runs in a separate `python -I` child process.
- Default timeout: 120 s per hypothesis, longer than the 60 s hypothesis runner
  because validation refits the model for `trial_sharpes`, `oos_decay`, and
  `pss`. Override with `--timeout`.
- Network access is blocked, the environment is minimal, and generated-code
  stdout/stderr never mixes with the JSON report.
- `oos_decay` fits once. `pss` fits one baseline plus two variants per
  non-degenerate numeric parameter.

## Commands

```powershell
python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo --validate
python -m whack26 validate-models runs/demo
python -m whack26 validate-models runs/demo --write-artifact
```

## Mock acceptance

The test suite covers:

- full four-layer pass on a strong synthetic linear model;
- zero-parameter Layer 4 skip;
- binary, multiclass, and count validation through `validate_model` with the
  expected primary metric and label round-trip;
- in-slice permutation for multiclass and count primary metrics;
- calibration and dispersion diagnostics and gates;
- in-sample fallback without `_fit`/`_predict`;
- adapter round-trips for numeric and categorical features;
- report persistence, artifact embedding, and preview rendering.

Run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_validation_runner.py tests/test_validation_adapter.py tests/test_validation_pipeline.py
```

## Known limitations

- The `stats_engine` backend remains return-only; general regression,
  classification, and count targets use the in-slice or general-engine path.
- `periods_per_year` applies only to the return-like path. General targets are
  not interpreted as financial return series.
- Legacy artifacts remain valid but validate in-sample only.
- A live two-dataset validation run is still pending provider quota.
