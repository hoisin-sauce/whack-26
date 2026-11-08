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

The engine supports continuous, return-like targets only. The runner pre-checks:

- numeric dtype (not bool, object, category, or string);
- at least 20 finite rows;
- at least 3 distinct values;
- non-zero variance;
- not a small integer-coded category set.

Failures become `skipped: target_contract`. The engine itself would return
`failed_layer: "Target contract"`; the runner translates that to a skip so
unsupported targets do not look like model failures.

## Layer mapping

| Report layer | Engine call | Pass condition |
|---|---|---|
| Layer 1 (DSR) | `trial_sharpes` + `dsr_dynamic` | DSR >= `min_dsr` (default 0.95) |
| Layer 2 (Monte Carlo) | `monte_carlo` on held-out predictions | p <= `max_p` (default 0.05) |
| Layer 3 (OOS Decay) | `oos_decay` on a chronological split | decay <= tolerance |
| Layer 4 (Parameter Sensitivity) | `pss` over `PARAMETERS` | drop <= `max_performance_drop` |

The full path calls `validation_engine`. When `PARAMETERS` has no non-degenerate
numeric value, Layers 1-3 run explicitly and Layer 4 is
`skipped: degenerate_hyperparameter`. Zero-valued floats are removed before the
engine because `pss` perturbs them to themselves and reports a vacuous pass.

## Report statuses

| Status | Meaning |
|---|---|
| `passed` | All four layers passed |
| `failed` | At least one layer failed |
| `partial` | Some layers passed; others were skipped (for example, in-sample fallback or no perturbable parameter) |
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
- binary-target skip;
- in-sample fallback without `_fit`/`_predict`;
- adapter round-trips for numeric and categorical features;
- report persistence, artifact embedding, and preview rendering.

Run with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_validation_runner.py tests/test_validation_adapter.py tests/test_validation_pipeline.py
```

## Known limitations

- The engine has no categorical target path; those reports are skips.
- `periods_per_year` defaults to 1 for non-time-series data. The report labels
  this as a directional-signal test, not a financial return series.
- Legacy artifacts remain valid but validate in-sample only.
- A live two-dataset validation run is still pending provider quota.
