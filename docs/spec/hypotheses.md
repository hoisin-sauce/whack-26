# Hypotheses: generation, execution, and the cheap screen

Status: decided in [ticket #9](https://github.com/hoisin-sauce/whack-26/issues/9). Applies to the 6–8 hour slice.

## Contract

Every hypothesis is a Python module with exactly two public functions:

```python
import pandas as pd

def applies(df: pd.DataFrame) -> bool:
    """Return True when the hypothesis can run on df. Must not raise."""

def run(df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit the model and return predictions. Must not mutate df."""
    ...
```

`run` returns a JSON-serializable dict:

| Key | Required | Type | Meaning |
|---|---|---|---|
| `predictions` | yes | list / Series / ndarray | One value per row in `row_index`, or per row of `df` when `row_index` is null |
| `target` | yes | `str \| None` | Target column, or `None` for descriptive hypotheses |
| `row_index` | no | list | Original index labels aligned with `predictions`; null means `df.index` |
| `model_summary` | no | dict | Coefficients, group means, fitted parameters, convergence flag |
| `notes` | no | str | Caveats, missing-data handling, transformations |

Metadata lives outside the code:

```json
{
  "id": "hyp_<sha256[:12]>",
  "title": "Median income and coastal proximity predict block-group house value",
  "method_family": "linear_model",
  "target": "median_house_value",
  "required_columns": ["median_income", "ocean_proximity"],
  "optional_columns": ["housing_median_age"],
  "min_rows": 100,
  "assumptions": [
    "The relationship between median_income and the target is approximately linear on the log scale.",
    "Rows are treated as independent block groups; spatial autocorrelation is not modelled."
  ],
  "claim_type": "prediction",
  "factors_used": ["fac_..."],
  "source_ids": ["openalex:W..."],
  "rationale": "...",
  "code": "...",
  "code_sha256": "..."
}
```

## Method families

Exactly three hypotheses per run, one per family. The family is a template, not a library.

| Family | Model | Typical target |
|---|---|---|
| `linear_model` | OLS via `numpy.linalg.lstsq`, or logistic/Poisson via `scipy.optimize.minimize`; optional log/sqrt transforms | numeric, binary, count |
| `group_comparison` | Group means/rates across a categorical or quantile-binned factor; predictions are the group statistic | numeric or binary |
| `interaction_or_derived` | OLS/GLM with one interaction between two factors, or one ratio/derived feature | numeric, binary, count |

Pick the link function from the target profile: numeric → identity; binary → logit; non-negative count → log. The generator must not use `sklearn` or `statsmodels`; both are outside the pinned stack. Implement the small amount of linear algebra with `numpy` and `scipy`.

## Generation

Three LLM calls, one per family, with the same context:

- dataset profile and summary;
- field path;
- factors with mappings and evidence IDs;
- top five research records (title, year, abstract ≤ 1,200 chars);
- target and time column;
- cheap-screen summary;
- the contract above;
- the allowed imports;
- the family description.

The LLM returns:

```json
{
  "title": "...",
  "rationale": "...",
  "assumptions": ["..."],
  "required_columns": ["..."],
  "optional_columns": ["..."],
  "min_rows": 100,
  "claim_type": "prediction",
  "factors_used": ["fac_..."],
  "code": "import pandas as pd\n..."
}
```

Rules enforced by the prompt and the validator:

- Do not use the target as a feature.
- Do not read files, environment variables, or the network.
- Do not mutate the input frame; start with `df = df.copy()`.
- Handle missing values explicitly: drop with a note or impute with a stated rule. Do not silently `dropna()` on a large fraction.
- Seed any randomness from `seed`; call `np.random.seed(seed)` and `random.seed(seed)` when sampling.
- Default to predictive or associative language. Use causal language only when the data description says the data came from a randomized or quasi-experimental design; otherwise the critic will reject the claim.
- Cite at least one factor and one source ID. If no factor applies, say so explicitly and justify using domain knowledge.
- Return predictions for every row, or provide `row_index` for the rows retained.
- Never import `sklearn`, `statsmodels`, `torch`, `tensorflow`, or any package outside the allowlist.

Allowed imports:

`pandas`, `numpy`, `scipy`, `math`, `statistics`, `datetime`, `itertools`, `functools`, `collections`, `re`, `warnings`.

## Static validation

Before execution:

1. `ast.parse` and `compile`.
2. Reject any `Import`/`ImportFrom` outside the allowlist.
3. Reject calls to `eval`, `exec`, `compile`, `open`, `input`, `breakpoint`, `__import__`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`.
4. Reject attribute names starting with `__`.
5. Require `def applies(df)` and `def run(df, seed=0)` with compatible signatures.
6. Reject module-level statements other than imports, constant assignments, and function definitions.
7. Reject the target column in `required_columns` and `optional_columns`.
8. Require `len(assumptions) >= 1`, non-empty `rationale`, and code ≤ 20 KB.
9. Reject string literals containing `http://`, `https://`, `socket`, or obvious file paths used in calls.

A static failure triggers one repair attempt. A second failure rejects the hypothesis and generates a replacement with the same family.

## Execution

The generated module is written to `runs/<run_id>/hypotheses/<hyp_id>.py`. The dataset is copied once to `runs/<run_id>/dataset.csv`.

The runner (`whack26/hypotheses/runner.py`) is a separate process:

```
python -I runner.py <module.py> <dataset.csv> <seed>
```

It:

1. Reads the CSV with pandas.
2. Blocks network access by replacing `socket.socket` and `socket.create_connection` with functions that raise `RuntimeError`.
3. Imports the module from its path with `importlib`.
4. Calls `applies(df)`. If false, exits 0 with `{"applies": false}`.
5. Calls `run(df.copy(), seed)`.
6. Validates the result keys and converts NumPy types to Python.
7. Writes `runs/<run_id>/predictions/<hyp_id>.csv` with `row_index,prediction`.
8. Prints a JSON envelope to stdout: `{"applies": true, "n_predictions": ..., "target": ..., "model_summary": ..., "notes": ...}`.

Parent limits: 60 s timeout, 8 MB stdout cap, 64 KB stderr cap, minimal environment without API keys, `cwd = runs/<run_id>`. `-I` ignores `PYTHON*` environment variables and user site-packages.

This is **best-effort isolation, not a security boundary**. The system is local and the code is generated from a trusted model; document that assumption in the runbook.

## Repair loop

| Failure | Attempts | Feedback |
|---|---|---|
| Static validation | 1 | The violated rule and the offending line |
| Missing/invalid `applies` or `run` | 1 | The contract and the AST error |
| Runtime error | 2 | Traceback (last 4,000 chars), code, contract |
| Wrong output shape | 1 | Expected and actual keys/lengths |
| Timeout | 1 | Timeout value; ask for a simpler model |

After the attempts, generate a replacement with the same family and a simpler template. Record every attempt in `generation_history` (no secrets). If fewer than three hypotheses survive, fail the run with a clear error rather than hand off a partial set.

## Cheap descriptive screen

`whack26/screen.py` computes, once per run:

- target distribution: numeric summary or category counts;
- missingness per column;
- Pearson and Spearman correlation matrices for numeric columns (values only, no p-values);
- group means/rates for categorical columns with ≤ 20 categories;
- a time trend summary (first/last, monthly means) when a time column exists.

No hypothesis tests and no p-values. The screen is context for generation and for the method critic; it is never used to accept or reject a hypothesis.

## Distinctness

- `method_family` is unique across the three.
- Code embedding cosine < 0.95 **or** normalized AST signatures differ.
- If a duplicate is detected, regenerate the later hypothesis.

## Acceptance

- Exactly three hypotheses, one per family.
- `applies(df)` is true on the demo dataset for all three.
- Each runs in ≤ 60 s and returns predictions of the expected length.
- No banned imports, calls, or file/network access.
- Each cites at least one factor or explicitly justifies having none; each cites at least one source ID when factors are used.
- The artifact contains the code, `code_sha256`, `generation_history`, and `execution` result.

## Module layout

```
whack26/hypotheses/
  generate.py   # prompts, three families, metadata
  validate.py   # AST and metadata checks
  execute.py    # subprocess orchestration, predictions CSV
  runner.py     # isolated child process
  repair.py     # feedback prompts
whack26/screen.py
```
