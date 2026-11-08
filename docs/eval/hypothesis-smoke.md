# Hypothesis and method-critic smoke test

Status: live end-to-end smoke on 2/2 demo datasets. Recorded 2026-11-08.

> The logical-flaws review fixes landed after this smoke. The recorded
> artifacts remain valid, but a fresh run may produce different research
> records, factor mappings, or hypotheses because classification cache identity,
> query caching, deduplication, and target-role validation changed. Later
> stages (model search, variants, generation validation, field pitfalls,
> statistical validation, and the final judge) were added after this record.

This page records the first full Referee run on California Housing and Palmer
Penguins: classification, research, factor mapping, the cheap screen, three
generated hypotheses, the method critic, the JSON artifact (then schema
`1.0.0`), the Markdown preview, and prediction reproducibility.

Provider: `gemini-3.5-flash-lite` for language calls and
`gemini-embedding-2` at 768 dimensions for embeddings. The default
`gemini-3.6-flash` model had exhausted its free-tier daily quota.

## Commands

```powershell
$env:GEMINI_MODEL='gemini-3.5-flash-lite'

.\.venv\Scripts\python.exe -m whack26 run `
  --input data\california_housing.csv `
  --description "1990 California census block-group housing data." `
  --target median_house_value `
  --out runs\live-california-5

.\.venv\Scripts\python.exe -m whack26 run `
  --input data\penguins.csv `
  --description "Palmer Penguins: body measurements for Adelie, Chinstrap, and Gentoo penguins across three islands." `
  --target body_mass_g `
  --out runs\live-penguins
```

Both runs were then checked with:

```powershell
.\.venv\Scripts\python.exe -m whack26 validate-artifact runs\<run>\artifact.json
.\.venv\Scripts\python.exe -m whack26 preview runs\<run>\artifact.json
```

## Results

| Dataset | Run | Field path | Confidence | Research | Factors | Hypotheses | Critic |
|---|---|---|---|---|---|---|---|
| California Housing | `run_20261108T003714Z_8a3727f4` | `Real Estate and Housing > Housing Valuation and Demographics` | 0.876 | 20 records / 5 queries | 5 (4 mapped) | 3/3 executed | refine 1, pass 2 |
| Palmer Penguins | `run_20261108T002912Z_f204db2c` | `Biology and Life Sciences > Zoology and Animal Behavior` | 0.841 | 20 records / 5 queries | 6 (6 mapped) | 3/3 executed | refine 3 |

The artifact validator passed for both runs with `schema_version 1.0.0`, three
unique method families, a critic result per hypothesis, and no credential
material or absolute run paths. The preview rendered every required section.

An earlier California run (`run_20261108T002524Z_8a3727f4`) returned one
`reject` and two `insufficient` verdicts for in-sample evaluation and missing
information. It exposed the critic sanitizer and verdict-rule defects listed
below. The final run uses the corrected code.

## Generated hypotheses

### California Housing

Target: `median_house_value`.

| Family | Hypothesis | Predictions | Critic |
|---|---|---|---|
| `linear_model` | OLS with median income, housing age, population, ocean proximity, and bedrooms | 20,640 | refine |
| `group_comparison` | Ocean-proximity group means | 20,640 | pass |
| `interaction_or_derived` | Median income x housing age interaction | 20,640 | pass |

The linear model's rationale calls median income a "primary driver" and the
claim is `prediction`, so the critic raised a major `S4-causal-predictive`
issue plus a minor `S5-diagnostics` issue about OLS assumptions. The other two
hypotheses had no material method gap in the supplied derivation.

### Palmer Penguins

Target: `body_mass_g`.

| Family | Hypothesis | Predictions | Critic |
|---|---|---|---|
| `linear_model` | OLS with flipper length, bill measurements, sex, species, island, and year | 333 | refine |
| `group_comparison` | Species and sex group means | 344 | refine |
| `interaction_or_derived` | Flipper-length and sex interaction | 333 | refine |

The deterministic `time.series` check flagged every hypothesis because the
profile identifies `year` as a time column and the generated code does not use
a lag, differencing, or a time-based split. The critic verdict is `refine`; the
LLM review itself found no additional material gap.

## Reproducibility

Each generated module was re-imported from its run directory, executed on the
stored `dataset.csv` with `seed=0`, and compared with the saved predictions:

| Dataset | Hypothesis | Rows compared | Result |
|---|---|---:|---|
| California Housing | all three | 20,640 each | exact match (`rtol=1e-9`) |
| Palmer Penguins | `linear_model` | 333 | exact match (`rtol=1e-9`) |
| Palmer Penguins | `group_comparison` | 344 | exact match (`rtol=1e-9`) |
| Palmer Penguins | `interaction_or_derived` | 333 | exact match (`rtol=1e-9`) |

The Penguins models that drop rows with missing measurements return a
`row_index` aligned with their predictions, and the saved CSV matches it.

## Fixes made during live acceptance

The first live California attempts exposed real defects:

1. **Relative run paths were doubled.** `run_module` received a relative
   `run_dir` and an absolute dataset path, and the child process resolved the
   dataset under the run directory a second time. `run_module` now resolves
   `module_path`, `dataset_path`, and `run_dir` to absolute paths before
   spawning the runner. A regression test executes a module with relative
   paths.
2. **The generator put the target in its feature lists.** The prompt now marks
   the target column in the column listing and states that
   `required_columns` and `optional_columns` are feature columns only. The
   validator error also says explicitly that the target is supplied separately.
3. **Generation history was lost on failure.** Every attempt is now persisted
   to `<run_dir>/logs/generation.json`, and the family error includes the last
   attempt's status and message.
4. **The critic leaked out-of-scope performance language through
   `missing_information`.** Items such as "out-of-sample performance metrics",
   "evaluation metrics", and "model generalization performance" are now
   removed with a warning, matching the existing summary and issue sanitizer.
5. **The critic could return an unexplained `insufficient`.** The verdict now
   follows the spec's rules directly: fatal issue -> `reject`; degraded or
   missing in-scope information -> `insufficient`; major issue -> `refine`;
   otherwise `pass`. If the LLM's `insufficient` is overridden, the artifact
   records `critic.llm_insufficient_overridden`.
6. **A deterministic check had a contradictory reason.** `time.series` now
   reports "No time column is present." when the dataset has no time column,
   instead of claiming one exists.

## Known limitations

- The full ten-dataset live gate still depends on provider quota. These two
  runs use `gemini-3.5-flash-lite`, which has a separate free-tier limit from
  the default `gemini-3.6-flash`.
- The hypotheses predict in-sample. The method critic flags this when it is a
  material gap, and the independent validator owns out-of-sample evaluation,
  per the slice boundary.
- The Palmer Penguins `year` column is treated as a time column by the profile.
  The deterministic time-series check therefore asks every hypothesis for a
  time-aware protocol even though the rows are repeated cross-sections.
