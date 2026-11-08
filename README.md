# whack26: research-to-hypothesis slice

Turn any pandas-readable tabular dataset into a cited research trail, a dynamic
field path, three runnable statistical hypotheses, a method-critic review, a
four-layer statistical validation pass, and a portable JSON artifact with a
Markdown preview.

The core slice is deliberately small: dynamic field classification, live
free-API research, factor mapping, generated Python hypotheses, a
corpus-grounded method critic, and the Brevan Howard four-layer engine as a
separate validation stage. Optional stages add dataset variants, automated
regression model search, research-to-model extraction, collection-aware
generation validation, and hierarchy operations. Final reporting is owned by a
separate component.

## Setup

Requires Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Fill GEMINI_API_KEY and OPENALEX_API_KEY in .env
python -m whack26 doctor
```

`.env` is gitignored. Never commit it. The required keys are
`GEMINI_API_KEY` and `OPENALEX_API_KEY`; `HF_TOKEN` and `ZENODO_TOKEN` are
optional. Wikipedia REST is the only web fallback.

## Demo run

```powershell
python -m whack26 fetch-demo --out data/california_housing.csv

python -m whack26 run `
  --input data/california_housing.csv `
  --description "1990 California census block-group housing data." `
  --target median_house_value `
  --out runs/demo `
  --validate

python -m whack26 validate-models runs/demo
python -m whack26 validate-models runs/demo --generation
python -m whack26 validate-artifact runs/demo/artifact.json
python -m whack26 preview runs/demo/artifact.json
```

The run directory contains:

```
runs/demo/
  artifact.json
  preview.md
  dataset.csv
  research.json
  factors.json
  validation.json   # when --validate is used
  generation_validation.json  # when --validate-generation is used
  model_search.json # when --model-search is used
  variants.json     # when --variants is used
  research_models.json
  scrape.json       # when --scrape is enabled
  scrape.md         # when --scrape is enabled
  hypotheses/<hyp_id>.py
  predictions/<hyp_id>.csv
  logs/run.log
```

Use `--mock` for deterministic fixtures without keys, `--offline` to read
research from cache only, and `--refresh` to ignore cache TTLs.

Unseen-dataset check:

```powershell
python -m whack26 run `
  --input data/penguins.csv `
  --description "Palmer Penguins measurements by species, island, and year." `
  --target body_mass_g `
  --out runs/penguins
```

## Semantic scraping

Semantic scraping is opt-in. It expands the one-hop research records through
API references and links, summarizes and embeds each document, classifies it
into the existing field tree, scores credibility, caches comparisons, and runs
the document method critic.

```powershell
# Full run with scraping enabled
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --scrape `
  --out runs/demo-scrape

# Scraping-only inspection command
python -m whack26 scrape `
  --input data/california_housing.csv `
  --target median_house_value `
  --max-resources 50 `
  --max-depth 2 `
  --out runs/scrape-demo

# Enable the compliant HTTP fetcher (robots-aware)
python -m whack26 scrape `
  --input data/california_housing.csv `
  --target median_house_value `
  --fetch `
  --out runs/scrape-fetch-demo

# Preload the statistics/validity corpus
python -m whack26 preload-statistics --limit 100
```

The HTTP fetcher is opt-in. When enabled it enforces `robots.txt` and
`Crawl-delay`, per-domain rate limits, redirect re-validation, SSRF/private
network blocking, optional domain allow/block lists, PDF/binary rejection,
size/time caps, and a text-excerpt cache. It never downloads PDFs or stores full
article text. Configure it with `WHACK26_SCRAPE_FETCH_ENABLED=1`,
`WHACK26_SCRAPE_ALLOWED_DOMAINS`, `WHACK26_SCRAPE_BLOCKED_DOMAINS`,
`WHACK26_SCRAPE_RATE_LIMIT_SECONDS`, `WHACK26_SCRAPE_MAX_BYTES`, and
`WHACK26_SCRAPE_MAX_TEXT_CHARS`.

The SQLite v4 knowledge store lives in `cache/fields.db` and holds `documents`
(Dataset/Search/Paper/Web), `document_links`, `document_fields`,
`document_comparisons`, `document_critiques`, `document_fetches`,
`field_operations`, `store_meta`, and `research_models`. `cache/research.db`
remains the raw HTTP cache. Scraping writes `scrape.json` into the run
directory and adds the optional `research.semantic_scraping` section to
`artifact.json`. Schema `1.3.0` adds optional `model_search`, `variants`,
`research_models`, `collection`, `generation_validation_summary`, and
`hypotheses[].generation_validation` sections; `1.0.0`–`1.2.0` artifacts remain
valid.

The mock end-to-end path is covered by the test suite. The live smoke is still
pending provider quota; the planned seed is OpenAlex references at depth 1.

## Statistical validation

`stats_engine.py` is wired in as an independent post-generation stage. The
validator reconstructs a numeric feature matrix from each hypothesis's declared
columns, fits the generated model on a chronological training split, and runs
the four layers:

1. Deflated Sharpe Ratio over real feature and model trials;
2. Monte Carlo permutation test on held-out predictions;
3. chronological out-of-sample decay;
4. parameter sensitivity over the model's `PARAMETERS` dict.

Generated modules expose two private helpers, `_fit(df, seed, params)` and
`_predict(model, df)`, plus a module-level `PARAMETERS` dict. `_predict` receives
feature columns only, so it cannot read the target. Legacy modules without the
helpers fall back to the saved predictions and are reported as partial
(in-sample) results.

The engine supports only continuous, return-like targets. Binary, categorical,
constant, non-finite, and short targets are reported as
`skipped: target_contract`; they never block artifact emission. Run validation
standalone with:

```powershell
python -m whack26 validate-models runs/demo
python -m whack26 validate-models runs/demo --write-artifact
```

`--write-artifact` embeds the reports in `artifact.json` and regenerates
`preview.md`. `run --validate` performs the same pass inside a normal run.

## Model search, variants, and generation validation

These stages are opt-in:

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --model-search `
  --variants `
  --validate-generation `
  --collection-json collection.json `
  --collection-field design=observational `
  --out runs/agentic
```

- `--model-search` types the target and features, screens candidates, evaluates
  baseline/linear/logistic/multinomial/Poisson/negative-binomial/interaction/
  group-comparison specifications over repeated holdouts, reserves a final
  holdout for `n >= 200`, and writes `model_search.json`.
- `--variants` writes `variants.json` with original, anonymized, and
  jumbled-label frames. Jumbled labels that still predict well are treated as
  leakage; PII-dependent models are rejected.
- `--validate-generation` reviews each hypothesis's claim and derivation
  against the collection metadata and writes `generation_validation.json`.
  `validate-models --generation` re-runs that review on an existing artifact.
- `--collection-json`/`--collection-field` supply the data-generation design:
  unit of analysis, sampling, treatment, instrument, running variable,
  identification strategy, known biases, and measurement notes. Causal claims
  without an identification strategy are rejected; quasi-experimental designs
  require a running variable or instrument.

The `hierarchy` package supports leaf assignment, branching with duplicate-sibling
merges, promotion with subtree re-levelling, and audited path recomputation.
`field_operations` records every change and `store_meta.hierarchy_version`
increments so caches can detect stale paths. The package is fully unit-tested but
is **not yet** the default classifier path; `classify.py` still uses the original
create-and-assign flow.

## Acceptance checks

| Check | Pass condition |
|---|---|
| Field path | Plausible domain root; depth 2-4; no dataset-name node |
| Research | At least one paper and one dataset record; at least 3 factors; at least one mapped factor |
| Hypotheses | Exactly 3, one per family; all `applies` true; all run in at most 60 s |
| Predictions | One value per row, or `row_index` alignment; saved CSV exists |
| Critic | 3 valid critic results; verdicts in `pass/refine/reject/insufficient`; no performance judgments |
| Validation | 3 reports when `--validate` is used; unsupported targets are `skipped`, not failed; failures never block the artifact |
| Model search | Selected candidate beats the baseline on the primary metric or is explicitly low-confidence; no target leakage; optional backends skip cleanly |
| Variants | Jumbled-label and anonymized checks report `pass`/`reject`/`insufficient`; a jumbled-label leak rejects the model |
| Generation validation | Claim/design checks return `pass`/`refine`/`reject`/`insufficient`; causal claims without an identification strategy are rejected |
| Hierarchy (module-level) | Leaf paths, `document_fields`, and dataset `path_json` agree; every change has a `field_operations` row |
| Artifact | `validate-artifact` passes; `preview.md` renders; no secrets or absolute portable paths |
| Runtime | Cold run at most 20 min; warm run at most 5 min |
| Offline | Completes from cache with a clear degraded-mode label |
| Unseen dataset | Palmer Penguins produces 3 runnable hypotheses without code changes |

The live two-dataset smoke is recorded in
[`docs/eval/hypothesis-smoke.md`](docs/eval/hypothesis-smoke.md). California
Housing returned one `refine` and two `pass` critic verdicts; Palmer Penguins
returned three `refine` verdicts. Every generated module reproduced its saved
predictions exactly.

## Validation dashboard (optional)

The same four-layer engine also powers the Brevan Howard desktop dashboard. The
dashboard is optional and uses PyQt6, which is kept out of the core
requirements:

```powershell
.\.venv\Scripts\pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe UI_frontEnd.py
```

The dashboard requires a dropped `.csv`/`.xls`/`.xlsx` file whose last column is
a continuous, return-like target with finite values, non-zero variance, and at
least three distinct values. Binary labels and constant targets are rejected.
It fits a ridge-regularised linear model on a chronological split, derives real
trial Sharpes from the feature signals and the fitted model, and evaluates
held-out predictions with an R-squared metric.

## Documentation

- [`docs/spec/README.md`](docs/spec/README.md): spec index and component boundaries
- [`docs/spec/semantic-scraping.md`](docs/spec/semantic-scraping.md): document catalog, SQLite v4 migration, frontier, credibility, comparisons, preload
- [`docs/spec/demo-runbook.md`](docs/spec/demo-runbook.md): demo dataset, commands, fallbacks
- [`docs/eval/hypothesis-smoke.md`](docs/eval/hypothesis-smoke.md): live step-4 results and fixes
- [`docs/eval/statistical-validation.md`](docs/eval/statistical-validation.md): four-layer validation contract and acceptance
- [`docs/eval/research-smoke.md`](docs/eval/research-smoke.md): live step-3 research results
- [`docs/eval/README.md`](docs/eval/README.md): ten-dataset field-classification gate
- [`GLOSSARY.md`](GLOSSARY.md): project vocabulary

## Status

Steps 1-4, semantic scraping, statistical validation, model search, variants,
generation validation, and the hierarchy package are implemented and tested; the
default suite is green. The two-dataset live demo is complete. The full
ten-dataset live field-classification gate is still pending provider quota, and
its human-score gates require a completed `eval/scores.json` filled in by a human
reviewer.
