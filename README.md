# whack26: research-to-hypothesis slice

Turn any pandas-readable tabular dataset into a cited research trail, a dynamic
field path, three runnable statistical hypotheses, a method-critic review, and a
portable JSON artifact with a Markdown preview.

The slice is deliberately small: dynamic field classification, live free-API
research, factor mapping, generated Python hypotheses, and a corpus-grounded
method critic. Independent finding validation and final reporting are owned by
separate components.

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
  --out runs/demo

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

# Preload the statistics/validity corpus
python -m whack26 preload-statistics --limit 100
```

The SQLite v2 knowledge store lives in `cache/fields.db` and holds `documents`
(Dataset/Search/Paper/Web), `document_links`, `document_fields`,
`document_comparisons`, `document_critiques`, and `document_fetches`.
`cache/research.db` remains the raw HTTP cache. Scraping writes
`scrape.json` into the run directory and adds the optional
`research.semantic_scraping` section to `artifact.json` (schema `1.1.0`).

The mock end-to-end path is covered by the test suite. The live smoke is still
pending provider quota; the planned seed is OpenAlex references at depth 1.

## Acceptance checks

| Check | Pass condition |
|---|---|
| Field path | Plausible domain root; depth 2-4; no dataset-name node |
| Research | At least one paper and one dataset record; at least 3 factors; at least one mapped factor |
| Hypotheses | Exactly 3, one per family; all `applies` true; all run in at most 60 s |
| Predictions | One value per row, or `row_index` alignment; saved CSV exists |
| Critic | 3 valid critic results; verdicts in `pass/refine/reject/insufficient`; no performance judgments |
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

The repo also contains the Brevan Howard four-layer validation dashboard. It
uses PyQt6, which is kept out of the core requirements:

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
- [`docs/spec/semantic-scraping.md`](docs/spec/semantic-scraping.md): document catalog, SQLite v2 migration, frontier, credibility, comparisons, preload
- [`docs/spec/demo-runbook.md`](docs/spec/demo-runbook.md): demo dataset, commands, fallbacks
- [`docs/eval/hypothesis-smoke.md`](docs/eval/hypothesis-smoke.md): live step-4 results and fixes
- [`docs/eval/research-smoke.md`](docs/eval/research-smoke.md): live step-3 research results
- [`docs/eval/README.md`](docs/eval/README.md): ten-dataset field-classification gate
- [`GLOSSARY.md`](GLOSSARY.md): project vocabulary

## Status

Steps 1-4 are implemented and tested (486 tests). The two-dataset live demo is
complete. The full ten-dataset live field-classification gate is still pending
provider quota, and its human-score gates require a completed
`eval/scores.json` filled in by a human reviewer.
