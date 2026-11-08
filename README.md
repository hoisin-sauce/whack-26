# Referee

> An autonomous research analyst that reads the literature, turns claims into
> runnable models, and refuses to ship what it can't defend.

Referee turns any pandas-readable tabular dataset into a cited research trail, a
dynamic field path, three runnable statistical hypotheses, a method-critic
review, an optional model search, a four-layer statistical validation pass, an
advisory final judge, and a portable JSON artifact with a Markdown preview and
a UI-ready HTML report.

The project is called **Referee**. The Python package and CLI keep the
`whack26` name from the hackathon repository, so commands still run as
`python -m whack26 ...`.

## What it does

1. **Profiles and classifies the dataset.** Schema, dtypes, missingness, roles,
   target/time detection, and a dynamically created general-domain field
   hierarchy (max depth 4, never pre-seeded).
2. **Reads the literature.** Generates queries and fetches live papers and
   datasets from free APIs, then normalizes, deduplicates, ranks, and caches
   the results.
3. **Extracts only what can be cited.** Factors must carry evidence IDs that
   resolve to a research record; surviving factors are mapped to dataset
   columns.
4. **Writes runnable hypotheses.** Three Python models, one per method family
   (linear, group comparison, interaction/derived), AST-validated and executed
   in an isolated subprocess with repair on failure.
5. **Audits the derivation.** An agentic method critic checks each model
   against the versioned statistical-pitfalls corpus and cannot judge model
   performance.
6. **Searches models (optional).** A typed candidate search compares baseline,
   regression, GLM, interaction, group-comparison, and optional tree/SINDy
   specifications over repeated holdouts.
7. **Checks the data-generation design (optional).** Anonymized and
   jumbled-label variants, research-to-model extraction, and a claim/design
   generation validator.
8. **Adds field-scoped caveats (optional).** Cited limitation literature for
   each node of the field path, with an explicit `applies_to_dataset` value.
9. **Scrapes semantically (optional).** Target-gated reference/link expansion
   with credibility scoring, cached comparisons, and a document method critic.
10. **Runs the statistical gauntlet.** A separate validator applies the
    Deflated Sharpe Ratio, Monte Carlo permutation, chronological
    out-of-sample decay, and parameter sensitivity, with general
    regression/classification/count support through `stats_engine.validate_model`.
11. **Judges the final model (optional).** Nine specialist critics plus a
    devil's advocate produce an advisory run-level decision and a UI-ready
    report. The judge never re-ranks candidates, never overrides the validator,
    and never blocks artifact emission.
12. **Hands off an audit trail.** A versioned `artifact.json`, `preview.md`,
    validation/model-search/variant/pitfall sidecars, and an optional
    dashboard report.

> **Every claim has an evidence ID. Every model has a code hash. Every verdict
> has an audit trail.**

## Setup

Requires Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Optional: add statsmodels for the full model-search stack
# python -m pip install -r requirements-model-search.txt
# Optional: offline embeddings via model2vec (set EMBED_DIMS=256)
# python -m pip install -r requirements-local-embeddings.txt
Copy-Item .env.example .env
# Fill DEEPSEEK_API_KEY, GEMINI_API_KEY, and OPENALEX_API_KEY in .env
python -m whack26 doctor
```

`.env` is gitignored. Never commit it. The default required keys are
`DEEPSEEK_API_KEY` (LLM), `GEMINI_API_KEY` (embeddings), and
`OPENALEX_API_KEY`; `HF_TOKEN`, `ZENODO_TOKEN`, and the optional source/search
keys are optional. DeepSeek has no embeddings API; Gemini is the default
embedding provider, and local `model2vec` embeddings are an optional
no-quota fallback (`requirements-local-embeddings.txt`, `EMBED_DIMS=256`).
Wikipedia REST is the default web fallback; the optional Google/Brave
web-search adapter registers only when configured. Batched OpenAlex author-metric
enrichment is opt-in with `WHACK26_RESEARCH_METRICS_ENABLED=1`.

## Quickstart

No keys needed:

```powershell
python -m whack26 fetch-demo --out data/california_housing.csv
python -m whack26 run `
  --input data/california_housing.csv `
  --description "1990 California census block-group housing data." `
  --target median_house_value `
  --out runs/demo-mock `
  --mock --offline
```

Live demo run with the default pipeline:

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --description "1990 California census block-group housing data." `
  --target median_house_value `
  --out runs/demo `
  --validate
```

Validate, preview, judge, or re-render an existing run:

```powershell
python -m whack26 validate-models runs/demo
python -m whack26 validate-models runs/demo --generation
python -m whack26 validate-models runs/demo --write-artifact
python -m whack26 validate-artifact runs/demo/artifact.json
python -m whack26 preview runs/demo/artifact.json
python -m whack26 judge runs/demo
python -m whack26 judge runs/demo --refresh
```

Enable the complete optional path in one run:

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --out runs/agentic `
  --full
```

`--full` enables scraping, model search, variants, generation validation,
field-scoped pitfalls, and statistical validation. It does **not** enable the
final judge; add `--judge` explicitly. Use `--mock` for deterministic fixtures
without keys, `--offline` to read research from cache only, and `--refresh` to
ignore cache TTLs.

The run directory contains:

```text
runs/demo/
  artifact.json
  preview.md
  dataset.csv
  research.json
  factors.json
  validation.json                 # when --validate is used
  generation_validation.json      # when --validate-generation is used
  final_judge.json / .md / .html / .fragment.html  # when --judge is used
  model_search.json               # when --model-search is used
  variants.json                   # when --variants is used
  research_models.json
  field_pitfalls.json             # when --field-pitfalls or --full is used
  scrape.json / scrape.md         # when --scrape is used
  hypotheses/<hyp_id>.py
  predictions/<hyp_id>.csv
  logs/run.log
```

Unseen-dataset check:

```powershell
python -m whack26 run `
  --input data/penguins.csv `
  --description "Palmer Penguins measurements by species, island, and year." `
  --target body_mass_g `
  --out runs/penguins
```

## Field classification and hierarchy

The default dataset classifier performs dynamic create/assign through
`whack26/classify.py`; it starts from no taxonomy and enforces a maximum depth
of 4. Scraped research documents use the hierarchy placement path in
`whack26/hierarchy/` by default (`WHACK26_HIERARCHY_PLACEMENT_ENABLED=1`),
including branching, duplicate-sibling merges, promotion with subtree
re-levelling, audited path recomputation, and `field_operations` rows.
`store_meta.hierarchy_version` increments whenever the tree changes so caches
can detect stale paths. Routing the user-dataset classifier through the same
placement path remains separate work.

## Semantic scraping

Semantic scraping is opt-in. It expands the one-hop research records through
API references and links, summarizes and embeds each document, classifies it
into the existing field tree, scores credibility, caches comparisons, and runs
the document method critic.

Credibility scoring, pairwise comparisons, and the document critic are part of
the scraping stage. Default one-hop research records are not credibility-scored
or compared to the dataset; enable `--scrape` when those gates are required.

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
`field_operations`, `research_models`, `store_meta`, and collection metadata.
`cache/research.db` remains the raw HTTP cache. Scraping writes `scrape.json`
into the run directory and adds the optional `research.semantic_scraping`
section to `artifact.json`. Schema `1.3.0` adds optional `model_search`,
`variants`, `research_models`, `collection`, `generation_validation_summary`,
and `hypotheses[].generation_validation` sections; `1.4.0` adds `final_judge`;
and `1.5.0` adds `field_pitfalls`. Earlier artifacts remain valid.

The mock end-to-end path is covered by the test suite. The full 20-resource live
scrape smoke is still pending provider quota; the recorded 5-document live smoke
is in [`docs/eval/semantic-scraping-smoke.md`](docs/eval/semantic-scraping-smoke.md).

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

The four-layer strategy engine is reserved for continuous, return-like series.
General continuous, binary, multiclass, and count targets are validated through
`stats_engine.validate_model` with the primary metric chosen per target kind:
R² for regression, log loss/AUC/accuracy for classification, and Poisson
deviance for counts. Optional calibration (`max_ece`) and dispersion
(`max_dispersion`) gates are reported in the diagnostics. Classification and
count reports are `partial` because Layer 1 (DSR) is return-series-only.
Constant, non-finite, and short targets are still reported as
`skipped: target_contract`; they never block artifact emission.

`--write-artifact` embeds the reports in `artifact.json` and regenerates
`preview.md`. `run --validate` performs the same pass inside a normal run.

## Model search, variants, and generation validation

These stages are opt-in. Use `--full` to enable the complete optional path
(scraping, model search, variants, generation validation, field-scoped
pitfalls, and statistical validation) in one run:

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --full `
  --collection-json collection.json `
  --collection-field design=observational `
  --out runs/agentic
```

Individual flags remain available:

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --model-search `
  --backend full `
  --variants `
  --field-pitfalls `
  --validate-generation `
  --collection-json collection.json `
  --collection-field design=observational `
  --out runs/agentic
```

- `--model-search` types the target and features, screens candidates, evaluates
  baseline/linear/logistic/multinomial/Poisson/negative-binomial/interaction/
  group-comparison specifications over repeated holdouts, reserves a final
  holdout for `n >= 200`, and writes `model_search.json`. `--backend sklearn`
  adds tree/forest/gradient-boosting candidates; `--backend full` also adds a
  SINDy candidate for time-series targets. The `core` backend stays
  numpy/scipy-only. Partial-dependence plots are not emitted.
- `--variants` writes `variants.json` with original, anonymized, and
  jumbled-label frames. Each generated hypothesis is re-run on all three
  variants; jumbled labels that still predict well are treated as leakage and
  PII-dependent models are rejected. The per-hypothesis flags feed generation
  validation.
- `--field-pitfalls` searches field-specific limitation literature for each
  field path node, leaf first, and writes `field_pitfalls.json`. Each cited
  caveat gets a stable `F:<field_id>:<n>` id and an `applies_to_dataset` value.
  The stage is advisory and never blocks a run.
- `--validate-generation` reviews each hypothesis's claim and derivation
  against the collection metadata and writes `generation_validation.json`.
  `validate-models --generation` re-runs that review on an existing artifact.
- `--collection-json`/`--collection-field` supply the data-generation design:
  unit of analysis, sampling, treatment, instrument, running variable,
  identification strategy, known biases, and measurement notes. Causal claims
  without an identification strategy are rejected; quasi-experimental designs
  require a running variable or instrument.

## Field-scoped pitfalls

`--field-pitfalls` (or `--full`) is opt-in. For each field in the run's field
path, leaf first, the stage runs a short literature query against the configured
research sources, extracts cited limitations, and tags each with
`applies_to_dataset=yes|unknown|no`. Pitfalls are advisory context for the
method critic and hypothesis context pack; they never create fields or block a
run. `unknown` means the profile cannot establish the precondition, so it is
background context only, not a finding. The stage writes
`field_pitfalls.json` with `F:<field_id>:<n>` citations and adds the optional
`field_pitfalls` section to `artifact.json`. Configure it with
`WHACK26_FIELD_PITFALLS_ENABLED`,
`WHACK26_FIELD_PITFALLS_MAX_QUERIES_PER_FIELD`,
`WHACK26_FIELD_PITFALLS_RECORDS_PER_QUERY`,
`WHACK26_FIELD_PITFALLS_MAX_PER_FIELD`,
`WHACK26_FIELD_PITFALLS_MAX_TOTAL`, and
`WHACK26_FIELD_PITFALLS_SOURCES`; the defaults are one query per field, five
records per query, three pitfalls per field, twelve total, and
`openalex,arxiv`.

## Final model judge

`--judge` runs the advisory final judge after model search, the method critic,
and statistical validation. It audits the selected model with nine specialist
critics plus a devil's advocate, computes deterministic flags in code, and
writes:

```text
final_judge.json
final_judge.md
final_judge.html
final_judge.fragment.html
```

The decision is one of `accepted`, `accepted_with_concerns`,
`requires_reaudit`, `rejected`, or `insufficient_evidence`. It is advisory: it
never re-ranks candidates, never overrides the statistical validator, and never
blocks artifact emission. The artifact gains a `final_judge` section and
`preview.md` gains a `## Final model judgement` section.

The UI consumes the CSS-free fragment directly:

```python
dashboard.load_final_judge(run_dir, tab_ref)
```

Re-run or refresh the judge on an existing run with:

```powershell
python -m whack26 judge runs/demo
python -m whack26 judge runs/demo --refresh
```

`judge --format markdown` (or another subset) writes only the requested files;
the UI handoff requires the default all-formats path so the fragment exists.
See [`docs/spec/final-judge.md`](docs/spec/final-judge.md) for the evidence
mapping, decision rules, and configuration.

## Research catalog and hierarchy

`run` records one Search document per query, with result links and embeddings.
Catalog queries over stored documents remain library-only
(`whack26/research/catalog.py`); there is no `catalog` CLI command yet.
Scraped documents use hierarchy placement by default, while the user-dataset
classifier still uses the create/assign flow described above.

## Validation dashboard (optional)

The dashboard is optional and uses PyQt6, which is kept out of the core
requirements. The launcher creates `.venv` if needed, installs the UI stack when
PyQt6 is missing, and starts the dashboard:

```powershell
.\run_ui.cmd
```

Use `.\run_ui.cmd -Mock -Offline` for a keyless demo, `-Reinstall` to force the
UI dependency install, or `-NoLaunch` to only prepare the environment. The
equivalent manual commands are:

```powershell
.\.venv\Scripts\pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe UI_frontEnd.py
```

The dashboard is tabbed. Drop a `.csv`/`.xls`/`.xlsx` file, click the drop zone,
or use **Choose dataset...**; the file loads, the **Target Column** selector
populates (default **Auto-detect**, with an explicit override), and the full
pipeline starts in a background thread (model search, statistical validation,
and final judge enabled). The tab renders `final_judge.fragment.html` in its
summary panel and keeps its own run, report, and log. The legacy
`DatasetWorkspace.run_validation` path still runs `stats_engine` directly for
the original continuous-target demo (last column as target, 30+ rows), but it is
not the main handoff.

The UI test suite is described in
[`docs/spec/ui-integration-test-plan.md`](docs/spec/ui-integration-test-plan.md).
It is optional: `requirements-ui-test.txt` installs PyQt6 and pytest-qt.

## Acceptance checks

| Check | Pass condition |
|---|---|
| Field path | Plausible domain root; depth 2-4; no dataset-name node |
| Research | At least one paper and one dataset record; at least 3 factors; at least one mapped factor |
| Hypotheses | Exactly 3, one per family; all `applies` true; all run in at most 60 s |
| Predictions | One value per row, or `row_index` alignment; saved CSV exists |
| Critic | 3 valid critic results; verdicts in `pass/refine/reject/insufficient`; no performance judgments |
| Validation | Reports when `--validate` is used; unsupported targets are `skipped`, not failed; failures never block the artifact |
| Model search | Selected candidate beats the baseline on the primary metric or is explicitly low-confidence; no target leakage; optional backends skip cleanly |
| Variants | Jumbled-label and anonymized checks report `pass`/`reject`/`insufficient`; a jumbled-label leak rejects the model |
| Generation validation | Claim/design checks return `pass`/`refine`/`reject`/`insufficient`; causal claims without an identification strategy are rejected |
| Field pitfalls | Cited `F:<field_id>:<n>` ids; `unknown` is background only; the stage never blocks a run |
| Final judge | Advisory decision only; report files and fragment validate; no re-ranking or validator override |
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

## Documentation

- [`docs/spec/README.md`](docs/spec/README.md): spec index, architecture, component boundaries, and documentation authority
- [`docs/spec/artifact.md`](docs/spec/artifact.md): run directory, artifact schema 1.5.0, stable IDs, preview, handoff
- [`docs/spec/field-classification.md`](docs/spec/field-classification.md): dynamic fields, classifier thresholds, evaluation gate
- [`docs/spec/hypotheses.md`](docs/spec/hypotheses.md): Python contract, three method families, execution, repair
- [`docs/spec/method-critic.md`](docs/spec/method-critic.md): corpus retrieval, deterministic checks, verdicts
- [`docs/spec/semantic-scraping.md`](docs/spec/semantic-scraping.md): document catalog, SQLite v4, frontier, credibility, comparisons, preload
- [`docs/spec/field-branching-and-model-search.md`](docs/spec/field-branching-and-model-search.md): hierarchy operations and model-search design
- [`docs/spec/agentic-hypotheses-validation-and-sources.md`](docs/spec/agentic-hypotheses-validation-and-sources.md): variants, research models, agent loop, collection metadata, sources
- [`docs/spec/field-pitfalls.md`](docs/spec/field-pitfalls.md): field-scoped pitfall queries, applicability, citations, settings
- [`docs/spec/final-judge.md`](docs/spec/final-judge.md): advisory final model audit, decision rules, report formats, UI handoff
- [`docs/spec/providers.md`](docs/spec/providers.md): LLM/embedding interfaces, DeepSeek, Gemini, local embeddings, mock mode
- [`docs/spec/api-notes.md`](docs/spec/api-notes.md): live-verified endpoints, auth, response shapes, limits
- [`docs/spec/demo-runbook.md`](docs/spec/demo-runbook.md): demo dataset, commands, fallbacks, time budget
- [`docs/spec/ui-integration-test-plan.md`](docs/spec/ui-integration-test-plan.md): dashboard-to-pipeline handoff and optional-feature test matrix
- [`docs/eval/README.md`](docs/eval/README.md): evaluation harnesses and recorded smoke results
- [`docs/devpost.md`](docs/devpost.md): submission narrative, demo storyboard, and asset checklist
- [`GLOSSARY.md`](GLOSSARY.md): project vocabulary

## Status

The core pipeline, semantic scraping, statistical validation, model search,
variants, generation validation, field-scoped pitfalls, the hierarchy package,
the final judge, and the optional PyQt6 dashboard are implemented and tested.
The latest full suite reported **1,476 passed and 18 documented xfails**, plus
one order-dependent teardown error in the optional-stage UI file (the file
passes in isolation; see
[`docs/spec/ui-integration-test-plan.md`](docs/spec/ui-integration-test-plan.md)).
Scraped documents use hierarchy placement by
default; `--variants` evaluates every generated hypothesis on the original,
anonymized, and jumbled frames; `run --judge` writes all four judge reports and
embeds the advisory decision in the artifact.

Known limits: the full ten-dataset live field-classification gate is still
pending provider quota and a completed `eval/scores.json`; the full 20-resource
live semantic-scrape smoke is pending quota; the live two-dataset statistical
validation run is pending quota. These are recorded in `docs/eval/`.
