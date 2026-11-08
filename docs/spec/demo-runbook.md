# Demo dataset and runbook

Status: decided in [ticket #12](https://github.com/hoisin-sauce/whack-26/issues/12). Applies to the 6–8 hour slice.

## Demo dataset

**California Housing** (1990 California census block groups).

- URL: `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv` (verified 200; 1,423,529 bytes).
- Shape: 20,640 rows × 10 columns.
- Columns: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `median_house_value`, `ocean_proximity`.
- Missing: `total_bedrooms` has 207 missing values (1.0%).
- Target: `median_house_value` (numeric, continuous).
- Categorical: `ocean_proximity` (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`).
- Why it fits: clear regression target, mixed numeric/categorical/geographic features, a real missing-data decision, and a rich research literature. It is general enough to exercise the general-domain taxonomy without finance-specific assumptions.
- Expected field path: a root such as `Housing & Real Estate` or `Economics`, a child such as `Housing Markets`, and a leaf such as `California Housing Prices`. Acceptable roots are semantic, not exact strings.
- Expected research queries: `determinants of house prices`, `housing market income elasticity`, `coastal proximity house prices`, `California housing dataset`.
- Expected factors: household income, coastal proximity, housing age, household size, population density, room count.
- Expected hypotheses: a log-linear income/coastal model; a group comparison by `ocean_proximity`; an income × coastal interaction or rooms-per-household derived feature.

The dataset is public StatLib/California census data. Check the source repository's licence before redistributing the CSV; the demo may download it or keep a local copy under `data/`.

## Setup

Requires Python 3.12.4 (3.15 is installed but blocked by `onnxruntime`/`faiss`/`litellm`; see `docs/research/python-stack.md`).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Fill GEMINI_API_KEY and OPENALEX_API_KEY in .env
python -m whack26 doctor
```

`.env` is gitignored. Never commit it. Required keys: `GEMINI_API_KEY`, `OPENALEX_API_KEY`. Optional: `HF_TOKEN`, `ZENODO_TOKEN`, Cloudflare/Groq/Jina. No general web search key is required; Wikipedia REST is the only web source.

## Commands

```powershell
# Download the demo dataset (or use a local path)
python -m whack26 fetch-demo --out data/california_housing.csv

# Cold run: fetch research, classify, generate, execute, critic, artifact
python -m whack26 run `
  --input data/california_housing.csv `
  --description "1990 California census block-group housing data. Rows are block groups; the target is median_house_value. Features include median_income, housing_median_age, rooms, bedrooms, population, households, latitude/longitude, and ocean_proximity." `
  --target median_house_value `
  --out runs/demo

# Warm run: cache and field store already exist
python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo-warm

# Offline: cache only
python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo-offline --offline

# Development without keys
python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo-mock --mock

# Opt-in agentic stages (model search, variants, generation validation)
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --model-search `
  --variants `
  --validate-generation `
  --collection-json collection.json `
  --collection-field design=observational `
  --out runs/demo-agentic

# Statistical validation (four-layer engine)
python -m whack26 validate-models runs/demo
python -m whack26 validate-models runs/demo --generation

# Validate the artifact
python -m whack26 validate-artifact runs/demo/artifact.json
```

The opt-in flags write `model_search.json`, `variants.json`, and
`generation_validation.json` into the run directory and add the corresponding
optional sections to `artifact.json` and `preview.md`. All four settings default
to off, so a plain `run` behaves exactly as before.

`doctor` checks Python, packages, the corpus file, `.env` keys, Gemini model responsiveness, and OpenAlex authentication. It prints a single status table and exits non-zero on a blocking problem.

## Acceptance checks

| Check | Pass condition |
|---|---|
| Field path | Root is a plausible housing/economics/real-estate concept; depth 2–4; no dataset-name node |
| Research | ≥ 1 paper and ≥ 1 dataset record; ≥ 3 factors; ≥ 1 mapped factor |
| Hypotheses | Exactly 3, one per family; all `applies` true; all run ≤ 60 s |
| Predictions | One value per row (or `row_index` alignment); saved CSV exists |
| Critic | 3 valid critic results; no performance judgments; verdicts in `pass/refine/reject/insufficient` |
| Artifact | `validate-artifact` passes; `preview.md` renders; no secrets/absolute portable paths |
| Runtime | Cold run ≤ 20 min including research; warm ≤ 5 min; classification-only ≤ 10 min |
| Offline | Completes from cache with a clear degraded-mode label |
| Unseen dataset | Palmer Penguins CSV produces 3 runnable hypotheses without code changes |

Unseen-dataset check:

```powershell
python -m whack26 run `
  --input data/penguins.csv `
  --description "Palmer Penguins measurements by species, island, and year." `
  --out runs/penguins
```

`https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv` was verified (200; 15,241 bytes; 344 rows). If no target is supplied, the pipeline may infer `body_mass_g` or `species`; the run remains valid.

## Fallbacks

| Failure | Fallback |
|---|---|
| `gemini-3.8-flash` times out | Default to `gemini-3.6-flash` (verified responsive; `gemini-3.5-flash` is an alias) via `GEMINI_MODEL` |
| Gemini unavailable / free-tier privacy concern | Cloudflare Workers AI (`WHACK26_LLM_PROVIDER=cloudflare`) or local `model2vec` embeddings |
| No keys at all | `--mock` deterministic fixtures; all pipeline stages still run |
| OpenAlex 429/403 | Keyless limited mode, exponential backoff, then cache |
| arXiv slow/timeout | Skip after 2 failures; continue with OpenAlex |
| HF/Zenodo down | Continue with the remaining sources; warn in the artifact |
| Research cache cold and offline | Continue with 0 records; hypotheses use the profile and domain knowledge; warn |
| Windows sandbox limits | Subprocess timeout + static allowlist are best-effort; do not run untrusted third-party code |

## Time budget

| Slice | Hours |
|---|---|
| Repo skeleton, config, CLI, doctor, mock provider | 0.5 |
| Ingest, profile, summary, field store, classifier | 1.0 |
| Research adapters, cache, normalization, factors | 1.5 |
| Hypothesis generation, validation, execution, repair | 1.5 |
| Method critic (corpus, checks, LLM, verdict) | 1.0 |
| Artifact, preview, validation | 0.5 |
| Demo run, 10-dataset eval, runbook fixes | 0.5 |
| Buffer | 0.5 |
| **Total** | **7.0** |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| LLM latency/rate limits | Default to the verified fast model; one retry; cache; mock mode |
| Thresholds are wrong | All thresholds configurable; eval harness calibrates; defaults are documented |
| Research quality on general-domain data | Rank by relevance; keep top 20; factors are evidence-linked; critic flags weak traceability |
| Free-tier privacy | Do not send sensitive data; Cloudflare/local embeddings for private data |
| Generated code hangs | 60 s timeout, static allowlist, no-network runner |
| Windows-only sandbox | Document best-effort isolation; do not claim container-grade security |
| Demo data licence | Use the public source; check the repository licence before redistribution |

## Deliverables at the end of the slice

- `runs/demo/artifact.json`, `preview.md`, three hypothesis modules, three prediction CSVs.
- `docs/eval/field-classification-results.md` with the 10-dataset gate.
- A short README section showing the command and the acceptance table above.
