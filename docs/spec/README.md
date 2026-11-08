# Revised spec: the smallest research-to-hypothesis slice

Status: **build-ready, mostly implemented**. The wayfinder map ([#1](https://github.com/hoisin-sauce/whack-26/issues/1)) resolved the decisions; this directory is the handoff to the build. Field classification, research, hypotheses, the critic, the artifact, semantic scraping, statistical validation, model search, variants, and generation validation are implemented. Catalog queries, Search-document emission, hierarchy placement, and three source adapters remain library-only.

## What this is

Given any pandas-readable dataset and an optional description, the slice:

1. profiles the data and classifies it into a **dynamically created, general-domain field hierarchy** (max depth 4);
2. fetches and caches papers and datasets from free APIs, then extracts **factors** and maps them to columns;
3. generates **three runnable Python hypotheses**, one per method family, with applicability predicates;
4. executes them in an isolated subprocess and repairs failures;
5. runs an **agentic method critic** over the derivation against a preloaded pitfalls corpus;
6. runs the **four-layer statistical validator** over each generated model;
7. emits a versioned **JSON artifact** plus a Markdown preview for the report owner.

## Scope

**In:** the seven steps above, local use, no auth, live free APIs with a per-dataset cache, one-hop research, a cheap descriptive screen, the field-classification evaluation gate, and the four-layer validation engine as a post-generation stage.

**Out:** final report rendering, auth/multi-user, Postgres/pgvector deployment, pre-seeded taxonomy, cross-run hypothesis deduplication. The optional PyQt6 dashboard ships separately (`requirements-ui.txt`), not in the core slice. Credibility scoring and semantic scraping are implemented (see [`semantic-scraping.md`](semantic-scraping.md)). Catalog queries, Search-document emission, and hierarchy placement are implemented as library surfaces but are not yet wired into the default run; three source adapters (`kaggle`, `semantic_scholar`, `websearch`) are implemented but unregistered.

## Architecture

```
input file + optional description
        |
        v
profile (schema, dtypes, missingness, target/time/roles) --> dataset summary
        |
        v
classifier  <-->  SQLite field store  <-->  Gemini embeddings + LLM
        |                                      (general-domain, max depth 4)
        v
research queries -> OpenAlex / arXiv / HF Hub / Zenodo / Wikipedia
                   + Dataverse / OpenML / data.gov / DataCite (registered)
                   + Kaggle / Semantic Scholar / web search (modules only)
        |                |
        |                v
        |          per-dataset cache (SQLite) -> normalize -> dedup -> rank
        v
factor extraction -> factor-to-column mapping -> optional one-shot refinement
        |
        v
3 hypothesis generators (linear / group comparison / interaction)
        |
        v
static validation -> subprocess execution -> repair loop -> predictions CSV
        |
        v
method critic (corpus retrieval + deterministic checks + LLM)
        |
        v
statistical validation (DSR / Monte Carlo / OOS decay / parameter sensitivity)
        |
        v
artifact.json + preview.md + validation.json + hypotheses/*.py + predictions/*.csv
```

## Spec index

| Document | Status | Covers |
|---|---|---|
| [`field-classification.md`](field-classification.md) | Decided; partially superseded by the branching plan | Dynamic fields, representation, prompts, thresholds, store schema, invariants, 10-dataset eval gate |
| [`research-pipeline.md`](research-pipeline.md) | Implemented | Ingest/profile, dataset summary, queries, source adapters, normalization, dedup, cache, factor mapping, refinement |
| [`semantic-scraping.md`](semantic-scraping.md) | Implemented; historical schema-v2 references are marked | Document catalog, link expansion, credibility, comparisons, preloaded statistics, and the schema v2→v4 migration |
| [`field-branching-and-model-search.md`](field-branching-and-model-search.md) | Implementation plan; model search wired, hierarchy library-only | Dynamic field branching/swapping and automated regression model search, with a parallel build wave and integration wave |
| [`agentic-hypotheses-validation-and-sources.md`](agentic-hypotheses-validation-and-sources.md) | Implementation plan; most stages wired, catalog/search sessions library-only | Variant datasets, research-to-model extraction, the model-building agent loop, collection metadata, generation/statistical validation, extra source adapters, web search, Search documents, and catalog queries |
| [`hypotheses.md`](hypotheses.md) | Implemented | Python contract, three method families, generation, AST validation, subprocess execution, repair, cheap screen |
| [`method-critic.md`](method-critic.md) | Implemented | Critic inputs, corpus retrieval, deterministic checks, rubric, output schema, verdicts, boundary with the validator |
| [`artifact.md`](artifact.md) | Implemented; schema 1.3.0 | Run directory, JSON schema, stable IDs, Markdown preview, handoff, versioning |
| [`demo-runbook.md`](demo-runbook.md) | Implemented | California Housing demo, setup, commands, acceptance checks, fallbacks, time budget |
| [`providers.md`](providers.md) | Implemented | LLM/embedding interfaces, Gemini and Cloudflare adapters, mock mode, privacy, retries |
| [`api-notes.md`](api-notes.md) | Reference (live-verified notes) | Live-verified endpoints, auth, response shapes, model latency, demo URLs |
| [`../../GLOSSARY.md`](../../GLOSSARY.md) | Reference | Shared language: field, level, field path, factor, hypothesis, critic, artifact |
| [`../corpus/statistical-pitfalls.md`](../corpus/statistical-pitfalls.md) | Reference | The preloaded method-critic corpus (11 retrievable chunks S0–S10, plus checklist and phrasebook appendices; cited) |

Original research notes live on their branches: `research/free-apis`, `research/llm-providers`, `research/python-stack`, `research/statistical-pitfalls`. They are never merged.

## Repo layout

```
whack26/
  cli.py            # run, scrape, preload-statistics, prefetch, fetch-demo,
                    # doctor, validate-artifact, validate-models, preview, schema
  config.py         # env, thresholds, model names
  io.py             # pandas-readable ingestion
  profile.py        # DatasetProfile, summary, target/time/role detection
  embeddings.py     # Gemini embedding adapter, cosine, mock provider
  llm.py            # native Gemini REST client, JSON schema, retries, mock provider
  classify.py       # field algorithm and refinement
  store.py          # SQLite field store and membership transactions
  screen.py         # descriptive screen (no p-values)
  factors.py        # extraction and mapping
  artifact.py       # Pydantic models, IDs, writers, validation
  preview.py        # Markdown renderer
  doctor.py         # environment/key/API checks
  research/
    base.py openalex.py arxiv.py huggingface.py zenodo.py wikipedia.py
    kaggle.py dataverse.py openml.py datagov.py datacite.py
    semantic_scholar.py websearch.py
    catalog.py search_sessions.py to_model.py
    normalize.py cache.py pipeline.py scrape/
  hypotheses/
    generate.py validate.py execute.py runner.py repair.py
    variants.py context.py agent_loop.py
  hierarchy/
    types.py place.py operations.py paths.py
  model_search/
    types.py data.py splits.py screen.py candidates.py fit.py
    metrics.py select.py diagnostics.py report.py pipeline.py
    trees.py sindy.py
  critic/
    corpus.py checks.py critic.py
  validation/
    models.py adapter.py runner.py execute.py pipeline.py
    collection.py generation.py generation_models.py
    statistical.py backends.py
stats_engine.py     # four-layer engine (unmodified)
eval/field_classification.py eval/field_hierarchy.py eval/model_search.py
tests/
docs/spec/
docs/corpus/
data/
runs/               # gitignored
cache/              # gitignored
requirements.txt
.env.example
```

## Build order

| Step | Deliverable | Hours |
|---|---|---|
| 1 | Skeleton, config, CLI, `doctor`, mock LLM/embedding providers | 0.5 |
| 2 | Ingest, profile, summary, field store, classifier, eval harness | 1.0 |
| 3 | Research adapters, cache, normalization, dedup, factors | 1.5 |
| 4 | Hypothesis generation, validation, subprocess execution, repair | 1.5 |
| 5 | Method critic (corpus, checks, LLM, verdict) | 1.0 |
| 6 | Artifact, preview, artifact validation | 0.5 |
| 7 | Statistical validation stage (fit/predict contract, runner, artifact reports) | 0.5 |
| 8 | Demo run, 10-dataset eval, fixes | 0.5 |
| 9 | Buffer | 0.5 |
| | **Total** | **7.5** |

Start with [`field-classification.md`](field-classification.md) and the eval harness; it is the highest-risk piece and gates the rest.

## Environment and keys

- **Python 3.12.4** in a `venv` (`py -3.12 -m venv .venv`). Python 3.15 is blocked by `onnxruntime`/`faiss`/`litellm`.
- **Required:** `GEMINI_API_KEY`, `OPENALEX_API_KEY`.
- **Optional:** `HF_TOKEN`, `ZENODO_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `GROQ_API_KEY`, `JINA_API_KEY`, `S2_API_KEY`, `DATAGOV_API_KEY`, `KAGGLE_API_TOKEN`.
- **No general web search key is required.** Wikipedia REST is the only web source, and it is labeled as an encyclopedic fallback. Brave is not used: it no longer has a free plan.
- **Defaults:** `GEMINI_MODEL=gemini-3.6-flash`, `GEMINI_EMBED_MODEL=gemini-embedding-2`, `EMBED_DIMS=768`, `WHACK26_LLM_PROVIDER=gemini`, `WHACK26_OFFLINE=0`.
- **No-key development:** `--mock` uses deterministic fixtures for the LLM and embeddings. The pipeline, artifact, and tests run end to end before real keys are configured.
- `.env` is gitignored; `.env.example` is the template. Never commit keys.

## Definition of done

- `python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo` produces three hypotheses, all applicable and runnable, plus a valid artifact and preview.
- The 10-dataset field-classification gate in [`field-classification.md`](field-classification.md) passes.
- A warm `--offline` run completes from cache and labels degraded sources.
- Palmer Penguins (unseen) produces three runnable hypotheses without code changes.
- `python -m whack26 validate-artifact runs/demo/artifact.json` passes; no keys or absolute portable paths appear in the artifact.
- `python -m whack26 validate-models runs/demo` writes `validation.json` and reports every hypothesis; unsupported targets are skipped, not failed.

## Decisions that differ from the original plan

| Original | Revised | Why |
|---|---|---|
| Finance-specific taxonomy | General-domain, dynamic | User rejected finance-specific and pre-seeded taxonomies |
| Max depth 3 (recommended) | Max depth 4 | User decision |
| Full promotion/rotation | Create + assign is the default; promotion/rotation is implemented as an opt-in hierarchy library | 6–8 hour slice; the hierarchy package landed later and is not yet the default classifier path |
| Gemini 3.8 Flash | Gemini 3.6 Flash default | 3.8 timed out at 75–120 s in live probes; 3.6 responds in ~1.1 s |
| OpenAI SDK with custom base URL | Native Gemini REST via `httpx` | OpenAI-compatible endpoint timed out at 120 s |
| Semantic scraping by references/links | Implemented as an opt-in stage: one-hop seeds plus target-gated link expansion | Time budget; legal/robots constraints; no PDF redistribution |
| sklearn/statsmodels for generated hypotheses | numpy/scipy only for hypotheses; optional sklearn/statsmodels for model search | Keeps the pinned stack small; the three hypothesis families are implementable directly |
| sqlite-vec / dedicated vector store | Brute-force cosine over SQLite BLOBs | The field count is small; fewer moving parts |
| RestrictedPython + subprocess | AST allowlist + subprocess | Simpler; best-effort isolation for local, trusted code |
| Google Dataset Search | Dropped | No public query API |
| Credibility scoring | Deferred | Nice-to-have, not this slice |
| Standalone classifier prototype | Absorbed into the eval harness | Same feasibility questions, less duplicated work |

## Existing repo code

| File | Role in this slice |
|---|---|
| `stats_engine.py` | The four-layer validation engine (Deflated Sharpe, Monte Carlo permutation, OOS decay, parameter sensitivity). It is wired in through `whack26/validation/` and remains unmodified. |
| `UI_frontEnd.py` | PyQt6 quant dashboard with CSV/XLS drag-and-drop. The UI remains optional and separate from the CLI validation path. |
| `__pycache__/stats_test.cpython-314.pyc` | Stale bytecode committed upstream. Ignore it; do not build on it. |

## Open items

- Cloudflare Workers AI fallback is untested (no credentials yet).
- Thresholds are documented defaults; the eval harness calibrates them.
- Windows subprocess isolation is best-effort, not container-grade.
- Research quality on general-domain datasets is the main unknown; the critic flags weak factor traceability.
- `gemini-3.7-flash` and `gemini-3.8-flash` may become responsive later; both are one env var away.
