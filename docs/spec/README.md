# Referee specification

Status: **implemented**. Field classification, research, hypotheses, the critic,
the artifact, semantic scraping, statistical validation, model search, variants,
generation validation, field-scoped pitfalls, the hierarchy package, the final
judge, and the optional PyQt6 dashboard are built and tested. Search documents
are emitted by `run`; scraped documents use hierarchy placement; catalog queries
remain library-only; the optional source adapters register when configured.

The project is **Referee**. The Python package and CLI keep the historical
`whack26` name, so the commands in this spec are `python -m whack26 ...`.

## What this is

Given any pandas-readable dataset and an optional description, Referee:

1. profiles the data and classifies it into a **dynamically created,
   general-domain field hierarchy** (max depth 4);
2. fetches and caches papers and datasets from free APIs, then extracts
   **factors** and maps them to columns;
3. generates **three runnable Python hypotheses**, one per method family, with
   applicability predicates;
4. executes them in an isolated subprocess and repairs failures;
5. runs an **agentic method critic** over the derivation against a preloaded
   pitfalls corpus;
6. optionally searches typed model specifications, builds dataset variants,
   extracts research models, and reviews the claim/design fit;
7. optionally retrieves **field-scoped pitfalls** for each field-path node;
8. runs the **statistical validator** over each generated model;
9. optionally runs the **advisory final judge** over the selected model,
   validation, and critic evidence;
10. emits a versioned **JSON artifact**, a Markdown preview, optional sidecars,
    and a UI-ready final-judge fragment.

## Scope

**In:** the stages above, local use, no auth, live free APIs with a per-dataset
cache, one-hop research, a cheap descriptive screen, semantic scraping,
credibility and comparisons, the hierarchy package, model search, variants,
generation validation, field-scoped pitfalls, the field-classification
evaluation gate, the four-layer validation engine, the advisory final judge, and
the optional PyQt6 dashboard.

**Out:** human sign-off and final editorial ownership of the report,
auth/multi-user, Postgres/pgvector deployment, a pre-seeded taxonomy, cross-run
hypothesis deduplication, and a catalog CLI command. The optional dashboard
ships separately (`requirements-ui.txt`), not in the core slice. The final judge
emits a structured report and a UI-ready HTML fragment, but the report owner
still owns the final presentation.

## Architecture

```
input file + optional description
        |
        v
profile (schema, dtypes, missingness, target/time/roles) --> dataset summary
        |
        v
classifier  <-->  SQLite v4 field/document store  <-->  DeepSeek LLM + Gemini embeddings
        |                                               (general-domain, max depth 4)
        v
research queries -> OpenAlex / arXiv / HF Hub / Zenodo / Wikipedia
                   + Dataverse / OpenML / data.gov / DataCite (default)
                   + Kaggle / Semantic Scholar / web search (optional, key-gated)
        |                |
        |                v
        |          per-dataset cache -> normalize -> dedup -> rank -> metrics enrichment
        v
optional variants -> optional semantic scraping -> factors -> screen
        |
        v
optional field pitfalls -> optional model search
        |
        v
optional research models + bounded context pack
        |
        v
3 hypothesis generators (linear / group comparison / interaction)
        |  (optional bounded agent loop)
        v
static validation -> subprocess execution -> repair loop -> predictions CSV
        |
        v
optional generation validation
        |
        v
method critic (corpus retrieval + deterministic checks + LLM)
        |
        v
statistical validation (DSR / Monte Carlo / OOS decay / parameter sensitivity;
                         general regression/classification/count via validate_model)
        |
        v
optional advisory final judge -> final_judge.json/.md/.html/.fragment.html
        |
        v
artifact.json + preview.md + sidecars + hypotheses/*.py + predictions/*.csv
```

## Spec index

| Document | Status | Covers |
|---|---|---|
| [`field-classification.md`](field-classification.md) | Implemented; superseded in part by the hierarchy package | Dynamic fields, representation, prompts, thresholds, invariants, 10-dataset eval gate |
| [`research-pipeline.md`](research-pipeline.md) | Implemented | Ingest/profile, dataset summary, queries, source adapters, metrics, normalization, dedup, cache, factor mapping, refinement |
| [`field-pitfalls.md`](field-pitfalls.md) | Implemented; opt-in | Field-scoped pitfalls: query generation, extraction categories, applicability, citations, artifact section, settings |
| [`semantic-scraping.md`](semantic-scraping.md) | Implemented; historical v2 notes marked | Document catalog, link expansion, credibility, comparisons, preloaded statistics, SQLite v4 |
| [`field-branching-and-model-search.md`](field-branching-and-model-search.md) | Implemented in part; historical plan sections marked | Dynamic branching/promotion and automated regression model search |
| [`agentic-hypotheses-validation-and-sources.md`](agentic-hypotheses-validation-and-sources.md) | Mostly implemented; catalog queries library-only | Variants, research-to-model extraction, the agent loop, collection metadata, generation validation, extra sources, Search documents |
| [`hypotheses.md`](hypotheses.md) | Implemented | Python contract, three method families, generation, AST validation, subprocess execution, repair, cheap screen |
| [`method-critic.md`](method-critic.md) | Implemented | Critic inputs, corpus retrieval, deterministic checks, rubric, output schema, verdicts, boundary with the validator |
| [`final-judge.md`](final-judge.md) | Implemented | Advisory final model audit, deterministic checks, decision rules, report formats, UI handoff |
| [`artifact.md`](artifact.md) | Implemented; schema 1.5.0 | Run directory, JSON schema, stable IDs, Markdown preview, handoff, versioning |
| [`demo-runbook.md`](demo-runbook.md) | Implemented | California Housing demo, setup, commands, acceptance checks, fallbacks, time budget |
| [`ui-integration-test-plan.md`](ui-integration-test-plan.md) | Implemented with documented production gaps | Dashboard-to-pipeline handoff, final judge fragment contract, optional-feature test matrix |
| [`providers.md`](providers.md) | Implemented for DeepSeek/Gemini/mock; fallbacks not implemented | LLM/embedding interfaces, provider selection, mock mode, privacy, retries |
| [`api-notes.md`](api-notes.md) | Reference (live-verified notes) | Live-verified endpoints, auth, response shapes, model latency, demo URLs |
| [`../../GLOSSARY.md`](../../GLOSSARY.md) | Reference | Shared language: field, level, field path, factor, hypothesis, critic, artifact, run |
| [`../corpus/statistical-pitfalls.md`](../corpus/statistical-pitfalls.md) | Reference | The preloaded method-critic corpus (11 retrievable chunks S0–S10, plus checklist and phrasebook appendices; cited) |

## Documentation contract

Each fact has one authority:

| Fact | Authority |
|---|---|
| Artifact fields, required/optional, version | `whack26/artifact.py` + `schemas/artifact.schema.json` |
| CLI commands and flags | `whack26/cli.py::build_parser` |
| Env vars and defaults | `whack26/config.py::Settings.from_env` + `.env.example` |
| SQLite schema and migrations | `whack26/store.py` |
| Source names, limits, priors | `whack26/research/base.py`, `whack26/research/pipeline.py` |
| Statistical and generation validation | `whack26/validation/models.py`, `generation_models.py` |
| Final judge evidence, checks, and report | `whack26/final_judge/` |
| Field-scoped pitfalls | `whack26/pitfalls/` |
| Hierarchy placement and operations | `whack26/hierarchy/`, `whack26/store.py` |

Status labels are literal:

- **implemented** — a test exercises it and it is reachable from `run`, the artifact, the preview, or the CLI;
- **library-only** — implemented and tested but not yet wired into the default path or exposed by the CLI;
- **planned** — not implemented; use forward-looking wording and link the ticket.

Plan documents are non-normative where they still describe an original build
sequence. Do not describe a symbol, flag, env var, schema field, or behavior as
implemented unless the authority above and a test agree.

Original research notes live on their branches: `research/free-apis`,
`research/llm-providers`, `research/python-stack`,
`research/statistical-pitfalls`. They are never merged.

## Repo layout

```
whack26/
  cli.py            # doctor, fetch-demo, classify, run, scrape,
                    # preload-statistics, prefetch, validate-artifact,
                    # validate-models, preview, judge, schema
  config.py         # env, thresholds, model names
  common.py ids.py io.py
  profile.py        # DatasetProfile, summary, target/time/role detection
  embeddings.py     # Gemini embedding adapter, cosine, mock provider
  llm.py            # DeepSeek chat + native Gemini REST clients, JSON schema, retries, mock provider
  classify.py       # field algorithm and refinement
  store.py          # SQLite v4 field/document store and migrations
  screen.py         # descriptive screen (no p-values)
  factors.py        # extraction and mapping
  pitfalls/         # field-scoped pitfall queries, extraction, report
  artifact.py       # Pydantic models, IDs, writers, validation
  preview.py        # Markdown renderer
  doctor.py         # environment/key/API checks
  run.py            # pipeline orchestration and sidecars
  research/
    base.py openalex.py arxiv.py huggingface.py zenodo.py wikipedia.py
    kaggle.py dataverse.py openml.py datagov.py datacite.py
    semantic_scholar.py websearch.py
    metrics.py catalog.py search_sessions.py to_model.py
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
    models.py corpus.py checks.py critic.py document_critic.py
  final_judge/
    types.py evidence.py checks.py prompts.py aggregate.py report.py pipeline.py
  validation/
    models.py adapter.py runner.py execute.py pipeline.py
    collection.py generation.py generation_models.py
    statistical.py backends.py engine_bridge.py
  stats_engine.py   # four-layer engine (unmodified)
eval/field_classification.py eval/field_hierarchy.py eval/model_search.py
UI_frontEnd.py      # optional PyQt6 dashboard
tests/
docs/spec/ docs/corpus/ docs/eval/ docs/agents/ docs/assets/
data/
runs/               # gitignored
cache/              # gitignored
requirements.txt
requirements-model-search.txt
requirements-ui.txt
requirements-ui-test.txt
.env.example
pyproject.toml
```

## Build order

This was the original build sequence; every slice is now implemented.

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

## Environment and keys

- **Python 3.12.4** in a `venv` (`py -3.12 -m venv .venv`). Python 3.15 is blocked by `onnxruntime`/`faiss`/`litellm`.
- **Required:** `DEEPSEEK_API_KEY` (LLM), `GEMINI_API_KEY` (embeddings), `OPENALEX_API_KEY`.
- **Optional:** `HF_TOKEN`, `ZENODO_TOKEN`, `S2_API_KEY`, `DATAGOV_API_KEY`, `KAGGLE_API_TOKEN`, `WEBSEARCH_PROVIDER`/`WEBSEARCH_API_KEY`/`WEBSEARCH_CX`, `WHACK26_RESEARCH_METRICS_ENABLED`.
- **No general web search key is required.** Wikipedia REST is the default web fallback; the Google/Brave `websearch` adapter registers only when configured. Brave has no free plan.
- **Model-search backends:** `requirements.txt` includes `scikit-learn` and `pysindy`; `requirements-model-search.txt` adds `statsmodels`. Select the candidate pool with `--backend core|sklearn|statsmodels|full` or `WHACK26_MODEL_SEARCH_BACKEND`.
- **Defaults:** `DEEPSEEK_MODEL=deepseek-flash`, `GEMINI_MODEL=gemini-3.6-flash`, `GEMINI_EMBED_MODEL=gemini-embedding-2`, `EMBED_DIMS=768`, `WHACK26_LLM_PROVIDER=deepseek`, `WHACK26_EMBED_PROVIDER=gemini`, `WHACK26_OFFLINE=0`.
- **Not implemented:** `WHACK26_LLM_PROVIDER=cloudflare` and `WHACK26_EMBED_PROVIDER=local` are parsed and validated but raise a clear "not implemented" error.
- **No-key development:** `--mock` uses deterministic fixtures for the LLM and embeddings. The pipeline, artifact, and tests run end to end before real keys are configured.
- `.env` is gitignored; `.env.example` is the template. Never commit keys.

## Definition of done

- `python -m whack26 run --input data/california_housing.csv --target median_house_value --out runs/demo` produces three hypotheses, all applicable and runnable, plus a valid artifact and preview.
- The 10-dataset field-classification gate in [`field-classification.md`](field-classification.md) passes when provider quota and `eval/scores.json` are available.
- A warm `--offline` run completes from cache and labels degraded sources.
- Palmer Penguins (unseen) produces three runnable hypotheses without code changes.
- `python -m whack26 validate-artifact runs/demo/artifact.json` passes; no keys or absolute portable paths appear in the artifact.
- `python -m whack26 validate-models runs/demo` writes `validation.json` and reports every hypothesis; unsupported targets are skipped, not failed.
- `python -m whack26 run ... --judge` writes all four judge reports and the artifact's optional `final_judge` section.

## Decisions that differ from the original plan

| Original | Revised | Why |
|---|---|---|
| Finance-specific taxonomy | General-domain, dynamic | User rejected finance-specific and pre-seeded taxonomies |
| Max depth 3 (recommended) | Max depth 4 | User decision |
| Full promotion/rotation | Create + assign remains the default for user datasets; scraped documents use the hierarchy placement/promotion path | Keeps the dataset classifier stable; hierarchy is used where it has the most value |
| Gemini 3.8 Flash | Gemini 3.6 Flash default | 3.8 timed out at 75–120 s in live probes; 3.6 responds in ~1.1 s |
| Gemini for LLM and embeddings | DeepSeek for LLM; Gemini for embeddings | DeepSeek is the default LLM after Gemini free-tier quota limits; DeepSeek has no embeddings API |
| OpenAI SDK with custom base URL | Native Gemini REST via `httpx` | OpenAI-compatible endpoint timed out at 120 s; DeepSeek uses plain `httpx` JSON calls |
| Semantic scraping by references/links | Implemented as an opt-in stage: one-hop seeds plus target-gated link expansion | Time budget; legal/robots constraints; no PDF redistribution |
| sklearn/statsmodels for generated hypotheses | numpy/scipy only for hypotheses; optional sklearn/statsmodels for model search | Keeps the generated-code stack small; the three hypothesis families are implementable directly |
| sqlite-vec / dedicated vector store | Brute-force cosine over SQLite BLOBs | The field count is small; fewer moving parts |
| RestrictedPython + subprocess | AST allowlist + subprocess | Simpler; best-effort isolation for local, trusted code |
| Google Dataset Search | Dropped | No public query API |
| Credibility scoring | Implemented in the semantic-scraping stage | One-hop research records are not scored unless `--scrape` is used |
| Standalone classifier prototype | Absorbed into the eval harness | Same feasibility questions, less duplicated work |

## Existing repo code

| File | Role in this slice |
|---|---|
| `stats_engine.py` | The four-layer validation engine (Deflated Sharpe, Monte Carlo permutation, OOS decay, parameter sensitivity). It is wired in through `whack26/validation/` and remains unmodified. |
| `UI_frontEnd.py` | Optional PyQt6 tabbed dashboard. It drops a dataset, runs the full pipeline in a background thread, and renders `final_judge.fragment.html`. The legacy `run_validation` path uses `stats_engine` directly for continuous-target demos. |

## Open items

- The full ten-dataset live field-classification gate is pending provider quota and completed human scores.
- The full 20-resource depth-1 live scrape smoke and the live two-dataset statistical-validation run are pending provider quota.
- Cloudflare Workers AI and local `model2vec` embeddings are documented seams but are not implemented.
- Catalog queries and the hierarchy placement path for the user-dataset classifier are library-only / not wired.
- Windows subprocess isolation is best-effort, not container-grade.
- Research quality on general-domain datasets is the main unknown; the critic flags weak factor traceability.
- `gemini-3.7-flash` and `gemini-3.8-flash` may become responsive later; both are one env var away.
