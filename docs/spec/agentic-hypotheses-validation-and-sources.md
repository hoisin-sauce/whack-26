# Agentic hypotheses, validation, and source/search breadth

Status: **implemented in part** — model search, variants, research-to-model
extraction, the agent loop, collection metadata, and generation validation are
wired behind opt-in settings and render into the artifact; `--full` enables the
matching stages together. The statistical backend registry is implemented and
tested, but the default run still uses the target-aware validation path rather
than consuming `WHACK26_STATISTICAL_BACKEND`. Variants are evaluated
per hypothesis, Search documents are emitted, scraped documents use hierarchy
placement, and the `kaggle`/`semantic_scholar`/`websearch` adapters register when
configured. Catalog queries remain library-only. The existing
`whack26/validation/` statistical pipeline is extended, not replaced.

> **Historical build plan.** The sections below describe the original
> implementation waves and the state before the v4 migration landed. The
> current authorities are `whack26/store.py`, `whack26/run.py`,
> `whack26/validation/`, and the spec index.

This is the follow-on to
[`field-branching-and-model-search.md`](field-branching-and-model-search.md)
and the live web-scraping task. It closes the remaining devplan gaps:

1. **Agentic hypothesis generation** — anonymized/jumbled-label variants,
   research-to-model extraction, the model-building agent loop, and
   parent-field/statistics context.
2. **Hypothesis validation** — collection-method metadata, generation-process
   validation, in-slice statistical validation, and the validation artifact.
3. **Source breadth and search sessions** — missing dataset/paper adapters,
   general web search, Search documents and links, and a catalog query API.

SQLite **v4 is implemented today**: v3 added `documents.leaf_field_id`,
`field_operations`, and `store_meta`; v4 added collection metadata, research
models, and search-session metadata. This plan's feature stages are wired on top
of that schema.

## 1. Goals and non-goals

### Goals

- Generate hypotheses against the original, anonymized, and jumbled-label
  dataset variants, and reject any model that exploits leakage or "unscrambles"
  the jumbled labels.
- Extract applicable model specifications from research documents and feed them
  to the generator and model search.
- Replace the one-shot generator with a bounded agent loop that receives
  diagnostics and either refines, replaces, or discards a model.
- Give the generator a bounded context pack built from the field path, ancestor
  fields, statistics, research, and model-search candidates.
- Capture the collection method and design metadata needed to validate claims.
- Run a generation-process validator and an in-slice statistical validator, and
  record the result in the artifact.
- Add the missing dataset/paper source adapters, a general web-search adapter,
  Search documents with to/from links, and a document-catalog query API.

### Non-goals

- No Postgres/pgvector; stay on SQLite.
- No formal causal identification. Validation checks claim/design alignment; it
  does not prove causality.
- No replacement for the independent validator; the artifact records the
  in-slice checks and the handoff.
- No general web crawl in this plan unless the live web-scraping task provides
  it; this plan defines the adapter and policy seam.
- No multi-user/API/auth layer.

## 2. Current state

- `whack26/hypotheses/`: `HypothesisGenerator`, static validation, subprocess
  execution, repair loop, three families, variants, the bounded agent loop, and
  the context pack.
- `whack26/profile.py`: schema, target, roles, description, and collection
  metadata persistence through `datasets.collection_json`.
- `whack26/critic/`: method critic and document critic; the critic audits the
  derivation, not the data-generation process.
- `whack26/validation/`: wired statistical validation (`adapter.py`,
  `execute.py`, `runner.py`, `pipeline.py`, `models.py`, `statistical.py`,
  `backends.py`, `engine_bridge.py`), collection metadata, and generation
  validation. It runs the four-layer `stats_engine` for continuous return-like
  targets and `stats_engine.validate_model` for general
  regression/classification/count targets, and is exposed through
  `run --validate` and `validate-models`.
- `stats_engine.py`: four-layer strategy validation engine; wired through the
  validation package for continuous targets, with a general-target path through
  `validate_model`.
- `whack26/research/`: OpenAlex, arXiv, HF Hub, Zenodo, Wikipedia, Dataverse,
  OpenML, data.gov, and DataCite adapters; optional key-gated Kaggle, Semantic
  Scholar, and Google/Brave web search; semantic-scraping catalog and
  robots-aware fetcher (`research/scrape/fetch.py`); metrics enrichment and
  research-to-model extraction.
- `run` creates one Search document per query with result links and embeddings.
  Catalog queries (`whack26/research/catalog.py`) remain library-only.
- The hierarchy placement path is used for scraped documents by default.

## 3. Part 1 — Agentic hypothesis generation

### 3.1 Dataset variants

New module: `whack26/hypotheses/variants.py`.

```python
class DatasetVariant(BaseModel):
    kind: Literal["original", "anonymized", "jumbled"]
    frame_sha256: str
    target_column: str | None
    rows: int
    seed: int
    dropped_columns: list[str] = []
    notes: list[str] = []

class VariantSet(BaseModel):
    original: DatasetVariant
    anonymized: DatasetVariant
    jumbled: DatasetVariant

def build_variants(
    profile: DatasetProfile,
    frame: pd.DataFrame,
    *,
    seed: int = 0,
) -> tuple[VariantSet, dict[str, pd.DataFrame]]: ...
```

**Anonymized variant**

- Drop direct identifiers: names, emails, phone numbers, addresses, account
  numbers, IP addresses, exact free-text identifiers.
- Hash or drop high-cardinality ID columns; keep dtypes and schema for the
  remaining columns.
- Redact PII from the profile/sample values before any LLM call; reuse
  `profile.py` sanitizers and `common.redact_secrets`.
- Keep column names unless the user requests a blind variant; the purpose is to
  prevent reliance on PII, not to hide the schema.
- Record dropped/masked columns in the variant report.

**Jumbled variant**

- IID data: random permutation of the target with a fixed seed.
- Time series: block permutation or circular shift that preserves
  autocorrelation; do not use a plain shuffle.
- Grouped data: permute within groups.
- Preserve target dtype and marginal distribution.
- Record the shuffle method and seed.

**Variant evaluation**

- Run the generated hypothesis on original, anonymized, and jumbled variants with
  the same split protocol.
- If the jumbled model's performance exceeds the permutation null (for example
  AUC > 0.55 or R² > 0.05 with a permutation p < 0.05 on multiple shuffles),
  flag `jumbled_label_leak` and reject the hypothesis.
- If the anonymized model uses a dropped column or its performance changes
  drastically for a reason tied to PII, flag `pii_reliance`.
- The devplan's "if the agent unscrambles the labels correctly, ignore its
  results" is implemented as this jumbled-variant gate.
- Store `VariantReport` in `runs/<id>/variants.json` and the artifact.

Tests: deterministic variants, no mutation, PII removal, jumbled distribution,
time-series block shuffle, grouped shuffle, a deliberately leaky model caught,
a clean model not falsely rejected.

### 3.2 Research-to-model bridge

New module: `whack26/research/to_model.py`.

```python
class ResearchModelSpec(BaseModel):
    document_id: str
    family: str
    outcome: str
    predictors: list[str]
    target_transform: str = "none"
    feature_transforms: dict[str, str] = {}
    interactions: list[tuple[str, str]] = []
    assumptions: list[str] = []
    reported_metrics: dict[str, float] = {}
    evidence: str
    mapped_columns: list[str] = []
    confidence: float = 0.0
    applied: bool = False
```

`extract_research_models(...)`:

- Input: the credible paper/web documents, their summaries, factors, and the
  dataset profile.
- One LLM call per document or a batched call, with a strict JSON schema.
- Require an evidence span from the summary/abstract; reject a model with no
  evidence.
- Map predictors to dataset columns with the existing factor mapping and
  embedding similarity; reject models whose predictors cannot be mapped.
- Store in `research_models` (schema v4) with document id, spec JSON, mapping,
  evidence, created time.
- Mark `applied=True` when the model is actually used by the generator or model
  search.

Integration:

- `HypothesisContext.research_models` includes the top applicable specs.
- The generator prompt includes the spec (family, predictors, transforms,
  assumptions, evidence) and can seed the closest contract family.
- Model search adds compatible research specs as candidates.
- Keep the three-family artifact contract. A paper model is mapped to the
  closest family (`linear_model`, `group_comparison`,
  `interaction_or_derived`) or recorded as context. A fourth
  `research_derived` family is a separate contract change and is not enabled by
  default.
- Artifact: `research_models` section with source ids, evidence, and whether the
  model was applied.

Tests: schema/evidence requirement, predictor mapping, unsupported model
rejection, no target leakage, integration into the prompt, `applied` flag.

### 3.3 Model-building agent loop

New module: `whack26/hypotheses/agent_loop.py`.

```python
class ModelAttempt(BaseModel):
    family: str
    draft: HypothesisDraft
    execution: ExecutionResult | None = None
    diagnostics: list[Diagnostic] = []
    validation: GenerationValidationResult | None = None
    decision: Literal["accept", "refine", "replace", "discard"] = "discard"
    reason: str = ""

class ModelAgentLoop:
    def run(self) -> HypothesisSet: ...
```

Loop:

1. Build a context pack (section 3.4) with the field path, ancestor context,
   model-search candidates, research models, factors, screen, and critic chunks.
2. Propose a hypothesis for the next family.
3. Validate statically and execute in the isolated runner.
4. Run model-search diagnostics, the generation validator, and the statistical
   validator (Part 2).
5. Decide:
   - `accept`: all checks pass and the model beats the baseline;
   - `refine`: same family, adjust transforms/features/interactions/assumptions
     based on the diagnostics;
   - `replace`: throw out the model and generate a new one for the same family
     with the accumulated feedback;
   - `discard`: family cannot produce a valid model.
6. Repeat within the per-family and global budgets.

Budgets:

- `AGENT_MAX_ATTEMPTS_PER_FAMILY=6`;
- `AGENT_MAX_TOTAL_ATTEMPTS=18`;
- `AGENT_MAX_SECONDS=900`;
- `AGENT_MAX_TOKENS=...`;
- stop on budget exhaustion, repeated identical failures, or a valid set of
  three families.

Feedback sources:

- execution errors (static/runtime/shape/timeout);
- model-search diagnostics (non-linearity, heteroskedasticity,
  multicollinearity, overdispersion, autocorrelation, missing data);
- generation-validator issues (claim/design mismatch, leakage, unit of
  analysis);
- statistical validator results;
- jumbled/anonymized variant report;
- critic issues.

Audit:

- `runs/<id>/agent/attempts.json` with structured attempts, diagnostics,
  decisions, and reasons;
- no prompts with secrets/PII; no raw dataset rows;
- generation history remains in `logs/generation.json`.

Integration:

- `run.py` uses `ModelAgentLoop` when `settings.agent_loop_enabled` is true;
  otherwise the current one-shot generator remains the fallback.
- The loop must produce exactly three unique families and pass the artifact
  contract. If a family cannot succeed, the run fails with a clear error and the
  attempt log.

Tests: simulated diagnostics, refine/replace/discard, budget exhaustion, family
uniqueness, jumbled-leak rejection, no infinite loop, artifact records.

### 3.4 Parent-field and statistics context pack

New module: `whack26/hypotheses/context.py`.

```python
class ContextPack(BaseModel):
    summary: str
    field_path: list[str]
    ancestor_documents: list[ContextDocument]
    ancestor_stats: dict[str, Any]
    factors: list[Factor]
    research_models: list[ResearchModelSpec]
    model_candidates: list[ModelCandidate]
    screen: dict[str, Any]
    critic_chunks: list[str]
    token_estimate: int
```

`build_context_pack(...)`:

- Field path from the classification.
- Ancestor documents: `documents` assigned to any field in the path, ranked by
  summary cosine to the dataset, credibility, and recency; include summaries,
  not full text.
- Ancestor statistics: counts, target distribution, and top factors per ancestor
  field, capped.
- Factors: existing factor list.
- Research models: applicable specs.
- Model candidates: top model-search candidates and diagnostics.
- Screen: the existing descriptive screen.
- Critic chunks: always the how-to chunk plus retrieved chunks.
- Token budget: `CONTEXT_MAX_CHARS` (for example 20,000); deterministic
  truncation; no PII; no secrets.

Integration: `HypothesisGenerator._base_prompt` consumes the pack instead of
recomputing the summary/factors/records. The prompt labels each block and keeps
the evidence ids.

Tests: ancestor inclusion, sibling exclusion, token cap, empty ancestors,
deterministic ordering, no PII, ids traceable.

## 4. Part 2 — Hypothesis validation

### 4.0 Current validation package (already implemented)

The repo already contains a wired statistical-validation stage:

- `whack26/validation/models.py` — `ValidationConfig`, `ValidationReport`,
  `ValidationRequest`, `ValidationLayerReport`; statuses
  `passed|failed|partial|skipped|error`; verdicts `pass|fail|inconclusive`.
- `adapter.py` — feature matrix/target contract/categorical encoding.
- `execute.py` / `runner.py` — isolated child process; target gate; four-layer
  `stats_engine` calls; in-sample fallback.
- `pipeline.py` — `validate_hypothesis`, `validate_run`, artifact/report writes.
- `run.py` — `--validate`, with `ValidationConfig` overrides.
- `cli.py` — `validate-models`.
- `artifact.py` — `HypothesisSection.validation` and
  `Artifact.validation_summary`.

**The plan extends this package; it does not replace it.** The existing
`ValidationReport` remains the statistical report. New claim/design checks use a
separate `GenerationValidationResult`, and a `ValidationBundle` combines both.

### 4.1 Collection-method metadata

New module: `whack26/validation/collection.py`.

```python
class CollectionMetadata(BaseModel):
    design: Literal["observational","experimental","quasi_experimental",
                    "survey","administrative","other"] = "observational"
    unit_of_analysis: str | None = None
    sampling: str | None = None
    time_period: str | None = None
    data_source: str | None = None
    collection_method: str | None = None
    known_biases: list[str] = []
    measurement_notes: list[str] = []
    treatment: str | None = None
    instrument: str | None = None
    running_variable: str | None = None
    identification_strategy: str | None = None
    ethical_constraints: list[str] = []
```

- Add `collection: CollectionMetadata | None` to `DatasetProfile`.
- CLI: `--collection-json <path>` or repeated `--collection-field key=value`.
- Persist in `datasets.collection_json` (schema v4).
- Validate:
  - causal claims require a design with an identification strategy, treatment,
    outcome, and unit;
  - quasi-experimental designs require the running variable/instrument and
    identification strategy;
  - missing required fields produce a validation issue, not a crash.
- Redact PII from the metadata before the LLM sees it.

Tests: schema validation, causal gate, missing fields, PII redaction, artifact
round-trip.

### 4.2 Generation-process validator

New module: `whack26/validation/generation.py`.

Inputs: collection metadata, profile, screen, model-search diagnostics,
research models, hypothesis, variant report.

Checks:

- claim type vs design (`causal` without a design is `reject`);
- treatment/instrument present when claimed;
- unit of analysis matches the model's row unit;
- sampling/selection bias acknowledged;
- known biases and measurement error addressed;
- time period/sample consistency;
- target leakage and target-derived features;
- model family matches the outcome;
- missing-data handling;
- reproducibility (seed, code hash, data hash);
- jumbled-label and PII-reliance reports.

Output:

```python
class GenerationValidationResult(BaseModel):
    hypothesis_id: str
    verdict: Literal["pass","refine","reject","insufficient"]
    checks: list[ValidationCheck]
    issues: list[ValidationIssue]
    missing_information: list[str]
    summary: str
    model: str
    corpus_version: str

class ValidationBundle(BaseModel):
    hypothesis_id: str
    generation: GenerationValidationResult | None = None
    statistical: ValidationReport | None = None  # existing validation.models
    backend: str = "in_slice"
    warnings: list[str] = []
```

`GenerationValidationResult` is deliberately separate from the existing
`ValidationReport`: the former is the claim/design review, the latter is the
four-layer statistical report. The bundle keeps both without changing the
existing artifact contract.

Reuse the critic corpus and LLM rubric; add deterministic checks for claim/design
mismatch and missing required metadata. The validator never judges performance;
it checks whether the derivation and claim match the data-generation process.

Integration: the agent loop consumes issues as feedback; the artifact records the
bundle.

Tests: observational causal claim rejected, experimental design accepted,
missing treatment rejected, known bias acknowledged, clean hypothesis passes,
deterministic checks override the LLM.

### 4.3 In-slice statistical validation and backend selection

New modules: `whack26/validation/statistical.py`, `engine_bridge.py`, and
`backends.py`.

```python
class StatisticalValidationBackend(Protocol):
    name: str
    def supports(self, context: ValidationContext) -> bool: ...
    def validate(self, context: ValidationContext) -> StatisticalValidationResult: ...
```

Backends:

- `in_slice` — default for general regression/classification/count data using
  model-search metrics, permutation, coefficient stability, and diagnostics;
- `stats_engine` — the current four-layer strategy engine, only for a
  return-like/strategy series. This backend **reuses the existing
  `whack26.validation.pipeline.validate_hypothesis` / `validate_run` path** and
  its `ValidationReport`; it does not reimplement the adapter or runner.
- `external` — an adapter for the general statistical-validation library
  currently being written; loaded by entry point or module path.
- `none` — returns `insufficient` with a clear reason.

Selection is configured with:

| Variable | Values | Default |
|---|---|---|
| `WHACK26_STATISTICAL_BACKEND` | `auto`, `in_slice`, `stats_engine`, `external`, `none` | `auto` |
| `WHACK26_EXTERNAL_VALIDATOR` | entry point or module path | unset |

> **Current wiring:** the registry, external seam, and tests exist, but
> `run --validate` and `validate-models` still call the target-aware
> `whack26/validation/pipeline.py` path directly. The settings are parsed and
> validated but not consumed by the default run.

`auto` rules:

- general data → `in_slice`; never call `stats_engine`;
- return-like/strategy series → `stats_engine` plus `in_slice` if the engine
  supports it;
- if `WHACK26_EXTERNAL_VALIDATOR` is configured, use the external backend (and
  optionally merge its result with `in_slice`); until the library is available,
  `in_slice` remains the fallback;
- if no backend supports the data and no external backend is configured, return
  `insufficient` with `statistical_backend_unavailable`; do not pass and do not
  fall back to `stats_engine`.

`in_slice`:

- baseline comparison (mean/median/majority);
- cross-validated metrics from model search;
- permutation test for the selected model;
- coefficient/sign stability across folds;
- diagnostics summary;
- multiple-hypothesis awareness: report the number of attempts and use a
  held-out/permutation test rather than raw p-values;
- explicit `not_applicable` when the data cannot support the test.

`engine_adapter.py`:

- Adapt a model to `stats_engine.validation_engine` only when
  `stats_engine` supports the target: a continuous, finite, return-like series
  with at least three distinct values.
- Never call the engine for binary, multiclass, count, constant, or general
  regression data. `not_applicable` is not a hypothesis failure; it means the
  strategy engine is the wrong tool.
- Use the four-layer engine where applicable: Monte Carlo permutation, OOS
  decay, parameter sensitivity; DSR only for a return/strategy series.
- Wrap the selected model as `(params, X, y) -> predict`; do not substitute
  `stats_engine.build_linear_model` unless that is the selected model.

`backends.py` / external library seam:

- `ExternalStatisticalValidator` protocol: `validate(context) ->
  StatisticalValidationResult`.
- The adapter loads the configured entry point/module, validates the returned
  result, and records `backend="external"` and the library version.
- The external library is an **in-progress dependency**, not a requirement for
  this plan. If it is absent or raises, the run falls back to `in_slice` and
  records a warning; it does not call `stats_engine` as a fallback.
- Provide a fake external backend in tests to prove the seam.

Output: metrics, permutation p-value, diagnostics, baseline delta, verdict,
`backend`, `engine_applicable`, and `external_status`.

Tests: clean model passes, leakage fails, NaN/inf fails, no-split fails,
jumbled-label leak fails, `stats_engine` is never called for binary/count/
multiclass/constant data, the external backend seam uses the configured adapter,
no applicable backend yields `insufficient`, and the engine path works on a
synthetic return series.

### 4.4 Validation output and handoff

The existing artifact fields stay:

- `hypotheses[].validation` — the existing `ValidationReport` (statistical
  four-layer result);
- `validation_summary` — the existing status counts.

Add:

```json
{
  "hypotheses": [
    {
      "id": "hyp_...",
      "validation": { "...existing ValidationReport..." : true },
      "generation_validation": {
        "hypothesis_id": "hyp_...",
        "verdict": "pass | refine | reject | insufficient",
        "checks": [],
        "issues": [],
        "missing_information": [],
        "summary": "",
        "model": "",
        "corpus_version": ""
      }
    }
  ],
  "generation_validation_summary": {
    "verdicts": {"pass": 1, "refine": 1, "reject": 1, "insufficient": 0}
  },
  "validation_summary": { "counts": { "passed": 1, "skipped": 2 } }
}
```

- Preview renders the existing statistical validation table and a separate
  generation-validation table.
- The external validator still owns formal tests; the artifact records the
  in-slice statistical report, the generation review, and the handoff status.
- CLI: keep `validate-models`; add `validate-models --generation` (or a
  `validate-generation` alias) to re-run the claim/design review on an existing
  artifact.
- `run.py`: keep `--validate`; add `--validate-generation` and pass the
  generation results through `build_artifact`.
- Do **not** add a duplicate `validate-hypotheses` command that shadows
  `validate-models`.

Tests: existing `tests/test_validation_*` stay green; new tests cover the
`generation_validation` field, summary, preview, no secrets, deterministic
output, and handoff status.

### 4.5 Compatibility with the current `stats_engine.py`

`stats_engine.py` is a **four-layer strategy validator** (DSR, Monte Carlo, OOS
decay, parameter sensitivity), not a general regression/classification library.
**For now, do not route data through it when it was not designed for that data.**
The general validation library currently being written plugs in through the
`external` backend when it is ready; until then, general data uses `in_slice`.
The existing `validation.pipeline` / `validation.runner` already wire the engine
for continuous targets via `validate-models` and `run --validate`; they are the
`stats_engine` backend. The rules below govern how the new backend registry
calls or extends that path:

1. **Target gate.** Call `validation_engine` only for a continuous, finite,
   return-like target with at least three distinct values. Binary, multiclass,
   count, and constant targets return `not_applicable`; never call the engine.
2. **Selection vs validation.** Model search owns model selection. The engine
   adapter runs *after* selection as a finalist check, on the same evaluation
   split or a dedicated final holdout. It must not re-rank the candidate pool or
   override the model-search selection.
3. **Metric contract.** Pass an explicit `metric_func` and
   `higher_is_better`. `oos_decay` requires a positive in-sample score; R² can
   be negative, and log-loss is lower-is-better. Use a positive metric
   transform, or skip Layer 3 with `not_applicable` when the metric cannot
   satisfy that contract.
4. **Model builder.** Wrap the *selected* model as
   `(params, X, y) -> predict`. Do not use `stats_engine.build_linear_model`
   unless that is the selected model. Expose real hyperparameters for PSS; if
   the model has none, mark Layer 4 `not_applicable` rather than failing on
   "no perturbable parameters".
5. **DSR inputs.** DSR requires `winning_returns` and at least two real
   `all_trial_sharpes`. Define them from the return series and the selected
   model's signals/trials. For non-return targets, skip Layer 1.
6. **Verdict mapping.** Map `passed`/`failed_layer` to the plan's
   `pass | reject | insufficient` vocabulary and carry the layer metrics and
   reasons into `GenerationValidationResult`.
7. **Additive, not invasive.** The adapter wraps the engine; it does not change
   `stats_engine.py`'s public behaviour. Add tests for binary, count,
   multiclass, negative-R², no-hyperparameter, and return-series cases.

Under those rules the plan does **not** contradict the current library. The
contradiction to avoid is calling the strategy engine on a general regression or
classification model, or letting it compete with model search as a second
selector. The general validator (`statistical.py`) remains primary; the engine
adapter is an optional finalist check for return-series hypotheses. The external
general library is an in-progress dependency: when it lands, configure
`WHACK26_EXTERNAL_VALIDATOR`; until then, `in_slice` is the default and
`stats_engine` is never a fallback.

## 5. Part 3 — Source breadth and search sessions

### 5.1 Source capability matrix and adapters

Add these sources, subject to a capability check:

| Source | API | Auth | Plan |
|---|---|---|---|
| Kaggle | Kaggle API | `KAGGLE_API_TOKEN` | Dataset metadata adapter |
| Dataverse | Dataverse REST | none | Dataset metadata adapter |
| OpenML | OpenML REST | none | Dataset metadata adapter |
| data.gov | CKAN API | optional key | Dataset metadata adapter |
| DataCite | DataCite REST | none | DOI/dataset metadata adapter |
| Semantic Scholar | Graph API | optional `S2_API_KEY` | Paper adapter; rate-limit aware |
| Google Dataset Search | no public API | — | Drop or document a fallback |
| General web search | Google Programmable Search or Brave | key + (Google) engine id | `WebSearchAdapter`; registers only when configured |

Each adapter:

- reuses the `SourceAdapter`/`ResearchRecord` contract;
- declares `name`, `ttl_seconds`, rate limits, and `limited_mode`;
- maps to `source`, `source_id`, title, abstract/description, authors, year, DOI,
  URL, license, and source-specific metadata;
- caches raw bodies in `cache/research.db`;
- returns a typed `SourceError` on auth/rate-limit/parse failure;
- has fixture-based tests and a live smoke.

Extend `SourceName`, `QUERY_SCHEMA` source enum, `SOURCE_LIMITS`, source priors,
query-generation prompt, and artifact source lists.

Tests: fixture parsing, auth failure, rate limit, empty results, dedup, license,
normalization, no secrets.

### 5.2 General web search

New module: `whack26/research/websearch.py`.

- `WebSearchAdapter` interface: `search(query, limit, client) -> list[ResearchRecord]`.
- Provider decision: Google Programmable Search or Brave (or reuse the provider
  chosen by the live web-scraping task).
- Auth via env var; free-tier quota; query normalization; result dedup by
  canonical URL; TTL; rate limit.
- If no key or quota is exhausted, skip the source with a warning; do not fail
  the run.
- Result records use `source="websearch"` and the canonical URL.
- Compliance: robots/ToS, no paywalled content, no PII, attribution.

Tests: fixture response, no-key fallback, quota warning, URL canonicalization,
dedup, query normalization.

### 5.3 Search documents and graph

New module: `whack26/research/search_sessions.py`.

- Create one `Document(kind='search')` per query/session:
  - id `srch_<hash>`;
  - title = query;
  - summary = query + source + result count + parameters;
  - metadata = query, source, parameters, result_count, session_id, timestamp;
  - status `classified`/`compared` as appropriate.
- Link search → each result document with relation `search_result`.
- Link search → follow-up search with `search_result` or `linked_from`.
- Link dataset → search/paper with `dataset_paper`/`search_result` when the
  relationship is explicit.
- Store in `documents`/`document_links`; no new table was needed.
- Integration (implemented): `run` creates one Search document per query,
  updates result links after semantic scraping, and exposes the count in the run
  log. The catalog query remains library-only.
- Tests: one search doc per query, idempotent rerun, links, dedup, no secrets.

### 5.4 Catalog query and filter API

New module: `whack26/research/catalog.py`.

```python
class CatalogQuery(BaseModel):
    kinds: list[str] = []
    sources: list[str] = []
    field_id: str | None = None
    include_descendants: bool = True
    year_min: int | None = None
    year_max: int | None = None
    credible_only: bool = False
    preloaded: bool | None = None
    text: str | None = None
    similar_to: str | None = None
    limit: int = 20
    offset: int = 0
```

`search_documents(store, query, embeddings=None)`:

- SQL filters on `documents`/`document_fields`;
- field subtree via `fields.parent_id` recursion or a materialized path;
- vector similarity over `summary_embedding` when `similar_to` is supplied;
- deterministic ordering by similarity, credibility, year, id;
- pagination;
- no target leakage or PII.

Proposed CLI (not implemented; catalog queries are library-only today):
`whack26 catalog query --kind paper --field ... --year ... --similar-to ...`.

Tests: each filter, subtree filter, vector ordering, empty result, pagination,
deterministic order, performance on 1,000 documents.

## 6. Parallel execution plan

This plan is also restructured for a single parallel launch. The contract freeze
is a short pre-flight step; every implementation and integration package starts
together against frozen types and fakes.

### 6.0 Pre-flight contract freeze — status

**Modules landed; fakes and integration partial (2026-11-08).**

Frozen in `whack26/validation/generation_models.py`:

- `CollectionMetadata`, `ValidationCheck`, `ValidationIssue`,
  `GenerationValidationResult`, `ValidationBundle`.

Frozen in their packages:

- `whack26/hypotheses/variants.py` — `DatasetVariant`, `VariantSet`,
  `VariantReport`;
- `whack26/research/to_model.py` — `ResearchModelSpec`;
- `whack26/research/catalog.py` — `CatalogQuery`;
- `whack26/validation/backends.py` — backend protocol and registry.

Still missing from `tests/contracts/fakes.py`: a fake validation corpus, fake
source HTTP responses, a fake document catalog, a fake external statistical
backend, and an in-memory hierarchy fixture. `pytest.ini`/`pyproject.toml` and
the `integration` marker are also absent.

`ModelSearchResult` and its fake landed with the plan-1 contract freeze.
`store.py` and `profile.py` already carry schema v4, so no further pre-flight
work is required there.

### 6.1 Parallel launch (19 + 1 agents)

All packages start together. S0 owns `store.py` and `profile.py`; feature packages
use the frozen types and fakes for unit tests. Cross-plan dependencies use the
plan-1 `ModelSearchResult` contract, so this plan does not wait for the model
search implementation.

| Package | Files | Contract | Unit-test dependency |
|---|---|---|---|
| S0 | `whack26/profile.py`, `whack26/store.py`, tests | Schema v4 (`datasets.collection_json`, `research_models`, optional `search_sessions`), migration v3→v4 | frozen types |
| G1 | `whack26/hypotheses/variants.py`, tests | Anonymized/jumbled variants, variant report, leak detection | fakes |
| G2 | `whack26/research/to_model.py`, tests | Research model extraction, evidence, column mapping, storage | fakes |
| G3 | `whack26/hypotheses/agent_loop.py`, tests | Propose → execute → diagnose → refine/replace/discard, budgets, audit | fake `ModelSearchResult` |
| G4 | `whack26/hypotheses/context.py`, tests | Parent-field/statistics context pack, token cap, deterministic ordering | fakes |
| V1 | `whack26/validation/collection.py`, tests | Collection metadata schema, CLI fields, persistence | frozen types |
| V2 | `whack26/validation/generation.py`, tests | Claim/design checks, corpus-grounded generation validator | fake corpus |
| V3 | `whack26/validation/statistical.py`, `engine_adapter.py`, tests | In-slice metrics/permutation; thin wrapper around the existing `validation.pipeline` for `stats_engine` | fake `ModelSearchResult` |
| V4 | `whack26/validation/generation.py`, `generation_models.py`, tests | `GenerationValidationResult` / `ValidationBundle` mapping; artifact/preview payload | existing `ValidationReport` |
| V5 | `whack26/validation/backends.py`, tests | Backend protocol/registry, external-library adapter seam, feature flags | fake external backend |
| S1 | `whack26/research/kaggle.py`, `dataverse.py`, `openml.py`, tests | Dataset metadata adapters | fake HTTP fixtures |
| S2 | `whack26/research/datagov.py`, `datacite.py`, `semantic_scholar.py`, tests | Dataset/paper metadata adapters | fake HTTP fixtures |
| S3 | `whack26/research/websearch.py`, tests | General web-search adapter with no-key fallback | fake HTTP fixtures |
| S4 | `whack26/research/search_sessions.py`, tests | Search documents, search/result links, search-to-search links | fake document store |
| S5 | `whack26/research/catalog.py`, tests | Filter + vector query over the document catalog | fakes |
| I1 | `whack26/run.py`, `whack26/hypotheses/generate.py`, `whack26/hypotheses/__init__.py`, tests | Variants → research models → context pack → agent loop | fakes until features land |
| I2 | `whack26/artifact.py`, `whack26/preview.py`, `whack26/cli.py`, tests | Validation, research-model, variant sections; `validate-hypotheses` | contract payloads |
| I3 | `whack26/research/pipeline.py`, `whack26/research/scrape/pipeline.py`, `whack26/config.py`, tests | New sources, web search, Search docs, catalog query, source enums | fake adapters |
| I4 | `tests/test_run_flow.py`, `tests/test_agentic_*.py`, docs | End-to-end mock/live tests, docs, acceptance run | fakes until features land |

The only cross-plan ordering is conceptual: this plan's S0 v4 migration follows
plan 1's v3 migration. If both plans launch together, use one schema owner for
the combined v3+v4 migration so the two `store.py` edits cannot collide.

### 6.2 Agent budget

| Scope | Packages | Recommended concurrent | Hard ceiling |
|---|---|---|---|
| This plan | 1 contract + 1 schema + 14 feature + 4 integration = **20** | **12–16** | **20** |
| Plan 1 (`field-branching-and-model-search`) | 1 contract + 1 schema + 10 feature + 4 integration = **16** | **12–14** | **16** |
| Both plans combined | 1 combined contract + 1 combined schema + 24 feature + 8 integration = **34** | **14–16** | **18** |

The practical sweet spot is **12–16 concurrent agents**. The hard ceiling is
around **18–20**; beyond that the merge queue, review, and integration-test
runtime become the bottleneck. File ownership is disjoint enough for 20, but the
34 combined work packages should be run in two or three shifts rather than
launched all at once.

### 6.3 Ownership and merge rules

- C0 owns the contract files and fakes; no feature package edits them.
- S0 owns `whack26/profile.py` and `whack26/store.py`; no other package edits
  them.
- G1–G4 own separate files under `whack26/hypotheses/`.
- V1–V5 own separate files under `whack26/validation/`; V5 owns `backends.py`.
- S1–S3 own separate adapter files; S4/S5 own search/catalog files.
- I1–I4 own `run.py`, `artifact.py`, `preview.py`, `cli.py`, the research
  pipeline, and the end-to-end tests.
- Each package merges the contract tip at start, then the integration branch tip
  before reporting.
- `integration`-marked tests run after S0 and the relevant feature packages land.

### 6.4 If fewer agents are available

| Agents | Launch order |
|---:|---|
| 14 | C0 + S0 + eight feature packages + four integration packages; second shift for the remaining six |
| 8 | C0 + S0 + G1/G3 + V1/V2/V3 + S1/S3 + I1; then the rest |
| 4 | C0+S0, agentic generation (G1–G4), validation (V1–V5), sources/search (S1–S5) |

## 7. Acceptance criteria

### Agentic generation

- Original, anonymized, and jumbled variants are generated deterministically.
- A deliberately leaky model is rejected by the jumbled-variant gate; a clean
  model passes.
- At least one research model is extracted with evidence and mapped to columns,
  or rejected with a clear reason.
- The agent loop records every attempt, respects budgets, and produces three
  unique families; it refines or replaces on diagnostics.
- The context pack includes ancestor documents/statistics, respects the token
  cap, and contains no PII.

### Validation

- Collection metadata is captured and persisted; a causal claim without a
  design is rejected or refined.
- The generation validator checks claim/design, unit of analysis, sampling,
  known biases, leakage, and reproducibility.
- The statistical validator reports metrics, permutation results, and
  diagnostics; the stats engine is used only where applicable.
- `stats_engine` is never called for binary, multiclass, count, constant, or
  general regression data; those use `in_slice` or the configured external
  backend.
- The external-backend seam works with a fake adapter; if the general library is
  absent, `in_slice` is used and no engine fallback occurs.
- If no statistical backend supports the data, the verdict is `insufficient`
  with `statistical_backend_unavailable`, not `pass`.
- The artifact and preview show the validation result and handoff.

### Sources and search

- At least three new source adapters work with fixtures and fail gracefully on
  auth/rate-limit/parse errors.
- General web search works with a key and falls back with a warning when absent.
- Each query creates a Search document and result links; reruns are idempotent.
- The catalog query supports kind/source/field/year/credibility/text/similarity
  filters and pagination.
- No secrets/PII in logs, cache metadata, or the artifact.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Jumbled-label false positives/negatives | Multiple shuffles, permutation null, threshold calibration, manual audit |
| Anonymization removes useful signal | Compare original vs anonymized, report dropped columns, keep schema |
| Research-model hallucination | Require evidence spans, column mapping, confidence threshold |
| Agent loop cost/latency | Per-family and global budgets, deterministic fallback, cache |
| Validation overreach | Separate in-slice checks from formal validation; explicit handoff |
| `stats_engine` used on unsupported data | Target gate; `not_applicable` for binary/count/multiclass/constant; no fallback |
| External library absent or changing | `in_slice` default; `ExternalStatisticalValidator` seam; versioned adapter; fake-backend tests |
| Backend disagreement | Model search selects; `in_slice` and optional external/engine results are reported side by side; no backend silently overrides another |
| Source API limits/legal | Cache, rate limits, keyless fallback, license/ToS review |
| Search graph complexity | Reuse `documents`/`document_links`, no new graph engine |
| Catalog query performance | Indexes, limit/offset, brute-force vector over a bounded candidate set |
| Schema v4 migration risk | Transactional, idempotent, v3→v4 tests, backup |

## 9. Effort

| Phase | Estimate |
|---|---|
| P0 contract freeze | 0.5–1 day |
| P1 agentic generation (G1–G4) | 5–6 days |
| P1 validation (V1–V5) | 5–6 days |
| P1 sources/search (S1–S5) | 5–7 days |
| P2 integration | 5–6 days |
| Acceptance/hardening/docs | 2–3 days |
| **Total** | **24–31 days** |

P0 is a one-agent pre-flight; P1 and P2 use the 12–16 concurrent-agent launch
described in §6. The combined 34 packages across both plans should run in two or
three shifts rather than one 34-agent launch.
