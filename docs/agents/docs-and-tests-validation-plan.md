# Documentation validation and additional test cases

**Date:** 2026-11-08
**Last verified:** 2026-11-08T05:53Z (confirmed green: 948 passed, 2 skipped)
**Scope:** the in-flight schema v4 / agentic-hypotheses / model-search update on
`main`
**Mode:** plan only. No source files are changed by this document.
**Audience:** the orchestrator and the package owners. Each section is written to
be split into independent work packages.

> Read this with `docs/agents/logical-flaws-review.md` as a precedent: findings
> are prioritized, cite evidence, and separate *implemented* behavior from
> *planned* behavior. Do not weaken existing tests to make new ones pass.

---

## 0. Snapshot: what is implemented, what is wired, what is contract

> **This inventory is a moving target, captured during an active integration
> wave.** Between 05:21 and 05:48 on 2026-11-08 another writer landed
> `model_search/pipeline.py`, the adapter registry, `run.py`, `artifact.py`,
> `preview.py`, `cli.py`, and several tests while this plan was being verified. The first scan saw 553
> tests, the second 880, the latest **950 collected** (948 passed, 2 skipped on
> the confirmation run). The table below reflects the tree at **05:48**; re-run
> the state check before scheduling, and do not treat a red suite run taken
> during active writes as evidence of a code defect.

**State check (run it again before planning):**

```powershell
git status --short --untracked-files=all
.\.venv\Scripts\python.exe -m pytest -q --collect-only
Get-ChildItem whack26\hierarchy,whack26\model_search,whack26\validation -File -Filter *.py
```

> **Use `.venv` explicitly.** Bare `python` on this machine resolves to
> `C:\Program Files\Python315\python.exe`, which has no `pandas`; collecting
> with it produces *72 import errors*, not a real failure. Every command in
> this plan assumes `.\.venv\Scripts\python.exe`.

At snapshot:

| State | Evidence |
|---|---|
| **Implemented and unit-tested** | SQLite v4 migration; statistical validation stage; artifact `1.2.0` + preview; `run --validate` / `validate-models`; new config fields; hierarchy `place`/`operations`/`paths`; model-search `types`/`data`/`splits`/`screen`/`candidates`/`fit`/`metrics`/`select`/`diagnostics`/`report`/`trees`/`sindy`; `model_search/pipeline.py::run_model_search`; `eval/model_search.py` + `eval/field_hierarchy.py` harnesses; `hypotheses/variants.py`; `research/to_model.py`; `hypotheses/agent_loop.py`; `hypotheses/context.py` (`ContextPack`, distinct from the older `generate.HypothesisContext`); `research/catalog.py`; `research/search_sessions.py`; `validation/generation.py`; `validation/collection.py`; `validation/statistical.py`; `validation/backends.py`; adapter modules for all seven new sources |
| **Wired end to end (as of 05:48)** | `run.py` computes variants, model search, research models, context pack, collection metadata, generation validation (`GenerationValidator` is instantiated), and the agent loop (opt-in). `build_artifact` accepts `generation_validations`, `model_search`, `variants`, `research_models`, and `collection`; `preview.py` renders each section; `cli.py` exposes `--model-search`, `--variants`, `--validate-generation`, and `--collection-json` on `run`. The four enabling settings still default to **False** |
| **Implemented but not wired** | `search_sessions` (no pipeline call, no Search documents emitted); `research/catalog.py` (no CLI or pipeline call); `hierarchy.place` (not routed from `classify.py`); 3 of 12 source adapters are unregistered (`kaggle`, `semantic_scholar`, `websearch`), so those queries fall through to `unknown_source` |
| **Contract only / partial** | `validation/generation_models.py` and `tests/contracts/fakes.py` are type/factory surfaces. Planned fakes for the validation corpus, fake source responses, fake catalog, fake external backend, and hierarchy fixture are still missing. `eval/model_search.py` prefers `whack26.model_search.pipeline` and has no test pinning the harness-to-pipeline signature |
| **Not found at snapshot** | `pytest.ini`/`pyproject.toml`; `requirements-model-search.txt`; `whack26/validation/generation_report.py`; CLI `catalog`, `verify-hierarchy`, and a standalone `validate-generation` command |

Baseline: **950 collected, 948 passed, 2 skipped in 273.03s** under
`.\.venv\Scripts\python.exe` (confirmation run 05:49–05:53). Earlier scans saw
553 and 880 tests; one run during the active writes reported 3 failures
(`test_model_search_eval.py::test_generators_are_deterministic_and_complete`,
`...::test_target_kind_inference_handles_every_golden_dataset`,
`test_validation_collection.py::test_redact_text_leaves_dates_alone`), but all
three pass in isolation and passed on the clean reruns, so they were write-race
artifacts. `tests/test_run_flow.py` changed during the confirmation run; it
passes on its own (13 passed). `README.md` still claims **551**.
`whack26 schema` was regenerated at 05:49 and now matches the model
(`artifact_json_schema() == committed`), so DOC-25 is closed; keep the equality
test to prevent drift. There is no `pytest.ini`/`pyproject.toml`, so the
`integration` marker promised by both plans is not registered.

> **Quiescence gate.** Do not schedule work or file defects from a suite run
> taken while `git status`/mtimes are moving. Confirm no concurrent writer with
> `Get-ChildItem whack26,tests,eval -Recurse -File -Include *.py |
> Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) }`, then run
> the suite once, with no second `pytest` process running.

### 0.1 Implementation gaps the docs must not paper over

| Gap | Evidence | Consequence |
|---|---|---|
| Default-off features | `model_search_enabled`, `variants_enabled`, `generation_validation_enabled`, and `agent_loop_enabled` all default to `False`; the CLI flags exist but are opt-in | Document the flags and the env equivalents; do not claim the run does this by default |
| Agent loop | **Wired 05:46** — `run.py` constructs `ModelAgentLoop` when `agent_loop_enabled=True` | Safe to document as opt-in |
| Adapter registry | `_build_adapters` now registers nine sources; `kaggle`, `semantic_scholar`, and `websearch` remain unregistered and fall through to `unknown_source` | Document those three as "implemented, not registered" |
| Model-search tests | `run_model_search` is wired into `run.py` and the artifact, but `tests/` has no `test_model_search_pipeline.py`; the eval harness is the only caller besides `run.py` | Do not claim the orchestrator is unit-tested |
| Search sessions | `search_sessions.py` exists and is tested, but no pipeline call site | Do not claim Search documents are emitted |
| Catalog | `research/catalog.py` exists and is tested, but there is no `catalog` CLI command or pipeline call | Document as library-only |
| Hierarchy CLI | `verify-hierarchy` is absent; `classify.py` does not call `hierarchy.place` | Document hierarchy as library-only |
| Marker registration | No `pytest.ini`/`pyproject.toml`/`conftest.py` | No `integration` marker yet |

**Freeze rule (put this in the docs index):**

> A plan document is non-normative. No document may say "implemented" for a
> symbol, CLI command, env var, schema field, or behavior unless a test in
> `tests/` exercises it and the full suite is green. A module is not
> "implemented" for user-facing docs until it is wired into the run/artifact/CLI
> path or explicitly labeled library-only.

---

## 1. Documentation validation plan

### 1.1 Authority map

Documentation must point at one authority per fact:

| Fact | Authority |
|---|---|
| Artifact fields, required/optional, version | `whack26/artifact.py` + generated `schemas/artifact.schema.json` |
| CLI commands and flags | `whack26/cli.py::build_parser` |
| Env vars and defaults | `whack26/config.py::Settings.from_env` + `.env.example` |
| SQLite schema and migrations | `whack26/store.py` (`_SCHEMA`, `_ensure_v3_schema`, `_ensure_v4_schema`) |
| Source names, limits, priors | `whack26/research/base.py`, `whack26/research/pipeline.py` |
| Validation contract and statuses | `whack26/validation/models.py` (statistical) + `whack26/validation/generation_models.py` (generation) |
| Planned but absent behavior | the two plan docs, each marked **implementation plan** |

### 1.2 Documentation defect register

> Verified at **05:33**, then **P0 executed 06:05–06:20**. Resolved in P0:
> DOC-01, DOC-02, DOC-03, DOC-04, DOC-05, DOC-06, DOC-07, DOC-08, DOC-09,
> DOC-10, DOC-11, DOC-12, DOC-13, DOC-16, DOC-17, DOC-18, DOC-19, DOC-20,
> DOC-21, DOC-23, DOC-24, DOC-25. Still open: **DOC-14** (source docstring, left
> to the schema owner), **DOC-15** (migrated vs fresh `field_operations`
> constraint, source), and **DOC-22** (test rename). The rows below keep the
> original evidence for auditability.

| ID | File:line | Defect | Action |
|---|---|---|---|
| DOC-01 | `README.md:264` | Says 551 tests; 950 collected / 948 passing in the latest run | Remove hardcoded counts or update at the final gate; prefer a generated status line |
| DOC-02 | `README.md:253` | Docs index calls the store "SQLite v2" | v4; link to the migration section |
| DOC-03 | `GLOSSARY.md:63` | **Resolved 05:48** — now says "SQLite v4 knowledge store" | Keep the v4 wording |
| DOC-04 | `docs/eval/README.md:94` | **Resolved 05:52** — now says "SQLite v4 migration" | Keep the v4 wording |
| DOC-05 | `docs/spec/semantic-scraping.md:30-32` | **Resolved 06:10** — added the schema-history note; present-tense lines now say v4 / `1.3.0` | Keep the v2→v3→v4 note |
| DOC-06 | `docs/spec/README.md:21` | "Out" list says credibility scoring and UI are out of scope, but both exist | Move implemented items into a "now in scope" paragraph |
| DOC-07 | `docs/spec/README.md:158` | Decision table says "create + assign only"; the branching plan now covers promote/swap | Mark the row "superseded by field-branching plan" |
| DOC-08 | `docs/spec/README.md:35,64` | Architecture still lists five sources and no status column; line 64 still says "SQLite v2 schema migration" | Add status column, show planned adapters separately, and fix the v2 wording |
| DOC-09 | `docs/spec/README.md:80-126` | **Resolved 05:49** — layout now lists `hierarchy/`, `model_search/`, `validation/`, the adapters, `variants.py`, `context.py`, `agent_loop.py`, and the eval harnesses | Add `tests/contracts/` when the fakes land |
| DOC-10 | `docs/spec/field-classification.md:7` | "Promotion, rotation, and level-shifting are deferred" contradicts the branching plan | Add a superseded-in-part banner and link the plan section |
| DOC-11 | `docs/spec/research-pipeline.md:170` | `SourceName` literal lists only five sources; code has twelve | Update the type and add an adapter-status table (implemented vs endpoint-only vs missing) |
| DOC-12 | `docs/spec/agentic-hypotheses-validation-and-sources.md:204,596,725` | Uses `ValidationResult` where code has `GenerationValidationResult`; §6.0 puts `CollectionMetadata` in `validation/models.py`, but it is in `generation_models.py` | Reconcile names and file paths; add a contract-freeze checklist |
| DOC-13 | `docs/spec/agentic-hypotheses-validation-and-sources.md:721-735` | The §6.0 pre-flight list is stale in both directions: `variants.py`, `to_model.py`, and `catalog.py` now exist and are tested, but the planned fake corpus/source/catalog/external-backend fakes do not; `generation_models.py` is the real home of the new validation types | Mark the freeze as "modules landed; fakes and integration partial"; list what is actually frozen |
| DOC-14 | `whack26/store.py:290` | Docstring says "Create the v2 schema or migrate a v1 database" | Describe v1→v2→v3→v4 |
| DOC-15 | `whack26/store.py:172-176` vs `whack26/store.py:477-495` | Fresh databases declare the `CHECK` on `field_operations.operation`; the v3 migration creates the same table without it, so migrated databases have a weaker constraint. The migration also uses `ON DELETE SET NULL` instead of a plain reference | Make fresh and migrated schemas identical, or document the migration difference; test both paths |
| DOC-16 | `.env.example` | None of the new `WHACK26_MODEL_SEARCH_*`, `WHACK26_AGENT_*`, `WHACK26_STATISTICAL_BACKEND`, `WHACK26_HIERARCHY_*`, `WHACK26_MAX_ROOTS/CHILDREN`, `WHACK26_VARIANTS_ENABLED`, or `WHACK26_GENERATION_VALIDATION_ENABLED` fields are documented | Add a commented block with defaults and mark each wired vs library-only |
| DOC-17 | `docs/spec/artifact.md:299-307` | **Resolved 06:12** — `1.3.0` now documents all emitted sections, and the schema example carries `1.3.0` | Keep the versioning history current |
| DOC-18 | `docs/eval/semantic-scraping-smoke.md:8-19,54,80` | Historical snapshot says v2 / `1.1.0` / 526 tests | Add a dated "historical record" banner; do not keep updating it |
| DOC-19 | `docs/spec/demo-runbook.md` | No `--model-search`, `--variants`, `--validate-generation`, or `--collection-json`/`--collection-field`; prerequisites still describe the old slice | Add the new run flags and v4 notes |
| DOC-20 | `GLOSSARY.md` | **Resolved 06:15** — all new terms are defined, including `directly related` and `statistical backend` | Keep the glossary in sync with new domain terms |
| DOC-21 | `requirements-model-search.txt` | Referenced by the model-search plan, but absent | Either add the optional requirements file or mark it pending |
| DOC-22 | `tests/test_store_documents.py:58` | Test is still named `test_new_store_creates_v2_schema` but asserts v4 | Rename to `..._v4_schema` |
| DOC-23 | `docs/spec/agentic-hypotheses-validation-and-sources.md:760` | Plans `whack26/validation/generation_report.py`, which does not exist; the code has `generation.py` + `generation_models.py` | Rename the planned file or mark it superseded |
| DOC-24 | `docs/spec/field-branching-and-model-search.md:3` | Opens with "the current SQLite v2 document" | Update to v4, consistent with DOC-02…DOC-05 |
| DOC-25 | `schemas/artifact.schema.json` vs `whack26/artifact.py:286-337` | **Resolved 05:49** — the model emits `model_search`, `variants`, `research_models`, `generation_validation`, `generation_validation_summary`, and `collection`; the schema was regenerated and now equals `artifact_json_schema()` | Keep the schema-equality test so it cannot drift again; decide on the version bump (DOC-17) |

### 1.3 Automated documentation tests

Add `tests/test_docs.py` (no live network, no provider calls). These are cheap,
deterministic, and catch the defect class above on every PR.

| Proposed test | Assertion |
|---|---|
| `test_markdown_links_resolve` | Every relative link in `README.md`, `GLOSSARY.md`, `docs/**/*.md` resolves to an existing path |
| `test_spec_index_lists_every_spec` | Every `docs/spec/*.md` except `README.md` appears in the index table |
| `test_every_spec_has_status_header` | Each `docs/spec/*.md` starts with a `Status:` line; plan docs contain "implementation plan" |
| `test_docs_do_not_claim_missing_symbols` | For every identifier promised by a plan doc, at least one of (symbol exists, doc says "planned") is true; start with a curated allowlist |
| `test_schema_file_matches_model` | `json.loads(schemas/artifact.schema.json) == artifact_json_schema()` |
| `test_docs_schema_version_matches_constant` | `SCHEMA_VERSION` appears in `README.md` and `docs/spec/artifact.md` |
| `test_cli_commands_documented` | The set of `sub.add_parser` names equals the command list in `README.md` and `docs/spec/README.md` |
| `test_env_vars_documented` | Every `WHACK26_*` read by `config.py` appears in `.env.example` or docs; every documented var is read |
| `test_source_lists_consistent` | `SourceName` values, `QUERY_SCHEMA` enum, `SOURCE_LIMITS`, `SOURCE_PRIOR`, and `SOURCE_RANK` keys are identical; docs source table matches |
| `test_artifact_example_validates` | Extract the JSON block from `docs/spec/artifact.md` and validate it with `jsonschema` or the Pydantic model |
| `test_no_secrets_in_docs` | Scan docs for key-like patterns (`AIza…`, `sk-…`, bearer tokens) |
| `test_glossary_terms_exist` | Terms marked `**Term**` in `GLOSSARY.md` that are used in `docs/spec/` are defined (advisory; warn-only until clean) |

Optional, but valuable: `test_referenced_paths_exist` — any backticked
`whack26/...py`, `tests/...py`, or `docs/...md` path in the spec docs exists.

Do **not** write a test that shells out to `pytest --collect-only` to verify the
README count. It is slow, recursive, and will rot again. Generate the count at
release time or remove it.

### 1.4 Review gates

Per package/PR:

1. The owning doc section is updated in the same PR as the code.
2. New public symbol, CLI flag, env var, or schema field has a doc line and a
   test.
3. The status label is accurate: `implemented`, `contract only`, or `planned`.
4. `pytest -q tests/test_docs.py` is green.
5. No forward claims: "will", "planned", "not yet" is used for absent behavior.

Merge gate for the whole update:

1. Full suite green under `.venv`.
2. `whack26 schema` regenerated; `git diff --exit-code schemas/artifact.schema.json`.
3. `tests/test_docs.py` green.
4. A human executes the README quickstart, `run --validate`, `validate-models`,
   and the four default-off run flags.
5. The defect register above is closed or explicitly deferred with an owner.

---

## 2. Additional test cases

### 2.1 Test infrastructure (do first)

- Add `pytest.ini` (or `pyproject.toml`) with:
  - `markers = integration: requires the real v4 schema, subprocesses, or optional deps`
  - `filterwarnings` only if the team agrees; do not hide Pydantic warnings.
- Extend `tests/contracts/fakes.py` with the pieces both plans promise:
  - in-memory hierarchy store / tree fixture;
  - fake validation corpus and critic result;
  - fake source HTTP responses;
  - fake document catalog;
  - fake external statistical backend;
  - builders for `DatasetVariant`/`VariantReport`, `ResearchModelSpec`, and
    `CatalogQuery` (the modules now exist; the fake builders do not).
- Add a contract smoke test that instantiates every frozen type from the spec
  examples. This is the only test allowed before behavior lands.
- Keep the default suite offline. `@pytest.mark.integration` for live APIs,
  subprocess-heavy paths, and optional `sklearn`/`statsmodels`/`pysindy`.

### 2.2 Implemented surface — write now

#### Schema and migrations — `tests/test_store_migrations.py` (new)

| Test | What it proves |
|---|---|
| `test_v2_to_v3_backfills_leaf_from_max_ordinal` | v3 leaf backfill chooses the last `document_fields.ordinal` |
| `test_v2_to_v3_backfills_leaf_from_path_json` | datasets without memberships fall back to the last `path_json` entry |
| `test_v2_to_v3_preserves_documents_datasets_and_embeddings` | no data loss; cache keys and vectors survive |
| `test_v3_to_v4_adds_collection_and_research_models` | v4 DDL on an existing v3 database |
| `test_v4_migration_is_idempotent_on_reopen` | reopening does not duplicate rows or reset `user_version` |
| `test_v1_to_v4_chain_preserves_data` | the documented v1→v2→v3→v4 path on one database |
| `test_migration_rolls_back_on_failure` | a failing migration leaves the prior version and data intact |
| `test_migration_runs_foreign_key_and_integrity_checks` | corruption or FK violations raise `FieldStoreError` |

#### Store APIs — `tests/test_store_documents.py` / `tests/test_store.py`

- `test_research_model_upsert_and_filter`: insert, update by id, filter by
  `document_id` and `applied`.
- `test_research_model_cascade_delete`: deleting a document removes its models.
- `test_collection_json_roundtrip_null_and_invalid`: null, malformed JSON,
  non-dict JSON, and missing dataset all return `None` without raising.
- `test_field_operation_audit_order_and_limit`: newest-first ordering, limit
  clamping, and the documented operation vocabulary.
- `test_field_operation_constraint_matches_doc`: either the DB rejects an
  unknown operation or the doc says it is free text (see DOC-15).
- `test_hierarchy_version_default_bump_persist_and_corruption`:
  default `1`, monotonic bump, survives reopen, corrupt value falls back to `1`.
- `test_set_leaf_missing_document_and_missing_field`: no-op for unknown document;
  FK failure for an unknown field when FKs are on.

#### Config — `tests/test_config.py`

- `test_new_settings_defaults`: model-search, agent, statistical, and hierarchy
  defaults match the plan tables.
- `test_new_settings_env_overrides`: each `WHACK26_*` maps to the right field.
- `test_new_settings_invalid_matrix`: parameterize `model_search_folds=0`,
  test/final fractions outside `[0,1)`, unknown backends, agent budgets `<= 0`,
  hierarchy taus outside `[0,1]`, negative margins, `max_roots/max_children <= 0`.
- `test_empty_statistical_backend_falls_back_to_auto` and
  `test_external_validator_unset_is_none`.

#### Artifact, schema, preview — `tests/test_artifact.py`

- `test_committed_schema_matches_generated_schema` (JSON equality; see 1.3).
- `test_schema_version_is_1_2_0` — or `1.3.0` if DOC-17 decides the new sections
  warrant a minor bump.
- `test_v1_0_and_v1_1_artifacts_still_load` (1.0 is partly covered; add 1.1).
- `test_unknown_minor_warns_but_loads` in addition to the existing major reject.
- `test_validation_summary_counts_only_present_hypotheses`: extra report IDs are
  ignored; missing IDs are not counted.
- `test_artifact_roundtrip_with_every_validation_status`: passed, failed,
  partial, skipped, error.
- `test_preview_renders_validation_statuses_and_no_paths`: all statuses appear;
  no absolute path, key, or raw prediction array leaks.
- `test_build_artifact_without_validations_keeps_summary_none`.

#### Statistical validation — `tests/test_validation_*.py`

Add to the existing coverage:

- Runner target gate matrix: binary (already), multiclass, count, constant,
  non-finite, short, and small-integer-coded.
- Prediction failures: NaN, `inf`, wrong length, missing predictions file,
  duplicate index labels, unaligned row index.
- Adapter: datetime and category encoding, `max_features`/`max_categories`
  truncation, all-null column, unseen category at predict time, `periods_per_year`
  inference for hourly/daily/monthly/yearly plus explicit override.
- Pipeline: relative vs absolute artifact path, missing artifact, legacy
  hypothesis with no `validation` fields, `--out`, `--write-artifact` false/true,
  and CLI exit codes for invalid input.
- Reproducibility: same seed and config produce the same report ignoring
  `runtime_ms`.
- Isolation: a hypothesis that writes to stdout/stderr must not corrupt the JSON
  report; a timeout is `error`, not a crash.

#### Hypothesis validation surface — `tests/test_hypotheses_validate.py`

- `_fit` first argument not `df`, `_predict` wrong second argument.
- `PARAMETERS` missing, non-dict, empty, boolean, zero, negative, and mixed.
- `_predict` reads the target; more than one warning is reported.
- Warnings remain warnings, not errors; the generated module still executes.
- Warnings surface in the artifact and preview.

#### Research registry — `tests/test_research_pipeline.py`, `test_research_base.py`

- `test_source_registry_is_consistent` (see 1.3).
- `test_unknown_query_source_warns_and_skips`: nine adapters are registered;
  `kaggle`, `semantic_scholar`, and `websearch` still produce `unknown_source`.
  Pin the current fallback for those three and update the test as each adapter is
  registered.
- `test_query_schema_accepts_every_source`: one valid query per enum value.
- `test_normalize_maps_new_dataset_sources_to_dataset_kind`: enum expansion must
  not change document kind for paper sources.
- `test_default_suite_makes_no_network_calls`: monkeypatch the HTTP client and
  assert no call in mock/offline paths.

#### CLI — `tests/test_cli.py`

- `validate-models` defaults and pass-through for every threshold flag.
- `validate-models` invalid `--out`, missing path, non-run directory.
- `run --validate` forwards all `--validation-*` values; omitting it performs no
  validation.
- `run --model-search` / `--variants` / `--validate-generation` set the
  corresponding settings override; omitting them preserves the default.
- `run --collection-json` and repeated `--collection-field KEY=VALUE` merge and
  reach the artifact; malformed JSON and non-dict JSON fail cleanly.
- `schema --out` writes the committed schema; `--help` lists the new command.
- `doctor` and `--version` remain unaffected.

### 2.3 Newly landed packages — verify coverage, then add integration

Most packages now have unit tests. Do not re-plan them from scratch: run the
existing files first, then add only the gaps. Exact counts at the latest
snapshot: hierarchy 42 (place 10, operations 11, paths 10, eval 11), model
search 154 across 11 files, variants 12, to-model 15, agent loop 17, context 9,
catalog 10, search sessions 9, generation 13, collection 16, backends 29,
adapters 3–21 each.

These are the acceptance tests both plans specify; keep new ones in the package
owner's PR and mark integration tests with the marker.

#### Hierarchy (`whack26/hierarchy/`)

**Status:** `test_hierarchy_place.py`, `test_hierarchy_operations.py`,
`test_hierarchy_paths.py`, and `test_field_hierarchy_eval.py` exist. Remaining:
real-store integration, property tests over random trees, and the
`verify-hierarchy` CLI.

- Place: attach at root, descend, branch, create root, duplicate merge,
  max-depth attach, deterministic margin fallback vs LLM judge.
- Operations: promote a leaf, promote a subtree (all descendants +1), reject a
  promotion that exceeds max depth, reassign closer documents, keep the rest.
- Paths: `recompute_paths` and `verify_paths`; cycles, missing parents, bad
  ordinals, `path_json` mismatch, sibling duplicates.
- Property tests: random trees and operation sequences preserve strict tree,
  `level = parent.level + 1`, reachable leaves, finite normalized centroids.
- Migration tests already listed in 2.2.
- Evaluation harness: `eval/field_hierarchy.py`; 80% exact/ancestor gate.

#### Model search (`whack26/model_search/`)

**Status:** tests exist for `types`, `data`, `splits`, `screen`, `candidates`,
`fit`, `metrics`, `select`, `diagnostics`, `report`, optional backends, and the
`eval/model_search.py` harness (`test_model_search_eval.py`, 22 tests).
`model_search/pipeline.py::run_model_search` landed at 05:21:26 and the harness
now prefers it over the local baseline, but there is **no dedicated test for the
orchestrator** and no test that pins the harness-to-pipeline signature. Missing:
`test_model_search_pipeline.py`, a golden test that asserts the production path
(not the fallback) recovers the known families, and run/artifact/preview wiring.

- Typing: continuous, binary, multiclass, count, time series; id/time/leakage
  exclusion.
- Splits: repeated holdout, stratified, chronological, group-aware; no
  train/test leakage; final holdout reserved before repeated splits.
- Screening: Pearson/Spearman, MI, ANOVA/chi-square, VIF, BH correction.
- Candidates: correct family per target kind; transforms; interactions;
  candidate cap.
- Fit/metrics: finite predictions, transform inversion, R²/RMSE/MAE, accuracy,
  balanced accuracy, AUC, log loss, deviance, AIC/BIC.
- Selection: baseline margin, stability, parsimony, parameter cap, top-N.
- Diagnostics: heteroskedasticity, VIF, autocorrelation, overdispersion,
  stationarity.
- Orchestrator (`run_model_search`): no-target and too-few-rows skip codes;
  warning propagation from splits/screen/fit; fold aggregation; baseline
  selection; deterministic seed; final holdout excluded from fitting; empty
  candidate list returns `model_search_no_candidates` rather than raising.
- Golden sets: linear, interaction, logistic, multiclass, overdispersed count,
  trend+seasonality, categorical group means, missing/collinear, small-n.
- Acceptance: correct family in top 2; no target-derived feature; fold variance
  reported; time budget on 20k×10.

#### Agentic generation

**Status:** `tests/test_hypotheses_variants.py`, `test_scrape_to_model.py`,
`test_agent_loop.py`, and `test_hypotheses_context.py` exist. `run.py` builds
variants, research models, the context pack, and `ModelAgentLoop` (opt-in),
injects them into generation, and persists variants/research models through the
artifact and preview. Still missing: an end-to-end jumbled-label gate and a test
proving the agent loop is selected when `agent_loop_enabled=True`.

- Variants: deterministic original/anonymized/jumbled, no mutation, PII removal,
  jumbled marginal preservation, time-series block shuffle, group shuffle,
  leaky model rejected, clean model accepted.
- Research-to-model: evidence required, predictor mapping, unsupported model
  rejection, no target leakage, `applied` flag.
- Agent loop: refine/replace/discard, budgets, family uniqueness, no infinite
  loop, jumbled leak rejection, audit file.
- Context pack: ancestor inclusion, sibling exclusion, token cap, empty
  ancestors, deterministic order, no PII, traceable IDs.

#### Validation plan

**Status:** `tests/test_validation_adapter.py`, `_backends.py`,
`_collection.py`, `_engine.py`, `_generation.py`, `_pipeline.py`, and
`_runner.py` exist. There is no dedicated `test_validation_statistical.py`;
verify `statistical.py` through `test_validation_backends.py` or add one.
`run.py` now instantiates `GenerationValidator`, and `build_artifact`/`preview`
accept the generation-validation payloads. Remaining: a test that the validator
actually runs under `--validate-generation`, backend selection through `run.py`,
and an artifact round-trip for a legacy artifact.

- Collection metadata: schema, causal gate, missing fields, PII redaction,
  artifact round-trip, CLI `--collection-json` / `--collection-field`.
- Generation validator: observational causal claim rejected, experimental
  accepted, missing treatment rejected, known bias acknowledged, clean pass,
  deterministic checks override the LLM.
- Statistical backends: `in_slice` metrics/permutation, `stats_engine` only for
  return-like series, external seam with a fake adapter, no applicable backend
  yields `insufficient`, no `stats_engine` fallback for binary/count/multiclass/
  constant/general regression.
- Backend selection: `auto`, `in_slice`, `stats_engine`, `external`, `none`;
  env var precedence; version recorded.
- Artifact/preview: `generation_validation`, summary, handoff; `run
  --validate-generation` (already exposed); no duplicate shadow command.

#### Sources and search

**Status:** per-adapter tests exist for all seven new sources; `test_research_catalog.py`
and `test_research_search_sessions.py` exist. Nine of twelve adapters are
registered. Remaining: register `kaggle`, `semantic_scholar`, and `websearch` in
`_build_adapters`, call search sessions from the pipeline, and expose the catalog
via CLI.

- Each adapter: fixture parse, auth failure, rate limit, empty results, dedup,
  license, normalization, no secrets.
- Web search: fixture response, no-key fallback with warning, quota warning,
  URL canonicalization, dedup.
- Search documents: one doc per query, idempotent rerun, result links,
  search-to-search links, dedup.
- Catalog: each filter, subtree filter, vector ordering, empty result,
  pagination, deterministic order, 1,000-document performance.

### 2.4 Integration wiring tests — highest priority

The packages are largely built and unit-tested, and `run.py` now wires most of
them into `build_artifact`/`preview`/CLI. What remains is tests that prove the
wiring, the agent-loop and search-session/catalog paths, and the adapter
registry. Write these before adding more package-level edge cases.

| Test | What it proves |
|---|---|
| `test_run_pipeline_with_model_search` | `run_pipeline` calls the orchestrator, injects `ModelSearchResult` into `HypothesisContext`, and persists it through `build_artifact` |
| `test_model_search_eval_uses_production_orchestrator` | `eval/model_search.py` resolves `whack26.model_search.pipeline.run_model_search` and calls the production path; assert the fallback is used only when the module is monkeypatched away, and pin the harness-to-pipeline signature |
| `test_artifact_contains_model_search_section` | `artifact.json` has candidates, metrics, diagnostics, selected model, backend, seed |
| `test_preview_renders_model_search_table` | preview renders the top candidates without leaking paths |
| `test_run_pipeline_generates_variants` | `run.py` writes `variants.json`, injects the report, and the artifact/preview include it; the jumbled gate rejects a leaky synthetic model |
| `test_generation_validation_is_actually_run` | `GenerationValidator` is instantiated when `WHACK26_GENERATION_VALIDATION_ENABLED=true`, results reach `generation_validation`/`generation_validation_summary`, and legacy artifacts still load |
| `test_run_pipeline_uses_agent_loop_when_enabled` | `agent_loop_enabled=True` uses `ModelAgentLoop`; false keeps the one-shot generator |
| `test_validate_generation_flag` | `run --validate-generation` enables the review; omitting it leaves the sections empty |
| `test_collection_cli_roundtrip` | `--collection-json` persists and appears in the artifact |
| `test_run_research_registers_all_adapters` | every `SourceName` has a registered adapter or an explicit "not implemented" sentinel |
| `test_search_sessions_created_by_pipeline` | one Search document per query, idempotent rerun, result links |
| `test_catalog_cli_query` | `catalog query` filters by kind/source/field/year/similarity and paginates |
| `test_verify_hierarchy_cli` | `verify-hierarchy` exits non-zero on a corrupted path |
| `test_hierarchy_placement_through_classifier` | `classify.py` routes to `hierarchy.place`, with the old create/assign fallback |

### 2.5 Cross-cutting tests

- **Optional dependencies:** import every new package with `sklearn`,
  `statsmodels`, and `pysindy` absent; core paths must still work.
- **Determinism:** same seed and inputs produce identical artifacts, reports,
  and variant reports (excluding timestamps/runtime).
- **Privacy:** no PII or secrets in prompts, logs, cache metadata, artifact, or
  preview. Reuse the existing redaction tests.
- **Windows:** tests must not assume LF line endings, POSIX paths, or
  `os.sep == "/"`. The repo runs on win32 and Git warns about CRLF.
- **Failure isolation:** one failed source, one failed hypothesis, or one failed
  validator must not abort the run.

---

## 3. Execution order

| Phase | Work | Exit gate |
|---|---|---|
| **P0 — Doc triage (0.5 day)** | Close the open DOC items (DOC-04, DOC-09, and DOC-25 were resolved during verification); add status banners; update the spec index; add `.env.example` block | `tests/test_docs.py` green; no "implemented" claim for absent behavior |
| **P1 — Remaining integration (1–2 days)** | §2.4 tests for the path that already landed (model search, variants, research models, generation validation, collection, artifact/preview, agent loop). Then wire what is still absent: search sessions, catalog + `verify-hierarchy` CLI, `classify.py` → `hierarchy.place`, and register the three remaining adapters (`kaggle`, `semantic_scholar`, `websearch`). | Each feature is reachable from `run`/`artifact`/`preview`/CLI or explicitly labeled library-only |
| **P2 — Coverage fill (1 day)** | `test_model_search_pipeline.py`, `tests/test_store_migrations.py`, config matrix, schema equality, validation edge cases, hypothesis warnings, property tests | Full suite green under `.venv`; no regressions |
| **P3 — Whole-update audit (0.5 day)** | Full acceptance matrix, README status line, GLOSSARY reconciliation, human quickstart | All gates in §4 |

Do not launch P1/P2 before P0 freezes the status labels. Otherwise every package
will reconnect its doc to a moving target. Re-run the §0 state check immediately
before scheduling: the tree changed continuously between 05:21 and 05:33, and
the integration wave landed while this plan was being verified.

---

## 4. Acceptance criteria

- `pytest -q` (default, offline) is green; `pytest -m integration` is green.
- `tests/test_docs.py` passes: links, status headers, schema equality, CLI
  coverage, env coverage, source-registry consistency.
- `whack26 schema` regenerates the committed `schemas/artifact.schema.json` with
  no diff; the new artifact sections (`model_search`, `variants`,
  `research_models`, `generation_validation`, `collection`) are present in it
  (DOC-25).
- Every new public symbol, CLI flag, env var, and schema field has at least one
  test.
- No plan document claims a behavior that has no test.
- The README quickstart, `run --validate`, `validate-models`, and the four
  default-off run flags all execute.
- The defect register is closed or explicitly deferred with an owner.
- Test counts quoted in docs are generated at release time, not hand-edited.

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Docs drift while packages land | Status manifest and per-PR doc gate; freeze labels in P0 |
| Plan docs are mistaken for specs | Banner + "implemented / contract only / planned" table in the index |
| Hardcoded counts rot | Remove or generate; do not test via nested pytest |
| Integration tests are flaky or network-bound | `integration` marker; default suite offline; fixture-based adapters |
| Concurrent suite runs share fixtures | Two `pytest` processes can write/read `tests/fixtures/model_search` at once and produce false failures; run one suite at a time and re-check any failure in isolation before filing it |
| Concurrent edits to `store.py`, `run.py`, `artifact.py` | One owner per file, as both plans already specify; doc updates in the owner's PR |
| Schema drift | Equality test + `whack26 schema` diff in CI |
| Windows path/CRLF assumptions | Cross-platform path tests; no LF assertions |
| Optional deps break core | Import-isolation tests with modules absent |
| `field_operations` constraint mismatch hides bad writes | Resolve DOC-15 before hierarchy operations land |

---

## 6. Immediate next actions

1. Re-run the §0 state check **with `.venv\Scripts\python.exe`** (bare `python`
   is a different interpreter and reports 72 bogus import errors) **and wait for
   the tree to quiesce** — the integration wave landed between 05:21 and 05:33
   while this plan was being verified. Fix the status labels before anything
   else.
2. Add the `implementation plan` banner to
   `docs/spec/agentic-hypotheses-validation-and-sources.md` and
   `docs/spec/field-branching-and-model-search.md`; state what is actually
   frozen, and correct DOC-12/DOC-13 (module names, `generation_models.py`,
   missing fakes) plus DOC-23/DOC-24 (absent `generation_report.py`, stale v2
   line).
3. Fix DOC-01/DOC-02/DOC-03/DOC-05 (950-vs-551 count and v2→v4) in `README.md`,
   `GLOSSARY.md`, and `docs/spec/semantic-scraping.md`. DOC-04 is already
   resolved in `docs/eval/README.md`.
4. Add `pytest.ini` with the `integration` marker.
5. Register the three remaining adapters (`kaggle`, `semantic_scholar`,
   `websearch`) in `research/pipeline._build_adapters` and add the registry
   consistency test. This is a real user-facing gap, not a doc issue.
6. Add the §2.4 tests for the wiring that just landed: model search, variants,
   research models, generation validation, collection, and artifact/preview
   round-trips.
7. Wire what is still absent: search sessions from the research pipeline,
   `catalog` and `verify-hierarchy` CLI, and `classify.py` → `hierarchy.place`.
   The four default-off CLI flags already exist on `run`.
8. Add `tests/test_model_search_pipeline.py` for `run_model_search`, then
   `tests/test_docs.py` with the link, status, schema, CLI, env, and source
   consistency checks.
9. Add `tests/test_store_migrations.py` for v2→v3→v4 and idempotency, including
   the migrated-vs-fresh `field_operations` constraint difference (DOC-15).
10. Add the config invalid/override matrix and the committed-schema equality test.
11. Extend `tests/contracts/fakes.py` with the fake corpus, source responses,
    catalog, external backend, and hierarchy fixture; replace the README count
    with a generated status line.
