# Field branching, swapping, and automated regression model search

Status: **implemented in part**. Model search is implemented and wired; the
hierarchy is implemented and used for scraped documents. The SQLite schema is
v4, the hierarchy package is `whack26/hierarchy/`, and model search is
`whack26/model_search/`. `run.py` wires model search behind `--model-search`
(or `--full`); `research/scrape/classify.py` uses hierarchy placement, while
`classify.py` still uses create/assign for user datasets. Tree/forest/boosting
and SINDy are wired through `--backend sklearn|full`; the default `core`
candidate pool remains numpy/scipy-only. Partial-dependence plots are not
implemented.

> **Historical build plan.** Sections that describe the original v3 schema,
> the pre-wiring state, and the parallel build waves are kept as the design
> record. The current authorities are `whack26/store.py`,
> `whack26/hierarchy/`, `whack26/model_search/`, and the spec index.

## 1. Goal

### Field branching and swapping

Every dataset, paper, search, and web document should be attached to the
**highest field in the hierarchy that it is directly related to**, rather than
being buried under an over-specific child. When no existing field expresses the
document's concept, create a branch at the correct level. When a newly created
field turns out to be more general than its intended parent, swap the two and
re-level the parent's subtree.

This is the `devplan.md` hierarchy-maintenance behaviour:

- find the closest field;
- if the field is too distant, ascend to its parent and repeat;
- if a field is related but too distant from its parent, create a subfield;
- if a new subfield is actually the parent of the current parent, promote it,
  move the current parent and its subtree down one level, and reassign
  documents that are closer to the new field.

### Automated regression model search

The devplan asks for static analysis that finds models: repeated holdouts,
correlation/mutual-information/ANOVA screening, regressions with transforms,
GLMs for binary/categorical/count outcomes, time-series checks, interactions,
tree models, and optionally SINDy. The current slice has a descriptive screen
and three hand-generated families; this plan adds an automated candidate search
that feeds the generator with ranked, validated model specifications.

The phrase "automated regression testing" is treated in two senses:

1. **Statistical search** — fit and compare regression/GLM/tree candidates.
2. **Software regression tests** — property and unit tests that prove hierarchy
   invariants and model-search behaviour do not regress.

## 2. Current state

- `whack26/store.py`: SQLite v4 `fields`, `documents`, `datasets`,
  `document_fields`, `document_links`, `document_comparisons`,
  `document_critiques`, `document_fetches`, `field_operations`,
  `research_models`.
- `whack26/classify.py`: top-down, top-k + LLM create/assign, max depth 4.
- `whack26/research/scrape/classify.py`: document counterpart; uses
  `whack26/hierarchy/place.py` for placement by default.
- `whack26/hierarchy/`: placement, branching, promotion, audited path
  recomputation, duplicate-sibling merges, and `field_operations` logging.
- `whack26/model_search/`: typed data/target analysis, screening, repeated and
  chronological splits, core candidates, optional tree/SINDy backends,
  selection, diagnostics, and report generation.
- `whack26/screen.py`: target distribution, missingness, Pearson/Spearman,
  group means/rates, time trend. No p-values, no model fitting.
- `whack26/hypotheses/`: three families (`linear_model`, `group_comparison`,
  `interaction_or_derived`), numpy/scipy, AST validation, subprocess runner,
  optional agent loop.
- `stats_engine.py`: four-layer validation engine, wired through
  `whack26/validation/` for continuous return-like targets and
  `stats_engine.validate_model` for general targets.
- `docs/devplan-gap-audit.md` lists the ambiguity in the devplan's hierarchy
  prose; this plan resolves it with explicit invariants and thresholds.

## 3. Part A — Field branching and swapping

### 3.1 Terminology

**Directly related**:
A document is directly related to field `f` when
`cos(document_embedding, field_score(f)) >= tau[level(f)]`, where
`field_score` blends the field description embedding and centroid with the
existing `desc_weight`/`centroid_weight`.

**Highest directly related field**:
The shallowest field (smallest level) that is directly related to the document
and is not merely a topical ancestor of a more specific field that also fits.
Operationally, start at the roots, descend only while a child is directly
related **and** the document is more specific than the current field; otherwise
attach at the current field.

**Branch**:
Create a new child under the highest directly related ancestor when no existing
child expresses the document's concept.

**Promote / swap**:
A newly created field `N` is judged to be the parent of its intended parent `P`.
Move `N` above `P`: `N` takes `P`'s old parent and level, `P` moves under `N`,
and `P`'s entire subtree moves down one level.

**Leaf**:
The field a document is attached to. Its path is the ancestor chain from the
root to the leaf. Existing `document_fields` stores that chain; this plan adds
`documents.leaf_field_id` as the explicit source of truth.

### 3.2 Original schema v3 (current is v4)

This is the **v3 target schema** from the original plan. It has since landed and
been extended to v4 by collection metadata and research models;
`whack26/store.py` is the current authority. Add to `cache/fields.db`:

```sql
ALTER TABLE documents ADD COLUMN leaf_field_id TEXT REFERENCES fields(id);

CREATE TABLE IF NOT EXISTS field_operations (
  id TEXT PRIMARY KEY,
  operation TEXT NOT NULL CHECK (
    operation IN ('create','merge','promote','reassign_leaf','recompute_path','prune','delete')
  ),
  field_id TEXT NOT NULL,
  old_parent_id TEXT,
  new_parent_id TEXT,
  old_level INTEGER,
  new_level INTEGER,
  document_id TEXT,
  affected_documents INTEGER NOT NULL DEFAULT 0,
  reason TEXT NOT NULL DEFAULT '',
  model TEXT,
  run_id TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS field_operations_field
  ON field_operations(field_id);
CREATE INDEX IF NOT EXISTS field_operations_created
  ON field_operations(created_at);

CREATE TABLE IF NOT EXISTS store_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
```

- Backfill `documents.leaf_field_id` from the last `document_fields.ordinal`
  (falling back to the last `datasets.path_json` entry).
- Backfill `store_meta['hierarchy_version'] = '1'`.
- Migration is transactional, idempotent, and runs after the v1→v2 migration.
- `field_operations` is the audit log: every create, promote, reassign, path
  rebuild, prune, and delete is recorded with the reason and the model.
- `hierarchy_version` is informational and audit-only; path correctness is
  enforced by `recompute_paths` and `verify_paths`, not by cache invalidation.

### 3.3 Placement algorithm

New module: `whack26/hierarchy/place.py`.

```python
def place_document(
    document: Document,
    *,
    store: FieldStore,
    llm: Any,
    embeddings: Any,
    settings: Settings,
) -> PlacementResult:
    ...
```

Pseudocode:

```
embedding = document.summary_embedding or embed(document.summary)
current = None
while True:
    candidates = store.children(current.id) if current else store.roots()
    scored = [(score(embedding, field), field) for field in candidates]
    direct = [item for item in scored if item.score >= tau[field.level]]
    current_direct = current is not None and score(embedding, current) >= tau[current.level]

    if not direct:
        if current is None:
            root = propose_root(document)          # no root fits
            return branch_and_attach(root, document)
        if current.level < settings.max_depth and is_more_specific(document, current):
            child = propose_child(current, document)
            return branch_and_attach(child, document)
        return attach(document, current)           # deepest valid concept

    best = max(direct, key=score)
    if current_direct and not is_more_specific(document, current, best):
        return attach(document, current)           # highest directly related

    if best.field.level >= settings.max_depth:
        return attach(document, best.field)

    current = best.field
```

`is_more_specific` is one LLM call per ambiguity, with a deterministic fallback:

- if `best.score - current.score > specificity_margin`: descend;
- if `current.score - best.score > specificity_margin`: attach at current;
- otherwise call the judge.

The walk only descends into a child that passes the level threshold, so a
non-null `current` is always directly related. The judge compares the document,
the current field, and the best direct child.

The judge schema:

```json
{
  "relation": "attach_here | needs_child | new_field_is_parent | new_field_is_child",
  "confidence": 0.0,
  "reason": "one sentence"
}
```

Prompt rules: prefer the highest field that genuinely covers the document; only
descend when the document is clearly about a narrower concept; never invent a
concept not supported by the summary; never create a root when a root already
covers the document.

`attach(document, field)`:

1. Set `documents.leaf_field_id = field.id`.
2. Recompute the document path from the field's ancestor chain.
3. Replace `document_fields` rows (ordinals, confidence) and, for `kind =
   dataset`, rewrite `datasets.path_json`.
4. Recompute affected field centroids.
5. Log a `reassign_leaf` or `create` operation.

### 3.4 Branch operation

New module: `whack26/hierarchy/operations.py`.

```python
def branch_field(
    *,
    parent: Field,
    name: str,
    description: str,
    description_embedding: Sequence[float],
    document: Document | None,
    store: FieldStore,
    llm: Any,
    settings: Settings,
) -> BranchResult:
    ...
```

Steps:

1. Validate `parent.level < max_depth`.
2. Reject an empty/overlong name or description.
3. Merge with a sibling if cosine >= `duplicate_cosine`; do not create, and log
   a `merge` operation.
4. Create the child.
5. Ask `specificity_judge(new_field, parent)`:
   - `new_field_is_parent`: call `promote(new_field, parent)`.
   - `new_field_is_child` or `uncertain`: keep the child branch.
6. If promotion is rejected because it would exceed `max_depth`, keep the new
   field as a child of `parent` and log `promotion_rejected_depth`.

### 3.5 Promote / swap operation

`promote(new_field, old_parent)`:

1. `grandparent = old_parent.parent_id`.
2. Compute the maximum descendant depth under `old_parent`.
3. If `old_parent.level + max_descendant_depth > settings.max_depth`, reject:
   - keep `new_field` as a child of `old_parent`;
   - log `promotion_rejected_depth`;
   - return the child branch.
   `max_depth` defaults to 4; raising the setting relaxes the guard and allows a
   deeper promoted subtree.
4. In one transaction:
   - `new_field.parent_id = grandparent`;
   - `new_field.level = old_parent.level`;
   - `old_parent.parent_id = new_field.id`;
   - `old_parent.level = new_field.level + 1`;
   - increment every descendant of `old_parent` by one level.
5. Reassign documents in the moved subtree:
   - For each document whose leaf is in `old_parent`'s subtree, including
     `old_parent`, compare `sim(document, new_field)` with
     `sim(document, old_parent)`.
   - If `sim(new_field) > sim(old_parent) + promotion_margin` and
     `sim(new_field) >= tau[new_field.level]`, set the document leaf to
     `new_field` (the document is directly about the new, more general concept).
   - Otherwise keep the existing leaf; its path now includes `new_field` above
     `old_parent`.
   - Borderline cases use the specificity judge; deterministic fallback is the
     margin rule above.
6. Recompute paths and centroids for every affected document and field.
7. Log a `promote` operation with the affected document count and reason.

The devplan's "if they are closer to the new field than the current parent,
assign them to the new field rather than the parent" is implemented exactly as
the comparison in step 5. Documents that remain under the old parent now have
the new field as an ancestor, preserving the hierarchy.

### 3.6 Path recomputation and consistency

New module: `whack26/hierarchy/paths.py`.

```python
def recompute_paths(
    store: FieldStore, *, field_ids: Iterable[str], run_id: str | None = None
) -> int:
    ...

def verify_paths(store: FieldStore) -> list[str]:
    ...
```

For every affected document:

1. Resolve `leaf_field_id`; if null, derive it from the maximum ordinal in
   `document_fields` or the last `datasets.path_json` entry.
2. Walk `parent_id` to the root; reject a cycle or a missing parent.
3. Build the root→leaf list.
4. Replace `document_fields` rows with ordinals and confidence.
5. For `kind='dataset'`, rewrite `datasets.path_json`.
6. Recompute centroids for every field in the old and new paths.

`verify_paths` checks:

- every non-null leaf exists and is reachable from a root;
- `document_fields` equals the ancestor chain in order;
- `datasets.path_json` equals the ids of that chain;
- no field level exceeds `max_depth`;
- no field is its own ancestor;
- every field has a unique normalized sibling name.

Run `verify_paths` after every hierarchy operation and expose it as
`whack26 verify-hierarchy`.

### 3.7 Thresholds and configuration

Add to `Settings` with the `WHACK26_` prefix:

| Variable | Default | Meaning |
|---|---:|---|
| `HIERARCHY_TAU_1` | `0.55` | Directly-related threshold at level 1 |
| `HIERARCHY_TAU_2` | `0.60` | Level 2 |
| `HIERARCHY_TAU_3` | `0.65` | Level 3 |
| `HIERARCHY_TAU_4` | `0.70` | Level 4 |
| `SPECIFICITY_MARGIN` | `0.05` | Deterministic descend/attach margin |
| `PROMOTION_MARGIN` | `0.05` | Reassign a document to the promoted field |
| `MAX_ROOTS` | `20` | Guard against root explosion |
| `MAX_CHILDREN` | `20` | Guard against branch explosion |
| `MAX_DEPTH` | `4` | Existing hard limit |
| `DUPLICATE_COSINE` | `0.90` | Existing sibling-merge threshold |
| `HIERARCHY_JUDGE_MODEL` | unset | Optional model override for specificity calls |

Thresholds are calibrated on the hierarchy evaluation set in §3.9. Until then,
they are documented defaults and the run records a `hierarchy_thresholds_default`
warning.

### 3.8 Interfaces and files

```
whack26/hierarchy/
  __init__.py
  types.py          # PlacementResult, BranchResult, PromotionResult, HierarchyError
  place.py          # place_document, specificity judge prompt/schema
  operations.py     # branch_field, promote_field, attach_document
  paths.py          # recompute_paths, verify_paths
  store_ops.py      # field_operations log, hierarchy_version, leaf helpers
```

`FieldStore` additions:

```python
def get_leaf(document_id: str) -> str | None: ...
def set_leaf(document_id: str, field_id: str) -> None: ...
def field_subtree(field_id: str) -> list[Field]: ...
def log_field_operation(...) -> None: ...
def hierarchy_version(self) -> int: ...
def bump_hierarchy_version(self) -> None: ...
```

### 3.9 Tests and evaluation

**Unit tests**

- Attach at root when the root is directly related and the document is not more
  specific.
- Descend when a child is directly related and the document is more specific.
- Branch when no child passes the threshold and the document is more specific.
- Create a root when no root passes the threshold.
- Duplicate sibling merges instead of creating.
- Max-depth attach instead of branching.
- Deterministic margin fallback vs LLM judge.
- Leaf/path/ordinal consistency after attach.

**Tree-operation tests**

- Promote a leaf with no descendants.
- Promote a field with a subtree; every descendant level increments.
- Reject promotion that would exceed the configured maximum depth.
- Reassign documents closer to the promoted field; keep the rest under the old
  parent with the new field as an ancestor.
- Prevent cycles and self-parenting.
- Audit rows for create/promote/reassign/prune.
- `verify_paths` catches corrupted paths.

**Property tests**

Random field trees and documents; random sequences of branch/promote/attach
operations; after every operation assert:

- strict tree, no cycles, `level = parent.level + 1`, `level <= max_depth`;
- every document leaf is reachable;
- `document_fields` and `datasets.path_json` match the ancestry;
- centroids are finite and normalized;
- no duplicate sibling names.

**Migration tests**

- v2→v3 backfills `leaf_field_id` for datasets and documents.
- Existing datasets, paths, embeddings, cache keys survive.
- Reopening is idempotent.
- A v1 database migrates through v2 to v3.

**Evaluation harness** (`eval/field_hierarchy.py`)

A labeled set of documents with a known "highest directly related field" (the
field that a human or a deterministic rule considers the correct level).
Metrics:

- exact match to the labeled field;
- ancestor match (attached to an ancestor of the labeled field);
- over-specificity rate (attached below the labeled field);
- under-specificity rate (attached above the labeled field);
- branch precision (new fields that a human accepts);
- promotion precision (swaps that improve the hierarchy);
- path consistency and audit completeness.

Start with 30–50 synthetic documents and 20–30 hand-labeled real documents. The
gate: no over-specific attachment when a directly related ancestor exists, and
at least 80% exact/ancestor match.

### 3.10 Failure handling and invariants

- All hierarchy operations run in a single SQLite `BEGIN IMMEDIATE`
  transaction; no concurrent writers.
- A failed operation rolls back; the tree and paths are unchanged.
- If the LLM judge fails, use the deterministic margin rule and log
  `hierarchy_judge_fallback`.
- If a promotion would exceed the configured maximum depth, keep the new field
  as a child and log
  `promotion_rejected_depth`.
- If `verify_paths` fails after an operation, roll back and raise
  `HierarchyIntegrityError`.
- Empty leaves are pruned with `delete_empty_leaf_fields`; intermediate fields
  may exist with descendants only.
- Root creation is guarded by `MAX_ROOTS` and duplicate checks.
- The audit log is append-only; never delete rows.

### 3.11 Lifecycle scope

This plan implements:

- create;
- duplicate-sibling merge;
- promote/swap and subtree re-leveling;
- reassign a document to the promoted field or keep it under the old parent;
- prune empty leaves and delete empty leaves.

The only merge/delete operations in this plan are **duplicate-sibling merge** and
**empty-leaf prune/delete**. Deleting or repairing a populated field and its
documents is not covered.

It does **not** implement:

- split a populated field into multiple fields;
- rename a field;
- manual delete/repair of a populated field and its documents;
- multi-writer concurrency or tree versioning beyond the audit log and
  `hierarchy_version`.

Those are follow-on lifecycle operations. The audit table and
`recompute_paths`/`verify_paths` are the seams for them.

## 4. Part B — Automated regression model search

### 4.1 Goals

Find a ranked set of statistically defensible model candidates for the dataset
and feed them to hypothesis generation and the artifact. The search must:

- infer the target type and feature types;
- use repeated holdouts or chronological splits with no leakage;
- screen features with correlation, mutual information, ANOVA/chi-square, and
  VIF;
- fit linear, logistic, multinomial, Poisson/negative-binomial, interaction,
  tree, and optional SINDy candidates;
- compare them against a baseline with appropriate metrics;
- report diagnostics and stability;
- select top candidates with a complexity penalty and multiple-testing
  awareness;
- never claim causal identification.

### 4.2 Package layout

```
whack26/model_search/
  __init__.py
  types.py          # TargetKind, FeatureKind, ModelSpec, FoldResult, ModelCandidate
  data.py           # target/feature typing, usable columns, missingness
  splits.py         # repeated holdout, stratified, chronological, group splits
  screen.py         # Pearson/Spearman, MI, ANOVA/chi-square, VIF, BH correction
  candidates.py     # candidate spec generation
  fit.py            # core numpy/scipy fitters and predict wrappers
  metrics.py        # R2, RMSE, MAE, accuracy, AUC, log-loss, deviance, AIC/BIC
  select.py         # ranking, baseline delta, stability, top-N
  diagnostics.py    # residual, heteroskedasticity, VIF, autocorrelation, overdispersion, stationarity
  trees.py          # optional sklearn tree/forest/boosting backend
  sindy.py          # optional pysindy backend
  report.py         # model_search.json / Markdown
  pipeline.py       # run_model_search(...)
```

### 4.3 Target and feature typing

`TargetKind = continuous | binary | multiclass | count | time_series`.

Rules:

- binary: exactly two unique non-null values;
- multiclass: 3–20 unique values, categorical or small integer codes;
- count: non-negative integer values, count-like name, or mean/variance ratio
  near 1;
- time_series: a detected `time_column` and an ordered target;
- otherwise continuous.

Feature kinds: numeric, categorical, datetime, id, leakage-risk. Exclude id,
time (unless the model is time-aware), and leakage-risk columns from features.
Record cardinality, missingness, and unusable columns.

### 4.4 Splits

- Default: 3 repeated random 70/30 train/test holdouts with distinct seeds, plus
  one full-data fit for final coefficients. The full-data fit is for the final
  coefficients, not an evaluation split.
- Classification: stratified by target.
- If a group/entity column is identified (for example a repeated site or
  subject), use group-aware splits so the same group cannot appear in both
  train and test.
- Time series: chronological splits (expanding window or 70/30 by time); never
  random shuffle.
- For `n >= 200`, reserve a final 20% holdout **before** generating the
  repeated splits, so it never appears in any training or test fold; use it once
  to confirm the selected model. There is **no separate untouched final holdout
  for `n < 200`**: report the repeated-holdout mean and standard deviation and
  mark the selection confidence lower.
- All imputation, encoding, scaling, and feature selection are fit inside the
  training fold.

### 4.5 Screening

For each feature:

- numeric: Pearson and Spearman correlation with the target; mutual
  information; VIF against other numeric features;
- categorical: ANOVA or chi-square; mutual information;
- missingness and cardinality;
- Benjamini–Hochberg correction across the feature tests;
- top-k features by corrected significance and MI, with a hard cap.

Screening results are diagnostics, not causal claims. The method critic does not
use p-values as a verdict; the independent validator owns formal inference.

### 4.6 Candidate generation

Every candidate is a `ModelSpec`:

```python
class ModelSpec(BaseModel):
    family: Literal["baseline","linear","logistic","multinomial","poisson",
                    "negative_binomial","interaction","tree","forest",
                    "gradient_boosting","sindy","group_comparison"]
    target_transform: Literal["none","log1p","sqrt"]
    feature_transforms: dict[str,int]   # none/log1p/sqrt per feature
    features: list[str]
    interactions: list[tuple[str,str]]
    regularization: float | None
    hyperparameters: dict[str, float | int | str]
```

Candidate pools:

- baseline: mean/median/majority;
- linear: OLS with target transforms and top feature transforms; interactions
  of the top 1–3 pairs; ridge for collinearity;
- logistic: binary target, with optional class weights;
- multinomial: multiclass target;
- Poisson/negative binomial: count target, with exposure/offset when available;
- group comparison: categorical or quantile-binned factor means/rates;
- tree/forest/boosting: sklearn backend, enabled by `--backend sklearn|full`;
- SINDy: time-series target with a library of polynomial/sin/cos/exp terms,
  enabled by `--backend full`.

Cap the candidate count (`model_search_max_candidates`, default 100) and rank
the pool before fitting.

### 4.7 Fitting and metrics

Core backend:

- OLS via `numpy.linalg.lstsq` or ridge normal equations;
- logistic/multinomial/Poisson/NB via `scipy.optimize.minimize` with analytic
  gradients where practical;
- optional `statsmodels` for GLM inference and diagnostics;
- optional `scikit-learn` for trees, metrics, splits, and mutual information;
- optional `pysindy` for SINDy.

Metrics by target:

- continuous: R², RMSE, MAE, MAPE (guarded), AIC/BIC;
- binary: accuracy, balanced accuracy, AUC, log-loss, Brier score;
- multiclass: accuracy, balanced accuracy, macro AUC/OVR, log-loss;
- count: Poisson deviance, RMSE, mean absolute error, overdispersion ratio;
- time series: the same metrics on the chronological test window, plus
  residual autocorrelation and stationarity diagnostics.

Per fold, store metric values; report mean, standard deviation, and the
baseline delta.

### 4.8 Selection

1. Discard candidates that fail to converge or produce non-finite predictions.
2. Require a minimum improvement over the baseline (`model_search_margin`).
3. Rank by the primary metric mean, then by stability (metric standard
   deviation), then by parsimony (AIC/BIC or parameter count).
4. Apply a complexity penalty and a hard parameter cap (`events per variable`
   for regression/logistic; `n/10`).
5. Keep top N (`model_search_top_n`, default 5).
6. If a final holdout exists, evaluate the top candidate once on it and report
   the result separately; do not use it to re-rank the full pool.
7. Mark selection confidence as `high`, `medium`, or `low` from fold stability
   and baseline margin.

### 4.9 Diagnostics

- residual summary and residual-vs-fitted;
- Breusch–Pagan / White heteroskedasticity;
- VIF and condition number;
- Durbin–Watson and residual autocorrelation;
- overdispersion for count models;
- ADF/KPSS stationarity and seasonality/periodogram for time series;
- Cook's distance / influence;
- calibration for binary/multiclass (optional);
- coefficient sign and magnitude stability across folds.

Diagnostics inform the critic and the artifact; they do not by themselves
reject a model.

### 4.10 Integration

- Run after `build_screen` and before hypothesis generation.
- `HypothesisContext` gains `model_search: ModelSearchResult | None`.
- The generator prompt includes the top candidates: family, formula/spec,
  features, transforms, metrics, diagnostics, and warnings. The three families
  are seeded as follows:
  - `linear_model`: the best linear/logistic/GLM candidate;
  - `group_comparison`: the best categorical/quantile group candidate;
  - `interaction_or_derived`: the best interaction/ratio/SINDy candidate.
- If the search has no candidate, keep the current generation fallback.
- The generated code must still pass the AST allowlist and runner contract; the
  model spec is context, not executable code.
- Artifact gains a `model_search` section: candidates, metrics, diagnostics,
  selected model, warnings, backend, seed.
- Model search does **not** call `stats_engine.py`. The strategy validator is a
  return-series-only finalist check in the follow-on validation plan; general
  model selection and evaluation use the model-search metrics and diagnostics.
- Preview renders a "Model search" table with the top candidates and metrics.
- CLI: `run --model-search` / `--no-model-search`; `model-search --input ...
  --target ... --out ...` for inspection; `--backend core|sklearn|statsmodels|full`.

### 4.11 Dependencies

The core install keeps numpy/scipy/pandas and now includes the two backends
used by the model search:

```text
scikit-learn>=1.5,<2
pysindy>=2,<3
```

`requirements-model-search.txt` adds the remaining optional backend:

```text
-r requirements.txt
statsmodels>=0.14,<1
```

`WHACK26_MODEL_SEARCH_BACKEND=core|sklearn|statsmodels|full` defaults to `core`.
The core backend must still produce linear/logistic/Poisson/interaction
candidates; optional backends unlock trees, richer diagnostics, and SINDy.

### 4.12 Tests and acceptance

**Golden datasets** (`tests/fixtures/model_search/`):

- linear relationship, known coefficient signs;
- interaction relationship;
- binary logistic relationship;
- multiclass relationship;
- count relationship with overdispersion;
- time-series trend + seasonality;
- categorical group means;
- missing values and collinear features;
- small-n and high-cardinality cases.

**Tests**

- target/feature typing;
- splits: no train/test leakage; stratified; chronological; group-aware;
- screening: known correlations/MI/ANOVA; BH correction; VIF;
- candidate generation: correct families for each target kind;
- fitting: coefficients/predictions finite; transforms invert correctly;
- selection: top model recovers the known family/feature; baseline is beaten;
- diagnostics: expected flags on synthetic heteroskedastic/overdispersed data;
- no leakage: a target-derived feature is never selected;
- integration: model search feeds hypotheses and artifact;
- regression: the existing full suite stays green.

**Acceptance**

- On every golden dataset, the correct model family or a strictly better
  alternative is in the top 2.
- No candidate uses the target or a target-derived column as a feature.
- Cross-validated metrics are reported with fold standard deviation.
- The selected model beats the baseline by at least the configured margin.
- The artifact validates and the preview lists the top candidates.
- Model search completes within a configurable time budget on 20k×10 data.

## 5. Parallel execution plan

The plan is restructured so there is **no serial implementation wave**. A single
short contract-freeze step publishes the interfaces and fakes; after that, every
implementation and integration package can start at the same time.

### 5.0 Pre-flight contract freeze (1 agent, 0.5–1 day)

Create and freeze:

- `whack26/hierarchy/types.py` — `PlacementResult`, `BranchResult`,
  `PromotionResult`, `HierarchyError`;
- `whack26/model_search/types.py` — `TargetKind`, `FeatureKind`, `ModelSpec`,
  `FoldResult`, `ModelCandidate`, `ModelSearchResult`;
- `tests/contracts/fakes.py` — an in-memory field store, fake model-search
  result, fake embeddings, and fixture builders;
- empty package directories, `__init__.py` files, and pytest markers for
  `integration`.

The contract agent writes no behaviour and does not edit `store.py`. The v3 DDL
in §3.2 is already frozen.

### 5.1 Parallel launch (15 + 1 agents)

All packages below start together. S0 owns `whack26/store.py`; feature packages
use the frozen types and fakes for unit tests. Tests that need the real v3
schema are marked `integration` and run once S0 lands.

| Package | Files | Contract | Unit-test dependency |
|---|---|---|---|
| S0 | `whack26/store.py`, `tests/test_hierarchy_store.py` | Schema v3, leaf field, operation log, hierarchy version, migration | frozen types |
| H1 | `whack26/hierarchy/place.py`, `tests/test_hierarchy_place.py` | `place_document`, specificity judge, thresholds, deterministic fallback | fakes |
| H2 | `whack26/hierarchy/operations.py`, `tests/test_hierarchy_operations.py` | `branch_field`, `promote_field`, `attach_document`, level updates, depth rejection | fakes |
| H3 | `whack26/hierarchy/paths.py`, `tests/test_hierarchy_paths.py` | `recompute_paths`, `verify_paths`, path consistency | fakes |
| H4 | `eval/field_hierarchy.py`, `tests/test_field_hierarchy_eval.py` | Labeled hierarchy set, metrics, regression gate | fakes |
| M1 | `whack26/model_search/data.py`, `splits.py`, tests | Target/feature typing, repeated/chronological/group splits | frozen types |
| M2 | `whack26/model_search/screen.py`, tests | Correlation, MI, ANOVA/chi-square, VIF, BH correction | frozen types |
| M3 | `whack26/model_search/candidates.py`, `fit.py`, `metrics.py`, tests | Core linear/logistic/multinomial/Poisson/NB/interaction candidates, fitters, metrics | frozen types |
| M4 | `whack26/model_search/trees.py`, `sindy.py`, tests | Optional sklearn tree/forest/boosting and pysindy backends | core fitters |
| M5 | `whack26/model_search/select.py`, `diagnostics.py`, `report.py`, tests | Ranking, baseline margin, stability, diagnostics, JSON/Markdown report | M1–M3 types |
| M6 | `eval/model_search.py`, `tests/fixtures/model_search/`, tests | Golden datasets, recovery tests, backend comparison | M1–M5 fakes |
| I1 | `whack26/classify.py`, `whack26/research/scrape/classify.py`, tests | Route placement through `hierarchy.place`, preserve create/assign fallback | fakes |
| I2 | `whack26/run.py`, `whack26/hypotheses/generate.py`, `hypotheses/__init__.py`, tests | Run model search after screen, inject into `HypothesisContext`, seed the three families | contract `ModelSearchResult` |
| I3 | `whack26/artifact.py`, `whack26/preview.py`, `whack26/cli.py`, tests | `model_search` artifact section, hierarchy audit/preview, `model-search` and `verify-hierarchy` commands | contract payloads |
| I4 | `tests/test_run_flow.py`, `tests/test_hierarchy_integration.py`, docs | End-to-end mock/live tests, docs, acceptance run | fakes until features land |

I1–I4 can start with the feature packages because their files are disjoint and
the contracts are frozen. They finish after the feature packages land; their
integration tests are the merge gate.

### 5.2 Agent budget

| Scope | Packages | Recommended concurrent | Hard ceiling |
|---|---|---|---|
| This plan | 1 contract + 1 schema + 10 feature + 4 integration = **16** | **12–14** | **16** |
| Other plan (`agentic-hypotheses-validation-and-sources`) | 1 contract + 1 schema + 14 feature + 4 integration = **20** | **12–16** | **20** |
| Both plans combined | 1 combined contract + 1 combined schema + 24 feature + 8 integration = **34** | **14–16** | **18** |

The limiting factor is not file ownership; it is review and merge capacity.
Twelve concurrent agents is the reliable sweet spot, sixteen is the practical
ceiling, and more than eighteen to twenty concurrent agents makes the merge
queue and integration tests the bottleneck rather than the work.

### 5.3 Ownership and merge rules

- C0 owns the contract files and fakes; no feature package edits them.
- S0 owns `whack26/store.py`; no other package edits it.
- H1–H4 own separate `whack26/hierarchy/*` files; M1–M6 own separate
  `whack26/model_search/*` files.
- I1–I4 own `classify.py`, `run.py`, `artifact.py`, `preview.py`, `cli.py`, and
  the end-to-end tests; feature packages never edit these.
- Each package merges the contract tip at start, then the integration branch tip
  before reporting.
- `integration`-marked tests run after S0 and the relevant feature packages land;
  the integration agents own that gate.

### 5.4 If fewer agents are available

| Agents | Launch order |
|---:|---|
| 12 | C0 + S0 + six feature packages + four integration packages; second shift for the remaining four |
| 8 | C0 + S0 + H1–H3 + M1–M3 + I1; then H4/M4–M6/I2–I4 |
| 4 | C0+S0, hierarchy (H1–H3), model search (M1–M3), integration (I1–I4) |

## 6. Acceptance criteria

### Hierarchy

- Every document is attached to the highest directly related field; no
  over-specific attachment when a directly related ancestor exists.
- Branching, promotion, and re-leveling preserve strict tree, max depth, levels,
  sibling uniqueness, and path consistency.
- Promotion reassigns documents closer to the new field; the rest keep the new
  field as an ancestor.
- `verify_paths` is clean after every operation.
- The audit log records every structural change.
- The existing full suite plus new hierarchy tests pass.

### Model search

- Golden datasets recover the known model family in the top 2.
- No target leakage; preprocessing is inside folds.
- Repeated holdout/chronological/group splits are implemented and tested.
- Cross-validated metrics with fold variance are reported.
- The selected model beats the baseline by the configured margin.
- Artifact/preview/CLI integration works; `model-search` inspection command
  exists.
- The existing full suite plus new model-search tests pass.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Promotion creates cycles or bad levels | Transactional operations, property tests, `verify_paths`, rollback |
| Depth cap blocks a valid promotion | Reject promotion, keep the new field as a child, log the reason |
| Over-specific attachment | Highest-directly-related walk, specificity judge, deterministic margin, eval gate |
| LLM judge cost/latency | Call only in ambiguous margins; cache judge results; temperature 0 |
| Path drift between `document_fields` and `path_json` | Single `recompute_paths` path, consistency check, integration tests |
| Model search overfits | Repeated holdout/nested final holdout, baseline margin, complexity penalty, stability |
| Target leakage | Exclude target/id/leakage columns, fit transforms inside folds, leakage regression tests |
| Multiple testing | BH correction for screening; model selection by CV, not p-values |
| Time-series leakage | Chronological splits, no shuffle, stationarity diagnostics |
| Optional dependencies unavailable | Core numpy/scipy backend; optional extras guarded by import checks |
| Runtime blow-up | Candidate cap, feature cap, time budget, parallelism disabled in the core path |

## 8. Effort

| Phase | Estimate |
|---|---|
| P0 contract freeze | 0.5–1 day |
| P1 hierarchy (H1–H4) | 4–5 days |
| P1 model search (M1–M6) | 6–8 days |
| P2 integration | 4–5 days |
| Evaluation/hardening/docs | 2–3 days |
| **Total** | **18–24 days** |

P0 is a one-agent pre-flight; P1 and P2 use the 12–14 concurrent-agent launch
described in §5. The 34 combined packages across both plans should run in two or
three shifts rather than one 34-agent launch.
