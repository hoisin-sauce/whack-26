# Field classification

Status: implemented in [ticket #2](https://github.com/hoisin-sauce/whack-26/issues/2).
This document is the original decision record for the user-dataset classifier.

> **Superseded in part (2026-11-08).** The original slice deferred promotion,
> rotation, and level-shifting. `whack26/hierarchy/` now implements branching,
> promotion with subtree re-levelling, audited path recomputation, and the v4
> `field_operations`/`store_meta` tables; see
> [`field-branching-and-model-search.md`](field-branching-and-model-search.md).
> Scraped research documents use hierarchy placement by default. The
> creation/assignment rules below still describe the user-dataset classifier,
> which is not routed through the hierarchy package.

## Scope

- The default classifier performs dynamic creation and assignment only; the optional hierarchy package adds promotion, rotation, and re-levelling (see the banner above).
- General-domain: no finance-specific taxonomy, no pre-seeded field list.
- Input: any pandas-readable dataset plus an optional description.
- Output: exactly one field path per dataset, with the fields created or reused along the way.

## Data model

```python
class Field:
    id: str                 # fld_<12 hex>
    parent_id: str | None   # None for roots
    level: int              # 1..4, equal to depth
    name: str               # human-readable, <= 5 words
    description: str        # 1-2 sentences, distinguishes it from siblings
    description_embedding: list[float]  # 768 dims, L2-normalized
    centroid: list[float] | None        # mean of member dataset embeddings, L2-normalized
    member_count: int
    created_at: str
    updated_at: str
```

`FieldPath` is a list of field references from root to leaf plus `confidence`, `created_fields`, and `refined_after_research`.

## Algorithm

1. Build the `DatasetProfile` and its text summary (see `research-pipeline.md`).
2. Embed the summary once: `d = embed(summary)`, L2-normalized.
3. Start at the root level (`parent_id = None`, `level = 1`).
4. At each level:
   1. Rank the children of the current parent by `score`:
      - `desc_sim = cos(d, field.description_embedding)`
      - `cent_sim = cos(d, field.centroid)` when `member_count > 0`, else `desc_sim`
      - `score = 0.5 * desc_sim + 0.5 * cent_sim`
      - Keep the top `TOP_K = 10`.
   2. Ask the LLM to choose an existing child or propose a new field. The prompt contains the dataset summary, the current path, the candidate list (`name`, `description`, `score`, `member_count`), and the depth. The LLM returns:
      ```json
      {"action": "assign" | "create" | "stop", "field_id": "fld_...", "name": "...", "description": "...", "reason": "..."}
      ```
   3. Validate:
      - `assign`: `field_id` must be one of the candidates. Move to that field and continue.
      - `create`: only when `level < 4`. Embed `name + ". " + description`; compare with every sibling. If cosine ≥ `DUPLICATE_COSINE = 0.90`, treat it as `assign` to the closest sibling and log a `duplicate_merge` warning. Otherwise insert the field at `level = parent.level + 1`, then continue.
      - `stop`: the dataset belongs at the current parent. Stop. `stop` is invalid at the root (level 1, no current field); at the root the LLM must `assign` or `create`.
      - Invalid `field_id`, an empty name/description, or `create` at level 4: re-ask once with the validation error. If the second response is still invalid, fall back to the top-scoring candidate or the current field and log a `llm_validation_fallback` warning.
   4. At level 4, `create` is invalid; the LLM must `assign` to an existing leaf or `stop` at the current field.
5. If no roots exist, the LLM creates the first root with no candidates.
6. In one transaction: upsert the dataset, replace its membership rows for every field in the path, and recompute the centroid of every affected field (new path and old path if the dataset moved).
7. `confidence` = mean of the scores of assigned existing fields. A created field contributes its parent's score; a newly created root contributes `0.5`. Record `created_fields` so the artifact shows what was invented.

## Prompts

### Field choice

System:

> You are classifying a tabular dataset into a dynamic topic hierarchy. The hierarchy is general-domain and is built from data; there is no fixed list. Choose an existing field only when it genuinely fits. Propose a new field only when no existing sibling is a reasonable parent. Names must be human-readable, at most five words, and must not be the dataset's filename or a raw column name. Descriptions must be one or two sentences and must distinguish the field from its siblings. Never propose a child at level 4.

User: dataset summary, current path, candidates, depth, instruction to return the JSON object.

Temperature 0; JSON schema enforced.

### Field naming

When `create` is chosen, the same call already returns `name` and `description`; no second call. The server validates length, rejects names containing the dataset id/filename, and re-asks once if invalid.

### Research refinement

After research, if `confidence < 0.62` or the top research titles contradict the path, run one refinement call with the summary plus the top five research titles and keywords. Allowed actions: `assign` to another existing path, `create` a child under the current path, or `stop`. Creating a new root is forbidden during refinement. At most one refinement per dataset. Set `refined_after_research = true` and record the reason.

## Thresholds

All values live in `config.py` and are overridable by environment variables using the `WHACK26_` prefix (for example, `WHACK26_TOP_K=10`).

| Name | Default | Meaning |
|---|---|---|
| `TOP_K` | 10 | Candidates shown to the LLM per level |
| `DUPLICATE_COSINE` | 0.90 | Proposed sibling is merged into an existing sibling at or above this |
| `REFINE_CONFIDENCE` | 0.62 | Refinement trigger |
| `MAX_DEPTH` | 4 | Hard tree depth |
| `EMBED_DIMS` | 768 | Embedding size |
| `DESC_WEIGHT` | 0.5 | Description weight in the score |
| `CENTROID_WEIGHT` | 0.5 | Centroid weight in the score |

## Invariants

- Strict tree: one parent per field, no cycles, `level = parent.level + 1`.
- `1 <= level <= 4`.
- Normalized names are unique among siblings (case-insensitive, punctuation-stripped).
- Every field has at least one member after the transaction commits; empty fields are deleted.
- A dataset has exactly one membership row per field in its path.
- Re-running a dataset replaces its membership; it does not duplicate it.
- Centroids are recomputed from member embeddings, never incrementally approximated in this slice.

## Store

> **Historical layout.** The DDL below is the original v1 field-store schema.
> The current store is **v4**: `documents` + the `datasets` extension table,
> `document_fields`, `document_links`, `document_comparisons`,
> `document_critiques`, `document_fetches`, `field_operations`,
> `research_models`, and `store_meta`. `whack26/store.py` is the authority; the
> v1→v2→v3→v4 migrations are tested.

SQLite, `cache/fields.db` (path configurable). Brute-force cosine is fine at this scale. Enable foreign keys on every connection:

```sql
PRAGMA foreign_keys = ON;
```

```sql
CREATE TABLE fields (
  id TEXT PRIMARY KEY,
  parent_id TEXT REFERENCES fields(id),
  level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 4),
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  desc_embedding BLOB NOT NULL,
  centroid BLOB,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX fields_sibling_name ON fields(COALESCE(parent_id, ''), lower(name));

CREATE TABLE datasets (
  id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  profile_json TEXT NOT NULL,
  summary TEXT NOT NULL,
  embedding BLOB NOT NULL,
  path_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  refined_after_research INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

CREATE TABLE field_members (
  field_id TEXT NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  PRIMARY KEY (field_id, dataset_id)
);
```

## Evaluation gate

The first build milestone is `eval/field_classification.py`, which replaces the separate throwaway prototype.

### Datasets

| # | Dataset | Domain | Description supplied |
|---|---|---|---|
| 1 | Palmer Penguins | biology / ecology | yes |
| 2 | Wine Quality | food / chemistry | yes |
| 3 | California Housing | housing / economics | yes |
| 4 | Titanic | demographics / history | no |
| 5 | Air Quality (UCI) | environment | yes |
| 6 | COVID country series | public health | yes |
| 7 | Spotify Tracks | music | no |
| 8 | Student Performance | education | yes |
| 9 | Online Retail | retail / commerce | no |
| 10 | Soccer match results | sports | no |

Each entry stores acceptable root concepts, forbidden labels, and a note on what a good path looks like. Acceptable roots are semantic sets, not exact strings.

### Runs

1. **Cold:** empty store, fixed order, no cache.
2. **Warm:** same datasets, same store, warm cache.
3. **Shuffled cold:** fresh store, reversed order; diagnostic only.

### Rubric

| Score | Meaning |
|---|---|
| 5 | Named as a practitioner would; parent-child relation obvious; description accurate |
| 4 | Minor naming quibble; relation clear |
| 3 | Acceptable, but a different path is also reasonable |
| 2 | A node is misleading, too broad, or too narrow |
| 1 | Nonsensical or dataset-specific label |

A dataset passes when the root and the full path both score ≥ 3.

The reviewer normally scores naming as well. If naming is omitted and the automated naming gate passes, the harness assumes the automated result is correct and uses score 4; pass `--no-assume-automated-naming` to require a human naming score.

### Gates

| Criterion | Gate |
|---|---|
| Top-level plausibility | ≥ 8/10 roots ≥ 3; mean ≥ 3.5 |
| Full-path plausibility | ≥ 8/10 paths ≥ 3; mean ≥ 3.5 |
| Hierarchy use | ≥ 6/10 paths have depth ≥ 2 |
| Max depth | 10/10 ≤ 4 |
| Duplicate roots | no exact duplicates; ≤ 1 pair with cosine ≥ 0.90 |
| Orphans | 0 fields with zero members |
| Naming | mean ≥ 4.0; no dataset proper nouns; no "Miscellaneous"/"Other" root |
| Sibling distinctness | ≥ 8/10 descriptions distinguish the field from siblings |
| Warm stability | same root ≥ 9/10; identical full path ≥ 7/10 |
| Order independence (diagnostic) | same root ≥ 8/10 |
| Runtime | cold classification ≤ 10 min; warm ≤ 5 min |
| Robustness | 0 crashes; 10/10 produce a path; same seed/model reproduces output |

This is a demo-grade sanity gate, not a benchmark. The harness writes
`docs/eval/field-classification-results.md` and exits non-zero if a gate fails.
The mock plumbing gate passes; the full live ten-dataset run is pending provider
quota and completed human scores. See
[`docs/eval/README.md`](../eval/README.md).

## Module layout

- `whack26/profile.py` — dataset profile and summary.
- `whack26/embeddings.py` — provider adapter, normalization, cosine.
- `whack26/classify.py` — algorithm, prompts, validation, refinement.
- `whack26/store.py` — SQLite v4 field/document store and migrations.
- `whack26/hierarchy/` — placement, branching, promotion, path recomputation.
- `eval/field_classification.py` — gate runner.
