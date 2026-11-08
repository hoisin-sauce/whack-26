# Semantic scraping implementation plan

Status: **implemented on the existing SQLite stack**. The 6–8 hour slice
originally deferred semantic scraping and credibility scoring; this document is
the follow-on design from `devplan.md`. The code now lives under
`whack26/research/scrape/` and is wired into `whack26 run --scrape`, the
`whack26 scrape` command, and `whack26 preload-statistics`. The mock end-to-end
path and a 5-document live smoke are validated; the full 20-resource depth-1
live smoke still depends on provider quota.

> **Schema history (2026-11-08).** This plan was written against SQLite schema
> v2 and artifact schema `1.1.0`. The store is now **v4** (v3 added leaf fields;
> v4 added `field_operations`, `research_models`, collection metadata, and the
> hierarchy version), and the artifact schema is **1.3.0** (adds
> `model_search`, `variants`, `research_models`, `collection`,
> `generation_validation_summary`, and `hypotheses[].generation_validation`).
> References to v2/`1.1.0` below describe the original slice; `whack26/store.py`
> and `schemas/artifact.schema.json` are the current authority.

## 0. Implemented modules

| Module | Responsibility |
|---|---|
| `models.py` | Document, link, comparison, critique, summary, credibility, and scrape-result models |
| `identity.py` | Canonical URLs, URL/content hashes, metadata/title identities |
| `store.py` | `DocumentStore` protocol; `FieldStore` is the SQLite implementation |
| `summarize.py` | Bounded summary prompt/schema, summary validation, summary embedding |
| `process.py` | Summarize/embed a document, merge links, set status |
| `links.py` | Extract and canonicalize references, DOIs, arXiv/OpenAlex IDs, URLs |
| `fetch.py` | Compliant HTTP fetcher: robots.txt, rate limits, SSRF/PDF/size/time controls, cache |
| `frontier.py` | Budgeted, target-gated BFS over links/references |
| `classify.py` | Assign documents into the existing dynamic field tree |
| `credibility.py` | Deterministic author/venue/source/recency/citation score |
| `compare.py` | Cached pairwise document comparisons |
| `preload.py` | Statistics/validity seed corpus |
| `pipeline.py` | End-to-end semantic-scraping orchestrator |
| `whack26/critic/document_critic.py` | Method critique for papers/web documents |

The store lives in `cache/fields.db` at **schema v4**; v2 was the original
semantic-scraping slice, v3 added leaf fields, and v4 added collection metadata
and research models. `cache/research.db` remains the raw HTTP cache. The
`whack26 schema` command exports the artifact schema at `1.3.0`, which includes
the optional `research.semantic_scraping` section.

## 1. What the devplan asks for

The semantic-scraping section of `devplan.md` requires:

1. A catalog of three resource kinds — datasets, searches, and papers — plus the
   web pages they link to. Datasets can link to searches or papers; searches
   link to other searches; papers link to other papers by reference.
2. A summary and vector embedding for every resource.
3. Identity checks before re-processing: compare the access link with tracking
   and access parameters removed, and compare a SHA-512 hash of the data.
4. A target (text or embedding) that decides whether to follow a resource's
   links: if the resource summary is close enough, add all its links; otherwise
   add only links whose summaries are close enough under the same or a tighter
   threshold.
5. Classification of each resource into the dynamic field hierarchy, using the
   same create/assign machinery as the user's dataset.
6. Credibility scoring from authors, publishers, and source metadata. Store
   low-credibility resources but do not use them as evidence.
7. Cached pairwise comparisons between resources, plus a direct comparison
   against the dataset currently being processed.
8. A method critique of papers and news articles against the preloaded
   statistical-pitfalls corpus.
9. A preloaded layer-1 statistics/validity corpus built from API searches.
10. A database that matches the devplan's document model: Dataset, Search, and
    Paper documents with summaries/embeddings, to/from links, field assignments,
    comparisons, and critiques. This plan keeps SQLite and does **not** use
    Postgres/pgvector.

## 2. Current state and gap

What already exists:

- `whack26/research/` fetches one hop from OpenAlex, arXiv, HF Hub, Zenodo, and
  Wikipedia, then normalizes, deduplicates, and ranks records.
- `ResearchRecord` carries title, abstract, authors, year, DOI, URL, license,
  source, query, and relevance. It has no summary embedding, link graph,
  credibility score, or field assignment.
- `whack26/store.py` holds a dynamic field hierarchy (`fields`), dataset
  membership (`field_members`), and dataset embeddings. The classifier walks
  the tree with create/assign only.
- `cache/research.db` caches API bodies; `cache/fields.db` holds fields and
  datasets. Both are SQLite with brute-force cosine.
- `whack26/critic/` already has a corpus parser, retrieval, deterministic
  checks, and an LLM rubric. It expects a generated hypothesis, not a paper.

What is missing: the devplan-aligned SQLite schema, the document catalog,
identity/dedup, summary embeddings, link-following frontier, document
classification, credibility, comparisons, document critique, and the statistics
preload.

## 3. Goals and non-goals

**Goals**

- Discover relevant resources at depth 1–2 from API-provided references and
  links, gated by target relevance and a hard budget.
- Never re-fetch a resource that is already catalogued; record the new link.
- Give every resource a standardized summary, embedding, field assignment,
  credibility score, and optional method critique.
- Cache comparisons so a repeated run does not re-call the LLM.
- Keep the current one-hop pipeline working unchanged when scraping is disabled.
- Keep the store behind a `DocumentStore` protocol so the SQLite implementation
  stays replaceable, without adding Postgres or pgvector in this plan.

**Non-goals for the first increment**

- No Postgres/pgvector. Stay on the current SQLite solution and brute-force
  cosine over BLOBs.
- No full-text PDFs, paywalled content, login walls, or redistribution. Store
  metadata, summaries, and links only.
- No general web search API. Start with references and links already returned by
  OpenAlex, arXiv, Wikipedia, HF Hub, and Zenodo. Add a compliant web-search
  adapter only after a legal review.
- No hierarchy level-shifting/promotion. Resource assignment reuses the current
  create/assign algorithm; dynamic re-parenting is a separate work package.
- The HTTP fetcher is opt-in (`WHACK26_SCRAPE_FETCH_ENABLED=1` or `--fetch`). It
  enforces `robots.txt`, per-domain rate limits, an optional allow/block list,
  private-network/SSRF blocking, redirect re-validation, PDF/binary rejection,
  size/time caps, and a text-excerpt cache. Arbitrary crawling remains off by
  default.
- No LLM-only credibility. Scores must be decomposable and evidence-backed.

## 4. Storage decision: stay on SQLite

The devplan names Postgres + pgvector, but this plan deliberately keeps the
current SQLite stack. The reasons are concrete:

- the field store (`cache/fields.db`) and the research cache
  (`cache/research.db`) are already SQLite;
- the project is local/offline-first and has no database operations budget;
- the expected scale is hundreds to a few thousand documents, where
  brute-force cosine over BLOB vectors is fast enough;
- keeping one storage engine keeps the tests, `--offline` mode, and the
  single-process runner simple.

The database target was originally a **versioned SQLite schema v2** in `cache/fields.db`; the current schema is **v4**.
`cache/research.db` remains the raw HTTP response cache. The devplan's
Dataset/Search/Paper document model, link graph, summaries, embeddings, and
field assignments all live in `fields.db`.

The domain layer talks to a `DocumentStore` protocol, not to `sqlite3`
directly. That keeps the door open for a future Postgres adapter without making
it part of this plan.

> Terminology: the devplan says “document” for Dataset/Search/Paper rows; the
> existing code and earlier drafts say “resource”. They are the same object.
> The database table and artifact section use `documents`; the Python domain
> model may keep `Resource` as an alias.

```python
class DocumentStore(Protocol):
    def find_by_url(self, url_hash: str) -> Document | None: ...
    def find_by_content(self, content_sha512: str) -> Document | None: ...
    def upsert(self, document: Document) -> Document: ...
    def add_link(self, link: DocumentLink) -> None: ...
    def assign_field(self, document_id: str, field_id: str, confidence: float) -> None: ...
    def neighbors(self, document_id: str, limit: int) -> list[Document]: ...
    def get_comparison(self, key: ComparisonKey) -> Comparison | None: ...
    def put_comparison(self, comparison: Comparison) -> None: ...
```

## 5. Database update to the devplan spec

### 5.1 Target schema (SQLite v4; originally v2)

The devplan's three document types share a `documents` base table; `kind`
discriminates `dataset`, `search`, `paper`, and `web` (a fetched link that is not
yet one of the three). Type-specific fields live in extension tables or
`metadata_json`. `fields` remains the devplan's field table.

```sql
PRAGMA user_version = 2;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,                 -- ds_/doc_/pap_/srch_/web_ + hash
  kind TEXT NOT NULL CHECK (kind IN ('dataset','search','paper','web')),
  source TEXT NOT NULL,                -- openalex, arxiv, wikipedia, hf, zenodo, web, user
  source_id TEXT,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  summary_embedding BLOB NOT NULL,
  embedding_model TEXT NOT NULL,
  canonical_url TEXT,
  url_hash TEXT UNIQUE,
  content_sha256 TEXT,
  content_sha512 TEXT UNIQUE,
  authors_json TEXT NOT NULL DEFAULT '[]',
  publisher TEXT,
  published_at TEXT,
  license TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  credibility_score REAL NOT NULL DEFAULT 0.0,
  credibility_json TEXT NOT NULL DEFAULT '{}',
  credible INTEGER NOT NULL DEFAULT 0,
  preloaded INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,                -- new|fetched|summarized|classified|compared|rejected
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS documents_kind ON documents(kind);
CREATE INDEX IF NOT EXISTS documents_credible ON documents(credible);
CREATE INDEX IF NOT EXISTS documents_preloaded ON documents(preloaded);

CREATE TABLE IF NOT EXISTS datasets (
  document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  profile_json TEXT NOT NULL,
  path_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  refined_after_research INTEGER NOT NULL DEFAULT 0,
  cache_key TEXT
);

CREATE VIEW IF NOT EXISTS searches AS
  SELECT * FROM documents WHERE kind = 'search';
CREATE VIEW IF NOT EXISTS papers AS
  SELECT * FROM documents WHERE kind = 'paper';

CREATE TABLE IF NOT EXISTS document_links (
  from_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  to_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  relation TEXT NOT NULL CHECK (
    relation IN ('references','cites','linked_from','same_as','dataset_paper','search_result')
  ),
  url TEXT,
  anchor_text TEXT,
  discovered_at TEXT NOT NULL,
  PRIMARY KEY (from_id, to_id, relation)
);
CREATE INDEX IF NOT EXISTS document_links_to ON document_links(to_id);

CREATE TABLE IF NOT EXISTS document_fields (
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  field_id TEXT NOT NULL REFERENCES fields(id) ON DELETE CASCADE,
  confidence REAL NOT NULL,
  assigned_at TEXT NOT NULL,
  PRIMARY KEY (document_id, field_id)
);
CREATE INDEX IF NOT EXISTS document_fields_field ON document_fields(field_id);

CREATE TABLE IF NOT EXISTS document_comparisons (
  left_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  right_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  method_version TEXT NOT NULL,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (left_id, right_id, kind, method_version)
);

CREATE TABLE IF NOT EXISTS document_critiques (
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  corpus_version TEXT NOT NULL,
  verdict TEXT NOT NULL,
  issues_json TEXT NOT NULL,
  model TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (document_id, corpus_version)
);

CREATE TABLE IF NOT EXISTS document_fetches (
  url_hash TEXT PRIMARY KEY,
  url TEXT NOT NULL,
  status INTEGER NOT NULL,
  etag TEXT,
  last_modified TEXT,
  body TEXT NOT NULL,                 -- API metadata or public abstract only; never PDFs/full article text
  fetched_at TEXT NOT NULL,
  ttl_seconds INTEGER NOT NULL
);
```

`document_fields` replaces `field_members` for the unified model. During the
transition, keep a compatibility view for any caller that still asks for the
old shape:

```sql
CREATE VIEW IF NOT EXISTS field_members AS
  SELECT field_id, document_id AS dataset_id
  FROM document_fields
  JOIN documents ON documents.id = document_fields.document_id
  WHERE documents.kind = 'dataset';
```

`datasets` is now an extension table keyed by `document_id`; the common summary,
embedding, and timestamps live in `documents`. This is the devplan's
Dataset/Search/Paper split with one shared vector-searchable summary column.

### 5.2 Devplan-to-schema mapping

| Devplan requirement | SQLite implementation (v4 today) |
|---|---|
| Three document types: Dataset, Search, Paper | `documents.kind` plus the `datasets` extension table and `searches`/`papers` views |
| Datasets linked to searches or papers | `document_links.relation IN ('dataset_paper','search_result')` |
| Searches linked to searches with to/from links | `document_links` with `relation='search_result'` or `'linked_from'` |
| Papers linked to papers by reference | `document_links.relation IN ('references','cites')` |
| Every document has a summary searchable by vector | `documents.summary` + `documents.summary_embedding` BLOB |
| Field table with level and vector encoding | existing `fields` table (`level`, `desc_embedding`, `centroid`) |
| Resource assigned to the closest field | `document_fields` |
| Comparisons recorded to avoid recalculation | `document_comparisons` with a method-version cache key |
| Paper/news method critique | `document_critiques` |
| Fetch cache with identity metadata | `document_fetches` (`url_hash`, status, ETag, body) |
| Schema evolution | `schema_migrations` + `PRAGMA user_version` |

### 5.3 Migration from the current v1 schema

Current v1: `fields`, `datasets` (id, sha256, profile_json, summary, embedding,
path_json, confidence, refined_after_research, cache_key, updated_at), and
`field_members` (field_id, dataset_id). `fields` is already the devplan field
table and needs no structural change.

Migration steps, all inside one transaction:

1. Back up `cache/fields.db` and `cache/research.db` to timestamped files.
   Refuse to migrate if the backup cannot be written.
2. `ALTER TABLE datasets RENAME TO datasets_v1;`
3. `ALTER TABLE field_members RENAME TO field_members_v1;` (SQLite updates the
   foreign-key references during the rename).
4. Create the v2 tables and indexes above.
5. Backfill each old dataset into `documents`:
   - `id = datasets_v1.id`
   - `kind = 'dataset'`, `source = 'user'`, `status = 'classified'`
   - `title` = basename of `profile_json.source_path`, else the dataset id
   - `summary = datasets_v1.summary`
   - `summary_embedding = datasets_v1.embedding`
   - `embedding_model` = the current settings value
   - `content_sha256 = datasets_v1.sha256`
   - `created_at = updated_at = datasets_v1.updated_at`
6. Backfill the `datasets` extension table with `document_id`, `profile_json`,
   `path_json`, `confidence`, `refined_after_research`, and `cache_key`.
7. Backfill `document_fields` from `field_members_v1` with
   `document_id = dataset_id`, `confidence = 1.0`, and `assigned_at = now`.
8. Verify row counts, a sample of `path_json` values, and the embedding round
   trip. Drop `field_members_v1` first, then `datasets_v1`.
9. Create the `field_members` compatibility view.
10. Set `PRAGMA user_version = 2`, insert a `schema_migrations` row, run
    `PRAGMA foreign_key_check` and `PRAGMA integrity_check`, then commit.

The migration is idempotent: if `user_version >= 2`, skip it. A failed
migration rolls back and leaves the v1 database and backup intact.

### 5.4 Store and code changes

- `FieldStore` becomes the SQLite `DocumentStore` implementation:
  - `get_dataset` joins `documents` and `datasets`.
  - `commit_path` upserts `documents` + `datasets` + `document_fields`.
  - `dataset_path` reads `document_fields`.
  - `recompute_centroids` reads `document_fields`; after Phase 3, resource
    membership contributes to the same centroid.
  - `delete_empty_leaf_fields` and `roots`/`children` are unchanged.
- `Classifier` keeps its public API; only the store internals change.
- `eval/field_classification.py` keeps using `FieldStore`; the migration must
  preserve its behavior.
- `cache/research.db` is unchanged; raw scraped-page bodies go into
  `document_fetches` in `fields.db`.
- Vector storage stays `BLOB` (little-endian float32 or float64) with
  brute-force cosine. No Postgres, no pgvector, no `sqlite-vec` dependency.
- Enable `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL`,
  `PRAGMA busy_timeout=5000`, and `PRAGMA foreign_keys=ON` on every connection.

### 5.5 Schema tests

- Build a v1 database with two datasets and a shared field path, run the
  migration, and assert `documents`, `datasets`, and `document_fields` counts.
- Assert `summary`, `summary_embedding`, `path_json`, `confidence`, and
  `cache_key` round-trip byte-for-byte.
- Run the migration twice; assert no duplicates and no errors.
- Inject a failure mid-migration; assert rollback to v1 and a readable backup.
- Run `PRAGMA foreign_key_check` and `PRAGMA integrity_check` after migration.
- Keep the existing field-store, classifier, and eval tests green.

## 6. Resource identity and deduplication

New module: `whack26/research/scrape/identity.py`.

- `canonicalize_url(url)`:
  - lowercase scheme and host, drop fragments;
  - remove tracking/access parameters: `utm_*`, `fbclid`, `gclid`, `mc_cid`,
    `mc_eid`, `igshid`, `ref`, `referrer`, `from`, `source`, `via`, `si`,
    `s`, `t`, `start`;
  - keep identity parameters such as `id`, `doi`, `q`, `search`, `page`;
  - sort remaining query parameters, strip a trailing slash except for the root;
  - reject non-http(s) schemes.
- `content_sha512`:
  - raw response bytes when a page is fetched;
  - otherwise a canonical JSON hash of the identity fields
    (`title`, `abstract`, `authors`, `year`, `doi`, `canonical_url`).
- Dedup order: `content_sha512` → `url_hash` → DOI/arXiv ID → normalized
  title+authors+year hash.
- On a hit, do not fetch or summarize again. Insert a `document_links` row with
  `relation='same_as'` or the relation that led to it, and return the existing
  resource.
- A link whose content hash matches a different URL is merged into the canonical
  resource; the new URL is recorded as an alias.

## 7. Resource processing

New module: `whack26/research/scrape/process.py`.

1. **Fetch** (only when a summary is not already available): official API
   metadata first, then the public page if allowed. Respect `robots.txt`,
   per-domain rate limits, `ETag`/`Last-Modified`, and the configured
   user-agent. Never fetch PDFs.
2. **Summarize** with one LLM call. Input: title, abstract/metadata, target,
   and at most 2,000 characters of extracted public text. Output schema:
   ```json
   {
     "summary": "2-4 sentences",
     "claims": ["..."],
     "methods": ["..."],
     "population": "..." or null,
     "time_period": "..." or null,
     "links": ["..."],
     "credibility_signals": {
       "peer_reviewed": true,
       "publisher": "...",
       "retracted": false
     }
   }
   ```
   Reject summaries over 1,200 characters or with no summary. Redact secrets and
   PII before the call.
3. **Embed** the summary once with the configured embedding model; store the
   L2-normalized vector and `embedding_model`. A model change invalidates cached
   similarities.
4. **Classify** the resource with the existing field hierarchy. Reuse the
   top-k + LLM choice pattern, but with a resource prompt (“classify a research
   resource, not a dataset”). Write `document_fields` rows for every field in
   the assigned path (root→leaf), matching the dataset membership invariant. A
   resource may be assigned to one leaf path; the path is stored in the artifact.
   Do not create a new root; if no field fits, assign to the closest existing
   root and warn.
5. **Credibility** (Phase 4) before the resource is eligible as evidence.

## 8. Semantic scraping frontier

New module: `whack26/research/scrape/frontier.py`.

Inputs: seed resources (the current one-hop `ResearchResult.records`), the target
text or embedding, thresholds, and budgets.

```
frontier = priority_queue(seeds, key=priority)
visited = set()
while frontier and budget_left:
    resource = pop(frontier)
    if resource.id in visited: continue
    visited.add(resource.id)
    process(resource)                     # summary, embed, classify
    sim = cos(resource.summary_embedding, target_embedding)
    if sim >= summary_threshold:
        candidates = all_links(resource)  # follow everything
    else:
        candidates = []
        for link in links(resource):
            linked = fetch_and_summarize(link)
            if cos(linked.summary_embedding, target_embedding) >= link_threshold:
                candidates.append(linked)
    for candidate in candidates:
        if candidate.id not in visited:
            frontier.push(candidate, priority(candidate))
```

- `summary_threshold` default 0.62; `link_threshold` default 0.68 (tighter).
- `priority = relevance * credibility * recency`.
- Budgets: `max_depth` (default 2), `max_resources` (default 100),
  `max_links_per_resource` (default 25), `max_fetches_per_run` (default 100).
- The frontier is breadth-first by depth, then priority. Stop on budget, no
  candidates, or a hard wall-clock limit.
- Never follow a link whose domain is not allowed, whose `robots.txt` disallows
  it, or whose scheme is not http(s).
- Record every attempted link and its outcome (`queued`, `skipped_threshold`,
  `skipped_robots`, `duplicate`, `error`) in the run log and artifact warnings.

## 9. Credibility scoring

New module: `whack26/research/scrape/credibility.py`.

Deterministic first version:

```
score = 0.35 * author_signal
      + 0.25 * venue_signal
      + 0.20 * source_type_signal
      + 0.10 * recency_signal
      + 0.10 * citation_signal
      - penalties
```

- `author_signal`: known author with citations/h-index from OpenAlex or Semantic
  Scholar; unknown author = 0.5; no authors = 0.2.
- `venue_signal`: peer-reviewed journal/conference = 1.0; book chapter = 0.8;
  preprint = 0.6; dataset repository = 0.7; news = 0.4; blog/unknown = 0.2.
- `source_type_signal`: OpenAlex/arXiv = 0.8, Zenodo/HF = 0.7, Wikipedia = 0.5,
  web = 0.3.
- `recency_signal`: `exp(-age_years / 5)`.
- `citation_signal`: `log1p(cited_by_count) / log1p(1000)`, capped at 1.
- Penalties: retracted = -1.0; predatory venue = -0.5; no date = -0.1.
- `credible = score >= 0.5`. Store the component breakdown in
  `credibility_json`.
- Low-credibility resources are still stored and linked, but are excluded from
  ranking, factor evidence, comparisons that require trust, and the artifact's
  “used evidence” list. The artifact records them under `excluded_documents`.

Calibrate the weights on a 50-resource labeled set before enabling the filter by
default. Until then, `credible` is advisory and the run warning says so.

## 10. Comparison engine and cache

New module: `whack26/research/scrape/compare.py`.

- When a document is added, select neighbors:
  - top 10 by summary cosine within the same field path;
  - plus the current dataset's own summary embedding;
  - plus any document sharing a DOI/dataset ID.
- Do not compare all pairs. The comparison set is `O(k)` per document.
- Comparison kinds: `dataset_relevance`, `method_overlap`, `finding_agreement`,
  `population_overlap`, `contradiction`.
- Output schema:
  ```json
  {
    "left_id": "doc_...",
    "right_id": "doc_...",
    "kind": "method_overlap",
    "similarity": 0.81,
    "verdict": "supports | contradicts | orthogonal | insufficient",
    "rationale": "...",
    "evidence": ["..."],
    "method_version": "1.0.0"
  }
  ```
- Cache key: canonical sorted `(left_id, right_id, kind, method_version)`. A
  repeat run reads the stored `document_comparisons` row and makes no LLM call.
- Invalidate a comparison when either summary embedding or `method_version`
  changes.
- The direct dataset comparison uses the dataset summary embedding and the
  existing factor/screen context; store it with `right_id` set to the dataset
  document. The dataset is registered as a `documents` row with
  `kind='dataset'` so the same cache and link tables apply.

## 11. Document method critique

New module: `whack26/critic/document_critic.py`.

- Reuse `Corpus`, `corpus.retrieve`, `CRITIC_SYSTEM_PROMPT`, and the
  deterministic checks where they apply (claim type, sample size, leakage,
  missing data, measurement error, time dependence, reproducibility).
- Input: document summary, claims, methods, population, time period, source
  metadata, and the retrieved corpus chunks.
- Do not expect Python code. Do not judge performance, p-values, effect sizes,
  or out-of-sample results; the independent validator owns those.
- Output: `verdict`, `issues`, `missing_information`, `summary`,
  `retrieved_chunks`, `corpus_version`, `model`.
- Store in `document_critiques`; include a compact form in the artifact.
- A `reject` critique excludes the document from factor evidence; a `refine`
  keeps it with a warning.

## 12. Preloaded statistics corpus

New module: `whack26/research/scrape/preload.py` and CLI
`whack26 preload-statistics`.

Seed queries (OpenAlex and arXiv; Wikipedia only for definitions):

- statistical validity and reproducibility
- p-hacking and publication bias
- multiple comparisons and false discovery rate
- causal inference assumptions and identification
- regression diagnostics and heteroskedasticity
- missing data mechanisms and imputation
- measurement error and misclassification
- sample size and statistical power
- time-series stationarity and dependence
- data leakage and train/test contamination
- external validity and distribution shift
- model selection and overfitting
- pre-registration and reporting standards

For each query: fetch top 10, deduplicate, summarize, embed, classify under a
`Statistics and Validity` root, and mark `documents.preloaded = 1`. Cap the
preload at 200 documents. The static `docs/corpus/statistical-pitfalls.md`
remains the fallback when the live preload is empty or offline.

## 13. Pipeline and CLI integration

Add a stage after the existing one-hop research and before factor extraction:

```
classify -> one-hop research -> semantic scraping -> factors -> hypotheses -> critic
```

- `run_semantic_scraping(seeds=research.records, target=summary, settings=...)`
  returns a `ScrapeResult` with `documents`, `links`, `comparisons`,
  `critiques`, `warnings`, and `stats`.
- Factor extraction receives the credible scraped documents as additional
  records. The existing `ResearchRecord` list stays unchanged for backward
  compatibility; scraped documents are a separate list.
- The method critic can use the preloaded statistics documents as extra
  retrieval context, but the static corpus remains the source of truth for the
  verdict vocabulary.
- CLI:
  - `whack26 scrape --input ... --target ... [--max-resources N] [--max-depth N]`
  - `whack26 run --scrape ...` (default off)
  - `whack26 preload-statistics [--limit N] [--refresh]`
- The scrape command writes `scrape.json` and `scrape.md` for inspection.

## 14. Artifact changes

Add an optional `research.semantic_scraping` section. The plan originally bumped
the schema to `1.1.0`; the current schema is `1.3.0` (see
`docs/spec/artifact.md`), and older handoffs remain valid when the section is
absent.

```json
{
  "semantic_scraping": {
    "enabled": true,
    "documents": [
      {
        "id": "doc_...",
        "kind": "paper",
        "source": "openalex",
        "title": "...",
        "url": "...",
        "summary": "...",
        "authors": ["..."],
        "year": 2024,
        "credibility": 0.78,
        "credible": true,
        "field_path": ["Statistics and Validity", "..."],
        "critique_verdict": "refine",
        "comparisons": 3
      }
    ],
    "links": [{"from": "doc_...", "to": "doc_...", "relation": "references"}],
    "comparisons": [
      {"left": "doc_...", "right": "doc_...", "kind": "method_overlap",
       "verdict": "supports", "similarity": 0.81}
    ],
    "warnings": ["scrape_budget_exhausted"],
    "stats": {"documents": 42, "credible": 31, "llm_calls": 18, "cache_hits": 7}
  }
}
```

No raw page text, secrets, or PII. The preview renders a compact “Semantic
scraping” section with the top credible documents, their field paths, and
comparison counts.

## 15. Configuration

Add to `Settings` with the `WHACK26_` prefix:

| Variable | Default | Meaning |
|---|---|---|
| `WHACK26_CACHE_DIR` | `cache` | Existing directory; holds `fields.db` (schema v4) and `research.db` |
| `WHACK26_SCRAPE_ENABLED` | `0` | Enable the semantic-scraping stage |
| `WHACK26_SCRAPE_MAX_DEPTH` | `2` | Maximum link/reference depth |
| `WHACK26_SCRAPE_MAX_RESOURCES` | `100` | Hard document budget per run |
| `WHACK26_SCRAPE_MAX_LINKS_PER_RESOURCE` | `25` | Link fan-out cap |
| `WHACK26_SCRAPE_SUMMARY_THRESHOLD` | `0.62` | Follow all links above this summary similarity |
| `WHACK26_SCRAPE_LINK_THRESHOLD` | `0.68` | Follow an individual link above this similarity |
| `WHACK26_SCRAPE_CREDIBILITY_THRESHOLD` | `0.50` | Evidence eligibility threshold |
| `WHACK26_SCRAPE_TTL_DAYS` | `7` | Fetch cache TTL |
| `WHACK26_SCRAPE_USER_AGENT` | project UA | Crawler user-agent with contact |
| `WHACK26_SCRAPE_FETCH_ENABLED` | `0` | Enable the compliant HTTP fetcher |
| `WHACK26_SCRAPE_ALLOWED_DOMAINS` | empty | Comma-separated domain allowlist |
| `WHACK26_SCRAPE_BLOCKED_DOMAINS` | empty | Comma-separated domain blocklist |
| `WHACK26_SCRAPE_RATE_LIMIT_SECONDS` | `1.0` | Minimum per-domain request interval |
| `WHACK26_SCRAPE_MAX_BYTES` | `2000000` | Maximum response body size |
| `WHACK26_SCRAPE_MAX_TEXT_CHARS` | `4000` | Maximum extracted text excerpt stored |
| `WHACK26_SCRAPE_FETCH_TIMEOUT` | `15.0` | Per-request timeout in seconds |
| `WHACK26_SCRAPE_MAX_REDIRECTS` | `3` | Maximum redirect hops |
| `WHACK26_SCRAPE_ALLOW_PRIVATE_NETWORKS` | `0` | Test-only; allows loopback/private hosts |

## 16. Testing and acceptance

**Unit tests**

- URL canonicalization: tracking parameters, fragments, case, trailing slash,
  IDN, non-http schemes.
- Identity: same URL, same content at a different URL, same DOI, different
  resources; no false merge for similar titles.
- Summary schema: valid, too long, missing summary, secret/PII redaction.
- Threshold gating: summary above/below threshold; link above/below the tighter
  threshold; budget exhaustion.
- Credibility: peer-reviewed vs blog, retracted penalty, unknown author,
  component breakdown.
- Comparison cache: repeat call makes no LLM call; invalidation on embedding or
  method-version change; symmetric pair normalization.
- Robots and rate limits: disallowed path, crawl-delay, per-domain budget.
- Resource classification: assigned path, no new root, duplicate merge.

**Integration tests**

- Mocked HTTP/LLM/embeddings with a 3-resource graph: seed → two references →
  one duplicate URL → one below-threshold link. Assert dedup, gating,
  classification, comparison reuse, and artifact validation.
- Offline mode: all resources from cache; a miss produces a warning, not a crash.
- Budget exhaustion: the run stops cleanly and records `scrape_budget_exhausted`.
- Non-credible exclusion: a low-score resource is stored but absent from factor
  evidence.
- No secrets/PII in `scrape.json`, logs, or the artifact.

**Live smoke**

- Start from 10 OpenAlex records for California Housing, follow
  `referenced_works` one hop, cap at 20 resources. Measure:
  - target relevance of followed resources (manual review of 20);
  - duplicate rate;
  - summary latency and LLM calls;
  - cache hit rate on a second run;
  - zero robots/rate-limit violations.
- Then add Wikipedia links and HF/Zenodo dataset links.

**Acceptance for the first increment**

- `--scrape` discovers at least 10 new credible resources at depth 1 from the
  demo seeds, with at least 70% judged relevant by a human.
- 100% of duplicate URLs/content hashes are deduplicated without a second
  summary call.
- A second identical run makes zero new LLM calls for summaries and
  comparisons.
- The artifact validates and the preview renders the semantic-scraping section.
- No full text, secrets, PII, robots violations, or paywalled content.
- The existing one-hop pipeline and all current tests still pass with scraping
  disabled.

## 17. Compliance and security

The compliant fetcher is implemented in `whack26/research/scrape/fetch.py` and
is **opt-in** (`WHACK26_SCRAPE_FETCH_ENABLED=1` or `--fetch`). It enforces:

- `robots.txt` for every request, with a per-origin cache and `Crawl-delay`;
- a per-domain minimum interval (default 1 s) and a fetch budget;
- manual redirect handling with robots/private-network/allowlist re-checks at
  every hop;
- private, loopback, link-local, reserved, and multicast address blocking
  unless `WHACK26_SCRAPE_ALLOW_PRIVATE_NETWORKS=1` (test-only);
- an optional domain allowlist and blocklist;
- rejection of PDFs, binaries, images, archives, and office documents by
  content type;
- response size and timeout caps;
- extraction of only title, meta description, text excerpt, and links; the
  `document_fetches` cache stores a capped text excerpt, never a PDF or full
  article body, and obeys the TTL;
- `ETag`/`Last-Modified` conditional requests and a fresh-cache fast path;
- a descriptive user-agent with contact information and `DNT: 1`.

The fetcher never downloads PDFs, never uses cookies, never logs in, and never
stores full article text. General web search remains out of scope; the fetcher
follows links already present in API metadata or seed documents.

Compliance is tested against a real Flask mock server in
`tests/test_scrape_fetch.py`: robots allow/disallow/404/5xx/redirect,
`Crawl-delay`, rate limits, redirects, redirect-to-private blocking, SSRF
blocking, allow/block lists, PDF/binary rejection, size/time caps, 404/429
handling, cache hits, ETag/304 reuse, user-agent, one-robots-fetch-per-origin,
text-excerpt caps, and the pipeline’s use of the compliant fetcher.

## 18. Phases and effort

| Phase | Deliverable | Estimate |
|---|---|---|
| 0 | SQLite schema v2, v1→v2 migration, `DocumentStore` protocol, settings, feature flag | 2–3 days |
| 1 | Identity/dedup, document store, summary + embedding | 3 days |
| 2 | Frontier, thresholds, budgets, link gating | 3 days |
| 3 | Document classification and field membership | 2 days |
| 4 | Credibility scoring and filtering | 2 days |
| 5 | Comparison engine, cache, document critic | 3 days |
| 6 | Statistics preload, pipeline/CLI/artifact integration | 3 days |
| 7 | Unit/integration/live smoke, migration hardening, docs | 3 days |
| | **Total** | **21–24 days** |

This is a follow-on feature, not a 6–8 hour slice. The first walking skeleton
(Phase 0–2 on OpenAlex references only) is the smallest demoable increment.

## 19. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Legal/robots exposure from link crawling | Start with API-provided references and links; allowlist; robots parser; no PDFs/full text; legal review before web search |
| LLM cost and latency | Summary/comparison caches; hard budgets; batch where possible; deterministic pre-checks before LLM calls |
| Hallucinated summaries | Strict JSON schema; require quotes/evidence; reject empty/oversized summaries; human spot checks |
| Embedding drift | Store `embedding_model`; invalidate similarities/comparisons on model change |
| Dedup false positives | Exact content hash first; conservative URL canonicalization; manual review of merges |
| Hierarchy pollution from documents | Reuse create/assign; require confidence; duplicate merge; defer level-shifting |
| Credibility bias | Deterministic, decomposable features; audit on a labeled set; keep low-score documents visible but excluded |
| SQLite migration data loss | Timestamped backup; one transaction; row-count and embedding round-trip checks; idempotent `user_version`; rollback on failure |
| SQLite brute-force slowdown | Indexed lookups by `url_hash`/`content_sha512`; keep embeddings as compact BLOBs; add a `DocumentStore` seam; revisit only if latency exceeds the budget |
| Schema drift between code and DB | `schema_migrations` + `PRAGMA user_version`; migration tests on a v1 fixture; `PRAGMA integrity_check` after migration |

## 20. First walking skeleton

Build this before the full plan:

1. Add the v2 `documents`, `datasets`, `document_links`, `document_fields`,
   `document_comparisons`, `document_critiques`, and `document_fetches` tables,
   plus the `DocumentStore` protocol.
2. Implement the v1→v2 migration and run it against a copy of `cache/fields.db`;
   assert the existing datasets, paths, embeddings, and `cache_key` survive.
3. Implement `canonicalize_url` and `content_sha512`.
4. Take the existing one-hop OpenAlex records for California Housing as seeds.
5. For each seed, store the document and follow `referenced_works` one hop,
   capped at 20 documents.
6. Summarize and embed each new document; deduplicate by URL/content hash.
7. Classify each document into the existing field tree.
8. Compare each document to the dataset summary with a deterministic cosine and
   one LLM comparison; store the comparison.
9. Write `scrape.json` and a short `scrape.md`; do not touch the main artifact
   yet.
10. Add mocked tests for the graph, dedup, thresholds, cache reuse, and the
    v1→v2 migration.

This proves the schema update, the identity model, the summary/embedding
contract, the link frontier, and the comparison cache without web crawling or a
storage-engine change. Once it is stable, wire the stage into `whack26 run`
behind `WHACK26_SCRAPE_ENABLED=1`.
