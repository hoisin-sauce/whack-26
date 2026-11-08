# Research, caching, and factor mapping

Status: decided in [ticket #8](https://github.com/hoisin-sauce/whack-26/issues/8). Applies to the 6–8 hour slice.

## Pipeline stages

```
ingest -> profile -> classify -> query -> fetch -> normalize -> cache
      -> deduplicate -> rank -> extract factors -> map factors
      -> refine classification (once) -> hypotheses -> critic -> artifact
```

Research runs **after** the first classification and **before** hypothesis generation. It may refine the field path once.

## 1. Ingest and profile

Supported inputs: CSV, TSV, Excel (`.xlsx`/`.xls`), JSON, Parquet. Pickle is rejected. Pandas reads with an explicit engine; a failure produces a typed `IngestError` with the format and the underlying message.

Reject a file with 0 rows or 0 columns: it cannot be classified or modelled. A one-row dataset is allowed but flagged in `warnings`. If `--target` names a missing column, fail before profiling with a clear list of available columns.

`DatasetProfile`:

```python
class ColumnProfile(BaseModel):
    name: str
    dtype: str                 # pandas dtype string
    role: Literal["feature", "target", "id", "time", "leakage_risk", "unknown"]
    missing_pct: float
    cardinality: int
    sample_values: list[str]   # <= 5, truncated
    numeric: dict | None       # min, max, mean, std, zeros, negatives
    top_values: list[tuple[str, int]] | None
    datetime: bool

class DatasetProfile(BaseModel):
    dataset_id: str            # ds_<sha256[:12]>
    source_path: str
    sha256: str
    format: str
    rows: int
    columns: list[ColumnProfile]
    description: str | None
    target: TargetChoice | None
    time_column: str | None
    warnings: list[str]
```

Target selection: `--target` wins. Otherwise rank candidates by name (`target`, `label`, `outcome`, `y`, `price`, `value`, `class`, `result`), dtype, and cardinality; the LLM picks one from the top five with a rationale. Record `source = "user" | "inferred"` and `confidence`. If no target is plausible, the run continues with `target = None`; hypotheses become descriptive and the critic marks claims accordingly.

Column roles:
- `id`: unique or near-unique identifier; excluded from features.
- `time`: datetime dtype or name matching `date`, `time`, `timestamp`, `year`, `month`; used by the time-series checks, not automatically as a feature.
- `leakage_risk`: names containing `outcome`, `result`, `target`, `future`, `post`, `after`, or columns that are deterministic functions of the target.
- `target`: the chosen target.
- `feature`: everything else.

## 2. Dataset summary

The summary is compact text, not raw rows:

```
Dataset: <filename>
Description: <optional user description>
Rows: <n>; Columns: <n>
Target: <name> (<dtype>, <n unique>, <missing>)
Time column: <name or none>
Columns:
- <name> (<dtype>, missing <pct>%, <n unique>): <roles and stats>
...
Sample rows (first 3, values truncated):
<pipe-separated table>
```

Truncate each sample value to 80 characters. Omit string columns that look like free text, names, emails, or IDs from the sample rows. Never send file paths or secrets.

## 3. Query generation

One LLM call with the dataset summary, field path, target, and source capabilities. Output:

```json
{
  "queries": [
    {"source": "openalex", "query": "determinants of house prices", "rationale": "..."},
    {"source": "openalex", "query": "housing market income elasticity", "rationale": "..."},
    {"source": "hf", "query": "california housing", "rationale": "..."},
    {"source": "zenodo", "query": "housing price dataset", "rationale": "..."},
    {"source": "wikipedia", "query": "housing market", "rationale": "..."}
  ]
}
```

Rules: 4–6 queries; at least two paper queries; at least one dataset query; at most one Wikipedia query; ≤ 12 words each; no boolean operators; no PII. Prefer 2–5 words per query and substantive domain factors rather than generic machine-learning terms: the Hugging Face search endpoint matches whole phrases, and long natural-language queries often return nothing. If the LLM fails, use deterministic templates:

- papers: `"<leaf field> <target concept> determinants"`, `"<leaf field> <target concept> prediction"`
- datasets: `"<leaf field> dataset"`, `"<target concept> dataset"`
- Wikipedia: `"<leaf field> <target concept>"`

## 4. Source adapters

A common protocol:

```python
class SourceAdapter(Protocol):
    name: str
    def search(self, query: str, limit: int, client: httpx.Client) -> list[ResearchRecord]: ...
```

### OpenAlex (primary papers)

```
GET https://api.openalex.org/works
  ?search=<query>
  &filter=is_retracted:false
  &per_page=10
  &select=id,doi,title,publication_year,authorships,primary_location,open_access,
          cited_by_count,abstract_inverted_index,topics,keywords,is_retracted
```

Auth: `api_key` query parameter or `Authorization: Bearer`. Both verified. Keyless is allowed but capped at `$0.10/day` and cursor paging is limited to page 1; the adapter uses keyless only as a fallback and sets `limited_mode=true`. Costs: search `$0.001` per call. Reconstruct the abstract from `abstract_inverted_index` (positions → words). Record `meta.cost_usd` in the cache log.

### arXiv (secondary papers)

```
GET https://export.arxiv.org/api/query
  ?search_query=all:<query>
  &start=0
  &max_results=5
  &sortBy=relevance
```

No key. One request every 3 seconds, one connection at a time. Parse Atom: `entry.id`, `entry.title`, `entry.summary`, `author.name`, `published`, `updated`, `category@term`, `link@href`, `arxiv:doi`. Metadata is CC0; never download or redistribute PDFs.

### Hugging Face Hub (datasets)

```
GET https://huggingface.co/api/datasets?search=<query>&limit=5
```

Keyless public search. Optional `Authorization: Bearer $HF_TOKEN` raises the rate limit. Map `id`, `author`, `description`, `downloads`, `likes`, `tags`, `createdAt`, `lastModified`, `private`, `gated`, `sha`. Respect gated/private datasets; each repo's licence tag governs.

### Zenodo (DOI-backed datasets)

```
GET https://zenodo.org/api/records?q=<query>&size=5&sort=bestmatch
```

`bestmatch` was chosen after live validation: `mostrecent` returned unrelated recent deposits for multi-word queries because Zenodo ranks the query loosely and then sorts by date. Public search keyless; optional `Authorization: Bearer $ZENODO_TOKEN`. Map `hits.hits[].id`, `doi`, `metadata.title`, `metadata.description`, `metadata.publication_date`, `metadata.creators`, `metadata.keywords`, `metadata.license`, `metadata.resource_type`, `links`. Metadata is CC0 unless stated; files vary; Zenodo terms restrict military use.

### Wikipedia REST (encyclopedic fallback)

```
GET https://en.wikipedia.org/w/api.php?action=query&list=search
  &srsearch=<query>&format=json&utf8=1&srlimit=3
GET https://en.wikipedia.org/api/rest_v1/page/summary/<title>
```

Keyless. Set a descriptive `User-Agent` with contact information (`whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)`); Wikimedia rejects agents without one with HTTP 403. Attribute CC BY-SA. This is an encyclopedic fallback, **not** general web search; label it as such in the artifact.

### Adapter status

Twelve source names are declared in `whack26/research/base.py`. Nine are
registered in `_build_adapters`: `openalex`, `arxiv`, `huggingface`, `zenodo`,
`wikipedia`, `dataverse`, `openml`, `datagov`, and `datacite`. Three modules are
implemented and unit-tested but **not registered**: `kaggle`,
`semantic_scholar`, and `websearch`; a generated query for those sources
currently yields `unknown_source`. General web search is therefore not part of
the default run. (Design note: Brave Search lost its free plan in 2026-11;
Google Programmable Search remains a key-gated option.)

## 5. Normalization and deduplication

`ResearchRecord`:

```python
class ResearchRecord(BaseModel):
    id: str                    # "<source>:<source_id>"
    source: Literal[
        "openalex", "arxiv", "huggingface", "zenodo", "wikipedia",
        "kaggle", "dataverse", "openml", "datagov", "datacite",
        "semantic_scholar", "websearch",
    ]
    source_id: str
    title: str
    abstract: str | None       # <= 4,000 chars
    authors: list[str]
    year: int | None
    doi: str | None
    url: str
    license: str | None
    retrieved_at: str
    query: str
    relevance: float
    also_seen_in: list[str] = []
```

- Strip HTML and collapse whitespace.
- Dedup priority: DOI → arXiv ID → OpenAlex ID → HF ID → Zenodo ID → `sha256(normalized title + first author + year)`.
- Merge duplicates; keep the record with the highest `SOURCE_RANK` for its kind (papers: OpenAlex > arXiv > Semantic Scholar; datasets: Zenodo > Dataverse/OpenML > Hugging Face/Kaggle/data.gov/DataCite; web: Wikipedia > web search). See `whack26/research/base.py` for the exact table.
- Rank: `0.55 * cos(embed(summary), embed(title + " " + abstract)) + 0.25 * source_prior + 0.20 * recency`, where `source_prior` is OpenAlex 0.8, arXiv/Semantic Scholar/Zenodo 0.7, Dataverse/OpenML 0.65, Hugging Face/Kaggle/data.gov/DataCite 0.6, Wikipedia 0.5, web search 0.4, and `recency = exp(-age_years / 5)`.
- Keep the top 20 overall. Preserve at least one paper and one dataset record when available.

## 6. Cache

SQLite `cache/research.db`:

```sql
CREATE TABLE api_cache (
  cache_key TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  source TEXT NOT NULL,
  query TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  ttl_seconds INTEGER NOT NULL,
  status INTEGER NOT NULL,
  body TEXT NOT NULL,
  headers_json TEXT
);

CREATE TABLE research_queries (
  cache_key TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  queries_json TEXT NOT NULL
);
```

- Key: `sha256(dataset_id | source | normalized_query | params_json)`.
- TTL: OpenAlex/arXiv/HF/Zenodo 24 h; Wikipedia 7 days.
- `--offline`: read cache only; a miss produces a warning and the source is skipped.
- `--refresh`: ignore TTL and refetch.
- Store raw bodies (JSON or Atom XML) up to 512 KB; truncate larger.
- Log OpenAlex `meta.cost_usd` and `X-RateLimit-Remaining`.
- Cache generated queries per dataset and summary hash. Query generation is nondeterministic, so without this a warm run would create new cache keys and miss every source response. The field path is deliberately excluded from the query-cache key: a one-time post-research refinement would otherwise invalidate a warm cache.
- Warm-run cache hit rate target: ≥ 90%.

## 7. Factor extraction and mapping

One LLM call over the profile, field path, and top 12 records:

```json
{
  "factors": [
    {
      "name": "household income",
      "description": "Median household income in the block group.",
      "direction": "positive",
      "claim_type": "association",
      "evidence_ids": ["openalex:W..."],
      "candidate_columns": ["median_income"],
      "rationale": "..."
    }
  ]
}
```

- 5–10 factors. Drop any factor with no valid `evidence_id`.
- Mapping, in order:
  1. Exact or synonym match after normalizing case, punctuation, plural forms, and parenthetical units.
  2. Embedding similarity between the factor text and each column profile text; take the top three above `FACTOR_MAP_COSINE = 0.72`.
  3. One LLM verification call returns `column`, `confidence`, `in_dataset`, and `rationale`.
- `in_dataset = false` factors are context only. A hypothesis may cite them but may not use them as features; a proxy must be explicitly mapped.
- Every factor and mapping is stored in `factors.json` and the artifact.

## 8. Refinement

If `classification.confidence < 0.62`, or the top five record titles contradict the path, run the classification refinement once with the summary plus those titles and keywords. Allowed: assign to another existing path, create a child under the current path, or stop. Creating a root is forbidden. Record `refined_after_research = true` and the reason.

## 9. Failure handling and rate limits

- Per-source `try/except`; record `SourceError(source, query, message)` in `warnings` and continue.
- Retries: 2, exponential backoff (1 s, 4 s) on 429/5xx/timeout.
- OpenAlex `403 key_not_activated`: retry keyless, set `limited_mode=true`, warn.
- arXiv: enforce 3 s between requests globally.
- Sequential calls are acceptable at this scale (≤ 6 queries × 5 sources = 30 calls).
- `httpx.Client(timeout=20.0, follow_redirects=True)`; descriptive `User-Agent`.
- If every source fails and the cache is empty: continue with 0 records, empty factors, and a warning. The run must not crash.

## 10. Module layout

```
whack26/research/
  base.py          # protocol, ResearchRecord, SourceError
  openalex.py
  arxiv.py
  huggingface.py
  zenodo.py
  wikipedia.py
  normalize.py     # abstract reconstruction, HTML strip, dedup, ranking
  cache.py         # SQLite cache
  pipeline.py      # query -> fetch -> normalize -> rank -> refine
whack26/factors.py # extraction + mapping
```

## 11. Acceptance

- 10-dataset evaluation: at least one paper and one dataset record for ≥ 8/10 datasets when sources are up.
- Zero crashes when any source returns 401/403/429/500 or times out.
- Warm-run cache hit rate ≥ 90%.
- At least 3 factors and at least one mapped factor for ≥ 8/10 datasets.
- No secrets, API keys, or PII in logs, cache metadata, or the artifact.
- Offline mode completes from cache and clearly labels degraded sources.
