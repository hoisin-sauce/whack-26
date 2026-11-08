# Semantic-scraping smoke

Status: implementation complete; mock end-to-end validated. The live smoke is
pending provider quota.

## What is implemented

- SQLite v2 knowledge store in `cache/fields.db`: `documents`, `datasets`,
  `document_links`, `document_fields`, `document_comparisons`,
  `document_critiques`, `document_fetches`, and `schema_migrations`.
- v1→v2 migration from the previous `datasets`/`field_members` schema.
- Document identity/dedup, summaries and embeddings, target-gated frontier,
  field classification, credibility scoring, cached comparisons, document
  method critique, and the statistics/validity preload.
- Artifact schema `1.1.0` with the optional
  `research.semantic_scraping` section and a Markdown preview section.

## Commands

```powershell
# Mock end-to-end: no keys or network
.\.venv\Scripts\python.exe -m whack26 run `
  --input data\california_housing.csv `
  --target median_house_value `
  --mock --offline --scrape `
  --out runs\mock-scrape

# Scraping-only inspection
.\.venv\Scripts\python.exe -m whack26 scrape `
  --input data\california_housing.csv `
  --target median_house_value `
  --mock --offline `
  --max-resources 10 --max-depth 1 `
  --out runs\mock-scrape-only

# Preload the statistics/validity corpus
.\.venv\Scripts\python.exe -m whack26 preload-statistics `
  --limit 0 --mock --offline
```

## Mock verification

The mock path verifies:

- `run --scrape` writes `scrape.json`, adds
  `research.semantic_scraping` to `artifact.json`, and renders
  `## Semantic scraping` in `preview.md`;
- the `scrape` command runs the same pipeline with scraping enabled;
- `preload-statistics` handles an offline/zero-limit run without network;
- the artifact validator accepts schema `1.1.0`;
- the full suite is green (`486 passed` at the time of writing).

The mock research cache is empty, so the mock run records a `scrape_no_seeds`
warning and an empty document list. The module-level tests exercise real
document graphs with mocked HTTP/LLM/embedding clients.

## Live smoke plan

The live smoke should start from the existing one-hop OpenAlex records for
California Housing and follow `referenced_works` one hop:

```powershell
$env:GEMINI_MODEL='gemini-3.5-flash-lite'
.\.venv\Scripts\python.exe -m whack26 scrape `
  --input data\california_housing.csv `
  --target median_house_value `
  --max-resources 20 `
  --max-depth 1 `
  --out runs\live-scrape
```

Record:

- number of discovered documents and credible documents;
- target relevance of a manual sample of 20;
- duplicate URL/content rate;
- summary/comparison LLM calls and cache hits;
- second-run cache hit rate;
- zero robots/rate-limit violations.

Acceptance: at least 10 new credible documents at depth 1, at least 70% judged
relevant by a human, 100% duplicate dedup, zero new summary/comparison calls on
a second identical run, and no full text, secrets, PII, or paywalled content.

## Live smoke result (2026-11-08)

A small live run used `gemini-3.5-flash-lite` and the five existing California
research records as seeds, with a copy of the live v1 field store:

```text
documents 5
links 5
comparisons 0
critiques 0
credible 3
warnings 5 (duplicate URLs)
```

All five documents were summarized with the real provider, embedded, and
classified into the existing path
`Real Estate and Housing > Housing Valuation and Demographics`. The three
Hugging Face dataset records scored above the credibility threshold
(0.552–0.566); the two paper records scored 0.447 and 0.478 and were correctly
excluded from evidence. Every document was a California Housing dataset or
paper, so the spot-check relevance was 5/5.

Comparisons and critiques were zero because the ad-hoc live script did not
supply a target dataset document or the critic corpus; those paths are covered
by `tests/test_scrape_compare.py`, `tests/test_document_critic.py`, and
`tests/test_scrape_pipeline.py`. The live run also exercised the v1→v2
migration against a real store that predated the `cache_key` column; that
exposed a migration edge case which is now fixed and covered by
`tests/test_store_documents.py`.

The full 20-resource depth-1 live smoke remains pending provider quota.

