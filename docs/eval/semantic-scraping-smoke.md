# Semantic-scraping smoke

Status: implementation complete; mock end-to-end validated. The full live smoke
is pending provider quota; a 5-document live smoke is recorded below.

> **Historical record (2026-11-08).** This smoke was captured when the store was
> schema v2 and the artifact schema was `1.1.0`. The store is now v4 and the
> artifact schema is `1.5.0`; `schemas/artifact.schema.json` is the authority.
> The results below are preserved as-is and are not updated.

## What is implemented

- SQLite v2 knowledge store in `cache/fields.db` at the time (now v4):
  `documents`, `datasets`, `document_links`, `document_fields`,
  `document_comparisons`, `document_critiques`, `document_fetches`, and
  `schema_migrations`.
- The v1→v2 migration from the previous `datasets`/`field_members` schema, now
  followed by v3→v4 migrations in `whack26/store.py`.
- Document identity/dedup, summaries and embeddings, target-gated frontier,
  field classification, credibility scoring, cached comparisons, document
  method critique, and the statistics/validity preload.
- A compliant HTTP fetcher (`whack26/research/scrape/fetch.py`) with robots.txt,
  crawl-delay, per-domain rate limits, redirect re-validation, SSRF/private
  network blocking, allow/block lists, PDF/binary rejection, size/time caps,
  ETag/304 caching, and a capped text-excerpt cache.
- Artifact schema `1.1.0` at the time (now `1.5.0`) with the optional
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
- the artifact validator accepts schema `1.1.0` at the time (current schema
  `1.5.0`);
- the full suite was green at the time of writing; see the current suite report
  for the present count.

The mock research cache is empty, so the mock run records a `scrape_no_seeds`
warning and an empty document list. The module-level tests exercise real
document graphs with mocked HTTP/LLM/embedding clients.

## Compliance fetcher tests

`tests/test_scrape_fetch.py` runs a real Flask mock server in a background
thread and covers:

- robots allow/disallow, longest-match and tie-to-allow, `Crawl-delay`,
  robots 404, robots 5xx, and robots redirects;
- one robots fetch per origin;
- public page extraction (title, text, links) and JSON content;
- PDF and binary rejection;
- redirect following, redirect-to-disallowed blocking, and redirect limits;
- private/loopback SSRF blocking and the test-only private-network override;
- domain allowlist and blocklist enforcement;
- per-domain rate limiting with an injected clock;
- response size and timeout caps;
- cache hits, ETag/304 reuse, user-agent, and 404/429 handling;
- text-excerpt caps that never store full HTML;
- the semantic-scraping pipeline using the compliant fetcher end to end.

The suite passes as part of the full test run. The compliance suite has 34
tests at the time of writing.

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

