# Verified API notes

Live checks performed on 2026-11-07 from the development machine. Keys are in `.env` and are never committed.

## Environment

| Item | Verified value |
|---|---|
| Python | 3.12.4 and 3.15.0 installed; `py -3.12` works |
| DeepSeek key | Valid; model listing and funded balance succeed |
| Gemini key | Valid; model listing succeeds |
| OpenAlex key | Valid and activated; both auth styles work |
| Hugging Face token | Valid; authenticated dataset search returns 200 |
| Zenodo | Keyless search returns 200 |
| arXiv | Keyless Atom API returns 200 |
| Wikipedia | Keyless Action API returns 200 |

## Gemini (native REST)

Base: `https://generativelanguage.googleapis.com/v1beta`

Auth header: `x-goog-api-key: <GEMINI_API_KEY>`

### Model availability and latency

A one-token prompt (`Reply with exactly: OK`) with `temperature=0`, `maxOutputTokens=64`, and `thinkingConfig.thinkingBudget=0`:

| Model | Result | Time |
|---|---|---|
| `gemini-3.5-flash` | HTTP 200, text `OK`; `modelVersion` reported as `gemini-3.6-flash` | ~1.0–1.6 s |
| `gemini-3.6-flash` | HTTP 200, text `OK` | ~1.08 s |
| `gemini-3.7-flash` | Timeout at 30 s | > 30 s |
| `gemini-3.8-flash` | HTTP 200 but no content, or a 120 s timeout; default thinking consumed the token budget | 75–120 s |

**Decision:** default `GEMINI_MODEL=gemini-3.6-flash` (verified ~1.08 s). `gemini-3.5-flash` is an alias that reports the same model version. `gemini-3.8-flash` is opt-in only; it is unusable on this machine/account as of the probe.

The OpenAI-compatible endpoint (`/v1beta/openai/chat/completions`) timed out at 120 s with 0 bytes for `gemini-3.8-flash`. Use the native REST API via `httpx`; the `openai` SDK is not required.

### Structured output

This request returned valid JSON (`{"ok": true, "reason": "test"}`) in ~1.6 s with `gemini-3.6-flash`:

```json
{
  "contents": [{"role": "user", "parts": [{"text": "Return a JSON object with ok=true and reason=\"test\"."}]}],
  "generationConfig": {
    "temperature": 0,
    "maxOutputTokens": 256,
    "thinkingConfig": {"thinkingBudget": 0},
    "responseMimeType": "application/json",
    "responseSchema": {
      "type": "object",
      "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}},
      "required": ["ok", "reason"]
    }
  }
}
```

Use `responseMimeType: "application/json"` plus `responseSchema` for every structured call. Validate the result with Pydantic; retry once on schema failure.

### Embeddings

`POST /v1beta/models/gemini-embedding-2:embedContent`:

```json
{
  "model": "models/gemini-embedding-2",
  "content": {"parts": [{"text": "housing prices and household income"}]},
  "outputDimensionality": 768
}
```

Returned a 768-element vector (first values `-0.00719808, -0.004311064, -0.035844736`). L2-normalize before cosine.

### Free-tier caveat

The Gemini free tier may use prompts and responses to improve Google products, and human reviewers may read them. Do not send sensitive, confidential, or personal data. Cloudflare Workers AI and local `model2vec` embeddings are documented seams but are not implemented; use `--mock` for plumbing tests with sensitive schemas.

## DeepSeek (OpenAI-compatible)

Base: `https://api.deepseek.com`

Auth header: `Authorization: Bearer <DEEPSEEK_API_KEY>`

Probes performed on 2026-11-08:

| Check | Result |
|---|---|
| `GET /models` | HTTP 200; `deepseek-flash` (DeepSeek-V4.1-Flash) and `deepseek-v4-pro` |
| `GET /user/balance` | HTTP 200; account funded |
| `POST /chat/completions` with `response_format={"type": "json_object"}` | HTTP 200; returned `{"ok": true}` |
| `POST /embeddings` and `POST /v1/embeddings` | HTTP 404; DeepSeek has no embeddings API |
| Default thinking (`reasoning_effort=high`) with `max_tokens=32` | Finish reason `length`; reasoning consumed the budget and content was empty |
| `thinking={"type": "disabled"}` | HTTP 200; content returned directly in ~1 s |

Structured-call shape:

```json
{
  "model": "deepseek-flash",
  "messages": [
    {"role": "system", "content": "<prompt + JSON Schema>"},
    {"role": "user", "content": "<task>"}
  ],
  "response_format": {"type": "json_object"},
  "thinking": {"type": "disabled"},
  "temperature": 0,
  "max_tokens": 4096
}
```

JSON mode guarantees valid JSON but does not enforce a schema, so the schema is included in the system message and callers still validate with Pydantic. Gemini remains the embedding provider.

## OpenAlex

Base: `https://api.openalex.org`

Auth (both verified identically):

- `?api_key=<OPENALEX_API_KEY>`
- `Authorization: Bearer <OPENALEX_API_KEY>`

Probe: `GET /works?search=housing&per_page=1` returned 2,295,667 results and `meta.cost_usd = 0.001` with the key. Keyless also works but is limited to `$0.10/day` and page 1.

Fields used:

```text
id, doi, title, publication_year, authorships, primary_location, open_access,
cited_by_count, abstract_inverted_index, topics, keywords, is_retracted
```

`abstract_inverted_index` maps each word to its positions; reconstruct by sorting positions. `topics[0].display_name` was `Housing Market and Economics` for the probe. Costs: search `$0.001` per call; free key budget `$1/day` after activation; new accounts are paced at `$0.10` per 5 minutes until the first dollar is spent.

Bibliometric enrichment (key-gated by `WHACK26_RESEARCH_METRICS_ENABLED`):

```text
GET /works?filter=doi:<d1>|<d2>&select=id,doi,cited_by_count,authorships,primary_location&per_page=50
GET /authors?filter=id:<A1>|<A2>&select=id,display_name,cited_by_count,summary_stats&per_page=50
```

Both OR filters accept up to 100 values; a list request returns up to 100 records. Live probe: three author IDs returned `summary_stats.h_index` 132/60/28 and `meta.cost_usd = 0.0001`. Works-by-DOI resolves arXiv/Zenodo/DataCite records to OpenAlex author IDs when a DOI exists.

## arXiv

Base: `https://export.arxiv.org/api/query`

Probe: `search_query=all:housing price&max_results=1` returned 200. Atom fields verified: `entry.id` (`http://arxiv.org/abs/1802.05870v1`), `entry.title`, `entry.summary`, `author.name`, `published`, `updated`, `category@term`, `link@href`, `arxiv:doi`. One request every 3 seconds; one connection at a time. Metadata CC0; do not redistribute PDFs.

## Hugging Face Hub

Base: `https://huggingface.co/api`

Probe: `GET /datasets?search=housing&limit=1` returned 200 with and without the token. Token is valid and raises the anonymous limit. Fields: `_id`, `id`, `author`, `disabled`, `gated`, `lastModified`, `likes`, `trendingScore`, `private`, `sha`, `description`, `downloads`, `tags`, `createdAt`, `key`. Respect gated/private datasets and each repo's licence tag.

## Zenodo

Base: `https://zenodo.org/api`

Probe: `GET /records?q=housing&size=1` returned 200. Top-level keys: `hits`, `aggregations`, `links`. Hit keys: `id`, `doi`, `conceptdoi`, `metadata`, `title`, `links`, `files`, `status`, etc. `metadata` keys include `title`, `description`, `publication_date`, `creators`, `keywords`, `license`, `resource_type`. Metadata CC0 unless stated; files vary; non-military use.

## Wikipedia

- Search: `GET https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=housing%20market&format=json&utf8=1&srlimit=3` returned 200; first title `Housing market`, snippet length 220.
- Summary: `GET https://en.wikipedia.org/api/rest_v1/page/summary/<title>` returned 200 (2,166 bytes for `Housing`).
- Keyless. Set a descriptive `User-Agent`; attribute CC BY-SA. This is an encyclopedic fallback, not general web search.

## General web search

The core path has no required general web-search provider; Wikipedia REST is the
default labeled encyclopedic fallback. An optional `websearch` adapter is
implemented and registers only when configured with
`WEBSEARCH_PROVIDER=google|brave` plus `WEBSEARCH_API_KEY` (and
`WEBSEARCH_CX` for Google). A missing provider/key is a normal skip, not a run
failure.

| Provider | Free tier | Status |
|---|---|---|
| Brave Search API | No free plan as of 2026-11; a $5 monthly credit, then $5 per 1,000 queries | Implemented in the optional adapter; configure `WEBSEARCH_PROVIDER=brave` |
| Google Programmable Search JSON API | 100 queries/day free; requires an API key and a Programmable Search Engine ID | Implemented in the optional adapter; configure `WEBSEARCH_PROVIDER=google`, `WEBSEARCH_API_KEY`, and `WEBSEARCH_CX` |
| Self-hosted SearXNG | Free software, but no hosted API or uptime guarantee | Out of scope |

## Demo data URLs

| Dataset | URL | Verified |
|---|---|---|
| California Housing | `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv` | 200; 1,423,529 bytes; 20,640 rows; 10 columns |
| Palmer Penguins | `https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv` | 200; 15,241 bytes; 344 rows |

## Things that did not work

| Attempt | Result | Consequence |
|---|---|---|
| OpenAI-compatible Gemini endpoint | 120 s timeout, 0 bytes | Use native REST via `httpx` |
| DeepSeek embeddings endpoint | HTTP 404 | Keep Gemini for embeddings |
| `gemini-3.8-flash` default thinking | 75–120 s and/or empty content | Default to `gemini-3.6-flash`; disable thinking |
| `gemini-3.8-flash` with `thinkingBudget=0` | 120 s timeout | Keep it opt-in only |
| Google Dataset Search | No public query API | Dropped |
| Keyless Semantic Scholar | 429 | Optional, key-gated |
| Keyless general web search | No lawful free API without a key | Wikipedia is the labeled encyclopedic fallback; general web search is out of scope |
| Brave Search API free tier | No free plan as of 2026-11; $5 monthly credit, then usage-based pricing | Removed; no Brave key is required |

## Operational defaults

| Setting | Value |
|---|---|
| `WHACK26_LLM_PROVIDER` | `deepseek` |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `WHACK26_EMBED_PROVIDER` | `gemini` |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` |
| `EMBED_DIMS` | 768 |
| DeepSeek `thinking` | `{"type": "disabled"}` for deterministic JSON/code calls |
| Gemini `thinkingBudget` | 0 for deterministic JSON/code calls |
| HTTP client | `httpx.Client(timeout=20.0, follow_redirects=True)` |
| User-Agent | `whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)` |
| OpenAlex `per_page` | 10 (slice); 100 only for bulk |
| Cache TTL | 24 h papers/datasets; 7 d Wikipedia |
