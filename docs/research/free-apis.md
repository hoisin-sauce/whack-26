# Free research APIs: capabilities, terms, and the minimal slice mix

- **Ticket:** #5 — Research free research APIs and their terms
- **Date:** 2026-11-07
- **Branch:** `research/free-apis`
- **Scope:** papers, datasets, and web results for a local-first Python slice; 6–8 hour build; no Docker/Postgres; Python 3.12 and 3.15 available; no API keys held at the start.

The ticket says rate limits are not the binding constraint for this project. That is fine: raw requests-per-second is rarely the problem. The binding constraints here are **whether a public API actually exists**, **free quotas and cost**, **what the terms allow**, and **how much adapter work fits in 6–8 hours**. This note records all four.

---

## 1. Bottom line

**Papers:** use **OpenAlex as the primary paper source** and **arXiv as the preprint/secondary source**. Add a free OpenAlex API key before the build: keyless access works, but it is limited to a `$0.10/day` budget and cursor paging past the first page requires a key. A free key raises this to `$1/day` and 30 requests/second. Treat **Semantic Scholar as optional**, behind a free key; its keyless pool returned `429` during live probing.

**Datasets:** use **Hugging Face Hub** and **Zenodo** as the two core dataset adapters. Add **DataCite** and **OpenML** only if there is spare time. **Dataverse** is a reasonable next adapter. **data.gov** works only through its new Catalog API with an API key (the old CKAN endpoint is gone for new use); `DEMO_KEY` is a 50-request/day demo, not a production plan. **Kaggle** requires the official API token; the keyless listing endpoint we probed is undocumented and its acceptable-use policy prohibits excessive crawling. Drop **Google Dataset Search**: it has no public query API.

**Web results:** there is **no lawful, keyless, general web-search API** among the usual options. Google Custom Search JSON API is closed to new customers and shuts down on 2027-01-01; Bing Web Search is retired; DuckDuckGo's public API returns instant answers, not web results, and its HTML results must not be scraped. Brave Search is the only current general option, but it needs an account/API key and is paid after free monthly credits. For the slice, build a **provider interface with a Wikipedia REST fallback**, and make real web search a key-gated adapter. This is the one stated requirement that cannot be met with zero keys and zero payment.

**Recommended core mix (fits 6–8 hours):**

- Papers: **OpenAlex + arXiv**
- Datasets: **Hugging Face Hub + Zenodo**
- Web: **Wikipedia REST** as a clearly labelled keyless fallback, plus a **key-gated Brave adapter** (inactive without a key)
- Optional stretch: **DataCite** (DOI-backed datasets), **OpenML** (ML benchmark datasets), **Dataverse** (social-science data)
- Deferred: **Semantic Scholar** (unless a free key is added), **data.gov**, **Kaggle**, **Google Dataset Search**

That is four active adapters plus one fallback adapter — enough source diversity for the demo without turning the build into an API-integration project.

---

## 2. How this was checked

Primary documentation was read and each public endpoint was probed on **2026-11-07**. Where a source requires an account, the note says so rather than assuming keyless access. Live probe results are in the appendix.

Limits and pricing change. OpenAlex changed its access model in 2026; data.gov changed its catalog API; Google Custom Search is being shut down; Bing Search APIs are retired. Re-check the linked pages before relying on any specific number.

---

## 3. Capability matrix

### 3.1 Papers

| Source | API / base | Auth and free access | Limits that bind | Main fields | License / attribution | Verdict |
|---|---|---|---|---|---|---|
| **OpenAlex** | `https://api.openalex.org` | Keyless works; free key recommended | Keyless: 10 req/s, `$0.10/day` shared per IP. Free key: 30 req/s, `$1/day`. Cursor paging past page 1 needs a key. `per_page` max 100; basic paging max 10,000 results. | `id`, `ids`, `doi`, `title`, `type`, `publication_year/date`, `authorships`, `primary_location`, `locations`, `open_access`, `abstract_inverted_index`, `topics`, `keywords`, `cited_by_count`, `referenced_works`, `is_retracted`, `fwci`, `language` | CC0. Cite Priem, Piwowar & Orr (2022), arXiv:2205.01833. | **Primary** |
| **Semantic Scholar** | `https://api.semanticscholar.org/graph/v1` | Keyless shared pool; free API key | Keyless probe returned `429`; free key is about 1 req/s. Search uses `offset`/`limit`; bulk search uses a token; batch lookups up to 500 IDs. | `paperId`, `externalIds`, `title`, `abstract`, `year`, `publicationDate`, `authors`, `venue`, `citationCount`, `referenceCount`, `influentialCitationCount`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf`, `url`, `references`, `citations` | S2 Data Licenses include CC BY-NC and ODC-BY; attribution to “Semantic Scholar” required; API License Agreement. | **Optional, key-gated** |
| **arXiv** | `https://export.arxiv.org/api/query` | No key | One request every 3 seconds; one connection at a time. `max_results` up to 30,000 total, sliced in 2,000s. Results only change after the daily processing run. | Atom: `id`, `title`, `summary`, `author/name`, `published`, `updated`, `category`, `primary_category`, `comment`, `journal_ref`, `doi`, `link` (alternate/PDF/DOI) | Metadata CC0. Full text is for personal/research use; do not redistribute PDFs or source from your own servers. | **Secondary** |

### 3.2 Datasets

| Source | API / base | Auth and free access | Limits that bind | Main fields | License / attribution | Verdict |
|---|---|---|---|---|---|---|
| **Hugging Face Hub** | `https://huggingface.co/api` | Public metadata search keyless; token optional | Anonymous: 500 API requests/5 min per IP. Free logged-in user: 1,000/5 min. `429` includes `RateLimit` headers. | `id`, `author`, `description`, `downloads`, `likes`, `tags` (includes license), `createdAt`, `lastModified`, `private`, `gated`, `sha`, `trendingScore` | HF ToS grants users a broad licence to public repo content, but each repo's stated licence still governs. Respect gated/private datasets. | **Core** |
| **Zenodo** | `https://zenodo.org/api` | Public search keyless; token optional | Guest: 60/min, 2,000/hr. Authenticated: 100/min, 5,000/hr. Search endpoints: 30/min. `size` max 25 anonymous, 100 authenticated. | `hits.hits[]`: `id`, `conceptrecid`, `doi`, `conceptdoi`, `metadata` (`title`, `description`, `publication_date`, `creators`, `keywords`, `license`, `resource_type`), `files`, `links` | Metadata CC0 unless stated otherwise. File licences vary. Zenodo terms restrict use to non-military purposes. | **Core** |
| **OpenML** | `https://www.openml.org/api/v1` | Public read keyless; key only for writes | No published global rate limit; be polite and cache. | `did`, `name`, `version`, `status`, `format`, `file_id`, `quality` (`NumberOfFeatures`, `NumberOfInstances`, `NumberOfMissingValues`, etc.), description/creator/collection date/URL | OpenML terms grant users a non-exclusive licence to use content for their own research, subject to each dataset's licence. | **Stretch** |
| **Dataverse (Harvard)** | `https://dataverse.harvard.edu/api` | Public search keyless; token for private/restricted | No published global rate limit for Harvard; each Dataverse installation is separate. `per_page` up to 1,000. | `data.items[]`: `name`, `type`, `url`, `global_id`/DOI, `description`, `published_at`, `publisher`, `citation`, `authors`, `subjects`, `fileCount`, `versionId`, `versionState` | Metadata and file licences are per dataset/installation; check the record's licence. | **Stretch** |
| **DataCite** | `https://api.datacite.org` | Public API keyless | 500 req/5 min unidentified; 1,000 with an identifying `User-Agent` containing `mailto:`; 3,000 authenticated. `page[size]` up to 1,000. | JSON:API `data[]`: `id`, `attributes.doi`, `creators`, `titles`, `publisher`, `publicationYear`, `subjects`, `types`, `descriptions`, `url`, `rightsList`, `relatedIdentifiers`, `geoLocations` | DataCite Data File (DOIs + deposited metadata) is CC0; linked resources are not covered. Attribute DataCite. | **Stretch/core for DOI data** |
| **data.gov** | `https://api.gsa.gov/technology/datagov/v4/` | Requires API key in `X-Api-Key` | Personal key: 1,000 req/hr. `DEMO_KEY`: 30 req/IP/hr and 50 req/IP/day; not for production or automated querying. | `results[]`: `title`, `description`, `identifier`, `slug`, `publisher`, `keyword`, `theme`, `has_spatial`, `popularity`, `last_harvested_date`, `distribution_titles`, `organization`, `dcat` (`license`, `accessLevel`, `modified`, `temporal`, etc.) | Per-record DCAT metadata; check `dcat.license` and `accessLevel` (`public`, `restricted public`, `non-public`). | **Defer/optional** |
| **Kaggle** | `https://www.kaggle.com/api/v1` | Official client requires a Kaggle token; keyless listing is undocumented | Official API token via `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, or legacy `kaggle.json`. Keyless listing returned `200` in a probe, but the acceptable-use policy prohibits excessive crawling. Downloads require auth and dataset-specific acceptance. | `id`, `ref`, `title`, `subtitle`, `creatorName`, `totalBytes`, `url`, `lastUpdated`, `downloadCount`, `licenseName`, `ownerName`, `usabilityRating`, `isPrivate`, `isFeatured` | Kaggle Terms and Acceptable Use Policy; dataset licences are self-reported and must be verified. Competition data has separate rules. | **Optional, token-gated** |
| **Google Dataset Search** | UI only | None | No public query API. Google's terms prohibit automated access in violation of machine-readable instructions; do not scrape. | N/A (UI; a bulk metadata dataset is published on Kaggle) | Google Terms of Service; Dataset Search help page. | **Drop** |

### 3.3 Web results

| Source | API / base | Auth and free access | Limits that bind | Main fields | Licence / terms | Verdict |
|---|---|---|---|---|---|---|
| **Brave Search** | `https://api.search.brave.com/res/v1/web/search` | API key; free monthly credits, then paid | Search plan: `$5` per 1,000 requests; `$5` free credits/month on that plan (up to `$20`/month advertised across products); 50 queries/s. | `web.results[].title`, `url`, `description`, `age`, `profile`, `thumbnail`, etc. | Brave API terms apply; review storage/display terms before persisting snippets. | **Key-gated** |
| **Google Custom Search JSON API** | `https://www.googleapis.com/customsearch/v1` | Key + search engine ID; existing customers only | 100 queries/day free; `$5`/1,000 after; closed to new customers; service ends 2027-01-01. | `items[].title`, `link`, `snippet`, `pagemap`, etc. | Google APIs Terms; display/attribution restrictions. | **Not viable for a new build** |
| **Bing Web Search API** | Retired | — | Docs archived; service retired. | — | — | **Not viable** |
| **DuckDuckGo Instant Answer** | `https://api.duckduckgo.com` | Keyless | Instant answers/related topics only; no general 20–50 web results. Do not scrape HTML results. | `AbstractText`, `AbstractURL`, `RelatedTopics`, `Results` | DuckDuckGo terms; no general web-results API. | **Fallback only** |
| **Wikipedia / Wikimedia REST** | `https://en.wikipedia.org/api/rest_v1` | Keyless | Set a descriptive `User-Agent`; cache; keep rates modest. | `title`, `extract`, `description`, `content_urls`, `timestamp`; search API for page lists | CC BY-SA; attribution required. | **Keyless fallback** |
| **SearXNG** | Self-hosted | No key | Requires running a service; upstream engines' terms still apply; public instances rate-limit/block. | Engine-dependent | Upstream engine terms. | **Not for this slice** |

---

## 4. Adapter notes

### 4.1 OpenAlex (papers, primary)

**Endpoint examples**

```text
GET https://api.openalex.org/works?search=climate%20change&per_page=100
    &select=id,doi,title,publication_year,authorships,primary_location,open_access,cited_by_count,abstract_inverted_index
GET https://api.openalex.org/works?filter=from_publication_date:2020-01-01,to_publication_date:2026-12-31&per_page=100
GET https://api.openalex.org/works/https://doi.org/10.1038/nature12373
```

**Auth and cost**

- Keyless: 10 req/s and `$0.10/day` per network IP.
- Free key: 30 req/s and `$1/day`; sign-up at <https://openalex.org/signup> is about 30 seconds, with no payment method. The agent can request a key with `POST https://api.openalex.org/api-keys` and the user's email; the user must click the activation link. New accounts are paced at `$0.10` per 5 minutes until the first `$1` is spent.
- Costs: search `$1` per 1,000 calls (`$0.001` each); list/filter `$0.10` per 1,000 (`$0.0001` each); single-entity and DOI lookups are free. Semantic search is 1 req/s.
- Cursor paging past the first page requires a key. Basic paging stops at 10,000 results; `per_page` max 100.

**Fields for the slice**

`id`, `ids`, `doi`, `title`/`display_name`, `type`, `publication_year`, `publication_date`, `authorships[].author.display_name`, `authorships[].institutions[].display_name`, `primary_location.source.display_name`, `primary_location.landing_page_url`, `best_oa_location.pdf_url`, `open_access.is_oa`/`oa_status`/`oa_url`, `abstract_inverted_index`, `topics[].display_name`, `keywords[].display_name`, `cited_by_count`, `referenced_works`, `related_works`, `is_retracted`, `fwci`, `language`.

`abstract_inverted_index` is not plain text. Reconstruct it by placing each word at its listed positions:

```python
def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positions = [(pos, word) for word, pos_list in inverted_index.items() for pos in pos_list]
    return " ".join(word for _, word in sorted(positions))
```

**Caching**

Cache search/filter responses for 24 hours by normalized query + filters + page + `select`. Cache single-entity/DOI lookups for 7 days. Log `meta.cost`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`; stop the run if the remaining budget drops below a configured floor. Prefer `select=` and `per_page=100`. Batch DOI lookups with the OR syntax (up to 100 values).

**Terms**

All OpenAlex data is CC0. Attribution is not legally required but is expected; cite the OpenAlex paper. No personal-use carve-out.

---

### 4.2 Semantic Scholar (papers, optional)

**Endpoint examples**

```text
GET https://api.semanticscholar.org/graph/v1/paper/search
    ?query=climate%20change&limit=100&fields=title,abstract,year,authors,externalIds,openAccessPdf,citationCount
GET https://api.semanticscholar.org/graph/v1/paper/search/bulk?query=climate%20change&fields=title,abstract,year
POST https://api.semanticscholar.org/graph/v1/paper/batch
```

**Auth and reliability**

Keyless access is technically allowed but shares a pool and is commonly throttled; a live probe returned `429 Too Many Requests` with a message to apply for a key. A free API key is sent as `x-api-key` and gives about 1 req/s. Do not make Semantic Scholar a core dependency unless the teammate can add the key.

**Fields**

`paperId`, `externalIds` (DOI, ArXiv, PubMed, etc.), `title`, `abstract`, `year`, `publicationDate`, `authors[].name`, `venue`, `citationCount`, `referenceCount`, `influentialCitationCount`, `fieldsOfStudy`, `publicationTypes`, `openAccessPdf.url`, `url`, `references`, `citations`.

**Caching**

Cache aggressively (24 hours; 7 days for paper lookups) because of the 1 req/s key limit. Use `/paper/batch` for up to 500 IDs and `/paper/search/bulk` for large keyword pulls.

**Terms**

The API License Agreement requires attribution to “Semantic Scholar” and citation of the Semantic Scholar Open Data Platform paper for scientific publications. S2 Data is offered under licences such as CC BY-NC and ODC-BY. **CC BY-NC is a real non-commercial restriction** — do not assume the data can be used in a commercial product. Check the specific record's licence.

---

### 4.3 arXiv (papers, secondary)

**Endpoint examples**

```text
GET https://export.arxiv.org/api/query?search_query=all:climate&start=0&max_results=50&sortBy=relevance
GET https://export.arxiv.org/api/query?search_query=cat:cs.LG+AND+submittedDate:[202501010000+TO+202612312359]&max_results=100
```

**Auth and limits**

No key. The API asks for **one request every 3 seconds** and one connection at a time. `max_results` can reach 30,000 total but is returned in slices of 2,000. Results only change after the daily processing run, so there is no reason to re-query the same search within a day.

**Fields**

Atom XML: `id` (abs URL), `title`, `summary`, `author/name`, `published`, `updated`, `category`, `primary_category`, `comment`, `journal_ref`, `doi`, `link` (alternate, PDF, DOI), plus `arxiv:doi` and `arxiv:journal_ref`.

**Caching**

Cache search results for 24 hours keyed by query + start + `max_results` + sort. Cache paper metadata by arXiv ID for 7 days. Store the arXiv ID, abstract, and landing page URL; do not store or redistribute the PDF.

**Terms**

Metadata is CC0. Full-text content is for personal/research use; the API terms prohibit redistributing PDFs/source from your own servers or circumventing limits. Link users to arXiv for the PDF.

---

### 4.4 Hugging Face Hub (datasets, core)

**Endpoint examples**

```text
GET https://huggingface.co/api/datasets?search=climate&limit=20&sort=downloads&direction=-1
GET https://huggingface.co/api/datasets/owner/name
```

**Auth and limits**

Public metadata search works without a token. Anonymous traffic is limited to 500 API requests per 5 minutes per IP; a free logged-in user gets 1,000 per 5 minutes. `429` responses carry `RateLimit` headers. Private and gated datasets need a token and acceptance of the dataset's conditions.

**Fields**

`id`, `author`, `description`, `downloads`, `likes`, `tags` (licence tags such as `license:apache-2.0` are in here), `createdAt`, `lastModified`, `private`, `gated`, `sha`, `trendingScore`. The dataset detail endpoint adds `cardData`, `siblings`, and file information.

**Caching**

Cache search results for 12–24 hours keyed by query + limit + sort; cache dataset info for 24 hours. Do not fetch file contents for discovery; store metadata and the repo URL.

**Terms**

The Hugging Face ToS grants each user a broad, perpetual, irrevocable, worldwide, royalty-free, non-exclusive licence to use public repository content through the service — but **the repository's stated licence still governs**. Read the `license:` tag and the dataset card. Gated datasets may have additional conditions. Do not treat “public on HF” as “free for any commercial use”.

---

### 4.5 Zenodo (datasets, core)

**Endpoint examples**

```text
GET https://zenodo.org/api/records?q=climate&size=25&sort=mostrecent&type=dataset
GET https://zenodo.org/api/records/1234567
```

**Auth and limits**

Public search works without a token. Guest: 60 requests/min and 2,000/hr; authenticated: 100/min and 5,000/hr; search endpoints: 30/min. Anonymous `size` max 25; authenticated max 100. Use OAI-PMH or the public data file for bulk harvesting, not the search endpoint.

**Fields**

`hits.hits[]`: `id`, `conceptrecid`, `doi`, `conceptdoi`, `doi_url`, `metadata.title`, `metadata.description`, `metadata.publication_date`, `metadata.creators`, `metadata.keywords`, `metadata.license`, `metadata.resource_type`, `files`, `links`.

**Caching**

Cache search results for 24 hours keyed by `q` + `size` + page + sort; cache record details by ID for 7 days. Store the DOI and landing page, not the files.

**Terms**

Metadata may be reused under CC0 unless the record says otherwise. File licences vary and must be checked per record. Zenodo's Terms of Use restrict access to non-military purposes. The uploader is responsible for the content; do not assume every record is suitable for commercial reuse.

---

### 4.6 OpenML (datasets, stretch)

**Endpoint examples**

```text
GET https://www.openml.org/api/v1/json/data/list/data_name/climate/limit/20
GET https://www.openml.org/api/v1/json/data/40945
```

**Auth and limits**

Public read endpoints work without a key; an API key is only needed for uploads/writes. There is no prominent global rate limit, but cache and keep request rates modest.

**Fields**

`did`, `name`, `version`, `status`, `format`, `file_id`, `quality` (`NumberOfFeatures`, `NumberOfInstances`, `NumberOfMissingValues`, etc.), plus description, creator, collection date, and URL in the standard list/detail responses.

**Caching**

Dataset lists: 7 days. Dataset detail: 7 days. These are curated benchmark datasets and change infrequently.

**Terms**

OpenML's terms grant each user a non-exclusive licence to access and use content for their own research purposes, subject to the licence attached to each dataset. Check the dataset's licence field/card. The site's honour code asks for attribution and polite, non-destructive use.

---

### 4.7 Dataverse, Harvard (datasets, stretch)

**Endpoint examples**

```text
GET https://dataverse.harvard.edu/api/search?q=climate&type=dataset&per_page=100&start=0
GET https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/...
```

**Auth and limits**

Public search works without a key. A token is needed for private/restricted data. There is no published global rate limit for Harvard's installation; each Dataverse installation is separate, so a generic adapter should take a base URL. `per_page` can be up to 1,000.

**Fields**

`data.total_count`, `data.items[]`: `name`, `type`, `url`, `global_id`/DOI, `description`, `published_at`, `publisher`, `citation`, `citationHtml`, `authors`, `subjects`, `fileCount`, `versionId`, `versionState`, `createdAt`, `updatedAt`, `contacts`.

**Caching**

Cache search results for 24 hours and dataset details for 7 days. Use `per_page=100` for discovery, not 1,000, to keep responses manageable.

**Terms**

Metadata and file licences are per dataset/installation; check the record. Harvard Dataverse generally publishes metadata under CC0, but do not assume the files are open. Restricted datasets require access requests.

---

### 4.8 DataCite (datasets, stretch/core for DOI data)

**Endpoint examples**

```text
GET https://api.datacite.org/dois?query=climate&page[size]=100&page[number]=1
GET https://api.datacite.org/dois/10.5281/zenodo.1234567
```

**Auth and limits**

The Public API needs no authentication and returns Findable DOIs only (not drafts or Registered DOIs). Rate limits: 500 req/5 min unidentified; 1,000 req/5 min with an identifying `User-Agent` containing a `mailto:`; 3,000 req/5 min authenticated. `page[size]` up to 1,000; page-number paging stops at 10,000 records, while cursor paging has no limit. Set a User-Agent such as `whack26-research/0.1 (mailto:you@example.org)`.

**Fields**

JSON:API `data[]`: `id`, `attributes.doi`, `creators`, `titles`, `publisher`, `publicationYear`, `subjects`, `contributors`, `dates`, `types`, `descriptions`, `url`, `rightsList`, `relatedIdentifiers`, `geoLocations`, plus `relationships`, `meta`, `links`.

**Caching**

Cache search responses for 24 hours keyed by query + page; cache DOI lookups for 7 days. Use the annual public data file or OAI-PMH for bulk harvesting.

**Terms**

The DataCite Data File (DOIs and deposited metadata) is CC0; linked resources are not covered by that waiver. Attribute DataCite. Check each record's `rightsList` for the resource licence.

---

### 4.9 data.gov Catalog API (datasets, defer)

**Endpoint examples**

```text
GET https://api.gsa.gov/technology/datagov/v4/search?q=climate&per_page=100
    Header: X-Api-Key: YOUR_KEY
GET https://api.gsa.gov/technology/datagov/v4/search?q=climate&after=<cursor>
```

**Important change**

The old CKAN endpoint (`https://catalog.data.gov/api/3/action/package_search`) returned `404` in our probe. The new Catalog API is `https://api.gsa.gov/technology/datagov/v4/`. It requires `X-Api-Key`; without it, the API returns `403 API_KEY_MISSING`. `DEMO_KEY` works for exploration (30 req/IP/hr, 50 req/IP/day) but is explicitly not for production or automated querying. A free personal key from <https://api.data.gov/signup> gives 1,000 req/hr.

**Fields**

`results[]`: `title`, `description`, `identifier`, `slug`, `publisher`, `keyword`, `theme`, `has_spatial`, `spatial_centroid`, `spatial_shape`, `popularity`, `last_harvested_date`, `distribution_titles`, `organization`, and a full `dcat` object (`license`, `accessLevel`, `modified`, `temporal`, `distribution`, etc.). Pagination is cursor-based via `after`.

**Caching**

Cache search responses for 24 hours keyed by query + filters + `after`; cache organization lists for 7 days. Stop when `after` is absent.

**Terms**

Each record carries DCAT metadata. Check `dcat.license` and `accessLevel`; do not download or redistribute restricted/non-public datasets. US federal works are often public domain, but the record's licence and rights statement control.

---

### 4.10 Kaggle (datasets, optional)

**Endpoint examples**

```text
GET https://www.kaggle.com/api/v1/datasets/list?search=climate&page=1
GET https://www.kaggle.com/api/v1/datasets/list?search=climate&page=2
```

**Auth and limits**

The official client and docs require a Kaggle account/token (`KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, or legacy `kaggle.json`). A keyless probe of the dataset-list endpoint returned `200` with useful metadata, but that is **undocumented** and not a supported contract. Kaggle's Acceptable Use Policy prohibits “excessive crawling of the content on the Services”. Do not build the slice on the keyless endpoint.

**Fields**

`id`, `ref`, `title`, `subtitle`, `creatorName`, `creatorUrl`, `totalBytes`, `url`, `lastUpdated`, `downloadCount`, `licenseName`, `ownerName`, `ownerRef`, `usabilityRating`, `isPrivate`, `isFeatured`, `thumbnailImageUrl`.

**Caching**

Cache metadata for 24 hours. Do not download dataset files for discovery; downloads require auth and may require accepting dataset-specific rules.

**Terms**

Kaggle Terms of Use and Acceptable Use Policy apply. Dataset licences are self-reported; verify the licence and description before use. Competition datasets have separate rules.

---

### 4.11 Google Dataset Search (datasets, drop)

There is **no public query API**. The Dataset Search help page describes the consumer UI and how publishers add schema.org markup; it does not document an API. Google's Terms of Service prohibit automated access that violates machine-readable instructions (e.g. robots.txt) and prohibit scraping content that does not belong to you. Do not scrape Dataset Search.

The only bulk artefact is a “dataset of metadata” published on Kaggle for citable datasets. It is not a live search API and is not part of the recommended slice. If dataset discovery needs Google's index, the lawful route is to use another repository's API or to ask Google for access, not to scrape.

---

### 4.12 Web search

**Brave Search API**

- Endpoint: `https://api.search.brave.com/res/v1/web/search?q=...`
- Header: `X-Subscription-Token: <key>`
- Key: <https://api-dashboard.search.brave.com/register>
- Pricing: Search plan `$5` per 1,000 requests; `$5` free credits per month on that plan (the site also advertises up to `$20`/month in free calls across products); 50 queries/s.
- This is a legitimate general web-search API with its own index, but it is **not a zero-cost, zero-key option** for the slice. Beyond the free credits, continued use needs a paid plan.
- Review Brave's API terms before storing snippets long-term or using results to train models.

**Google Custom Search JSON API**

- Closed to new customers. Existing customers have until **2027-01-01**.
- 100 queries/day free; `$5`/1,000 after; needs an API key and a Programmable Search Engine ID.
- Not a viable choice for a new build.

**Bing Web Search API**

- Retired; the Microsoft docs are archived. Not viable.

**DuckDuckGo Instant Answer API**

- `https://api.duckduckgo.com/?q=climate&format=json&no_html=1`
- Keyless and returns `200`, but the payload is instant answers/related topics, not a list of 20–50 web results. It is not a general web-search API. Do not scrape DuckDuckGo's HTML results.

**Wikipedia / Wikimedia REST**

- `https://en.wikipedia.org/api/rest_v1/page/summary/Climate_change`
- Keyless, stable, and useful for encyclopedic context. Set a descriptive `User-Agent`; cache; attribute CC BY-SA.
- This is the recommended keyless fallback, but it is not a substitute for web search.

**SearXNG**

- Self-hostable metasearch with no API key, but it requires running a service and still depends on upstream engines' terms. Public instances rate-limit/block automated use. Too much operational work for a 6–8 hour local-first slice.

**Stack Exchange API / Hacker News Algolia**

- Keyless or low-quota, but technical-content-only. Useful as optional enrichment, not as general web search.

---

## 5. Caching, normalization, and deduplication

### 5.1 Cache design

Use a local SQLite table (no Docker/Postgres needed):

```sql
CREATE TABLE api_cache (
  cache_key TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  request_url TEXT NOT NULL,
  status INTEGER NOT NULL,
  etag TEXT,
  last_modified TEXT,
  body_json TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
```

Cache key: `source | normalized_query | filters | page/cursor | selected_fields | date_bucket`. Normalize queries by lowercasing, trimming, and collapsing whitespace; keep the original query for display.

Suggested TTLs:

| Source | TTL | Notes |
|---|---|---|
| OpenAlex search/filter | 24 h | Cache DOI lookups 7 days; log `meta.cost` |
| Semantic Scholar | 24 h | 7 days for paper lookups; 1 req/s key |
| arXiv | 24 h | Results only change after the daily run |
| Hugging Face | 12–24 h | Search + dataset info |
| Zenodo | 24 h | 7 days for record details |
| OpenML | 7 d | Curated and slow-changing |
| Dataverse | 24 h | 7 days for dataset details |
| DataCite | 24 h | 7 days for DOI lookups |
| data.gov | 24 h | Cursor pagination |
| Kaggle | 24 h | Metadata only |
| Wikipedia | 7 d | Page summaries |
| DuckDuckGo IA | 24 h | Instant answers only |
| Brave | 1 h | Paid; cache to control cost |

Use `If-None-Match` / `If-Modified-Since` where the API supports them. On `429`, honour `Retry-After`; on `5xx`, exponential backoff with jitter, max 3 attempts; do not retry ordinary `4xx` except `429`.

### 5.2 Normalized record

Map every source to one shape before storage:

```json
{
  "source": "openalex",
  "source_id": "W2741809807",
  "doi": "10.1038/nature12373",
  "arxiv_id": null,
  "title": "...",
  "abstract": "...",
  "authors": ["..."],
  "published_at": "2024-01-01",
  "url": "https://...",
  "pdf_url": null,
  "license": "CC0",
  "resource_type": "paper",
  "citation_count": 123,
  "tags": ["climate change", "..."],
  "retrieved_at": "2026-11-07T12:00:00Z",
  "raw_json": {}
}
```

Keep the raw response for debugging and provenance, but do not render it directly to users. Store `api_endpoint`, query, retrieval time, and a response hash alongside each record.

### 5.3 Deduplication

Deduplicate in this order:

1. **DOI**, lowercased and stripped of `https://doi.org/`, `doi:`, and whitespace.
2. **arXiv ID**, stripped of the `arXiv:` prefix and version suffix.
3. **Source + source ID** (e.g. `openalex:W...`, `zenodo:1234567`).
4. **Normalized title + first author + year** as a last resort.

When duplicates merge, prefer the record with the richer abstract/fields, keep all source IDs, and keep the earliest `retrieved_at`. Do not merge records with the same title but different DOIs without a human-readable note.

---

## 6. Legal and terms checklist

- **Metadata and links only.** Store titles, abstracts/descriptions, authors, dates, licences, and URLs. Do not store or redistribute full-text PDFs or dataset files.
- **arXiv:** full text is for personal/research use; do not proxy or redistribute PDFs; link to arXiv.
- **Semantic Scholar:** S2 Data includes CC BY-NC material; do not assume commercial reuse. Attribute “Semantic Scholar”.
- **OpenAlex:** CC0; attribution expected; cite the paper.
- **Hugging Face:** public repos grant a broad licence through the service, but each repo's licence tag/card controls. Respect gated/private data.
- **Zenodo:** metadata CC0 unless stated; files vary; non-military use only.
- **OpenML:** research use per the site terms and each dataset's licence.
- **Dataverse:** per-dataset/installation licences; restricted data needs access requests.
- **DataCite:** metadata CC0; linked resources are not covered; attribute DataCite.
- **data.gov:** check `dcat.license` and `accessLevel`; do not redistribute restricted/non-public records.
- **Kaggle:** official API/token; no excessive crawling; verify self-reported dataset licences; competition rules are separate.
- **Google Dataset Search:** no public API; do not scrape.
- **Google/DDG web results:** do not scrape HTML; use an authorised API.
- **Wikipedia:** CC BY-SA attribution and share-alike for reused text.
- **robots.txt:** for any link-following outside APIs, fetch and obey robots.txt; one hop by default; no paywall circumvention.
- **Keys:** environment variables only; add `.env` to `.gitignore`; never commit tokens.
- **User-Agent:** identify the tool and a contact (`whack26-research/0.1 (mailto:...)`) for DataCite and any polite scraping.
- **Terms can change.** Re-check the linked pages before launch, especially OpenAlex pricing, data.gov keys, and Brave/Google terms.

---

## 7. Recommended minimal slice

### 7.1 Active adapters

| Layer | Adapter | Why it earns its place | Auth | Rough effort |
|---|---|---|---|---|
| Papers | **OpenAlex** | Broad coverage, CC0, DOI/abstract/citation fields, good search | Free key | 1.5 h |
| Papers | **arXiv** | Preprints and full metadata, keyless, simple Atom | None | 0.75 h |
| Datasets | **Hugging Face Hub** | ML datasets, licence tags, keyless search | None | 0.5 h |
| Datasets | **Zenodo** | DOI-backed research data, keyless search | None | 0.75 h |
| Web | **Wikipedia REST** | Keyless encyclopedic fallback, clearly labelled | None | 0.5 h |
| Web | **Brave adapter** (inactive without key) | Real web results when a key is available | Key/paid | 0.75 h |
| Cross-cutting | Cache + normalizer + dedup | Required for reproducibility and quota control | None | 1.5 h |
| Cross-cutting | Error/empty-state handling | “No web key”, “rate limited”, “no results” | None | 0.75 h |

Core build: roughly **6.5 hours** including a small test suite. That leaves 1–1.5 hours for integration and demo polish, or for one stretch adapter.

### 7.2 Stretch adapters (only if time remains)

- **DataCite** — best stretch choice if the demo needs cross-repository DOI dataset discovery. Keyless, CC0 metadata.
- **OpenML** — best if the demo is ML/benchmark-oriented. Keyless, simple JSON.
- **Dataverse** — best if social-science data is a target. Keyless search, per-installation base URL.

Do **not** add data.gov or Kaggle to the critical path. Both need accounts/keys, and Kaggle's keyless listing is not a supported contract.

### 7.3 Build order

1. **OpenAlex** search + DOI lookup + abstract reconstruction (1.5 h).
2. **arXiv** search + Atom parsing (0.75 h).
3. **Hugging Face** dataset search (0.5 h).
4. **Zenodo** record search (0.75 h).
5. **SQLite cache + normalized record + dedup** (1.5 h).
6. **Wikipedia** fallback + “web search requires a key” state (0.5 h).
7. **Brave adapter** behind an env flag (0.75 h, can be a stub).
8. Optional: **DataCite** or **OpenML** (0.5–0.75 h).
9. Tests: fixture-based adapter tests, dedup tests, cache-hit/miss tests (included above).

### 7.4 Environment variables

```text
OPENALEX_API_KEY=          # free key; keyless works for smoke tests
SEMANTIC_SCHOLAR_API_KEY=  # optional
BRAVE_SEARCH_API_KEY=      # optional; real web search
KAGGLE_API_TOKEN=          # optional; not in core
DATA_GOV_API_KEY=          # optional; not in core
```

Do not require any of these for the core demo except, ideally, `OPENALEX_API_KEY`. If the key is absent, the OpenAlex adapter should fall back to keyless mode, cap itself to first-page/basic paging, and surface a “limited mode” badge.

---

## 8. Feasibility flags

1. **Google Dataset Search has no public query API.** Confirmed; drop it. The only bulk artefact is a Kaggle metadata dataset, not a live API.
2. **Zero-key general web search is not available lawfully.** Google CSE is closed to new customers and ends in 2027; Bing is retired; DuckDuckGo is instant-answer only; Brave requires a key and payment after credits. This is the main requirement that cannot be met in the stated constraints. Use Wikipedia as a labelled fallback and make real web search key-gated.
3. **OpenAlex keyless is demo-only.** `$0.10/day` and no cursor paging past page 1. Get the free key; it is the highest-value five minutes in the build.
4. **Semantic Scholar keyless is unreliable.** A live probe returned `429`. Treat it as optional and key-gated; do not make it the only paper source.
5. **data.gov migrated.** The old CKAN endpoint returned `404`; the new v4 API needs `X-Api-Key`. `DEMO_KEY` is 50 requests/day. Defer unless a key is added.
6. **Kaggle is not keyless in practice.** The official API needs a token; the keyless listing endpoint is undocumented and the acceptable-use policy prohibits excessive crawling.
7. **Rate limits are not the issue; quotas, cost, and terms are.** The project can make few requests and still be non-compliant if it stores PDFs, scrapes Google, or ignores a non-commercial licence.

---

## 9. Sources

All accessed 2026-11-07.

**Papers**

- OpenAlex access and pricing: <https://help.openalex.org/access/pricing> and <https://help.openalex.org/api/authentication/>
- OpenAlex example costs: <https://help.openalex.org/access/example-costs/>
- OpenAlex work attributes: <https://help.openalex.org/data/works/attributes>
- OpenAlex citation guidance: <https://help.openalex.org/how-to/citing-openalex/>
- Semantic Scholar API: <https://www.semanticscholar.org/product/api>
- Semantic Scholar API licence: <https://www.semanticscholar.org/product/api/license>
- arXiv API manual: <https://info.arxiv.org/help/api/user-manual.html>
- arXiv API terms: <https://info.arxiv.org/help/api/tou.html>

**Datasets**

- Hugging Face Hub API: <https://huggingface.co/docs/hub/api>
- Hugging Face rate limits: <https://huggingface.co/docs/hub/rate-limits>
- Hugging Face terms: <https://huggingface.co/terms-of-service>
- Hugging Face dataset cards: <https://huggingface.co/docs/hub/datasets-cards>
- Zenodo developer docs: <https://developers.zenodo.org/>
- Zenodo terms: <https://about.zenodo.org/terms/>
- OpenML docs: <https://docs.openml.org/>
- OpenML terms: <https://docs.openml.org/intro/terms/>
- Dataverse search API: <https://guides.dataverse.org/en/latest/api/search.html>
- DataCite REST API: <https://support.datacite.org/docs/api>
- DataCite rate limits: <https://support.datacite.org/docs/rate-limit>
- DataCite data file use policy: <https://support.datacite.org/docs/datacite-data-file-use-policy>
- data.gov Catalog API: <https://resources.data.gov/catalog-api>
- Kaggle API docs: <https://github.com/Kaggle/kaggle-api>
- Kaggle acceptable use policy: <https://www.kaggle.com/acceptable-use-policy>
- Kaggle terms: <https://www.kaggle.com/terms>
- Google Dataset Search help: <https://datasetsearch.research.google.com/help>

**Web**

- Brave Search API: <https://brave.com/search/api/>
- Google Custom Search JSON API: <https://developers.google.com/custom-search/v1/overview>
- Bing Web Search API (archived): <https://learn.microsoft.com/en-us/previous-versions/bing/search-apis/bing-web-search/overview>
- DuckDuckGo Instant Answer API: <https://duckduckgo.com/api>
- Wikimedia REST API: <https://www.mediawiki.org/wiki/API:REST_API>
- Google Terms of Service: <https://policies.google.com/terms>

---

## Appendix A: live probe log

| Probe | Result | Note |
|---|---|---|
| Semantic Scholar `/graph/v1/paper/search?query=climate&limit=1` | `429` | “Too Many Requests … apply for a key” |
| OpenAlex `/works?search=climate&per_page=1` | `200` | Keyless works; budget headers present |
| arXiv `/api/query?search_query=all:climate&max_results=1` | `200` | Atom XML |
| Hugging Face `/api/datasets?search=climate&limit=2` | `200` | JSON array |
| Zenodo `/api/records?q=climate&size=2` | `200` | `hits.hits` |
| Dataverse Harvard `/api/search?q=climate&type=dataset&per_page=2` | `200` | `data.items` |
| DataCite `/dois?query=climate&page[size]=1` | `200` | JSON:API |
| data.gov v4 `/search?q=climate` without key | `403` | `API_KEY_MISSING` |
| data.gov v4 `/search?q=climate` with `DEMO_KEY` | `200` | Demo quota only |
| OpenML `/api/v1/json/data/list/data_name/climate/limit/2` | `200` | JSON |
| Kaggle `/api/v1/datasets/list?search=climate&page=1` | `200` | Keyless, undocumented; do not rely on it |
| DuckDuckGo `api.duckduckgo.com/?q=climate&format=json` | `200` | Instant answers only |
| Wikipedia `/api/rest_v1/page/summary/Climate_change` | `200` | Summary JSON |
| Google Dataset Search query endpoint | N/A | No public query API found |

## Appendix B: minimal adapter interface

A small protocol keeps the four core sources interchangeable:

```python
from typing import Protocol

class ResearchSource(Protocol):
    name: str
    def search(self, query: str, *, limit: int = 25) -> list[dict]: ...
    def get(self, source_id: str) -> dict | None: ...
```

Each adapter should:

1. Build a cache key from source + query + pagination + selected fields.
2. Return a cached response if it is fresh.
3. Apply a per-source rate limiter and retry policy.
4. Map the raw response to the normalized record.
5. Store the raw JSON and provenance.
6. Never raise on a single source failure; return an empty list plus a structured error for the UI.

This is enough structure to add DataCite, OpenML, or Dataverse later without changing the research layer.
