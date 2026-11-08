# Research and factor smoke test

Status: live smoke on 2/2 demo datasets. Recorded 2026-11-07.

The full ten-dataset research gate is still pending LLM quota. This page records
the two-dataset smoke that validates the adapters, cache, query generation,
ranking, factor extraction, and warm-run behavior end to end.

Provider: `gemini-3.5-flash-lite` for language calls and
`gemini-embedding-2` for embeddings. The default `gemini-3.6-flash` model had
exhausted its free-tier daily quota.

## Commands

```powershell
$env:GEMINI_MODEL='gemini-3.5-flash-lite'
.\.venv\Scripts\python.exe -m whack26 prefetch `
  --input data\california_housing.csv `
  --target median_house_value `
  --description "Block-group housing statistics: median house value, median income, housing age, rooms, population, and ocean proximity." `
  --db C:\Users\FireF\AppData\Local\Temp\opencode\california-live2.db `
  --out C:\Users\FireF\AppData\Local\Temp\opencode\california-prefetch4.json

.\.venv\Scripts\python.exe -m whack26 prefetch `
  --input data\penguins.csv `
  --target species `
  --description "Morphological measurements (bill, flipper, body mass) and sex for three penguin species observed on three islands in the Palmer Archipelago." `
  --db C:\Users\FireF\AppData\Local\Temp\opencode\penguins-live.db `
  --out C:\Users\FireF\AppData\Local\Temp\opencode\penguins-prefetch2.json
```

The second identical invocation of each command was the warm run.

## Research results

| Dataset | Field path | Records | Sources | Warm run |
|---|---|---|---|---|
| California Housing | `Housing and Real Estate > Housing Market and Property Statistics` | 20 | OpenAlex 11, arXiv 1, Hugging Face 5, Wikipedia 3 | 9 cache hits, 0 misses, 0 writes, hit rate 1.000 |
| Palmer Penguins | `Biology and Life Sciences > Ecology and Wildlife` | 20 | OpenAlex 8, arXiv 2, Hugging Face 3, Zenodo 4, Wikipedia 3 | 10 cache hits, 0 misses, 0 writes, hit rate 1.000 |

Queries used by the warm California run:

- `housing market affordability determinants` (OpenAlex)
- `spatial econometrics real estate prices` (OpenAlex)
- `urban housing price prediction` (arXiv)
- `california housing` (Hugging Face)
- `California housing market` (Wikipedia)

Queries used by the warm Palmer Penguins run:

- two OpenAlex paper queries, one arXiv paper query, one Hugging Face dataset
  query, one Zenodo dataset query, and one Wikipedia query. The query cache
  persisted the exact terms from the cold run.

Every query set met the 4-6 query, two-paper, one-dataset, one-Wikipedia rules.
No source returned an error in the final runs.

## Factor results

A separate smoke loaded the cached records and ran factor extraction. All
evidence IDs resolve to a record in the same snapshot.

### California Housing

| Factor | Direction / claim | Column | Evidence |
|---|---|---|---|
| Median Income | positive / association | `median_income` | `openalex:W7105857935`, `openalex:W4410418134` |
| Geographic Coordinates | unclear / descriptive | `longitude` | `openalex:W7105857935` |
| Housing Median Age | unclear / association | `housing_median_age` | `openalex:W7105857935` |
| Population | unclear / descriptive | `population` | `openalex:W7105857935` |

Result: 4 factors, 4 mapped, 0 unmapped. This triggered the intentional
`factor_extraction_small: only 4 evidence-backed factor(s)` warning; the
acceptance threshold is at least 3 factors and at least one mapped factor.

### Palmer Penguins

| Factor | Direction / claim | Column | Evidence |
|---|---|---|---|
| Bill Length | unclear / descriptive | `bill_length_mm` | `arxiv:2604.01619v3`, `openalex:W2974107072` |
| Bill Depth | unclear / descriptive | `bill_depth_mm` | `arxiv:2604.01619v3`, `openalex:W2974107072` |
| Flipper Length | unclear / descriptive | `flipper_length_mm` | `openalex:W2974107072` |
| Body Mass | unclear / descriptive | `body_mass_g` | `arxiv:2604.01619v3` |
| Island Habitat | unclear / association | `island` | `huggingface:apreshill/palmerpenguins` |
| Sex | unclear / descriptive | `sex` | `huggingface:apreshill/palmerpenguins` |

Result: 6 factors, 6 mapped, no warnings.

## Fixes found by live validation

- **Wikipedia 403:** Wikimedia rejects a user agent without contact information.
  The shared agent is now
  `whack26-research/0.1 (https://github.com/hoisin-sauce/whack-26)`.
- **Zenodo noise:** `sort=mostrecent` returned unrelated recent deposits for
  multi-word queries. The adapter now uses `sort=bestmatch`; the spec records
  why.
- **Hugging Face exact-phrase search:** a long query or even the qualifier
  `dataset` can return nothing (`palmer penguins dataset` misses while
  `palmer penguins` hits). The pipeline retries an empty Hugging Face search
  with the first two words of the query.
- **Nondeterministic warm runs:** generated queries are cached per dataset and
  summary hash. Without this, a second run produced different queries and every
  source response missed even though the data was unchanged.

## Limitations

- Factor evidence is selected by the LLM. A paper that only fits a prediction
  model may not establish that a factor is associated with the target; external
  validation must check each citation, not just that the ID exists.
- California produced 4 factors, below the 5-10 target, even after the prompt
  required direct evidence. The warning is visible in the run.
- The ten-dataset acceptance run remains blocked by provider quota.
- `prefetch` exposes research and cache warming. Factor extraction is currently
  called from Python; it becomes part of the artifact pipeline when `run` lands.
