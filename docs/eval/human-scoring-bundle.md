# Human scoring bundle — field-classification gate

**Purpose:** give a human scorer everything needed to fill `eval/scores.json`.
**Date:** 2026-11-08
**Current commit:** `d74836a` (plus local uncommitted docs)
**Score file:** `eval/scores.json` (generated template; fill the nulls)

> **Important:** score the **live cold run**, not a mock run. Mock mode exercises plumbing only. Live cold-run paths currently exist for **California Housing** and **Palmer Penguins**; the other eight need a live cold pass, which is blocked by Gemini quota. Do not score mock paths as if they were live results.

---

## 1. Rubric

| Score | Meaning |
| --- | --- |
| 5 | Named as a practitioner would; parent-child relation obvious; description accurate |
| 4 | Minor naming quibble; relation clear |
| 3 | Acceptable, but a different path is also reasonable |
| 2 | A node is misleading, too broad, or too narrow |
| 1 | Nonsensical or dataset-specific label |

A dataset passes when **root and full path both score ≥ 3**.

## 2. Gates

| Criterion | Gate |
| --- | --- |
| Top-level plausibility | ≥ 8/10 roots ≥ 3; mean ≥ 3.5 |
| Full-path plausibility | ≥ 8/10 paths ≥ 3; mean ≥ 3.5 |
| Naming (human) | mean ≥ 4.0 |
| Sibling distinctness | ≥ 8/10 descriptions distinguish the field from siblings |

The harness never invents root, path, or sibling-distinctness scores (`eval/field_classification.py:1054`; `docs/eval/README.md:70`). `naming` is optional: if it is null and the automated naming gate passes, the harness assumes the automated result is correct and uses score 4. Without root, path, and sibling_distinctness for a dataset, its gates stay `pending`.

## 3. Reference data for all ten datasets

| Slug | Dataset | Domain | Target | Acceptable roots | Good path | Forbidden labels | Effective file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `penguins` | Palmer Penguins | biology / ecology | `species` | biology; ecology; zoology; life sciences; biological sciences; animal science; ornithology | Life sciences > Ecology or zoology > Morphology / species | palmer penguins; penguins dataset; penguin data | `data/penguins.csv` |
| `wine` | Wine Quality | food / chemistry | `quality` | food science; chemistry; agriculture; food and beverage; oenology; food | Food science > Oenology > Wine quality | wine quality dataset; winequality; uci | `data/eval/wine/winequality-red.csv` |
| `california` | California Housing | housing / economics | `median_house_value` | housing; economics; real estate; urban planning; geography; social science | Housing or economics > Real estate > Housing prices | california housing; housing.csv; california_housing | `data/california_housing.csv` |
| `titanic` | Titanic | demographics / history | `Survived` | history; demography; sociology; maritime history; disaster studies; social science | History > Maritime disasters or demography > Survival | titanic; titanic dataset | `data/eval/titanic/titanic.csv` |
| `air_quality` | Air Quality (UCI) | environment | `CO(GT)` | environment; environmental science; air quality; atmospheric science; public health; chemistry | Environment > Air quality > Pollution monitoring | airqualityuci; uci air quality; air quality dataset | `data/eval/air_quality/AirQualityUCI.xlsx` |
| `covid` | COVID Country Series | public health | `new_cases` | public health; epidemiology; health; demography; infectious disease | Public health > Epidemiology > Infectious disease surveillance | owid; owid-covid; covid-19 dataset; coronavirus dataset | `data/eval/covid/owid-covid-data.csv` |
| `spotify` | Spotify Tracks | music | `track_popularity` | music; media; culture; entertainment; arts; audio | Music > Musicology or audio analysis > Track popularity | spotify; tidytuesday; spotify_songs | `data/eval/spotify/spotify_songs.csv` |
| `student` | Student Performance | education | `G3` | education; social science; sociology; demography; learning | Education > Student outcomes > Academic performance | student performance dataset; student-mat; student-por; uci | `data/eval/student/student-mat.csv` |
| `retail` | Online Retail | retail / commerce | none | retail; commerce; business; economics; marketing | Retail / commerce > E-commerce > Transactions | online retail; online_retail; uci | `data/eval/retail/Online Retail.xlsx` |
| `soccer` | Soccer Match Results | sports | `FTR` | sports; sports science; recreation; statistics | Sports > Association football > Match results | football-data; e0.csv | `data/eval/soccer/E0.csv` |

Dataset descriptions:

- **Penguins:** morphological measurements (bill, flipper, body mass) and sex for three penguin species observed on three islands in the Palmer Archipelago.
- **Wine:** physicochemical measurements of red wine samples (acidity, sugar, sulphates, alcohol) paired with sensory quality ratings.
- **California:** block-group housing statistics: median house value, median income, housing age, rooms, population, and ocean proximity.
- **Titanic:** no description supplied.
- **Air quality:** hourly air-quality sensor readings from an Italian city, including carbon monoxide, nitrogen oxides, benzene, temperature, and humidity.
- **COVID:** country-level daily COVID-19 cases, deaths, vaccinations, testing, and policy indicators.
- **Spotify:** no description supplied.
- **Student:** secondary-school student grades, demographics, family background, study habits, and school support.
- **Retail:** no description supplied.
- **Soccer:** no description supplied.

## 4. Live evidence available now

### California Housing — `runs/live-california-5/artifact.json`

| Level | Field | Description |
| ---: | --- | --- |
| 1 | Real Estate and Housing | Datasets containing information on housing markets, property characteristics, and real estate valuations. |
| 2 | Housing Valuation and Demographics | Datasets focusing on housing characteristics, regional demographics, and property values based on census or market data. |

- Confidence: `0.876`
- Refined after research: `true`
- Created fields: none

### Palmer Penguins — `runs/live-penguins/artifact.json`

| Level | Field | Description |
| ---: | --- | --- |
| 1 | Biology and Life Sciences | Datasets concerning living organisms, ecology, and biological measurements. |
| 2 | Zoology and Animal Behavior | Datasets detailing animal characteristics, physiology, morphology, and behavioral observations across various species and habitats. |

- Confidence: `0.841`
- Refined after research: `true`
- Created fields: `Biology and Life Sciences`, `Zoology and Animal Behavior`

### Sibling context

The live store (`cache/fields.db`) currently contains only these four fields:

| Level | Field | Parent |
| ---: | --- | --- |
| 1 | Biology and Life Sciences | — |
| 1 | Real Estate and Housing | — |
| 2 | Housing Valuation and Demographics | Real Estate and Housing |
| 2 | Zoology and Animal Behavior | Biology and Life Sciences |

Each root has one child, so sibling distinctness for these two live paths is easy: the two roots are distinct, and the two children are distinct within their own parents.

## 5. Missing live paths

The other eight datasets have **no live cold-run paths yet**. To produce them once quota is available:

```powershell
.\.venv\Scripts\python.exe eval\field_classification.py `
  --datasets wine,titanic,air_quality,covid,spotify,student,retail,soccer `
  --output docs\eval\field-classification-results.md `
  --json-output docs\eval\field-classification-results.json
```

Then score the cold-run paths from the generated report. The harness also runs warm and shuffled passes; only the cold pass is scored by `evaluate_human`.

If you want a **dry run** now, add `--mock` and write to a scratch path:

```powershell
.\.venv\Scripts\python.exe eval\field_classification.py `
  --mock --allow-pending `
  --datasets wine,titanic,air_quality,covid,spotify,student,retail,soccer `
  --output C:\Users\FireF\AppData\Local\Temp\opencode\mock-eval.md `
  --json-output C:\Users\FireF\AppData\Local\Temp\opencode\mock-eval.json `
  --work-dir C:\Users\FireF\AppData\Local\Temp\opencode\mock-eval-work
```

Label any mock scores as mock; do not use them for the acceptance gate.

## 6. Scoring worksheet

Fill `eval/scores.json`. The generated template already has the reference roots and good path for each slug.

`naming` is optional. Leave it blank/null to trust the automated naming gate (assumed score 4); fill 1–5 only if you want to override it.

| Slug | root (1–5) | path (1–5) | naming (1–5, blank = auto) | sibling distinct (yes/no) | notes |
| --- | ---: | ---: | ---: | --- | --- |
| `penguins` |  |  |  |  |  |
| `wine` |  |  |  |  |  |
| `california` |  |  |  |  |  |
| `titanic` |  |  |  |  |  |
| `air_quality` |  |  |  |  |  |
| `covid` |  |  |  |  |  |
| `spotify` |  |  |  |  |  |
| `student` |  |  |  |  |  |
| `retail` |  |  |  |  |  |
| `soccer` |  |  |  |  |  |

## 7. Commands

```powershell
# Regenerate the template if needed
.\.venv\Scripts\python.exe eval\field_classification.py --write-scores-template eval\scores.json

# Run the gate with your scores
.\.venv\Scripts\python.exe eval\field_classification.py --scores eval\scores.json

# Mock plumbing check with pending human gates
.\.venv\Scripts\python.exe eval\field_classification.py --mock --repeat --allow-pending
```

## 8. What each score means in practice

- **`root`** — would a practitioner accept this top-level field for the dataset, or is it too broad, too narrow, or in the wrong domain?
- **`path`** — does each child genuinely follow from its parent, and is the leaf specific enough to be useful?
- **`naming`** — is the wording practitioner-quality, or is it generic, awkward, or dataset-specific?
- **`sibling_distinctness`** — do the descriptions of fields at the same level actually distinguish them, or could they be swapped without loss?
