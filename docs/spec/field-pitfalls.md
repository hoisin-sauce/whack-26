# Field-scoped pitfalls

Status: implemented optional advisory stage. It is enabled with
`--field-pitfalls` (or `--full`) and `WHACK26_FIELD_PITFALLS_ENABLED=1`. The
stage is unit- and integration-tested; its live literature retrieval is not
validated at scale.

## Purpose and boundary

The stage searches field-specific limitation and bias literature for each node
of a dataset's field path and turns the results into cited, bounded evidence
records. It gives the method critic, the hypothesis context pack, and the
artifact a field-scoped vocabulary for asking *what usually goes wrong in this
kind of data?*

A field pitfall is a candidate caveat, not a finding. It does not show that the
dataset suffers from the limitation, does not judge model performance, and does
not discover causal structure. It never creates fields, mutates the field
store, or blocks a run.

## Inputs

- The root-to-leaf field path: names, ids, levels, and descriptions.
- The dataset profile and summary: target, time column, column roles,
  missingness, derived and aggregate-looking names, and the description.
- The loaded frame when available: boundary concentration and rare-category
  support.
- The one-hop research records, used as a fallback when a field query returns
  nothing.
- The existing research adapters and per-dataset research cache.

Extraction is deterministic and makes no additional LLM call.

## Query generation

Queries ask for limitations rather than for the field itself:

| Order | Template |
|---|---|
| 1 | `<field name> limitations bias` |
| 2 | `<field name> measurement error confounding` |
| 3 | `<field name> common pitfalls` |

- One query per field by default; `WHACK26_FIELD_PITFALLS_MAX_QUERIES_PER_FIELD`
  allows 1–3.
- Fields are searched leaf first, then ancestors. Leaf results win when the
  report is capped.
- Field names are capped at five words; queries stay under 12 words and contain
  no PII or boolean operators.
- Default sources are `openalex` and `arxiv`. `semantic_scholar` and `websearch`
  are optional and are used only when their adapters are configured. A query
  tries its assigned source, then the other configured sources.
- Results use the normal adapter limits, rate limits, retries, and cache keys.

## Extraction

A record is kept only when its title or abstract matches a category pattern or
a generic limitations/caveat phrase. The primary category is the one with the
most matched terms; the fallback is `other`.

Categories: `censoring`, `selection`, `aggregation`, `measurement_error`,
`missing_data`, `spatial_dependence`, `temporal_dependence`, `confounding`,
`multiplicity`, `rare_group`, `other`.

For each kept record the stage stores the source title, a one-sentence excerpt
(≤ 500 characters), the category, the source id, URL, year, relevance, a
deterministic confidence, and a category-specific `check` sentence. Excerpts
are redacted for secrets.

## Applicability

`applies_to_dataset` is one of:

| Value | Meaning |
|---|---|
| `yes` | The profile or frame shows the precondition, such as a time column, geographic columns, missing values, or a large boundary concentration. |
| `unknown` | The profile cannot establish the precondition. This is background context only. |
| `no` | The profile rules the precondition out. The record is dropped from the report. |

`unknown` is never a finding and must not raise a critic issue by itself. The
method critic may raise an issue from a field pitfall only when the value is
`yes` or when the hypothesis code or collection metadata demonstrates the
defect.

## Citations, dedupe, and caps

- Citation id: `F:<field_id>:<n>`. `<field_id>` is the stored `fld_…` id, or a
  name slug when the field has no id. `<n>` is the report-wide position after
  the final sort and cap, starting at 1.
- A research record is used at most once across the report; duplicate titles
  within a field are dropped.
- Per-field cap: `WHACK26_FIELD_PITFALLS_MAX_PER_FIELD` (default 3).
- Records per query: `WHACK26_FIELD_PITFALLS_RECORDS_PER_QUERY` (default 5).
- Total cap: `WHACK26_FIELD_PITFALLS_MAX_TOTAL` (default 12). The report is
  sorted leaf first, then deeper field level, then higher confidence, then
  citation id before renumbering.
- `applies_to_dataset=no` records are removed before the total cap.

The id convention is stable for a given field path and record set. It is not a
literature identifier; `evidence_id` carries the source record id.

## Artifact and consumers

When the stage runs, `run` writes `field_pitfalls.json` and adds the optional
`field_pitfalls` section to `artifact.json` (schema 1.5.0). The section holds
`schema_version`, `generated_at`, `field_path`, `queries`, `pitfalls`,
`warnings`, `degraded`, `offline`, and `cache`.

The method critic receives at most six applicable pitfalls (never `no`) as
additional cited chunks, leaf first. The hypothesis context pack receives up to
eight applicable pitfalls in its own section. A pitfall remains advisory
context; it is not a statistical validation and does not re-rank hypotheses or
models.

## Failure and degraded behavior

- Source errors, adapter warnings, cache misses in `--offline`, and empty
  result sets become warnings on the report and are copied into the run's
  research warnings.
- `degraded` is set when a source or adapter failed; an empty or missing field
  path returns an empty, degraded report.
- Mock mode does not search external sources; it extracts from the existing
  one-hop records and records a warning.
- The stage is wrapped by `run`: an unexpected failure becomes a
  `field_pitfalls_failed` warning, the run continues, and no sidecar is written.
- The stage never blocks hypothesis generation, the critic, validation, or
  artifact emission.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `WHACK26_FIELD_PITFALLS_ENABLED` | `0` | Enable the stage without `--field-pitfalls`. |
| `WHACK26_FIELD_PITFALLS_MAX_QUERIES_PER_FIELD` | `1` | Queries per field, 1–3. |
| `WHACK26_FIELD_PITFALLS_RECORDS_PER_QUERY` | `5` | Adapter result limit per query. |
| `WHACK26_FIELD_PITFALLS_MAX_PER_FIELD` | `3` | Pitfalls kept per field. |
| `WHACK26_FIELD_PITFALLS_MAX_TOTAL` | `12` | Pitfalls kept per report. |
| `WHACK26_FIELD_PITFALLS_SOURCES` | `openalex,arxiv` | Configured search sources. |

## Non-goals

- Not a performance judgement, model ranking, or validation verdict.
- Not causal discovery; confounding and selection remain questions for the
  design and the critic.
- Not a field creator or classifier; it never writes to the field store.
- Not an exhaustive literature review or a PDF fetch.
- Not a guarantee that a retrieved caveat applies to the dataset.
