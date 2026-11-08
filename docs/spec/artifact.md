# Output artifact and component handoff

Status: decided in [ticket #11](https://github.com/hoisin-sauce/whack-26/issues/11). Applies to the 6–8 hour slice.

## Run directory

```
runs/<run_id>/
  artifact.json
  preview.md
  dataset.csv
  factors.json
  research.json
  validation.json             # when --validate is used
  generation_validation.json  # when --validate-generation is used
  model_search.json           # when --model-search is used
  variants.json               # when --variants is used
  research_models.json        # research-to-model extraction
  scrape.json / scrape.md     # when scraping is enabled
  hypotheses/<hyp_id>.py
  predictions/<hyp_id>.csv
  logs/run.log
```

`run_id = "run_" + UTC timestamp (YYYYMMDDTHHMMSSZ) + "_" + dataset_id[3:11]`.

All paths inside the artifact are relative to the run directory. `dataset.source_path` is informational and may be absolute; `dataset.stored_path` is the portable copy.

## Schema

```json
{
  "schema_version": "1.3.0",
  "run": {
    "run_id": "run_20261107T213000Z_a1b2c3d4",
    "started_at": "2026-11-07T21:30:00Z",
    "finished_at": "2026-11-07T21:36:12Z",
    "mode": "live",
    "seed": 0,
    "tool_version": "0.1.0",
    "models": {
      "llm": "gemini-3.6-flash",
      "embedding": "gemini-embedding-2@768"
    }
  },
  "dataset": {
    "id": "ds_a1b2c3d4e5f6",
    "source_path": "data/california_housing.csv",
    "stored_path": "dataset.csv",
    "sha256": "…",
    "format": "csv",
    "rows": 20640,
    "description": "1990 California census block-group housing data.",
    "target": {"name": "median_house_value", "source": "user", "confidence": 1.0},
    "time_column": null,
    "columns": [
      {
        "name": "median_income",
        "dtype": "float64",
        "role": "feature",
        "missing_pct": 0.0,
        "cardinality": 12928
      }
    ]
  },
  "classification": {
    "path": [
      {
        "field_id": "fld_…",
        "name": "Housing & Real Estate",
        "level": 1,
        "description": "Housing supply, demand, prices, and residential property markets."
      }
    ],
    "confidence": 0.74,
    "created_fields": ["fld_…"],
    "refined_after_research": false,
    "refinement_reason": null
  },
  "research": {
    "queries": [
      {"source": "openalex", "query": "determinants of house prices", "rationale": "…"}
    ],
    "records": [
      {
        "id": "openalex:W…",
        "source": "openalex",
        "source_id": "W…",
        "title": "…",
        "abstract": "…",
        "authors": ["…"],
        "year": 2024,
        "doi": "10.…",
        "url": "https://…",
        "license": "CC0",
        "retrieved_at": "2026-11-07T21:31:00Z",
        "query": "determinants of house prices",
        "relevance": 0.82,
        "also_seen_in": []
      }
    ],
    "cache": {"hits": 4, "misses": 2, "offline": false},
    "limited_mode": false,
    "semantic_scraping": {
      "enabled": true,
      "documents": [
        {
          "id": "doc_a1b2c3d4e5f6",
          "kind": "paper",
          "source": "openalex",
          "title": "…",
          "url": "https://…",
          "summary": "…",
          "authors": ["…"],
          "year": 2024,
          "credibility": 0.78,
          "credible": true,
          "field_path": ["Statistics and Validity", "…"],
          "critique_verdict": "refine",
          "comparisons": 3,
          "preloaded": false
        }
      ],
      "links": [{"from_id": "doc_…", "to_id": "doc_…", "relation": "references"}],
      "comparisons": [
        {
          "left_id": "doc_…",
          "right_id": "doc_…",
          "kind": "method_overlap",
          "verdict": "supports",
          "similarity": 0.81
        }
      ],
      "critiques": [
        {
          "document_id": "doc_…",
          "verdict": "refine",
          "corpus_version": "1.0.0+…",
          "summary": "…"
        }
      ],
      "warnings": ["scrape_budget_exhausted"],
      "stats": {"documents": 42, "credible": 31}
    }
  },
  "factors": [
    {
      "id": "fac_…",
      "name": "household income",
      "description": "Median household income in the block group.",
      "direction": "positive",
      "claim_type": "association",
      "evidence_ids": ["openalex:W…"],
      "mapping": {
        "column": "median_income",
        "confidence": 0.91,
        "in_dataset": true,
        "rationale": "The column is the block group's median income."
      },
      "rationale": "…"
    }
  ],
  "screen": {
    "target_summary": {"mean": 206855.8, "std": 115395.6, "missing": 0},
    "missingness": {"total_bedrooms": 0.0100},
    "top_correlations": [{"a": "median_income", "b": "median_house_value", "pearson": 0.688}],
    "time_trend": null
  },
  "hypotheses": [
    {
      "id": "hyp_…",
      "title": "…",
      "method_family": "linear_model",
      "claim_type": "prediction",
      "target": "median_house_value",
      "required_columns": ["median_income", "ocean_proximity"],
      "optional_columns": ["housing_median_age"],
      "min_rows": 100,
      "assumptions": ["…"],
      "factors_used": ["fac_…"],
      "source_ids": ["openalex:W…"],
      "rationale": "…",
      "code_path": "hypotheses/hyp_….py",
      "code_sha256": "…",
      "generation_history": [
        {"attempt": 1, "status": "ok", "error": null}
      ],
      "execution": {
        "status": "ok",
        "runtime_ms": 842,
        "n_predictions": 20640,
        "predictions_path": "predictions/hyp_….csv",
        "error": null
      },
      "critic": {
        "claim_type": "prediction",
        "verdict": "pass",
        "issues": [],
        "missing_information": [],
        "checks": [],
        "summary": "…",
        "retrieved_chunks": ["S0-how-to-use", "S2-leakage"],
        "model": "gemini-3.6-flash",
        "corpus_version": "1.0.0+…"
      },
      "validation": {
        "hypothesis_id": "hyp_…",
        "status": "passed",
        "verdict": "pass",
        "target": "median_house_value",
        "rows_used": 20640,
        "dropped_rows": 0,
        "feature_columns": ["median_income", "housing_median_age"],
        "periods_per_year": 1.0,
        "periods_per_year_source": "default",
        "in_sample_fallback": false,
        "has_fit_predict": true,
        "layers": [
          {"layer": "Layer 1 (DSR)", "status": "passed", "metrics": {"dsr_score": 0.99}},
          {"layer": "Layer 2 (Monte Carlo)", "status": "passed", "metrics": {"p_value": 0.001}},
          {"layer": "Layer 3 (OOS Decay)", "status": "passed", "metrics": {"oos_score": 0.58}},
          {"layer": "Layer 4 (Parameter Sensitivity)", "status": "passed", "metrics": {}}
        ],
        "dsr_score": 0.99,
        "p_value": 0.001,
        "oos_score": 0.58,
        "decay_rate": 0.04,
        "tolerance": 0.31,
        "runtime_ms": 2400,
        "warnings": []
      }
    }
  ],
  "critic_summary": {
    "verdicts": {"pass": 1, "refine": 1, "reject": 1, "insufficient": 0},
    "degraded": false
  },
  "validation_summary": {
    "counts": {"passed": 2, "failed": 1, "partial": 0, "skipped": 0, "error": 0}
  },
  "warnings": [],
  "provenance": {
    "python": "3.12.4",
    "packages": {"pandas": "3.1.1", "numpy": "2.5.4"},
    "corpus_version": "1.0.0+…",
    "git_commit": null
  }
}
```

### Optional 1.3.0 sections

These are absent unless the corresponding stage runs, and legacy artifacts
remain valid without them:

- `model_search` — typed target, candidate list, selected model, metrics,
  diagnostics, backend, and seed (`run --model-search`).
- `variants` — original, anonymized, and jumbled-label reports plus leak
  verdicts (`run --variants`).
- `research_models` — evidence-backed model specs extracted from papers and web
  documents (research-to-model stage).
- `collection` — the declared collection-method metadata
  (`--collection-json`/`--collection-field`).
- `generation_validation_summary` and `hypotheses[].generation_validation` —
  claim/design review results (`run --validate-generation` or
  `validate-models --generation`).

The corresponding sidecar files (`model_search.json`, `variants.json`,
`generation_validation.json`, `research_models.json`) are written next to
`artifact.json` even when a consumer only reads the artifact.

## Stable IDs

| Entity | ID | Rule |
|---|---|---|
| Dataset | `ds_<12 hex>` | `sha256(file bytes)[:12]` |
| Field | `fld_<12 hex>` | `uuid5(NAMESPACE_URL, "whack26/field/<parent_id or root>/<slug(name)>")[:12]` |
| Factor | `fac_<12 hex>` | `sha256("<dataset_id>:<normalized name>")[:12]` |
| Hypothesis | `hyp_<12 hex>` | `sha256(run_code + sorted(factors_used))[:12]` |
| Research record | `<source>:<source_id>` | Stable source identity |
| Run | `run_<timestamp>_<dataset id>` | Time-ordered and dataset-scoped |

IDs are content-derived where possible so re-runs and cross-run dedup can use them. Cross-run deduplication is not implemented in this slice.

## Markdown preview

`preview.md` is generated only from `artifact.json` (no extra state). Sections:

1. Title and run metadata (mode, models, timings).
2. Dataset: shape, target, time column, missingness, top columns.
3. Field path: breadcrumb with descriptions.
4. Research: the top 10 records with title, year, source, link, and one-line relevance.
5. Factors: table of factor, direction, evidence, mapped column, confidence.
6. Hypotheses: one section per hypothesis with title, family, claim type, assumptions, factors, rationale, fenced code, execution status/runtime, predictions path, critic verdict, and the statistical-validation layer statuses and metrics.
7. Critic summary: verdict counts and the issues that are `major`/`fatal`.
8. Validation summary: report-status counts.
9. Warnings and degraded modes.
10. Provenance: Python/package versions, corpus version, git commit if available.

No API keys, absolute paths, raw PII, or full prediction arrays.

## Handoff

**Statistical-tests owner** reads:

- `dataset.stored_path` and `dataset.sha256` (portable copy);
- `dataset.target`, `dataset.time_column`;
- `hypotheses[].code_path`, `required_columns`, `method_family`;
- `hypotheses[].predictions_path` (aligned `row_index,prediction`);
- `hypotheses[].execution` to know whether the function ran;
- `hypotheses[].validation` when the validator ran, and `validation_summary`
  for counts.

The in-repo validator (`python -m whack26 validate-models`) consumes the
artifact, calls the generated module's `_fit`/`_predict` helpers when present,
and writes `validation.json`. Legacy modules without those helpers fall back to
the saved predictions and are marked `in_sample_fallback: true`. The
`validation` section is advisory: a failed layer never invalidates the
artifact. The statistical-tests owner can rerun the engine independently and
write results outside this artifact.

**Report owner** reads `artifact.json` and `preview.md`, plus the statistical owner's results. They own the final report.

**Independent validator** reads the artifact to know the method and sources, but writes its own validation record. The critic verdict is advisory input, never a validation.

## Versioning

- `schema_version` is semver. Breaking changes bump the major.
- `1.1.0` adds the optional `research.semantic_scraping` section.
- `1.2.0` adds the optional `hypotheses[].validation` and
  `validation_summary` sections. `1.0.0` and `1.1.0` artifacts remain valid.
- `1.3.0` adds the optional `model_search`, `variants`, `research_models`,
  `collection`, `generation_validation_summary`, and
  `hypotheses[].generation_validation` sections. `1.0.0`–`1.2.0` artifacts
  remain valid.
- Consumers must reject an unknown major and warn on an unknown minor.
- `schemas/artifact.schema.json` is generated from the Pydantic model and checked into the repo.
- `corpus_version` and `tool_version` are separate from the schema.

## Acceptance

- `python -m whack26 validate-artifact runs/<id>/artifact.json` passes.
- No API keys, absolute paths in `stored_path`/`code_path`/`predictions_path`, or full prediction arrays.
- `preview.md` renders the three hypotheses and the critic verdicts.
- A test imports a generated hypothesis from `code_path` and reproduces the saved predictions within tolerance.
- The artifact is sufficient for the stats owner to run all three hypotheses without reading the pipeline code.

## Module layout

```
whack26/artifact.py   # Pydantic models, IDs, writers, validation
whack26/preview.py    # Markdown renderer
schemas/artifact.schema.json
```
