# Agentic method critic

Status: decided in [ticket #4](https://github.com/hoisin-sauce/whack-26/issues/4). Applies to the 6–8 hour slice.

## Purpose and boundary

The critic audits **the method used to derive a model** against the preloaded statistical-pitfalls corpus. It answers: *does this derivation support the claim being made?*

**In scope:** claim type, estimand, assumptions, leakage, split protocol, multiplicity and selection, causal versus predictive language, model-family fit, sample size, missing data, measurement error, time-series dependence, factor traceability, and reproducibility.

**Out of scope:** p-values, effect sizes, accuracy, AUC, RMSE, calibration, out-of-sample performance, benchmark comparisons, and the truth of the finding. A separate independent validator owns those. The critic may say a diagnostic is missing; it must not compute or judge the diagnostic result.

## Inputs

```python
class CriticInput(BaseModel):
    hypothesis_id: str
    dataset: DatasetProfile
    field_path: FieldPath
    factors: list[Factor]
    hypothesis: Hypothesis          # metadata + code
    screen: ScreenSummary           # descriptive only
    research: list[ResearchRecord]  # top 12
```

The critic never receives API keys, absolute file paths outside the run, or raw PII.

## Corpus and retrieval

Corpus: `docs/corpus/statistical-pitfalls.md` (copied from the research branch; version 1.0.0). Its `##` sections are self-contained chunks:

`S0-how-to-use`, `S1-assumptions`, `S2-leakage`, `S3-multiplicity`, `S4-causal-predictive`, `S5-diagnostics`, `S6-count-models`, `S7-sample-size`, `S8-missing-measurement`, `S9-time-series`, `S10-reporting`.

Retrieval:

1. Always include `S0-how-to-use` (claim-type gate, verdict vocabulary, output shape).
2. Embed `hypothesis.rationale + dataset summary + factor names`.
3. Rank sections 1–10 by cosine similarity; include the top 4.
4. Maximum 5 chunks per call. If embedding fails, include `S0-how-to-use`, `S1-assumptions`, `S2-leakage`, `S5-diagnostics`.
5. If the corpus file is missing, return `verdict = "insufficient"` with a `critic.corpus_missing` issue; never silently pass.

Citation convention: each issue cites a chunk ID and, when available, a source from that chunk. The artifact stores `retrieved_chunks` and `corpus_version`.

## Deterministic checks

These run before the LLM and cannot be overridden by it.

| ID | Check | Severity | Trigger |
|---|---|---|---|
| `code.static` | AST/import/call allowlist | fatal | Any banned import or call, or missing `applies`/`run` |
| `target.leakage` | Target or target-derived column used as a feature | fatal | Target in `required_columns`/`optional_columns`, or code references it on the feature side |
| `split.preprocessing` | Preprocessing fitted before the split | major | Scaler/imputer/encoder `.fit()` on the full frame before `train_test_split` (AST heuristic) |
| `split.temporal` | Future data used for training | fatal | Negative `shift`, future-dated rows, or a random split when a time column exists |
| `missing.complete_case` | Silent complete-case analysis | major | `dropna()` when a column has > 5% missing and no note |
| `sample.epv` | Events per variable too low | major | < 10 events per parameter for regression/logistic; < 5 for a count model |
| `sample.small` | Small sample | minor | `n < 100` for a multi-parameter model |
| `multiplicity.selection` | Undisclosed search | major | More than one hypothesis/factor set and a significance or "best model" claim without a correction or pre-specification |
| `claim.causal` | Causal language without a design | major | "causes", "effect of", "leads to", "impact" in the rationale when the data description is observational |
| `model.family` | Model family cannot represent the outcome | fatal | OLS presented as a rate/count model; classification model on a continuous target |
| `time.series` | Dependence ignored | major | Time column exists and neither lag, differencing, nor a time-based split is used |
| `research.traceability` | Factors not traceable | major | A factor lacks an evidence ID, or a hypothesis uses a factor not in the run |
| `assumptions.specific` | Generic assumptions | minor | Empty, one-word, or "all assumptions met" |
| `reproducibility` | Not reproducible | minor | No seed, no data hash, or no code hash |

The static check is shared with `hypotheses/validate.py`; the critic reuses its result rather than re-parsing.

## LLM rubric

The LLM reviews the retrieval chunks and returns one issue per material gap, using the corpus vocabulary. It must:

- classify the claim as `description | prediction | causal | explanatory`;
- use the corpus's verdict vocabulary: `pass | refine | reject | insufficient`;
- use severities `fatal | major | minor`;
- cite a chunk ID for every issue;
- list missing information rather than guessing;
- not mention performance metrics or statistical significance.

Rubric areas: estimand and assumptions (`S1`), leakage (`S2`), multiplicity (`S3`), causal versus predictive (`S4`), regression diagnostics (`S5`), count models (`S6`), sample size (`S7`), missing data and measurement error (`S8`), time series (`S9`), reporting and reproducibility (`S10`).

## Output

```json
{
  "hypothesis_id": "hyp_...",
  "claim_type": "description | prediction | causal | explanatory",
  "verdict": "pass | refine | reject | insufficient",
  "issues": [
    {
      "chunk": "S2-leakage",
      "severity": "fatal | major | minor",
      "finding": "Preprocessing was fitted on the full dataset before the split.",
      "evidence": "run() scales the frame before train_test_split",
      "why": "Test-set information entered training; the estimate is optimistic.",
      "fix": "Fit the scaler inside the training fold and re-evaluate."
    }
  ],
  "missing_information": ["unit of analysis", "split protocol"],
  "checks": [
    {
      "id": "split.preprocessing",
      "status": "pass | flag | fail | not_applicable",
      "severity": "fatal | major | minor | none",
      "reason": "...",
      "citations": ["S2-leakage"]
    }
  ],
  "summary": "One paragraph, no performance claims.",
  "retrieved_chunks": ["S0-how-to-use", "S2-leakage", "S5-diagnostics"],
  "model": "gemini-3.6-flash",
  "corpus_version": "1.0.0+<sha256[:8]>"
}
```

## Verdict rules

- `reject` if any `fatal` issue exists.
- `insufficient` if required information is missing and no fatal issue exists, or if the critic cannot run.
- `refine` if any `major` issue exists and no fatal/insufficient issue exists.
- `pass` otherwise (minor issues may still be listed).

The critic never returns "validated" or "works". `pass` means "no material method gap found in the supplied derivation", not "the finding is true".

## Prompt

System:

> You are a method critic for statistical modelling. You audit whether the derivation method can support the claim being made. You do not evaluate model performance, p-values, effect sizes, accuracy, AUC, RMSE, calibration, or out-of-sample results; another component owns that. Use only the supplied corpus chunks and derivation trace. If the information needed to judge a check is absent, say so in `missing_information` and use `insufficient` rather than guessing. Return JSON matching the schema.

User: dataset profile summary, field path, factors, hypothesis metadata, code, rationale, assumptions, screen summary, retrieved corpus chunks.

Temperature 0; JSON schema enforced; `thinkingConfig.thinkingBudget = 0` when the provider supports it.

## Failure handling

- Invalid JSON or schema mismatch: retry once with the validation error.
- Second failure or timeout: return `verdict = "insufficient"`, one `major` issue `critic.unavailable`, and a summary explaining that the critic could not run. The artifact is still emitted; the run is marked `critic_degraded = true`.
- No blocking of hypothesis execution or the artifact on critic failure.
- Log the model, latency, and token counts; never log prompts containing secrets.

## Corpus versioning

- Copy the research corpus to `docs/corpus/statistical-pitfalls.md` on `main`.
- `corpus_version = "<frontmatter version>+<sha256[:8]>"`.
- Store `corpus_version`, `retrieved_chunks`, and the model in every critic result.

## Acceptance

- Unit tests seed three examples: a clean split, a scaler-before-split leak, and a target-as-feature leak. The critic must pass the first and fail the other two.
- On the demo's three hypotheses, returns valid JSON with a verdict and at least one citation per issue.
- Never emits a performance judgment; a test asserts that banned metric terms are absent from `summary` and `issues`.
- Latency ≤ 30 s per hypothesis with thinking disabled; ≤ 90 s with thinking enabled.

## Module layout

```
whack26/critic/
  corpus.py    # parse, chunk IDs, embeddings, retrieval
  checks.py    # deterministic checks
  critic.py    # LLM call, merge, verdict
```
