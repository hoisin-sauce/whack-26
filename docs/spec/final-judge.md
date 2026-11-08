# Final model judge

Status: implemented. The package is `whack26/final_judge/`; `run --judge` and
`python -m whack26 judge` are the entry points.

## Purpose and boundary

The final judge is the advisory, run-level audit over the **selected model**,
its **statistical validation**, and the **method-critic results**. It answers:
*given the evidence the pipeline produced, is this model result trustworthy
enough to report, and what must be checked next?*

It is not a second method critic and not a second validator.

| Component | Owns |
|---|---|
| `whack26/critic` | derivation method, claim type, corpus-grounded method gaps |
| `whack26/validation` | statistical validation, folds, permutation, diagnostics |
| `whack26/final_judge` | performance/evidence audit, run-level decision, UI-ready report |
| Report owner / UI | final presentation and any human sign-off |

The judge never re-ranks model-search candidates, never overrides a
`ValidationReport.verdict`, and never blocks artifact emission. A `rejected`
decision is a report-level recommendation, not a deletion.

## Inputs

The evidence adapter consumes whichever of these are available:

- `ModelSearchResult` (selected candidate, folds, train/test metrics,
  diagnostics, coefficients, screening, dataset typing);
- `ValidationReport` / `StatisticalValidationResult` (status, verdict, failed
  layers, DSR, permutation p-value, OOS decay, warnings);
- method-critic results;
- generation-validation results;
- variant and collection metadata;
- dataset profile/description, target, factors, and research titles.

If a critic's evidence is absent, that critic is **unevaluated**. The judge
does not ask the LLM to pass something it cannot see.

## Critic roster

1. `overfitting_skeptic` — train/CV gap and fold variance.
2. `residual_analyst` — Ljung-Box, Durbin-Watson, heteroskedasticity.
3. `leakage_detective` — feature provenance and target leakage.
4. `assumptions_auditor` — stationarity, independence, linearity,
   multicollinearity, calibration, dispersion.
5. `subgroup_critic` — sliced metrics (unevaluated until slices are supplied).
6. `multiple_comparisons_skeptic` — screening/selection multiplicity.
7. `extrapolation_critic` — prediction range versus training support.
8. `simplicity_advocate` — Nadeau–Bengio corrected paired interval versus the
   baseline.
9. `domain_sanity_critic` — coefficient signs/magnitudes against domain context.
10. `devils_advocate` — runs last, must add a novel alternative or report none.

The prompts and JSON schema live in `whack26/final_judge/prompts.py`.
The deterministic checks live in `whack26/final_judge/checks.py`.

## Decision rules

- `rejected` — an established, high-confidence model defect, or a
  corroborated high-severity model finding.
- `requires_reaudit` — a high-severity established finding, or an
  evaluation-design finding that must be confirmed or refuted.
- `accepted_with_concerns` — no blocking finding, but concerns, code-verified
  flags, missing evidence, or unevaluated critics remain.
- `insufficient_evidence` — fewer than half the critics produced a usable
  verdict. This is **not** a pass.
- `accepted` — zero concerns, zero flags, zero missing evidence, and every
  critic evaluated.

A suspected finding can never be high-confidence; code normalises it to
`medium`. Missing statistical validation always prevents a plain `accepted`.

## Outputs

`run --judge` writes into the run directory:

```
final_judge.json           # structured report
final_judge.md             # Markdown section for preview/report owner
final_judge.html           # standalone full page
final_judge.fragment.html  # semantic HTML fragment for the UI
```

The artifact gains an optional `final_judge` section (`schema_version` 1.5.0)
with the decision, findings, flags, re-audit steps, and relative paths to the
HTML report and UI fragment. `preview.md` gains a `## Final model judgement`
section.

## UI handoff

The PyQt dashboard accepts the fragment directly:

```python
dashboard.load_final_judge(run_dir, tab_ref)
```

or, equivalently:

```python
dashboard.set_report(
    (run_dir / "final_judge.fragment.html").read_text(encoding="utf-8"),
    tab_ref,
)
```

The fragment contains no CSS, JavaScript, external resources, or `<details>`
elements. `UI_frontEnd.py` applies its own stylesheet. The full
`final_judge.html` is for standalone viewing.

## CLI and configuration

```powershell
python -m whack26 run ... --model-search --validate --judge
python -m whack26 judge runs/<id>
python -m whack26 judge runs/<id> --refresh --mock
```

`run --full` enables the other optional stages but **not** the judge; add
`--judge` explicitly. `judge --format markdown` (or another subset) writes only
the requested files; the UI handoff needs the default all-formats path so the
fragment exists.

| Variable | Default | Meaning |
|---|---:|---|
| `WHACK26_FINAL_JUDGE_ENABLED` | `0` | Enable the judge in `run` |
| `WHACK26_FINAL_JUDGE_MODEL` | unset | Optional critic model override |
| `WHACK26_FINAL_JUDGE_DA_MODEL` | unset | Optional devil's-advocate model |
| `WHACK26_FINAL_JUDGE_MAX_WORKERS` | `3` | Parallel critic calls |
| `WHACK26_FINAL_JUDGE_MAX_TOKENS` | `2000` | Per-critic output cap |
| `WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION` | `0.50` | Acceptance floor |

## Failure handling

- A critic call failure becomes an unevaluated verdict with the error in
  `evidence`; it never becomes a pass.
- The devil's advocate failure is reported separately.
- A whole-judge failure is advisory: `run.py` records
  `final_judge_failed: ...` and still emits the artifact.
- Mock mode returns explicitly unevaluated verdicts, so the report shows
  `insufficient_evidence` rather than a fabricated clean pass.

## Tests

- `tests/test_final_judge_checks.py` — deterministic checks.
- `tests/test_final_judge_aggregate.py` — decision rules.
- `tests/test_final_judge_evidence.py` — model-search/validation mapping.
- `tests/test_final_judge_report.py` — JSON/Markdown/HTML/fragment rendering.
- `tests/test_final_judge_pipeline.py` — orchestration and provider failure.
- `tests/test_final_judge_integration.py` — run/artifact/preview/CLI handoff.
- `tests/test_ui_pipeline.py` — `run --judge` output loaded by the dashboard, tab isolation, and fragment rendering.
