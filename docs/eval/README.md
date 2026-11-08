# Field-classification evaluation

`eval/field_classification.py` is the ten-dataset gate from
[`docs/spec/field-classification.md`](../spec/field-classification.md). It runs
the live classifier in three passes (cold, warm, shuffled cold), optionally a
second cold pass for reproducibility, evaluates the automated gates, merges
human scores, and writes:

- `docs/eval/field-classification-results.md`
- `docs/eval/field-classification-results.json`

The harness exits non-zero when a gate fails. Human-score gates are reported as
`pending` until every dataset has a score.

## Commands

```powershell
# Deterministic plumbing check; human gates stay pending.
.\.venv\Scripts\python.exe eval\field_classification.py --mock --repeat --allow-pending

# Create a score template, fill it in, then run the live gate.
.\.venv\Scripts\python.exe eval\field_classification.py --write-scores-template eval/scores.json
.\.venv\Scripts\python.exe eval\field_classification.py --scores eval/scores.json

# Faster smoke test on a subset.
.\.venv\Scripts\python.exe eval\field_classification.py --datasets penguins,california --mock --repeat --allow-pending

# Optional live run with a second cold pass for reproducibility.
.\.venv\Scripts\python.exe eval\field_classification.py --scores eval/scores.json --repeat
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Every blocking gate passed, or only human gates are pending with `--allow-pending` |
| 1 | A blocking gate failed, or a dataset could not be loaded/classified |
| 2 | Configuration error, or human scores are still pending |

## Datasets

| # | Slug | Dataset | Domain | Description |
|---|---|---|---|---|
| 1 | `penguins` | Palmer Penguins | biology / ecology | yes |
| 2 | `wine` | Wine Quality | food / chemistry | yes |
| 3 | `california` | California Housing | housing / economics | yes |
| 4 | `titanic` | Titanic | demographics / history | no |
| 5 | `air_quality` | Air Quality (UCI) | environment | yes |
| 6 | `covid` | COVID country series | public health | yes |
| 7 | `spotify` | Spotify Tracks | music | no |
| 8 | `student` | Student Performance | education | yes |
| 9 | `retail` | Online Retail | retail / commerce | no |
| 10 | `soccer` | Soccer match results | sports | no |

Downloads are cached under `data/eval/` (gitignored). The California Housing and
Palmer Penguins demo files under `data/` are reused when present.

## Human scores

Use the 1-5 rubric from the spec. A dataset needs `root`, `path`, and
`sibling_distinctness`; `naming` is optional. Leave all four null to mark it
unscored. If `naming` is null and the automated naming gate passes, the harness
assumes the automated result is correct and uses score 4; pass
`--no-assume-automated-naming` to require a human naming score. The human gates
share one set of criteria, also printed in the score template:

- at least 80% of datasets score at least 3 on the root;
- at least 80% score at least 3 on the full path;
- the mean root and path scores are at least 3.5;
- the mean naming score is at least 4.0;
- at least 80% of descriptions distinguish the field from its siblings.

The harness never invents root, path, or sibling-distinctness scores.

## Research and factor smoke

The step-3 live smoke is recorded in
[`research-smoke.md`](research-smoke.md). California Housing and Palmer Penguins
both produced 20 ranked records, cited factors, and fully warm second runs
(9/9 and 10/10 cache hits). California produced 4 factors and raised the
small-count warning; Penguins produced 6 without warnings.

## Hypothesis and method-critic smoke

The step-4 live end-to-end smoke is recorded in
[`hypothesis-smoke.md`](hypothesis-smoke.md). Both demo datasets produced a
valid `artifact.json`, a rendered `preview.md`, three executed hypotheses with
unique method families, a critic result per hypothesis, and predictions that
reproduce exactly from the saved code. California returned one `refine` and two
`pass` verdicts; Penguins returned three `refine` verdicts.

## Current status

The harness and its gate logic are implemented and tested. A full deterministic
mock run over all ten datasets completes without crashes and leaves only the
human-score gates pending.

The warm pass reuses a dataset's committed path when its stable content
identity is unchanged (content hash, target, description, schema, time column,
and classification settings); source path and warning text are excluded. A live
warm pass therefore makes no provider calls. The shuffled pass still classifies
against a fresh store and therefore spends calls.

A full live ten-dataset run has not been recorded yet. On 2026-11-07 the
configured Gemini key reported a free-tier daily limit for `gemini-3.6-flash`
(`generate_content_free_tier_requests`). The exact per-model limit is
account-specific and should be checked in AI Studio. A cold classification needs
several calls per dataset, so the live gate requires billing or a key with a
higher quota. `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`
remain available with separate free-tier limits; set `GEMINI_MODEL` to use one
for a smoke test. The client automatically drops `thinkingConfig` when a model
rejects it. Once quota is available, run the live command with a completed
`eval/scores.json`; the report will be written to
`docs/eval/field-classification-results.md`.
