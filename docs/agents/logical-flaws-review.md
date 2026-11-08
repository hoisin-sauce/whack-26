# Logical-Flaws Review — Findings for Subagent Work

> **Resolution (2026-11-08):** All findings in this review have been implemented.
> The validation engine was renamed to `stats_engine.py`; the full suite is now
> **341 passed**. The fixes cover Work Packages 1–7, including the continuous
> target contract, real trial Sharpes, finite-value guards, store path
> invariants, stable classification cache identity, profile/factor role fixes,
> research cache and ranking fixes, eval-gate alignment, ingestion/config/doctor
> hardening, and the `PyQt6` optional requirements file. The findings below are
> retained as the historical review record.

**Date:** 2026-11-08
**Scope:** `whack26/`, `eval/`, `tests/`, `stats_test.py`, `UI_frontEnd.py`
**Mode:** Review only. No source files were edited by this review; the only authored artifact is this document. A `git pull --ff-only` fast-forwarded the working tree to `d74836a` (UI/stats changes).
**Audience:** An orchestrator that will split the work below across subagents.

> Line numbers refer to the working tree at review time and may drift while another agent edits the codebase. Re-read the cited code before patching. Do not treat this document as a specification; treat it as a prioritized list of suspected logical flaws with evidence.

---

## Ground rules for subagents

1. **Use the repo virtualenv**, not the system `python` on PATH:
   ```powershell
   .venv\Scripts\python.exe -m pytest -q --tb=short
   ```
   The system Python here is 3.15 and lacks the project dependencies. Running pytest with it produces 27 collection errors and says nothing about the repo suite.

2. **Do not weaken tests to make them pass.** Several existing tests encode the bugs listed below. Update those tests deliberately and explain the behavior change.

3. **Do not make unrelated changes.** Another agent may be editing the codebase concurrently. Keep patches scoped to one work package.

4. **Prefer test-first fixes.** Add a failing test that demonstrates the flaw, then fix it.

5. **Confirm before changing behavior.** Items marked *design smell* or *judgment* are not proven bugs. Ask the owner before changing public behavior.

6. **Run targeted tests first**, then the full suite under `.venv`.

---

## Verification status

| Check | Result |
| --- | --- |
| `.venv` interpreter | Python 3.12.4 |
| `.venv` numpy / scipy | 2.5.4 / 1.18.1 |
| `.venv` `pytest --collect-only` | rc=0, **247 tests collected** (post-pull `d74836a`) |
| Full suite under `.venv` | **247 passed in 90.24s** (0:01:30), post-pull `d74836a` |
| System Python 3.15 pytest | 27 collection errors, `Interrupted`, rc=2 — **not** evidence about the repo suite |
| Temp-dir pytest experiment | Default collection error → `Interrupted`, rc=2, no tests run; `--continue-on-collection-errors` → `1 passed, 1 error`, rc=1 |

**Environment caveat:** `importlib.util.find_spec` returns `None` for a missing module, not `False`. Earlier notes that said "returns False" were using a `bool(...)` wrapper.

**Post-pull revision:** Remote `d74836a` changed `UI_frontEnd.py` and `stats_test.py`. Work Package 1 below is rewritten for the pulled code. The other work packages are unchanged.

---

## Severity summary

| ID | Severity | Area | Finding |
| --- | --- | --- | --- |
| VAL-01 | **Critical** | UI | UI builder signature mismatch: engine calls `(params, X, y)`, UI defines `(params)`. Layers 3–4 always fail. |
| VAL-02 | **Critical** | UI | Layer 2 compares binary predictions to continuous actuals; accuracy ≈ 0, p ≈ 1.0. |
| VAL-03 | **Critical** | UI/Stats | No target contract: pure binary fails Layer 1; pure continuous fails Layer 2. |
| VAL-04 | **High** | UI/Stats | `min_sharpe` is still not the DSR threshold; engine reads `min_dsr_limit`, UI passes `min_sharpe_limit`. |
| VAL-05 | **High** | UI | Trial Sharpes are still fabricated from `min_sharpe`. |
| VAL-06 | **Medium** | Stats | NaN OOS score passes Layer 3 (`success: True`, `decay_rate: nan`). |
| VAL-07 | **Medium** | Stats | `pss` returns True when zero trials run. |
| VAL-08 | **Medium** | Stats | `pss` passes with partial trial failures. |
| VAL-09 | **Medium** | Stats | `np.isnan` guards miss `inf`. |
| VAL-10 | **Low** | UI/Stats | `max_decay` silently clipped; `model_predict_func` is dead; `data_description` unused. |
| VAL-11 | **Low** | Packaging | `PyQt6` is still missing from `requirements.txt`. |
| VAL-12 | **Low** | Tests | `__test__ = False` does not prevent collection-time import; rename/exclude is still needed. |
| STORE-01 | **High** | Store | `commit_path` accepts inconsistent `path_json` / `field_ids`. |
| STORE-02 | **Medium** | Store | `dataset_path` silently skips missing fields. |
| STORE-03 | **Medium** | Tests | `test_store.py` encodes the path/ID inconsistency as valid. |
| CLASS-01 | **Medium** | Classify | `_cached_path` compares the full `DatasetProfile`, including `warnings` / `source_path`. |
| CLASS-02 | **Medium** | Classify/Eval | Fallback root `"General Research"` is banned by the eval gate. |
| PROF-01 | **High** | Profile | LLM target choice accepts id/time/leakage columns. |
| PROF-02 | **High** | Profile | Target role overrides id/time role. |
| PROF-03 | **Medium** | Profile | `_looks_like_id` over-flags high-cardinality integer/string columns. |
| PROF-04 | **Low** | Profile | `_name_looks_sensitive` flags `grid`, `valid`, `solid`. |
| FACT-01 | **High** | Factors | Verification can overwrite an exact mapping (off-prompt response). |
| FACT-02 | **Medium** | Factors | `_singular("houses")` → `"hous"`. |
| FACT-03 | **Low** | Factors | Exact-mapping precedence prefers a short column name over a description match. |
| RES-01 | **Medium** | Research | Cache truncates bodies at 512K characters with no marker. |
| RES-02 | **Low/latent** | Research | Cached `status` is never checked on read. |
| RES-03 | **Medium** | Research | Wikipedia offline partial cache aborts the whole source. |
| RES-04 | **Medium** | Research | Dedupe ranks source before abstract length, losing better metadata. |
| RES-05 | **Medium** | Research | Embedding failure silently degrades to zero similarity. |
| RES-06 | **Medium** | Research | Query cache key omits the field path. |
| EVAL-01 | **High** | Eval | Gate bans the classifier's own fallback root. |
| EVAL-02 | **Medium** | Eval | Near-duplicate root gate tolerates one pair. |
| EVAL-03 | **Medium** | Eval | Naming gate uses plain substring matching. |
| EVAL-04 | **Medium** | Eval | Human rubric and automated gates disagree. |
| EVAL-05 | **Medium** | Tests | `evaluate_naming` has no test. |
| IO-01 | **Low** | Ingestion | Duplicate-header restoration only works when widths match. |
| IO-02 | **Low** | Ingestion | JSON retry conflates schema errors with layout errors. |
| CFG-01 | **Low** | Config | `desc_weight` / `centroid_weight` are not validated. |
| DOC-01 | **Low** | Doctor | Python check requires exactly 3.12. |
| DOC-02 | **Low** | Doctor | `_version_of` fallback is evaluated eagerly. |
| DOC-03 | **Low** | Doctor | NaN embedding vector can pass the check. |
| DEP-01 | **Medium** | Packaging | `PyQt6` is not in `requirements.txt`. |
| TEST-01 | **Medium** | Tests | `stats_test.py` is collected by pytest; `__test__ = False` does not prevent the import, so an interpreter missing numpy/scipy still aborts the suite. Under `.venv` it is harmless. |
| TEST-02 | **Medium** | Tests | Several tests encode the bugs above instead of catching them. |

---

# Work Package 1 — Validation engine and GUI (post-pull)

**Files:** `stats_test.py`, `UI_frontEnd.py`, `requirements.txt`
**Suggested subagents:** one for the engine, one for the GUI, one for integration tests.

> **Post-pull status:** Remote `d74836a` fixed several pre-pull findings and introduced new API mismatches. This section reflects the pulled code. The pre-pull section that follows is retained for history only; do not implement its VAL-01..10 findings.

## Fixed in `d74836a`

- Import fallback removed; `UI_frontEnd.py:11-12` now imports `validation_engine` directly.
- `actuals = targets` replaced the self-referential `actuals = predictions` (`UI_frontEnd.py:176-178`).
- `max_decay` is forwarded as `base_params["max_decay_tolerance"]` (`UI_frontEnd.py:196-202`) and used by `oos_decay` (`stats_test.py:117-118`).
- `validation_engine` returns a result dict with `passed` / `failed_layer` instead of a bool (`stats_test.py:250-326`).
- `oos_decay` returns a dict; the `-1.0` sentinel is gone (`stats_test.py:98-161`).
- `oos_decay` now calls `base_model_builder(base_params, X_train, y_train)`, so the engine can fit (`stats_test.py:132-137`).
- NaN guards added for DSR, p-value, in-sample score, base score, and variant scores (`stats_test.py:25-26, 36-37, 142-143, 204-205, 233-235, 274, 284`).
- `pss` skips `min_sharpe_limit` / `max_decay_tolerance`, fixes integer perturbation skips, and detects all-fail (`stats_test.py:210-244`).
- `stats_test.py` sets `__test__ = False` (`stats_test.py:4-5`), though this does not prevent collection-time import.
- UI summary report added (`UI_frontEnd.py:102-120, 219-229`).

## VAL-01 — UI builder signature mismatch (Critical, new)

**Where:** `UI_frontEnd.py:193-194`; `stats_test.py:133, 199, 227`

```python
# UI
def base_model_builder(params):
    return model_predict_func
```

```python
# Engine
model = base_model_builder(base_params, X_train, y_train)
...
base_model = base_model_builder(base_params, features, targets)
variant_model = base_model_builder(test_params, features, targets)
```

Every call raises `TypeError: base_model_builder() takes 1 positional argument but 3 were given`. `oos_decay` catches it and returns `success: False`; `pss` catches it and returns `False`. The dashboard can never pass Layer 3 or Layer 4.

**Verified:** running `oos_decay` with a one-argument builder returns `{'success': False, ..., 'error': 'Model training/prediction failed: one_arg_builder() takes 1 positional argument but 3 were given'}`; `pss` returns `(False, 'Baseline model construction failed: ...')`.

**Fix direction:** Change the UI builder to `def base_model_builder(params, X, y):` and return a fitted callable; or change the engine API to a single, documented signature. Add an integration test that calls `validation_engine` with the UI's builder.

**Tests:** Engine + UI builder integration; assert no `TypeError`; assert fit/predict behavior.

## VAL-02 — Layer 2 compares binary predictions to continuous actuals (Critical, new)

**Where:** `UI_frontEnd.py:176-178`; `stats_test.py:77-80`

```python
actuals = targets
predictions = np.where(targets > np.median(targets), 1, 0)
```

`monte_carlo`'s default metric is `np.mean(preds == acts)`. If `targets` is continuous (price, return, count), exact equality between a 0/1 prediction and a continuous actual is almost always false. `real_score ≈ 0`, permuted scores ≈ 0, and `p` tends to `1.0`, so Layer 2 fails.

**Fix direction:** Binarize the actuals consistently (`actuals = (targets > median).astype(int)`) or use a regression metric with an explicit direction. Do not mix a classification prediction with a continuous label.

**Tests:** Continuous target; binary target; constant target.

## VAL-03 — Layer 1 and Layer 2 require incompatible target types (Critical, new)

**Where:** `UI_frontEnd.py:179-180`; `stats_test.py:10-49, 55-92`

- Layer 1 uses `winning_returns = targets[targets > 0]`. For a binary 0/1 target, that is all 1s, `std = 0`, and `dsr_dynamic` returns `0.0`; Layer 1 fails.
- Layer 2's default metric needs binary/categorical labels. For a continuous target, it fails as in VAL-02.

A pure binary 0/1 target fails Layer 1; a purely continuous target fails Layer 2. Only accidental mixed data satisfies both. The UI needs separate return/label inputs or a documented target contract (e.g., "last column must be a continuous return series; Layer 2 will binarize it").

**Fix direction:** Decide the target contract. If the last column is returns, use a regression metric for Layer 2 and a return-based Layer 1. If it is labels, use a separate return series for Layer 1. Add a matrix test: continuous, binary, constant.

**Tests:** The three target types above.

## VAL-04 — `min_sharpe` is still not the DSR threshold (High)

**Where:** `UI_frontEnd.py:197-202`; `stats_test.py:265, 210-212`

`validation_engine` reads `min_dsr_limit` (default `0.95`), but the UI passes `min_sharpe_limit`. `pss` explicitly skips `min_sharpe_limit` and `max_decay_tolerance`. `min_sharpe` only affects `all_trial_sharpes` (`UI_frontEnd.py:180`).

**Fix direction:** Pass `min_dsr_limit=min_sharpe` (or rename the UI field and the engine parameter consistently), remove `min_sharpe_limit` from `base_params`, and add a test that a high `min_sharpe` rejects a low DSR.

**Tests:** `min_sharpe` below/above the observed DSR; assert the gate uses it.

## VAL-05 — Trial Sharpes are still fabricated (High)

**Where:** `UI_frontEnd.py:180`

```python
all_trial_sharpes = [min_sharpe + 0.3, min_sharpe + 0.1, 0.8, 1.1]
```

The DSR multiple-testing penalty is not based on actual trials. `min_sharpe` influences the trial list, which is the opposite of its intended role.

**Fix direction:** Compute trial Sharpes from real trials, or mark the GUI as a demo and do not claim a DSR result.

**Tests:** Trial list derived from data; `min_sharpe` does not change the trial list.

## VAL-06 — NaN OOS score passes Layer 3 (Medium, remaining)

**Where:** `stats_test.py:139-160`

`is_score` is checked for NaN (142), but `oos_score` is not. If `oos_score` is NaN, `decay_rate` is NaN, `if decay_rate > dynamic_tolerance` is false, and the function returns:

```python
{"success": True, "oos_score": nan, "decay_rate": nan, "tolerance": ...}
```

`validation_engine` only checks `success`, so Layer 3 passes with a NaN score.

**Verified:** a metric returning `1.0` on train and `nan` on test yields `{'success': True, 'oos_score': nan, 'decay_rate': nan, ...}`.

**Fix direction:** Check `np.isfinite(oos_score)` and `np.isfinite(decay_rate)` before returning success.

**Tests:** Metric returns NaN on the test split; assert failure.

## VAL-07 — `pss` returns True with zero trials (Medium, remaining)

**Where:** `stats_test.py:207-244`

`total_trials` is incremented only for perturbable parameters. If `base_params` contains only `min_sharpe_limit` / `max_decay_tolerance` (both skipped), `total_trials == 0` and the function returns `(True, "Parameter stability confirmed.")`.

**Verified:** `pss(..., {"min_sharpe_limit": 1.0, "max_decay_tolerance": 0.2}, ...)` returns `(True, 'Parameter stability confirmed.')`.

**Fix direction:** Return `False`/warning when `total_trials == 0`.

**Tests:** Only-threshold params; assert failure/warning.

## VAL-08 — `pss` passes with partial trial failures (Medium, remaining)

**Where:** `stats_test.py:241-242`

The function fails only when `failed_trials == total_trials`. A 50% exception rate still returns `True`.

**Fix direction:** Require a minimum number of successful trials or fail above a failure ratio; report the counts in the result.

**Tests:** Builder raises for half the variants; assert failure or a warning.

## VAL-09 — `np.isnan` guards miss `inf` (Medium, remaining)

**Where:** `stats_test.py:25-26, 36-37, 142-143, 204-205, 233-235, 274, 284`

The new guards use `np.isnan`, not `np.isfinite`. `inf` can propagate through DSR, OOS, and PSS.

**Fix direction:** Use `np.isfinite` for every scalar guard.

**Tests:** `np.inf` / `-np.inf` inputs per layer.

## VAL-10 — Silent clipping and dead parameters (Low)

**Where:** `stats_test.py:117-118, 257`; `UI_frontEnd.py:148-150`

- `max_decay_tolerance` is clipped to `[0.15, 0.50]` without telling the user. A user entering `0.8` gets `0.5`.
- `validation_engine` accepts `model_predict_func` but never uses it; `oos_decay`/`pss` use `base_model_builder`.
- `data_description` is displayed but never passed to the engine.

**Fix direction:** Validate/report the clamp; remove the dead parameter or use it; pass the description if it is meant to inform validation.

**Tests:** `max_decay=0.8` reports the clamp; no unused parameter in the engine signature.

## VAL-11 — `PyQt6` still missing (Low)

**Where:** `requirements.txt:1-19`; `UI_frontEnd.py:5`

The GUI cannot run from a clean install. Add `PyQt6` to an optional UI requirements file.

## VAL-12 — `__test__ = False` does not prevent import (Low)

**Where:** `stats_test.py:4-5`

`__test__ = False` tells pytest not to collect items, but pytest still imports the module to read the flag. If numpy/scipy are missing, the collection error still aborts the run. Under `.venv` it imports cleanly and is harmless.

**Fix direction:** Rename the file (e.g., `stats_engine.py`) or exclude it via `collect_ignore` / `python_files`; do not rely on `__test__ = False` for dependency isolation.

**Tests:** A temp module with a missing import plus `__test__ = False` still errors (verified).

---

# Work Package 1 — Validation engine and GUI (pre-pull; superseded)

> **Superseded:** This section describes the pre-pull code. Use the post-pull section above.

**Files:** `stats_test.py`, `UI_frontEnd.py`, `requirements.txt`
**Suggested subagents:** one for the engine, one for the GUI, one for tests.

## VAL-01 — Layer 2 is self-certifying (Critical)

**Where:** `UI_frontEnd.py:142-143`

```python
predictions = np.where(targets > 0, 1, 0)
actuals = predictions
```

**What:** On the file path, the "actual" labels are a copy of the predictions. `monte_carlo` then scores identical arrays, so the real score is 1.0 by construction. The gate is measuring agreement with itself, not model skill.

**Nuance:** For non-constant targets the p-value is approximately `1/501`, so it usually passes. For a constant target column, every shuffle is identical and `p = 1.0`, so Layer 2 fails. Either way the result says nothing about a model.

**Fix direction:** Use real held-out labels for `actuals`; derive `predictions` from a model. If this is only a demo, label it as a demo and do not print "passed all validation gates."

**Tests:** A test where `actuals != predictions`; a constant-target test; a small/imbalanced binary target test (see VAL-09).

## VAL-02 — UI data, trial Sharpes, model, and thresholds are fake or ignored (Critical)

**Where:** `UI_frontEnd.py:127, 139-160, 164-175`

- `winning_returns = targets[targets > 0]` (141) assumes the last column is a return series and positive values are wins.
- `all_trial_sharpes` is hardcoded at 144 and 148; on the file path it includes `min_sharpe + 0.2`.
- `dummy_predict` returns random labels (154-155) and is passed as the model (171).
- `dummy_builder(params)` ignores `params` (157-158).
- `base_params` includes `min_sharpe_limit` (160); `pss` sweeps it as a hyperparameter.
- `max_decay` is parsed at 127 but never forwarded to `validation_engine` (164-175); the engine has no such parameter.
- The import fallback returns `True` (12-16), so a missing engine reports success.

**Fix direction:** Either wire the GUI to real artifacts (model, predictions, trial Sharpes, thresholds) or make it an explicit demo with a clear warning. Pass `max_decay` into the engine or remove the input. Make the import failure a hard error. Add `PyQt6` to an optional extra or `requirements.txt`.

**Tests:** GUI-level smoke test with a fake engine that records arguments; assert `max_decay` and `min_sharpe` reach the engine; assert the fallback does not return `True`.

## VAL-03 — NaN passes every threshold guard (Critical)

**Where:** `stats_test.py:273, 283, 158, 234`

- `if dsr_score < 0.95` — `NaN < 0.95` is false, so NaN passes Layer 1.
- `if p_value > 0.05` — NaN passes Layer 2.
- `if decay_rate > dynamic_tolerance` — NaN passes Layer 3.
- `if drop > max_performance_drop` — NaN passes Layer 4.
- `if oos_score == -1.0` (293) — NaN is not `-1.0`, so NaN passes Layer 3.
- `if base_score <= 0` (200) — `NaN <= 0` is false, so a NaN baseline passes Layer 4.

**Fix direction:** Use `math.isfinite` / `np.isfinite` guards before every comparison; treat non-finite as failure. Add explicit tests with `np.nan`, `np.inf`, and `-np.inf`.

**Tests:** One per layer; assert failure, not pass.

## VAL-04 — `oos_decay` never fits a model (High)

**Where:** `stats_test.py:144-145`

```python
train_preds = model_predict_func(X_train)
test_preds = model_predict_func(X_test)
```

The function accepts a predict callable; nothing in the function fits a model. The GUI passes a random predictor. A caller could pass a pre-trained model, but the engine itself never trains.

**Fix direction:** Accept a `fit` callable or a model object with `fit`/`predict`; fit on `X_train, y_train`, then predict on both splits. If predict-only is intentional, rename the layer and document that the caller must pre-train.

**Tests:** A spy model that records `fit`; assert `fit` is called once on the training split before predictions.

## VAL-05 — DSR cutoff hardcoded; `min_sharpe` misused (High)

**Where:** `stats_test.py:273`; `UI_frontEnd.py:144, 160`

`validation_engine` has no min-Sharpe parameter and compares DSR to a literal `0.95`. `min_sharpe` is never used as a threshold. It is instead:
- injected into `all_trial_sharpes` as `min_sharpe + 0.2` (UI 144), and
- swept by `pss` as `min_sharpe_limit` (UI 160; stats 206-218).

**Fix direction:** Add an explicit `min_dsr` parameter (default 0.95) and pass the user's threshold. Remove `min_sharpe_limit` from `base_params`; it is a validation threshold, not a model hyperparameter. If trial Sharpes are meant to be real, compute them from trials, not from the threshold.

**Tests:** `min_dsr` below/above observed DSR; assert `pss` does not perturb a validation threshold.

## VAL-06 — `pss` swallows all exceptions and can pass without testing (High)

**Where:** `stats_test.py:224-228`

```python
except Exception:
    continue
```

Every perturbation can fail and the function still returns `True`. For integer parameters, `int(round(...))` can equal the base value, so all trials can be skipped (215-218).

**Fix direction:** Count attempted, skipped, and failed perturbations. Fail or warn if no perturbation was actually evaluated. Re-raise programming errors; only skip known-invalid parameter combinations.

**Tests:** Builder raises for every variant → `pss` must not return `True`; integer parameter whose ±10% rounds to itself → explicit skip count.

## VAL-07 — Metric direction assumed higher-is-better (Medium)

**Where:** `stats_test.py:231-234`

```python
drop = (base_score - variant_score) / base_score
```

A lower-is-better metric (MSE, MAE) that worsens has a higher score, so `drop` is negative and the check passes. `oos_decay` has the same assumption.

**Fix direction:** Add a `higher_is_better` flag or a metric object with a direction. Compute degradation accordingly.

**Tests:** MSE-style metric where a variant is worse; assert failure.

## VAL-08 — `-1.0` sentinel conflates failure and a legitimate score (Medium)

**Where:** `stats_test.py:135, 152, 160, 293`

`oos_decay` returns `-1.0` for insufficient data, non-positive in-sample score, and excessive decay. A legitimate metric value of `-1.0` is indistinguishable from failure.

**Fix direction:** Return a result object (`score`, `status`, `reason`) or raise typed exceptions. Do not use an in-band sentinel.

**Tests:** A metric that returns `-1.0` legitimately; assert it is not treated as a failure unless the status says so.

## VAL-09 — Layer 2 p-value depends on permutation identity rate (Medium)

**Where:** `stats_test.py:97-98`; `UI_frontEnd.py:142-143`

The exact p-value is:

```text
p = (1 + B * P(identity)) / (B + 1)
```

`P(identity)` is the probability that a random shuffle restores the original array. For a binary target with one minority value in `n` rows, `P(identity) = 1/n`; for `n = 20`, expected identity shuffles ≈ 25, so `p ≈ 26/501 ≈ 0.052 > 0.05` and Layer 2 fails even when `actuals = predictions`. For a constant target, `P(identity) = 1` and `p = 1.0`. For all-distinct values, `P(identity) = 1/n!`, so `p ≈ 1/501`.

**Fix direction:** Use a permutation test that excludes identity permutations, or compute the exact null distribution, or report the identity rate. The key is that the p-value must not depend on how many repeated values the target has.

**Tests:** Small/imbalanced binary target; constant target; all-distinct target.

## VAL-10 — Import fallback and missing GUI dependency (Low)

**Where:** `UI_frontEnd.py:12-16`; `requirements.txt:1-19`

The fallback returns `True` on `ImportError`. `PyQt6` is imported but not declared.

**Fix direction:** Remove the fallback or return a typed failure; add `PyQt6` to an optional extra or requirements. If the GUI is a demo, put it in a separate `demo/` directory with its own requirements.

**Tests:** Import failure path; packaging smoke test.

---

# Work Package 2 — Store and classification consistency

**Files:** `whack26/store.py`, `whack26/classify.py`, `tests/test_store.py`, `tests/test_classifier.py`
**Suggested subagents:** one for store invariants, one for classification cache/fallback.

## STORE-01 — `commit_path` accepts inconsistent `path_json` / `field_ids` (High)

**Where:** `whack26/store.py:507-602`

`path_json` is inserted/upserted verbatim. `field_ids` drives `field_members`. Nothing checks that `path_json` decodes to the same IDs or order. A caller can commit a dataset whose displayed path disagrees with its memberships.

**Fix direction:** Decode `path_json`, validate it is a list of strings, compare it to `field_ids` (same IDs, same order), and reject mismatches. If order is intentionally independent, store an explicit ordering field and document it.

**Tests:** Mismatched order; mismatched IDs; malformed JSON; valid case.

## STORE-02 — `dataset_path` silently skips missing fields (Medium)

**Where:** `whack26/store.py:654-657`

```python
field = self.get_field(field_id)
if field is not None:
    fields.append(field)
```

A deleted or unknown field is dropped from the path with no error. The caller cannot distinguish a shorter path from a corrupted one.

**Fix direction:** Raise `FieldStoreError` on a missing field, or return a typed partial result with the missing IDs. Prefer raising for a committed dataset path.

**Tests:** Delete a field referenced by `path_json`; assert a clear error.

## STORE-03 — `test_store.py` encodes the inconsistency (Medium)

**Where:** `tests/test_store.py:280-307`

The test commits `path_json=[child, grandchild, root]` with `field_ids=[root, child, grandchild]` and asserts both survive. This is the bug expressed as expected behavior.

**Fix direction:** Decide the invariant (order must match or order is independent), then update this test to assert the chosen behavior. Add a rejection test for the invalid case.

## CLASS-01 — `_cached_path` compares the full profile (Medium, design smell)

**Where:** `whack26/classify.py:247-279`; `whack26/profile.py:95-105`

`stored_profile != profile` compares the full pydantic model, including `source_path` and `warnings`. Re-reading the same data from a different path, or any warning-text change, invalidates the cache. The returned `FieldPath` also sets `refinement_reason=None`.

**Fix direction:** Key the cache on a stable content identity (SHA-256 + schema + relevant settings), not the full profile. If warnings are intentionally part of identity, document why.

**Tests:** Same content at a different path reuses the path; changed warning text does not invalidate; changed schema does invalidate.

## CLASS-02 — Fallback root is banned by the eval gate (Medium)

**Where:** `whack26/classify.py:692-699`; `eval/field_classification.py:706-718`

The classifier's invalid-response fallback creates a root named `"General Research"`. `banned_roots` includes `"general research"` after normalization. Any run that hits the fallback is guaranteed to fail the naming gate.

**Fix direction:** Pick one: remove `"general research"` from `banned_roots`, rename the fallback, or treat the fallback as an explicit eval failure with a clear reason. Coordinate with the eval work package.

**Tests:** Fallback path passes the naming gate, or is explicitly reported as a fallback failure.

---

# Work Package 3 — Profiling and factor mapping

**Files:** `whack26/profile.py`, `whack26/factors.py`, `tests/test_profile.py`, `tests/test_factors.py`
**Suggested subagent:** one owner for both files, because target-role and factor-mapping semantics interact.

## PROF-01 — LLM target choice accepts id/time/leakage columns (High)

**Where:** `whack26/profile.py:387-393`

The LLM's target is accepted if it is any string in `frame.columns`. The prompt asks the model to avoid id/time/leakage, but nothing enforces it.

**Fix direction:** After parsing the LLM choice, run `_column_role` and reject `id`, `time`, and `leakage_risk` targets; fall back to the heuristic target or ask again with a constrained prompt.

**Tests:** LLM returns `row_id`, `order_date`, `outcome_next_week`; assert rejection and fallback.

## PROF-02 — Target role overrides id/time (High)

**Where:** `whack26/profile.py:234-239`

```python
if target is not None and name == target:
    return "target"
if _looks_like_id(...):
    return "id"
if _looks_like_time(...):
    return "time"
```

A column named `id_target` becomes the target rather than an identifier. This compounds PROF-01.

**Fix direction:** Check id/time first, or reject a target that also matches id/time. Keep the role mutually exclusive and explain the precedence.

**Tests:** `id_target`, `date_target`, `outcome_target`; assert id/time wins or the target is rejected.

## PROF-03 — `_looks_like_id` over-flags high-cardinality columns (Medium)

**Where:** `whack26/profile.py:207-219`

Any column with `cardinality / rows >= 0.98` and an integer **or string** dtype is an id. A high-cardinality integer measure (price, count) or a near-unique free-text column is misclassified.

**Fix direction:** Require a name hit or a stronger structural signal (monotonic integer sequence, UUID/email pattern, exact uniqueness with no repeats). Do not use dtype alone.

**Tests:** Near-unique integer price; near-unique string category; real id.

## PROF-04 — Sensitive-name token false positives (Low)

**Where:** `whack26/profile.py:535-545`

Tokens ending in `"id"` with length > 2 match `grid`, `valid`, `solid`.

**Fix direction:** Match whole tokens (`id`, `ids`, `uuid`, `guid`, `key`, `index`) or known prefixes/suffixes, not any token ending in `"id"`.

**Tests:** `grid`, `valid`, `solid` are not sensitive; `user_id` is.

## FACT-01 — Verification can overwrite an exact mapping (High)

**Where:** `whack26/factors.py:287-335`

`unresolved` is what the prompt shows, but `by_name` is built from all `factors`, and the loop does not check that the mapped factor is unresolved. An off-prompt LLM response naming an already exact-mapped factor overwrites it; setting `in_dataset=False` also clears `mapped_column`.

**Fix direction:** Build `by_name` from `unresolved` only, or reject any mapping for a factor not in the prompted set. If an off-prompt response should be honored, require an explicit reason and do not clear an exact mapping without evidence.

**Tests:** Exact mapping followed by an off-prompt verification naming the same factor; assert the exact mapping survives.

## FACT-02 — `_singular("houses")` → `"hous"` (Medium)

**Where:** `whack26/factors.py:168-175`

```python
if len(token) > 4 and token.endswith("ses"):
    return token[:-2]
```

`"houses"` → `"hous"`; singular `"house"` is unchanged, so plural and singular stop matching.

**Fix direction:** Handle `-ses` → `-s`/`-` correctly, or use a small stemming library. Add a table of common cases.

**Tests:** `house/houses`, `class/classes`, `analysis/analyses`, `bus/buses`.

## FACT-03 — Exact-mapping precedence (Low)

**Where:** `whack26/factors.py:209-216`

The order is: exact name match (1.0), `column_tokens <= name_tokens` (0.9), then description (0.75). A short column name can beat a longer, more specific description match.

**Fix direction:** Score name and description independently and choose the best, or require the name match to be exact before using the subset rule.

**Tests:** Short column name vs. specific description; assert the intended winner.

---

# Work Package 4 — Research pipeline

**Files:** `whack26/research/cache.py`, `base.py`, `wikipedia.py`, `normalize.py`, `pipeline.py`, and their tests
**Suggested subagents:** cache/base, Wikipedia, normalize/rank, pipeline cache key.

## RES-01 — Cache truncates bodies at 512K characters with no marker (Medium)

**Where:** `whack26/research/cache.py:13, 185`

```python
MAX_BODY_CHARS = 512 * 1024
stored_body = body[:MAX_BODY_CHARS]
```

Truncated JSON fails to parse on an offline warm cache; online it silently refetches. There is no marker, so callers cannot tell the body is incomplete.

**Fix direction:** Store the full body, or store a marker and a hash; on read, detect truncation and fail clearly offline / refetch online with a warning. Consider compressing or chunking large bodies.

**Tests:** Body > 512K with valid JSON; offline read either succeeds or raises a clear "truncated" error; online refetch is logged.

## RES-02 — Cached status is never checked (Low/latent)

**Where:** `whack26/research/base.py:223-251`; `whack26/research/cache.py:158-169`

`CachedResponse.status` is stored but no caller reads it; `base.search` parses `cached.body` regardless. In practice adapters cache only after `raise_for_status`, so the stored status is normally success. Impact is limited to future adapters that cache errors.

**Fix direction:** Check `cached.status` before parsing; if it is not 2xx, treat it as a cached failure (raise or skip) rather than parsing the body. Add a regression test with a cached 500.

**Tests:** Cache a 500 body; assert it is not parsed as success.

## RES-03 — Wikipedia offline partial cache aborts the whole source (Medium)

**Where:** `whack26/research/wikipedia.py:44-50, 127-169`

On a cache hit, `_parse_summary` is tried. On failure, offline returns `None`; otherwise it falls through to the network fetch. If there is no cache entry at all, `_summary_record` raises `SourceError` (145-148), and `search` does not catch it per title. A partially warm cache therefore aborts all Wikipedia records.

**Fix direction:** Skip a missing summary offline (return `None`) and continue with the other titles; report a per-title warning. If the whole source must fail, make that explicit and test it.

**Tests:** Seed one summary key, leave another missing; offline search returns the available record(s) and warns.

## RES-04 — Dedupe ranks source before abstract length (Medium)

**Where:** `whack26/research/normalize.py:135-163`

`_winner_sort_key` is `(-SOURCE_RANK, -len(abstract), id)`. OpenAlex outranks arXiv, so an OpenAlex record with a shorter abstract wins. `_merge_group` only fills the abstract when the winner has none (`if not abstract and loser.abstract`).

**Fix direction:** Rank by metadata quality first (abstract length, authors, year, DOI), with source as a tiebreaker. Merge the best abstract even when the winner has one but a loser has a longer/better one, or document why source wins.

**Tests:** OpenAlex short abstract vs. arXiv long abstract; assert the longer abstract is retained or the chosen policy is explicit.

## RES-05 — Embedding failure silently degrades to zero similarity (Medium)

**Where:** `whack26/research/normalize.py:298-318`

```python
except EmbeddingError:
    similarities = [0.0] * len(items)
```

Ranking continues with zero similarity, so a total embedding outage looks like poor relevance. There is no warning.

**Fix direction:** Surface a warning/error and either fail the ranking or fall back to a documented lexical score. Do not silently pretend the embeddings were computed.

**Tests:** `FailingEmbeddings`; assert a warning or error, not a silent zero-similarity ranking.

## RES-06 — Query cache key omits the field path (Medium)

**Where:** `whack26/research/pipeline.py:230-242`

The comment says the field path is deliberately excluded; the key is dataset id + summary hash. A query cached before reclassification is reused afterwards, even if the new field path implies different queries.

**Fix direction:** Include the field path (or a classification version/hash) in the key, or invalidate the query cache when the path changes. Document the tradeoff if reuse is intentional.

**Tests:** Same dataset/summary with two field paths; assert different query keys or explicit invalidation.

---

# Work Package 5 — Evaluation harness

**Files:** `eval/field_classification.py`, `tests/test_eval_harness.py`
**Suggested subagent:** one owner.

## EVAL-01 — Gate bans the classifier's own fallback (High)

**Where:** `eval/field_classification.py:706-718`; `whack26/classify.py:692-699`

`banned_roots` includes `"general research"`; the classifier fallback root is `"General Research"`; `_normalize` lowercases. Any fallback run fails the naming gate by construction.

**Fix direction:** Remove `"general research"` from `banned_roots`, rename the fallback, or mark fallback runs as an explicit eval failure with a distinct status. Coordinate with CLASS-02.

**Tests:** Fallback root is not banned; a genuinely banned root still fails.

## EVAL-02 — Near-duplicate root gate tolerates one pair (Medium)

**Where:** `eval/field_classification.py:616-621`

```python
"pass" if near_pairs <= 1 else "fail"
```

Exactly two near-duplicate roots pass.

**Fix direction:** Decide the tolerance explicitly. If zero near-duplicates is the goal, use `near_pairs == 0`. If one is allowed, document why.

**Tests:** Two near-duplicate roots → fail (or the documented pass); three → fail.

## EVAL-03 — Naming gate uses plain substring matching (Medium)

**Where:** `eval/field_classification.py:720-724`

```python
if _normalize(label) in normalized_node:
```

A forbidden label like `e0.csv` flags any node merely containing it, not just the dataset proper noun.

**Fix direction:** Match whole tokens or known proper-noun patterns; distinguish "contains the string" from "is the dataset name."

**Tests:** A node containing a forbidden substring incidentally; assert no false positive.

## EVAL-04 — Human rubric and automated gates disagree (Medium)

**Where:** `eval/field_classification.py:760-789, 845-849`

The template says a dataset passes when root and path are both at least 3. The gates require 80% of roots/paths at least 3, a mean at least 3.5, plus a naming mean at least 4.0.

**Fix direction:** Align the template and gates, or document the stricter gate as an additional requirement. Make the pass criteria a single source of truth.

**Tests:** A dataset at the rubric threshold; assert the documented outcome.

## EVAL-05 — `evaluate_naming` has no test (Medium)

**Where:** `eval/field_classification.py:699, 1268`; `tests/test_eval_harness.py:9-17`

The test module imports `evaluate_automated` and `evaluate_human`, not `evaluate_naming`.

**Fix direction:** Add tests for banned roots, forbidden labels, substring false positives, and the fallback root.

---

# Work Package 6 — Ingestion, config, doctor, packaging

**Files:** `whack26/io.py`, `whack26/config.py`, `whack26/doctor.py`, `requirements.txt`
**Suggested subagent:** one owner.

## IO-01 — Duplicate-header restoration only works when widths match (Low)

**Where:** `whack26/io.py:117-126`

```python
if raw_header is None or len(raw_header) != frame.shape[1]:
    return
```

Ragged rows are left mangled with no warning.

**Fix direction:** Reconcile widths explicitly (pad/truncate with a warning) or raise `IngestError` when the header cannot be restored.

**Tests:** Ragged CSV; assert a clear warning/error.

## IO-02 — JSON retry conflates schema and layout errors (Low)

**Where:** `whack26/io.py:41-51`

The first parse catches every exception and retries the other layout. Both error strings are preserved in the final `ValueError`, so the message is not lost, but a schema error is treated as a layout error.

**Fix direction:** Distinguish `JSONDecodeError` (layout) from schema/type errors; retry only on the former.

**Tests:** Valid JSON with the wrong shape; assert the schema error is surfaced directly.

## CFG-01 — Weights are not validated (Low)

**Where:** `whack26/config.py:136-150`

`desc_weight` and `centroid_weight` are not checked for range or sum. A typo silently skews similarity scoring.

**Fix direction:** Require `0 <= weight <= 1` and `desc_weight + centroid_weight ≈ 1` (or document that they are independent).

**Tests:** Negative, >1, and non-summing weights; assert a clear error.

## DOC-01 — Python check requires exactly 3.12 (Low)

**Where:** `whack26/doctor.py:20, 46-56`

`sys.version_info[:2] == (3, 12)` fails on 3.13 even if the code works.

**Fix direction:** Use a supported range (e.g., `(3, 12) <= version < (3, 14)`) or check against `requires-python`.

**Tests:** Simulate 3.13; assert PASS if supported, WARN/FAIL with a clear message if not.

## DOC-02 — `_version_of` fallback is eager (Low)

**Where:** `whack26/doctor.py:69`

```python
getattr(imported, "__version__", _version_of(module) or "?")
```

Python evaluates the default argument before `getattr`, so `_version_of` runs even when `__version__` exists.

**Fix direction:** Use an explicit `getattr` check or `try/except AttributeError`.

**Tests:** Module with `__version__`; assert `_version_of` is not called (spy).

## DOC-03 — NaN embedding passes (Low)

**Where:** `whack26/doctor.py:224-233`

`abs(norm - 1.0) > 1e-3` is false for `norm = NaN`, so both the normalize and FAIL branches are skipped.

**Fix direction:** Check `math.isfinite(norm)` before the tolerance check; fail on NaN/inf.

**Tests:** NaN vector; assert FAIL.

## DEP-01 — `PyQt6` is missing from requirements (Medium)

**Where:** `requirements.txt:1-19`; `UI_frontEnd.py:5`

The GUI cannot run from a clean install.

**Fix direction:** Add `PyQt6` to an optional extra or a separate `requirements-ui.txt`; document the install.

**Tests:** Packaging smoke test / docs check.

---

# Work Package 7 — Test hygiene

**Files:** `tests/`, `stats_test.py`, optional `pytest.ini` / `pyproject.toml`
**Suggested subagent:** one owner, after the other packages land.

## TEST-01 — `stats_test.py` is collected by pytest (Medium)

**Where:** `stats_test.py:1-2`; pytest default `python_files = ["test_*.py", "*_test.py"]`

The file is a script, not a test module. It imports numpy/scipy at top level and defines no test functions. `stats_test.py:4-5` sets `__test__ = False`, but pytest still imports the module to read that flag; it does not prevent a collection-time import error. Under `.venv` it imports cleanly and contributes nothing (247 tests collected; 247 passed). Under an interpreter missing numpy/scipy it still produces a collection error, and pytest's default behavior is to abort the whole run with `Interrupted` and exit 2 before any test executes. `--continue-on-collection-errors` is the only escape hatch. Renaming or excluding the file is the robust fix.

**Fix direction:** Rename the file (e.g., `stats_engine.py`), move it out of the repo root, or add a pytest config that excludes it. If the file is meant to be a demo, put it under `demo/`.

**Tests:** `pytest --collect-only` under `.venv`; assert no unintended module is collected.

## TEST-02 — Tests that encode bugs

| Test | Issue |
| --- | --- |
| `tests/test_store.py:280-307` | Commits mismatched `path_json`/`field_ids` and asserts success. |
| `tests/test_research_base.py:145-160` | Asserts 512K truncation is acceptable. |
| `tests/test_research_normalize.py:115-148` | Tests abstract back-filling, not the shorter-abstract preference; the code bug is untested. |
| `tests/test_research_normalize.py:416-446` | Encodes the silent zero-similarity fallback. |
| `tests/test_eval_harness.py:9-17` | Never imports/calls `evaluate_naming`. |
| `tests/test_classifier.py` mapping test | Only tests an invalid column name; the caller pre-filters columns. |
| "Warm classification reuses existing path" test | Does not actually verify cache reuse. |
| `tests/test_research_wikipedia.py` | Tests cold, warm, and offline miss; missing a partially warm cache. |

**Fix direction:** Update each test to assert the intended behavior, and add the missing regression tests listed in the work packages.

---

## Retracted / corrected findings

These were in earlier review drafts and are **not** valid as stated. Do not chase them.

| Earlier claim | Correction |
| --- | --- |
| `classify.py:387` create path can name the wrong parent chain | Retracted. `_create_field` is called with `parent=current` and searches `children(current.id)` / `find_sibling(current.id, ...)`; any merge target is a child of `current`, and `current = path.fields[-1]`. The path is valid. |
| `monte_carlo` uses a `-1.0` sentinel | Retracted for `monte_carlo`; it returns a p-value in `(0, 1]`. Only `oos_decay` uses `-1.0`. |
| `test_research_normalize.py:115-148` encodes a shorter-abstract preference | Retracted. The test gives OpenAlex `abstract=None` and asserts arXiv's abstract is back-filled. The code preference is real but untested. |
| `test_research_wikipedia.py` only covers fully-cold or fully-warm | Partially retracted. It does test an offline cache miss; what is missing is a *partially* warm cache. |
| `min_sharpe_limit` is never read | Corrected again after `d74836a`: `pss` now skips it, and the engine reads `min_dsr_limit` instead. It is still not used as the DSR threshold (VAL-04). |
| The 400/`thinkingConfig` retry doubles requests per call | Downgraded. Only the first rejected call re-posts once; later calls send one POST. Not a finding. |
| A collection error always prevents the suite from running | Confirmed for the default configuration: pytest reports all collection errors, then interrupts before the run loop (rc=2). `--continue-on-collection-errors` allows the rest to run (rc=1). |
| `importlib.util.find_spec` returns `False` for a missing module | Corrected: it returns `None`; `bool(None)` is `False`. |
| numpy/scipy are absent in this workspace | Corrected: they are present in `.venv` (2.5.4 / 1.18.1). The 27-error run used the system Python 3.15, not `.venv`. |
| UI `actuals = predictions` | Fixed in `d74836a`: `actuals = targets`, `predictions = np.where(...)`. Replaced by VAL-02/VAL-03 (metric/target mismatch). |
| `max_decay` not forwarded | Fixed in `d74836a`: forwarded as `base_params["max_decay_tolerance"]` and used by `oos_decay`. |
| Import fallback returns `True` | Fixed in `d74836a`: direct import, hard error. |
| `oos_decay` `-1.0` sentinel | Fixed in `d74836a`: returns a dict with `success`/`error`. |
| `pss` integer perturbation can skip | Fixed in `d74836a`: adjusts by ±1 when rounding collapses. |
| `pss` all-exceptions pass | Partially fixed: fails when all trials fail, but still passes with zero trials or partial failures (VAL-07/VAL-08). |

---

## Suggested parallelization

| Wave | Work packages | Notes |
| --- | --- | --- |
| 1 | VAL-04/06/07/08/09/10 (engine), STORE-01/02, PROF-01/02/03/04, FACT-01/02/03, RES-01/03/04/05/06, EVAL-01..05, IO-01/02, CFG-01, DOC-01..03 | Independent files; can run in parallel. |
| 2 | VAL-01/02/03/05/11/12 (GUI/target contract), CLASS-01/02, RES-02, DEP-01 | Depends on wave 1 decisions (engine API, target contract, fallback policy, cache status policy). |
| 3 | TEST-01/02, integration tests, full-suite run | After code changes settle. |

**Conflict notes:**
- `whack26/store.py` and `whack26/classify.py` share the path/ID invariant; coordinate STORE-01/02 with CLASS-02.
- `whack26/profile.py` and `whack26/factors.py` share target/role semantics; one owner is safer.
- `eval/field_classification.py` and `whack26/classify.py` share the fallback-root decision; coordinate EVAL-01 with CLASS-02.
- `stats_test.py` and `UI_frontEnd.py` share the engine API; define the API before wiring the GUI.

---

## Copy-paste subagent briefs

Each brief is self-contained. Replace `<repo>` with the workspace path.

### Brief A — Validation engine

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix VAL-04, VAL-06, VAL-07, VAL-08, VAL-09, and VAL-10 in `stats_test.py`. Use `.venv\Scripts\python.exe` for all commands. Add tests under `tests/test_validation_engine.py`. Do not modify `UI_frontEnd.py` in this task; keep the public function names unless a rename is necessary for the fix, and report any API change. Acceptance: `min_sharpe`/`min_dsr_limit` is the DSR threshold; NaN and inf fail every layer; `oos_decay` rejects non-finite OOS scores; `pss` fails with zero trials or partial failures; `model_predict_func` is removed or used; `max_decay` clipping is reported; no `-1.0` sentinel. Run targeted tests and the full suite under `.venv`.

### Brief B — GUI wiring

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix VAL-01, VAL-02, VAL-03, VAL-05, VAL-11, and VAL-12 in `UI_frontEnd.py`, `requirements.txt`, and `stats_test.py` (filename only). Use `.venv\Scripts\python.exe`. The goal is a real validation path or an explicitly labeled demo: the builder must accept `(params, X, y)`; actuals and predictions must use the same label space; Layer 1 and Layer 2 must use compatible target types; `min_sharpe` must reach `min_dsr_limit`; trial Sharpes must not be fabricated; `PyQt6` must be declared; rename/exclude `stats_test.py` if needed. Coordinate the engine API with Brief A. Acceptance: a fake engine records all arguments; the UI builder works with the engine; a continuous target does not silently fail Layer 2; a clean install can launch the GUI.

### Brief C — Store and classification

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix STORE-01/02/03, CLASS-01, and CLASS-02 in `whack26/store.py`, `whack26/classify.py`, and their tests. Use `.venv\Scripts\python.exe`. Decide and document the `path_json`/`field_ids` invariant; reject mismatches; raise on missing fields; key the classification cache on stable content identity; align the fallback root with the eval gate. Acceptance: mismatch/missing-field tests fail before the fix and pass after; cache reuse works across paths; the fallback root is not banned. Coordinate CLASS-02 with Brief E.

### Brief D — Profile and factors

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix PROF-01..04 and FACT-01..03 in `whack26/profile.py`, `whack26/factors.py`, and their tests. Use `.venv\Scripts\python.exe`. Reject id/time/leakage targets; make roles mutually exclusive; tighten id heuristics; fix sensitive-token matching; restrict verification to unresolved factors; fix `_singular`; document exact-mapping precedence. Acceptance: tests for `row_id`, `order_date`, `outcome_next_week`, `grid`, `valid`, `solid`, `houses`, and an off-prompt verification overwrite.

### Brief E — Research pipeline

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix RES-01..06 in `whack26/research/` and its tests. Use `.venv\Scripts\python.exe`. Decide the cache-truncation policy (full body or explicit marker); check cached status; make Wikipedia skip per-title misses offline; rank by metadata quality before source; surface embedding failures; include the field path or a classification version in the query cache key. Acceptance: tests for >512K bodies, cached 500s, partially warm Wikipedia caches, better-abstract dedupe, embedding failure warnings, and reclassification cache invalidation.

### Brief F — Eval harness

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix EVAL-01..05 in `eval/field_classification.py` and `tests/test_eval_harness.py`. Use `.venv\Scripts\python.exe`. Align the banned-root list with the classifier fallback; decide the near-duplicate tolerance; replace substring naming checks with token/proper-noun matching; make the rubric and gates agree; add `evaluate_naming` tests. Coordinate EVAL-01 with Brief C. Acceptance: fallback root is handled intentionally; two near-duplicate roots fail (or the documented policy); substring false positives are tested; rubric and gate outcomes match.

### Brief G — Ingestion, config, doctor, packaging

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix IO-01/02, CFG-01, DOC-01/02/03, and DEP-01. Use `.venv\Scripts\python.exe`. Handle ragged duplicate headers explicitly; separate JSON schema errors from layout errors; validate similarity weights; accept a supported Python range; make `_version_of` lazy; reject NaN embeddings; declare `PyQt6` in an optional requirements file. Acceptance: targeted tests for each item; full suite under `.venv`.

### Brief H — Test hygiene

> Work in `<repo>`. Read `docs/agents/logical-flaws-review.md`, then fix TEST-01/02. Use `.venv\Scripts\python.exe`. Rename or exclude `stats_test.py` from pytest collection; update the tests that encode known bugs; add the missing regression tests from Briefs A–G. Do not weaken assertions to make the suite green. Acceptance: `pytest --collect-only` collects only intended tests; the full suite runs under `.venv`; every fixed bug has a regression test.

---

## Final note

The highest-value change is still making the validation path honest. The pull fixed the self-referential `actuals = predictions`, the import fallback, the `max_decay` plumbing, and the `-1.0` sentinel, but the UI and engine now disagree on the builder signature, the target type, and the threshold name. Until those are fixed, the dashboard will reject every run (or pass a NaN OOS score) rather than validate a strategy.
