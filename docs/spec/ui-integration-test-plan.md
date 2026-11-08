# UI-to-system integration test plan

Status: **implemented and green**. Every test slice in this plan exists, and
the full suite passes:

- **1432 passed, 2 skipped, 13 xfailed, 0 failed** in 437.15s on 2026-11-08.
- New modules: `tests/test_ui_widgets.py`, `tests/test_ui_dataset.py`,
  `tests/test_ui_validation.py`, `tests/test_ui_report.py`,
  `tests/test_ui_pipeline.py`, `tests/test_ui_optional_stages.py`,
  `tests/test_ui_async.py`, `tests/test_ui_security.py`.
- New shared infrastructure: `tests/ui_fixtures.py`,
  `tests/test_ui_fixtures.py`, the collection hook in `tests/conftest.py`,
  `requirements-ui-test.txt`, and the `integration` marker in `pyproject.toml`.
- pytest-qt 4.5 is installed in the development `.venv`; PyQt6 is installed.
  The existing `tests/test_ui_frontend.py` still runs without pytest-qt.

The 13 xfails are documented production gaps, not silent passes:

- 6 async UI tests (UI-90 to UI-95): `run_validation` is synchronous; there is
  no worker thread, streamed progress, busy state, or cancel control.
- 2 pandas 3 `StringDtype` tests (UI-14/UI-19): `np.issubdtype` raises
  `TypeError` for text columns in `_prepare_arrays`; use
  `pandas.api.types.is_numeric_dtype`.
- 2 optional-stage tests (UI-82/UI-83): the fragment renderer has no
  generation-specific or collection-specific section.
- 1 failed-judge-stage test (UI-79): `load_final_judge` raises instead of
  rendering a degraded state.
- 2 credential-redaction tests (UI-99): the UI has no redaction seam.

The two skips are unrelated optional-backend skips in
`tests/test_model_search_optional_backends.py`.

## Goal

Prove that the optional PyQt6 dashboard can be connected to the rest of
`whack26` without weakening the core slice:

1. accept the same datasets the pipeline accepts;
2. run the pipeline (mock/offline in CI, live manually) or attach to a completed
   run directory;
3. display the final judge's UI fragment, `final_judge.fragment.html`;
4. keep per-dataset tabs isolated;
5. surface failures in the activity log without crashing the window;
6. skip cleanly on machines that do not install the optional UI stack.

The payoff is one tested handoff instead of two parallel validation paths: the
CLI writes the run directory, and the dashboard renders its report.

## Definition of connected

A run is **connected** when all of the following are true:

- `python -m whack26 run ... --judge --mock --offline` exits `0` and writes
  `artifact.json`, `final_judge.json`, `final_judge.html`, and
  `final_judge.fragment.html`.
- `validate_artifact` accepts the artifact, including the optional
  `final_judge` section and its relative `report_path`/`fragment_path`.
- `QuantDashboard.load_final_judge(run_dir, tab_ref)` renders the fragment in
  the selected tab's `HtmlSummaryView`.
- The rendered panel shows the decision label, the critic count, and the model
  summary when present.
- A second tab can load a different run directory without cross-contamination.
- A missing or unreadable fragment produces an actionable UI error and leaves
  the previous report intact.

## Current seams and gaps

| Seam | Producer | Consumer | Contract today | Gap |
|---|---|---|---|---|
| Dataset input | `whack26.io.read_dataset` | `DatasetWorkspace.handle_dropped_file` | CSV/XLS/XLSX, last column is the target | UI reads with pandas directly instead of `read_dataset` |
| Legacy validation | `stats_engine.validation_engine` | `DatasetWorkspace.run_validation` | Continuous return-like target, R², chronological split | Remains synchronous and separate from the pipeline; no artifact |
| End-to-end run | `whack26.run.run_pipeline` | `DatasetWorkspace.start_pipeline` via `_PipelineTask` | Background thread, streamed log lines, busy state, final-judge fragment on completion | No cancel control; `run_validation` still bypasses it |
| Final judge files | `whack26.final_judge.report.write_reports` | `DatasetWorkspace.load_final_judge_report` | Reads `final_judge.fragment.html`; `write_reports` can emit a format subset | No artifact check; a `judge --format` subset that omits the fragment raises `FileNotFoundError` |
| Final judge routing | `QuantDashboard.load_final_judge` | tab workspace | Tab index/workspace/`None` | Index/workspace/detached refs are now covered; keep the guard |
| Optional stages | `model_search`, `variants`, `validate_generation`, collection | final judge evidence bundle | Report sections when enabled | Generation and collection-specific sections are still absent from the fragment |
| Artifact schema | `artifact_json_schema()` | `schemas/artifact.schema.json` | Schema 1.5.0 includes optional `final_judge` and `field_pitfalls`; checked-in file equals the model schema | Keep the equality test as a regression guard |
| CLI | `run --judge`, `judge` | run directory | `run --judge` writes all four files; `judge --format` may write a subset | UI-74 now covers the `run --judge` path |

Current implementation anchors (symbols, not line numbers, so the plan survives
refactors):

- `UI_frontEnd.py`: `QuantDashboard`, `DatasetWorkspace`, `HtmlSummaryView`,
  `_PipelineSignals`, `_PipelineTask`, `start_pipeline`,
  `load_summary_html`, `load_final_judge_report`, `run_validation`,
  `set_report`, `load_final_judge`.
- `whack26/final_judge/report.py`: `render_fragment`, `render_document`,
  `write_reports` (accepts `formats=`, default all four).
- `whack26/run.py`: `run_pipeline`, the final-judge stage, and
  `RunOutcome.final_judge`.
- `whack26/cli.py`: `run --judge`, the `judge` subcommand, and `_cmd_judge`.
- `whack26/config.py`: `final_judge_enabled` and related settings.
- `docs/spec/final-judge.md`: the implemented final-judge contract.

Existing test inventory: `tests/test_final_judge_aggregate.py`,
`tests/test_final_judge_checks.py`, `tests/test_final_judge_evidence.py`,
`tests/test_final_judge_pipeline.py`, `tests/test_final_judge_integration.py`,
and `tests/test_final_judge_report.py` cover the final-judge backend, the
`run_pipeline(..., final_judge=True)` handoff, the `judge` subcommand, report
format subsets, and all decision banners. `tests/test_ui_pipeline.py` now covers
the `python -m whack26 run --judge` CLI path (UI-74), the richer decision/count/
model-summary rendering, and two-run tab isolation.
`tests/test_ui_auto_pipeline.py` covers the drop-to-pipeline auto-run and
background-thread handoff. `tests/test_run_flow.py` covers the mock pipeline.
`tests/test_ui_frontend.py` covers the workspace fragment load, the
missing-fragment error, and current-tab dashboard routing. The remaining UI gaps
are the legacy synchronous `run_validation` path, pandas 3 `StringDtype`
handling, generation/collection rendering, a failed-judge degraded state, and
credential redaction.

## Test strategy

Use the cheapest level that can catch the defect. Do not use pixel screenshots:
the theme is not a contract, and Qt rendering differs by platform.

| Level | What it proves | Needs Qt | Needs keys/network | Runs in default suite |
|---|---|---|---|---|
| L0 static/import | Optional dependency boundary is honest | no | no | yes (skips) |
| L1 backend contract | final judge files are UI-safe and stable | no | no | yes |
| L2 widget unit | tabs, routing, HTML loading, errors | yes | no | no (`ui` marker) |
| L3 renderer-to-widget | real fragment renders in a real widget | yes | no | no (`ui` marker) |
| L4 pipeline-to-widget | mock/offline run reaches the panel | yes | no | no (`ui` + `integration`) |
| L5 CLI-to-widget | CLI output is consumable by the UI | yes | no | no (`ui` + `integration`) |
| L6 live smoke | real providers produce a usable report | yes | yes | manual only |

The table has seven levels: **L0** is the optional-dependency boundary, and
**L1-L6** are the functional levels. Any shorthand count of "six levels" refers
to L1-L6 only.

### Optional features this plan intentionally uses

The UI and the final judge are both optional, so the tests must treat them as
optional too:

- **pytest-qt** for widget interaction, signal waiting, and teardown. Do not
  hand-roll `QApplication` loops when the plugin is available.
- **PyQt6 from `requirements-ui.txt`**. It is not in the core
  `requirements.txt`, so a clean core environment skips UI tests and stays
  green. The current development `.venv` has PyQt6 installed; a clean install
  does not.
- **`QT_QPA_PLATFORM=offscreen`** so the UI job runs headless on CI.
- **`--mock`** for deterministic LLM/embedding fixtures. The mock final judge
  deliberately returns `insufficient_evidence`; this is the correct "not a
  pass" fixture.
- **`--offline`** so no test touches the network or provider quotas.
- **`--judge` / `judge`** as the opt-in final-judge stage under test.
- **`--model-search`, `--variants`, `--validate-generation`, and
  `--collection-json`** to exercise the optional report sections.
- **Module-scope `pytest.importorskip`** plus a
  `pytest_collection_modifyitems` hook so missing extras skip at collection
  time rather than fail. The hook checks PyQt6 only. Modules that specifically
  need pytest-qt add `pytest.importorskip("pytestqt")`; this keeps the existing
  PyQt6-only `tests/test_ui_frontend.py` runnable without pytest-qt. Do not put
  `importorskip` inside a `qtbot` fixture: pytest resolves `qtbot` before
  running the fixture body.
- **A `ui` marker** so the hook can target only the optional tests.
- **`pytest.mark.integration`** for the mock pipeline tests. Add it to the
  existing `[tool.pytest.ini_options]` marker list in `pyproject.toml`; the
  `ui` marker is already registered there, and the plan does not depend on an
  external issue landing first.

Supported dependency combinations:

| PyQt6 | pytest-qt | Result |
|---|---|---|
| absent | absent | All `ui` tests skip; core suite stays green |
| present | present | All UI tests run |
| present | absent | `qtbot`-based tests skip; PyQt6-only tests such as `tests/test_ui_frontend.py` still run |
| absent | present, no binding supported by the installed pytest-qt version | **Unsupported**: pytest-qt aborts at configure before the hook runs. Install PyQt6 or uninstall pytest-qt. |
| absent | present, PySide6 or PyQt5 installed | pytest-qt configures, but the collection hook skips `ui` tests because `UI_frontEnd.py` requires PyQt6 |

The abort is triggered by the absence of a Qt binding that the installed
pytest-qt version supports. pytest-qt 4.5.0 (the current resolution of
`>=4.4`) supports PyQt6, PySide6, and PyQt5; it dropped PySide2. A PySide2-only
machine therefore aborts at configure on a fresh install. This repository's UI
module is PyQt6-only, so treat PyQt6 + pytest-qt as the only fully supported
combination. `requirements-ui-test.txt` installs both; the base
`requirements.txt` installs neither. Never install pytest-qt on its own.

If an optional feature is unavailable, tests that need it **skip with a reason**;
they never pass silently and never fail the core suite. PyQt6-only tests remain
runnable when pytest-qt is absent.

## Environment and fixtures

### Test dependencies

Add a separate file; do not put the UI test stack in `requirements.txt`.

```text
# requirements-ui-test.txt
-r requirements-ui.txt
pytest-qt>=4.4
```

Install with:

```powershell
.\.venv\Scripts\pip.exe install -r requirements-ui-test.txt
```

Use two clean CI environments so the unsupported pytest-qt-without-PyQt6
combination cannot occur:

| Job | Install | Run |
|---|---|---|
| Core | `requirements.txt` only | full suite; UI tests skip |
| UI | `requirements-ui-test.txt` | full suite; UI tests run |

Never install pytest-qt in the core job. It is not enough to rely on
`pytest.importorskip` there, because pytest-qt fails during its own configure
step if no Qt binding supported by that pytest-qt version is present.

### Pytest configuration

`pyproject.toml` already registers `optional`, `ui`, `slow`, and `live`. Add the
missing `integration` marker to that list; do not create a separate
`pytest.ini`.

```toml
[tool.pytest.ini_options]
markers = [
  "optional: requires the optional model-search dependencies",
  "ui: requires PyQt6 and a headless Qt platform",
  "integration: runs a full mock/offline pipeline",
  "slow: long-running test",
  "live: requires network access or live provider credentials",
]
```

Only the `integration` line is new; the rest already exists.

### Shared fixtures

Create `tests/ui_fixtures.py` and reuse the existing fixture style from
`tests/test_run_flow.py` and `tests/artifact_fixtures.py`.

```python
def make_ui_dataset(path: Path, rows: int = 80, seed: int = 0) -> Path:
    """Small dataset whose last column is a continuous target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    z = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": ["a", "b"] * (rows // 2),
            "y": 2.0 * x - z + rng.normal(scale=0.2, size=rows),
        }
    )
    frame.to_csv(path, index=False)
    return path
```

```python
def make_report(
    decision: str = "insufficient_evidence",
    **overrides,
) -> FinalJudgeReport:
    """Build a report without invoking an LLM."""
    values = {
        "generated_at": "2026-11-08T10:00:00Z",
        "decision": decision,
        "n_critics_evaluated": 0,
        "min_required": 5,
        "unevaluated_critics": ["overfitting_skeptic"],
        "note": "The audit did not complete; this is NOT a pass.",
    }
    values.update(overrides)
    return FinalJudgeReport(**values)
```

```python
# tests/conftest.py
# Keep the existing sys.path setup in this file; add the hook below it.
import importlib.util
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_collection_modifyitems(config, items):
    """Skip UI tests when PyQt6 itself is unavailable."""
    if importlib.util.find_spec("PyQt6") is not None:
        return
    reason = "optional UI stack not installed: PyQt6"
    for item in items:
        if "ui" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))
```

```python
# tests/test_ui_widgets.py (qtbot-based UI test modules start this way)
import pytest

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from UI_frontEnd import QuantDashboard


@pytest.fixture
def dashboard(qtbot):
    widget = QuantDashboard()
    qtbot.addWidget(widget)
    return widget
```

The PyQt6 guard is required for every UI module. The pytest-qt guard is
additional and only needed by modules that use `qtbot`; this is why the
collection hook checks PyQt6 only. The module-level guard matters because pytest
resolves the `qtbot` fixture before executing the fixture body, so an
`importorskip` inside `dashboard(qtbot)` is too late. The unsupported
"pytest-qt without any binding it supports" combination still aborts inside
pytest-qt's own `pytest_configure`, before either guard runs; keep the UI stack
all-or-nothing in the requirements files.

`tests/conftest.py` already exists for `sys.path` setup; add the hook to it
instead of replacing it. `tests/test_ui_frontend.py` already uses the
module-level PyQt6 guard and runs without pytest-qt.

Keep the old `tests/artifact_fixtures.py` as the artifact factory; extend it with
an optional `final_judge=` argument rather than creating a second artifact
builder. Promote the existing `make_report` helper from
`tests/test_final_judge_report.py` into the shared fixture module instead of
duplicating it.

## Test matrix

### 0. Optional-dependency boundary

File: `tests/test_ui_fixtures.py`. Level L0. No `ui` marker, so this runs in
every environment.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-00 | Collection skip hook | A fake `ui` item receives a skip marker when PyQt6 is missing; no skip marker is added when PyQt6 is present, even if pytest-qt is absent | P0 |

UI-00 is the regression test for the collection hook. It must not skip
PyQt6-only tests merely because pytest-qt is absent; the `qtbot`-based modules
handle that with their own module-level `pytest.importorskip("pytestqt")`.

### A. Bootstrap and tab lifecycle

File: `tests/test_ui_widgets.py`. Level L2. Marker `ui`.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-01 | Construct `QuantDashboard` | Exactly one tab exists; window title contains the tab name | P0 |
| UI-02 | Add a second tab | Two workspaces, independent inputs, unique auto names | P0 |
| UI-03 | Load two datasets with the same stem | Second tab becomes `name (2)` | P1 |
| UI-04 | Rename a tab, then reset with `None` | Explicit name wins; reset restores dataset-based naming | P1 |
| UI-05 | Close a tab | Remaining tabs renumber names; closing the last tab creates a fresh one | P1 |
| UI-06 | `Ctrl+T` / `Ctrl+W` via `qtbot.keyClick` | Add/close behave like the buttons | P2 |
| UI-07 | Description edit | Tab tooltip follows the description, or falls back to the title | P2 |

### B. Dataset input and target contract

File: `tests/test_ui_dataset.py`. Level L2. Marker `ui`.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-10 | Drop a valid CSV | Frame loads; log shows shape and columns; tab title changes | P0 |
| UI-11 | Drop a valid XLSX | Frame loads through `read_excel` | P1 |
| UI-12 | Drop `.txt` or unknown extension | Log explains the supported formats; loaded frame unchanged | P1 |
| UI-13 | Drop a malformed CSV | Log shows the pandas error; no exception escapes | P1 |
| UI-14 | Last column non-numeric | `_prepare_arrays` raises the target-type message; summary shows `REJECTED` | P0 |
| UI-15 | Constant target | Rejected with the variance/distinct-values reason | P0 |
| UI-16 | Binary target with only two distinct values | Rejected; message names the continuous-target contract | P0 |
| UI-17 | Fewer than 30 complete rows | Rejected with the row count | P1 |
| UI-18 | Fewer than 10 train or test rows after split | Rejected with the split message | P1 |
| UI-19 | Non-numeric feature columns mixed with numeric ones | Non-numeric columns are logged and ignored; numeric ones are used | P1 |
| UI-20 | Missing values in a text feature but complete numeric features | The sparse text column does not shrink the validation sample | P2 |
| UI-21 | `NaN`/`inf` in used features or target | Rejected as non-finite | P1 |

Regression note: these tests also pin the fixes recorded in
`docs/agents/logical-flaws-review.md` (builder signature, actuals/predictions in
the same label space, `min_sharpe` reaching `min_dsr_limit`).

### C. Legacy validation path

File: `tests/test_ui_validation.py`. Level L2. Marker `ui`.

The dashboard currently calls `stats_engine` directly. Keep these tests until
that path is replaced by the pipeline; they prevent a silent regression in the
shipped demo.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-30 | Valid continuous dataset and valid fields | Log shows `Validation cycle completed`; summary shows `Status: PASSED` or `REJECTED` and every engine key | P0 |
| UI-31 | Missing/invalid restriction fields | Log asks for valid numbers; no engine call; no crash | P0 |
| UI-32 | Fewer than two real trial signals | `REJECTED` with the DSR trial-count reason | P1 |
| UI-33 | `max_decay` outside `[0.15, 0.50]` | Summary shows the clamping warning | P1 |
| UI-34 | Engine raises | Log shows `Engine Execution Error`; summary shows the error; window stays responsive | P1 |
| UI-35 | Fake engine arguments | Builder receives `(params, X, y)`; predictions and actuals share a label space | P0 |

### D. Report rendering and tab routing

File: `tests/test_ui_report.py`. Level L2. Marker `ui`.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-40 | `set_report` with the active tab | Summary HTML changes; returns the workspace | P0 |
| UI-41 | `set_report` by index and by workspace | Both route to the intended tab | P0 |
| UI-42 | `set_report` with an invalid index/type/detached workspace | `IndexError` or `TypeError` with an actionable message | P1 |
| UI-43 | `load_summary_html` with `bytes` | UTF-8 decoded and rendered | P1 |
| UI-44 | `load_summary_html` with a non-string, non-bytes value | `TypeError` | P2 |
| UI-45 | Empty/whitespace report | Summary unchanged; log says the report was empty | P1 |
| UI-46 | `load_summary_html` alias | Delegates to `set_report` with the same result | P2 |
| UI-47 | Two tabs, two reports | Tab A never shows tab B's sentinel text | P0 |
| UI-48 | Report contains `<script>`, event attributes, or external URLs | Text is escaped or inert; no script runs; no resource request is made | P0 |
| UI-49 | Report load | Scroll position resets to the top; log records the embed | P2 |

### E. Final judge fragment contract

File: `tests/test_final_judge_report.py`. Level L1. No Qt, no keys, no marker.

The renderer is implemented; these tests are the interface the UI can rely on.
Existing coverage: UI-50 to UI-53 and UI-55 live in
`tests/test_final_judge_report.py`; UI-61 lives in `tests/test_artifact.py`;
UI-60 is partly covered by `tests/test_final_judge_integration.py` and
`tests/test_artifact.py`. Extend the renderer file for UI-54, UI-56 to UI-59,
and complete UI-60 rather than creating a parallel renderer test module.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-50 | `write_reports` default and subset | Default writes all four files as UTF-8; a `formats=` subset writes only the requested files; unknown formats raise `ValueError` | P0 |
| UI-51 | Fragment shape | No `<html>`, `<head>`, `<body>`, `<style>`, `<script>`, `<link>`, `<iframe>`, `<img>`, or `javascript:` | P0 |
| UI-52 | Escaping | Model text containing `<`, `>`, `&`, and quotes appears escaped, never as markup | P0 |
| UI-53 | Decision labels | All five decisions render the matching human label | P0 |
| UI-54 | Counts and model table | Critic count `n/min`, blocking count, and selected-model rows appear when supplied | P0 |
| UI-55 | Empty sections | Missing sections are omitted, not rendered as empty cards | P1 |
| UI-56 | Unevaluated critics and missing evidence | The "not a pass" wording and missing items are visible | P0 |
| UI-57 | JSON round-trip | `FinalJudgeReport.model_validate_json` accepts `final_judge.json` | P1 |
| UI-58 | Determinism | The same report renders byte-identical fragments | P2 |
| UI-59 | Size | A dense report stays below a documented cap (for example 1 MB) | P2 |
| UI-60 | Artifact handoff | An artifact with `final_judge` validates; `report_path` and `fragment_path` are relative file names | P0 |
| UI-61 | Checked-in schema | `schemas/artifact.schema.json` equals `artifact_json_schema()` and includes `final_judge` | P0 |

UI-61 is currently green in `tests/test_artifact.py`: the checked-in schema is
1.5.0, includes `final_judge`, and equals the model schema. Keep it as a
regression guard and regenerate with `python -m whack26 schema` if the model
changes.

### F. Pipeline-to-widget integration

File: `tests/test_ui_pipeline.py`. Level L4. Markers `ui` + `integration`.

Use a small synthetic dataset, `mock=True`, `offline=True`, and
`run_pipeline(..., final_judge=True, out_dir=tmp_path / "run")`.

`tests/test_final_judge_integration.py` covers UI-70
(`run_pipeline(..., final_judge=True)`) and UI-75 (`judge` subcommand reuse),
including file emission, artifact embedding, and fragment safety.
`tests/test_ui_pipeline.py` covers UI-74 (`python -m whack26 run --judge`), the
richer decision/count/model-summary rendering in a real tab, and two-run tab
isolation. Do not re-author the backend-covered cases; the remaining gap is
UI-79's degraded state.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-70 | Mock/offline run with `final_judge=True` | `RunOutcome.final_judge` is present; all four report files exist; artifact validates | P0 |
| UI-71 | Report and fragment agree | `outcome.final_judge.decision` maps to the decision label in the fragment | P0 |
| UI-72 | Load the real fragment into a real widget | Summary shows the decision label, `n/min`, and `Final model judgement` | P0 |
| UI-73 | `final_judge=False` | No `final_judge.*` files; `load_final_judge` raises the `--judge` hint; summary remains usable | P1 |
| UI-74 | CLI `run --judge --mock --offline` | Exit `0`; the same files exist; the UI loads them | P0 |
| UI-75 | CLI `judge <run_dir>` when `final_judge.json` exists | Reuses the report; the default all-formats path rewrites the fragment, while a `--format` subset may omit it (UI-78) | P1 |
| UI-76 | Two run directories, two tabs | Each tab shows its own decision/report | P0 |
| UI-77 | `run_dir` passed as `str` and as `Path` | Both work | P2 |
| UI-78 | Run directory with `artifact.json` but no fragment | Actionable error; no stale report from a previous run | P1 |
| UI-79 | Pipeline warning or failed judge stage | UI shows a degraded/error state, not a traceback or a blank panel | P1 |

UI-70, UI-71, UI-72, UI-74, UI-76, UI-77, and UI-78 are exercised by
`tests/test_final_judge_integration.py`, `tests/test_ui_pipeline.py`, and
`tests/test_ui_frontend.py`. UI-79 remains an `xfail` until
`load_final_judge` renders a degraded state instead of raising. Note that
`judge --format markdown` writes no fragment; the default `judge` and
`run --judge` write all four files.

### G. Optional-stage report coverage

File: `tests/test_ui_optional_stages.py`. Level L4. Markers `ui` + `integration`.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-80 | `model_search=True` | Fragment has the selected-model table; UI displays it | P0 |
| UI-81 | `variants=True` | Fragment still renders; variant-driven flags/concerns appear when present | P1 |
| UI-82 | `validate_generation=True` | Fragment still renders; generation evidence or missing evidence appears | P1 |
| UI-83 | `collection` metadata | Fragment still renders; collection context is not dropped | P1 |
| UI-84 | All optional stages off | Minimal fragment renders without empty placeholder sections | P1 |
| UI-85 | Mock final judge | Decision is `insufficient_evidence`; text says this is not a pass; UI never shows `Accepted` | P0 |

### H. Responsiveness, progress, and cancellation

File: `tests/test_ui_async.py`. Level L4. Markers `ui` + `integration` + planned
implementation.

The drop-to-pipeline path now runs in a background `_PipelineTask`, streams log
lines, disables the run button, and surfaces failures without crashing the tab;
`tests/test_ui_auto_pipeline.py` covers that handoff. These six tests still
target the legacy `DatasetWorkspace.run_validation` path, which remains
synchronous on the GUI thread and has no cancel control, so they stay `xfail`
until that path delegates to the pipeline worker or gains its own async seam.
`load_final_judge` itself is instantaneous.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-90 | Slow pipeline while the event loop runs | Window remains responsive; a timer tick fires during the run | P1 |
| UI-91 | Progress/log updates | Pipeline log lines arrive in `log_output` before completion | P1 |
| UI-92 | Close the tab mid-run | No crash, no update to a deleted widget | P2 |
| UI-93 | Two concurrent runs | Each result lands in its own tab | P2 |
| UI-94 | Pipeline exception | Run button re-enables; log shows the error | P1 |
| UI-95 | Cancel button | Cancels the worker and leaves a clear terminal log line | P2 |

### I. Security and robustness

File: `tests/test_ui_security.py`. Level L2. Marker `ui`.

| ID | Scenario | Pass condition | Priority |
|---|---|---|---|
| UI-96 | Report with model-generated HTML/URLs | No script execution; no external resource loads; text appears literally | P0 |
| UI-97 | Paths with spaces and non-ASCII characters | Load succeeds | P2 |
| UI-98 | Repeated loads in one tab | No widget leak; only the latest report is shown | P2 |
| UI-99 | Streaming log lines | Credentials are redacted by the backend before display | P1 |

## Parallel subagent execution

The suite is slow because the mock pipeline starts LLM/embedding fixtures, opens
SQLite, and spawns hypothesis subprocesses, while the Qt tests share a single
`QApplication`. Running the whole suite in every subagent repeats that setup and
creates resource contention. The plan therefore separates **authoring** from
**integration**:

- each subagent owns one test slice and runs only that slice;
- no subagent runs a bare `pytest`;
- the full suite is run once, by the final integrator, after all slices land.

| Lane | Runner | Scope | Cost |
|---|---|---|---|
| Fast | SA-1 to SA-7 | one test file or marker expression | seconds per slice |
| Slow | SA-8 only | the entire suite | minutes, paid once |

The same model works for a subagent that owns a feature branch: derive the
smallest test slice from the branch-impact map below, add the tests for that
branch, and run only that slice.

### Rules

1. **One owner per file.** No two subagents edit the same test file, fixture, or
   pytest configuration.
2. **Targeted commands only.** A subagent runs its own test file(s) or marker
   expression. It never runs `pytest`, `pytest -m ui`, or the full suite.
3. **Shared infrastructure first.** One subagent lands the shared fixtures,
   `conftest.py`, markers, and UI test requirements. Everyone else consumes
   them; only the owner edits them.
4. **Test-authoring subagents do not edit production code.** If a test needs a
   production change, the subagent reports it and marks the test `xfail`/planned.
   The production owner makes the change in a separate branch. A
   branch-validation subagent may fix production code on its own branch, but it
   still runs only its branch-impact slice and never edits another workstream's
   tests.
5. **No live providers.** Every subagent uses `--mock --offline` or pure
   fixtures. Live smoke is manual and last.
6. **No cross-slice fixes.** A failure outside the subagent's slice is reported
   with the failing test ID and owner, not fixed in that branch.
7. **Rebase before handoff.** Each branch is rebased onto the integration branch
   so the final run measures the merged state.
8. **The final integrator is the only full-suite runner.** SA-8 is the only
   subagent with explicit permission to run the full suite; every other
   subagent's handoff must state that it did not.

### Workstreams

Each row is one subagent, one worktree/branch, and one test slice. The test IDs
refer to the matrix above.

| ID | Owner slice | Files (exclusive) | Tests | Depends on | Target wall clock |
|---|---|---|---|---|---|
| SA-0 | Shared test infrastructure | `tests/ui_fixtures.py`, `tests/test_ui_fixtures.py`, `tests/conftest.py`, `pyproject.toml` (add `integration` marker), `requirements-ui-test.txt` | UI-00, fixture smoke, marker registration | none | 10 min |
| SA-1 | Final-judge fragment contract | `tests/test_final_judge_report.py` | UI-50 to UI-59 (extend existing coverage); UI-60 partly covered, UI-61 already in `tests/test_artifact.py` | none (SA-0 only for shared `make_report`) | 15 min |
| SA-2 | Dashboard shell and tabs | `tests/test_ui_widgets.py`, existing `tests/test_ui_frontend.py` | UI-01 to UI-07; maintain existing smoke tests | SA-0 | 15 min |
| SA-3 | Dataset contract and legacy validation | `tests/test_ui_dataset.py`, `tests/test_ui_validation.py` | UI-10 to UI-21, UI-30 to UI-35 | SA-0 | 20 min |
| SA-4 | Report rendering and tab routing | `tests/test_ui_report.py` | UI-40 to UI-49 | SA-0 | 15 min |
| SA-5 | Pipeline-to-widget handoff | `tests/test_ui_pipeline.py` | UI-71, UI-72, UI-74, UI-76 to UI-79 (UI-70/75 are backend-covered) | SA-1, SA-4 | 20 min |
| SA-6 | Optional-stage report coverage | `tests/test_ui_optional_stages.py` | UI-80 to UI-85 | SA-5 | 15 min |
| SA-7 | Async and security | `tests/test_ui_async.py`, `tests/test_ui_security.py` | UI-90 to UI-99 | SA-2, SA-4; async tests need the UI thread change | 20 min |
| SA-8 | Final integrator | integration branch only; no new test files | Full suite, merge resolution, final report | all of the above | 30 min |

Suggested waves:

| Wave | Subagents | Purpose |
|---|---|---|
| 0 | SA-0 | Land fixtures, markers, requirements; unblock everyone |
| 1 | SA-1, SA-2, SA-3, SA-4 | Backend contract and independent widget slices in parallel |
| 2 | SA-5, SA-7 | Pipeline handoff and security/async work; can start once their dependencies are merged |
| 3 | SA-6 | Optional-stage matrix on top of the working pipeline slice |
| 4 | SA-8 | Merge everything and run the full suite once |

Do not start SA-5 or SA-6 before SA-4's routing tests are green: they would
duplicate the same widget failures and hide the real handoff defect. SA-5
branches from the integration branch after SA-1 and SA-4 merge; SA-6 branches
after SA-5 merges. SA-7 can start its security slice in wave 1 and add the async
tests later.

### Orchestration in an OpenCode session

When the coordinator is an OpenCode agent, use the harness rather than starting
all work in one context:

1. Create an integration worktree and branch for SA-8
   (`tools.opencode.worktree_create`).
2. Spawn SA-0 as a foreground `general` subagent with the prompt template. Merge
   its branch first.
3. Spawn SA-1, SA-2, SA-3, and SA-4 as **background** `general` subagents, one
   worktree each, using the prompt template. Their sessions are independent;
   collect each `sessionID` and final report.
4. Merge only slices whose targeted command is green. If a slice is red, send
   the failure back to that `sessionID`; do not restart the whole wave.
5. Spawn SA-5 and SA-7, then SA-6, with the same pattern.
6. Spawn SA-8 with explicit full-suite permission and the merged integration
   worktree as its directory.

Never spawn a subagent with a bare "run the tests" prompt. The prompt must name
the workstream, the exclusive file list, and the single targeted command.

### Branch-impact map

Use this for a subagent that owns a feature branch rather than an authoring
slice. Derive the changed files, map each one, union the listed test files, and
run only that union plus any new test file the branch adds:

```powershell
git diff --name-only origin/main...HEAD
```

| Branch changes | Run only |
|---|---|
| `UI_frontEnd.py` tab shell, `QuantDashboard`, `set_report`, `load_final_judge` | `tests/test_ui_widgets.py`, `tests/test_ui_report.py` |
| `UI_frontEnd.py` file loading, `_prepare_arrays`, `run_validation` | `tests/test_ui_dataset.py`, `tests/test_ui_validation.py` |
| `whack26/final_judge/report.py` | `tests/test_final_judge_report.py`, `tests/test_ui_report.py` |
| Other `whack26/final_judge/*.py` | `tests/test_final_judge_aggregate.py`, `tests/test_final_judge_checks.py`, `tests/test_final_judge_evidence.py`, `tests/test_final_judge_pipeline.py` |
| `whack26/run.py`, `whack26/cli.py` judge path | `tests/test_run_flow.py`, `tests/test_final_judge_integration.py`, `tests/test_ui_pipeline.py` |
| `whack26/artifact.py`, `schemas/artifact.schema.json` | `tests/test_artifact.py`, `tests/test_final_judge_report.py` |
| Optional-stage flags or evidence | `tests/test_ui_optional_stages.py` |
| Docs only | No pytest run; run the link/schema checks in the docs test if present |

### Commands per workstream

Always use `-q --tb=short -x` for the authoring loop. Drop `-x` only when the
subagent wants the full failure list for its own slice.

```powershell
# SA-0: shared infrastructure smoke, no product test authoring
.\.venv\Scripts\python.exe -m pytest --collect-only -q
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_fixtures.py -x --tb=short

# SA-1: backend contract, no Qt
.\.venv\Scripts\python.exe -m pytest -q tests/test_final_judge_report.py -x --tb=short

# SA-2: widget shell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_widgets.py -x --tb=short

# SA-3: dataset and legacy validation
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_dataset.py tests/test_ui_validation.py -x --tb=short

# SA-4: report routing
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_report.py -x --tb=short

# SA-5: pipeline and CLI handoff
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_pipeline.py -m "ui and integration" -x --tb=short

# SA-6: optional stages
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_optional_stages.py -m "ui and integration" -x --tb=short

# SA-7: security first, then async when implemented
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_security.py -x --tb=short
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_async.py -x --tb=short
```

A subagent must not use these commands:

```powershell
pytest                      # full suite: integrator only
pytest -m ui                # all UI: integrator only
pytest tests/               # full suite by another name
pytest -m "not ui"          # core suite: integrator only
```

### Handoff report

Every subagent returns the same compact report so the integrator can decide
whether the slice is safe to merge:

```text
Workstream: SA-3
Branch/worktree: test/ui-dataset-validation
Files touched: tests/test_ui_dataset.py, tests/test_ui_validation.py
Test IDs: UI-10..UI-21, UI-30..UI-35
Command run: python -m pytest -q tests/test_ui_dataset.py tests/test_ui_validation.py -x --tb=short
Result: 14 passed, 1 xfailed, 2 skipped (reason: PyQt6 absent), 8.2s
Failures outside slice: none
Contract assumptions: last column is the target; <30 rows is rejected
Full suite run: no (by design)
```

If the slice is red, the report includes the failing test ID, the first
traceback line, and whether the defect is in the test, a shared fixture, or
production code. The integrator routes it back to the owner; it does not fix it
in the integration branch.

### Failure routing at the end

The final full-suite run is expected to find some cross-slice issues. Route them
by the same ownership map:

| Failing test file | Owner |
|---|---|
| `tests/test_final_judge_report.py` | SA-1 |
| `tests/test_ui_widgets.py`, `tests/test_ui_frontend.py` | SA-2 |
| `tests/test_ui_dataset.py`, `tests/test_ui_validation.py` | SA-3 |
| `tests/test_ui_report.py` | SA-4 |
| `tests/test_ui_pipeline.py` | SA-5 |
| `tests/test_ui_optional_stages.py` | SA-6 |
| `tests/test_ui_async.py`, `tests/test_ui_security.py` | SA-7 |
| `tests/test_final_judge_aggregate.py`, `tests/test_final_judge_checks.py`, `tests/test_final_judge_evidence.py`, `tests/test_final_judge_pipeline.py`, `tests/test_final_judge_integration.py` | existing final-judge owner, not a UI subagent |
| `tests/test_run_flow.py`, `tests/test_artifact.py` | existing owner, not a UI subagent |
| Shared fixture or marker failure | SA-0 |

Owner fix loop:

1. Integrator sends the failing test ID and traceback to the owner.
2. Owner creates a `fix/ui-<slice>` branch from integration, runs only that test
   file, fixes, and reports.
3. Integrator reruns only the failed test to confirm.
4. Integrator reruns the full suite once at the end. Do not rerun the full suite
   after every individual fix.

### Subagent prompt template

Copy this into each child session and replace the bracketed fields:

```text
You own workstream [SA-N] in the UI integration test plan:
docs/spec/ui-integration-test-plan.md.

Worktree/branch: [path/branch]. You may edit only: [files].

Task: add or extend the tests for [test IDs] using the existing fixtures and
the contracts in the plan. Do not modify UI_frontEnd.py, shared fixtures,
pytest configuration, or another workstream's tests.

Run only this command:
  [targeted pytest command]

Do not run a bare `pytest`, the full suite, `-m ui`, `-m "not ui"`, or any live
provider test. Do not install packages. If a test needs a production change,
mark it xfail/planned and report the change instead of editing production code.

Report back exactly:
- workstream, branch, files touched
- test IDs covered
- command run and pass/fail/skip counts with runtime
- first failing test and traceback line, if any
- failures outside your slice
- contract assumptions and blockers
- confirmation that you did not run the full suite
```

### Time budget

| Wave | Wall clock | Notes |
|---|---|---|
| SA-0 | 10 min | Blocks everyone; land it first |
| SA-1 to SA-4 | 20 min | Parallel; bounded by SA-3 |
| SA-5, SA-7 | 20 min | Parallel after dependencies merge |
| SA-6 | 15 min | Serial on SA-5 |
| SA-8 full suite | 30 min | One run plus targeted rechecks |
| Buffer | 20 min | Cross-slice failures and rebases |

Expected wall clock is roughly 90 minutes with four concurrent workers, versus
running the slow full suite in each of eight subagents. The full-suite cost is
paid exactly once.

## Contract with the final-judge team

The UI must not depend on the current card markup. The following is the minimum
stable contract; anything beyond it is a rendering detail.

1. **File names and formats.** `write_reports(directory, report, formats=None)`
   writes all four files by default: `final_judge.json`, `.md`, `.html`, and the
   UI fragment `final_judge.fragment.html`. A `formats=` subset writes only the
   requested files; the UI requires the `fragment` entry, so callers that want a
   UI handoff must include it. `run --judge` uses the default all-formats path.
   The fragment is UTF-8 semantic HTML with no document wrapper.
2. **No active or external content.** The fragment contains no `<style>`,
   `<script>`, `<link>`, `<iframe>`, `<object>`, `<embed>`, `<img>`, forms, or
   `javascript:` URLs. The dashboard supplies all styling.
3. **Qt-supported tags only.** Use `h1`-`h3`, `p`, `table`/`tr`/`th`/`td`,
   `ul`/`ol`/`li`, `dl`/`dt`/`dd`, `div`/`span`, `b`/`strong`, `em`, `hr`,
   `blockquote`, and `pre`/`code`.
4. **Escaping.** Every value that can contain model output is HTML-escaped.
   Tests inject `<script>alert(1)</script>` into a finding and require it to
   appear as text.
5. **Stable text.** The fragment contains the human decision label (for
   example `Insufficient evidence` and the "NOT a pass" note), the critic count
   `n/min`, and the selected-model summary when present. Tests match these
   strings, not CSS classes or card order.
6. **Paths.** `FinalJudgeReport.report_path` and `.fragment_path` are relative
   file names in the run directory, and `validate_artifact` rejects absolute or
   escaping paths.
7. **Artifact.** The optional `final_judge` section round-trips through
   `artifact.json`. The checked-in `schemas/artifact.schema.json` is 1.5.0,
   includes `final_judge`, and equals `artifact_json_schema()`; keep that
   equality test. Regenerate with `python -m whack26 schema` and commit if the
   model changes.
8. **Empty states.** A report with no findings still renders the decision
   banner, counts, and unevaluated-critic section; missing sections are omitted.
9. **Optional version marker.** Recommended: add a `fragment_schema_version`
   field or a single HTML comment so the UI can detect an incompatible future
   fragment. This is an open decision, not a current requirement.

## Execution

### Subagents: targeted only

Each subagent uses its command from "Commands per workstream" and stops there.
The non-negotiables:

- no bare `pytest`, no `-m ui`, no `-m "not ui"`, no `tests/`;
- `-q --tb=short -x` during authoring so the first failure is actionable;
- `QT_QPA_PLATFORM=offscreen` for every UI slice;
- no live providers, no package installs, no full-suite "just to check".

If the targeted slice passes, hand off. If it fails, fix only that slice. The
integrator owns every cross-slice question.

### Final integrator: everything once

Run this only after all branches are merged and rebased. It is the one full-suite
run the plan allows.

```powershell
# 1. Fast collection check; catches marker/config/import breaks in seconds.
.\.venv\Scripts\python.exe -m pytest --collect-only -q

# 2. Everything, once. Add -x only if you want the first failure to stop the run.
.\.venv\Scripts\python.exe -m pytest -q

# 3. If step 2 fails: route by owner, owner runs only the failing file, then
#    rerun that file, then rerun the full suite once more at the end.
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_report.py -x --tb=short
```

Do not run the full suite after every merge. Batch the merges, run once, fix the
owned failures, run once more. The second full run is the release gate.

Manual live smoke (never in CI, after the full suite is green):

```powershell
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --out runs/demo-judge `
  --judge
python UI_frontEnd.py
# Load runs/demo-judge and visually check the decision, counts, and findings.
```

## Rollout phases

| Wave | Owner | Deliverable | Gate |
|---|---|---|---|
| 0 | SA-0 | Shared fixtures, markers, UI test requirements | `--collect-only` and fixture smoke pass |
| 1 | SA-1 | Final-judge fragment contract | `tests/test_final_judge_report.py` green; UI-61 already green in `tests/test_artifact.py` |
| 1 | SA-2 | Dashboard shell and tabs | `tests/test_ui_widgets.py` green |
| 1 | SA-3 | Dataset contract and legacy validation | `tests/test_ui_dataset.py` and `tests/test_ui_validation.py` green |
| 1 | SA-4 | Report rendering and routing | `tests/test_ui_report.py` green |
| 2 | SA-5 | Pipeline and CLI handoff | `tests/test_ui_pipeline.py` green; depends on SA-1 and SA-4 |
| 2 | SA-7 | Security and async | `tests/test_ui_security.py` green; async green when implemented |
| 3 | SA-6 | Optional-stage matrix | `tests/test_ui_optional_stages.py` green; depends on SA-5 |
| 4 | SA-8 | Merged integration branch | Full suite green once |
| 5 | manual | Live report smoke | Recorded in `docs/eval/`; never a CI gate |

Waves 1 and 2 are the parallel window. The slow pipeline slice stays serial, so
only one subagent pays the mock-pipeline setup cost.

## Exit criteria

- Every subagent handoff names its targeted command, result, and confirms the
  full suite was not run.
- The merged integration branch passes `pytest --collect-only -q`.
- The merged integration branch passes **one** full `pytest -q` run with the UI
  extra installed and `QT_QPA_PLATFORM=offscreen`.
- In an environment without PyQt6, the same full run is green and every UI test
  skips with a clear reason.
- Every test ID in section 0 and sections A-I is owned by exactly one
  workstream; there are no orphaned IDs.
- The mock pipeline E2E test genuinely renders `final_judge.fragment.html` in a
  `QuantDashboard` tab.
- The missing-fragment, no-judge, and invalid-report paths are covered.
- The final judge contract tests pass against the current renderer.
- `schemas/artifact.schema.json` includes `final_judge` and matches the model.
- No automated test requires a network connection, API key, or provider quota.
- Each targeted slice completes in under 30 seconds; the full suite completes
  once in under five minutes on the development machine.
- A live smoke run is recorded separately and is not a CI gate.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| The final judge markup is still changing | Test semantic text and the file contract; keep a static fragment fixture for widget tests |
| PyQt6 is absent from the base environment | Module-scope `pytest.importorskip`, a `pytest_collection_modifyitems` skip hook, the `ui` marker, and separate UI test requirements |
| pytest-qt is installed without PyQt6 | Unsupported combination; it aborts at pytest-qt configure before any hook runs. Keep the UI stack all-or-nothing in `requirements-ui-test.txt` |
| Headless CI lacks a display | `QT_QPA_PLATFORM=offscreen`; no pixel snapshots |
| Qt parses only a subset of HTML | Fragment contract restricts tags and forbids CSS/JS |
| `QApplication` singleton or teardown crashes | `qtbot.addWidget` and the plugin's session-scoped `qapp` |
| Mock pipeline is slow or flaky | Small synthetic dataset, deterministic seed, `--mock --offline`, `integration` marker |
| In-process pipeline freezes the window | SA-7's async work (wave 5) offloads it; until then, keep the legacy path clearly labelled |
| Artifact schema drifts again | Equality test between the checked-in schema and `artifact_json_schema()` |
| `__getattr__` passthrough hides typos | Test explicit public methods; do not rely on dynamic attribute access |
| Tab indices shift after close | Route by workspace object where possible; test index invalidation |
| Model text becomes injected HTML | Escaping test in SA-1; no active-content tags in the fragment contract |
| A mock judge pass is mistaken for a real pass | The mock fixture returns `insufficient_evidence`; UI-85 asserts the "not a pass" text |
| Cross-slice failures surface only in the final run | Contract-first SA-1, shared fixtures from SA-0, ownership map, and a dedicated SA-8 fix loop |
| A subagent strays outside its slice | One-owner file matrix, prompt template, and handoff report that requires the targeted command |
| Several subagents run the slow suite and contend for Qt/SQLite/subprocesses | Hard rule: only SA-8 runs the full suite; everyone else runs one file or marker expression |
| `pytest-xdist` puts multiple Qt tests in one worker | Do not enable xdist for `-m ui`; each subagent runs its own process, and the integrator keeps UI tests serial |
| A branch is stale when it reaches integration | Rebase onto the integration branch before handoff; SA-8 reruns collection and full suite after merges |
| The final E2E test is flaky | Deterministic seed, `--mock --offline`, one retry allowed, then route to SA-5 |
| A subagent quietly installs the UI stack or changes shared config | Explicit no-install rule; SA-0 is the only owner of `pyproject.toml`, `conftest.py`, and requirements |

## Open decisions

1. **How does the UI launch a run?** Recommend a worker that calls
   `run_pipeline` in-process for log streaming, or a subprocess/CLI call for
   isolation. This choice determines UI-90 to UI-95.
2. **Does the UI read only the fragment or also `artifact.json`?** Recommend
   the fragment for display and `artifact.json`/`final_judge.json` for metadata
   such as the decision enum and report paths.
3. **Should `--judge` be on by default in the UI?** Recommend a "full report"
   toggle that defaults to on, while the CLI default stays off.
4. **Where does the run directory come from?** Drop a folder, choose it in a
   dialog, or always create one under `runs/`. The test plan works with any of
   these once the seam is fixed.
5. **Should `load_final_judge_report` validate the artifact first?** Recommend
   warning (not blocking) when `artifact.json` is missing or invalid, so a
   fragment-only handoff still renders.
6. **Fragment version marker.** Add one if the renderer is expected to change
   shape again after the hackathon.
7. **Marker ownership.** Register both `ui` and `integration`; if issue #15 is
   available and covers `integration`, coordinate to avoid duplicate marker
   definitions, then document both in this plan and the repo README.
8. **Concurrency width.** Recommend four test-authoring subagents in wave 1
   (SA-1 to SA-4), then SA-5 and SA-7, then SA-6. More workers do not help because
   the mock pipeline slice is serial.
9. **Worktree policy.** Recommend one worktree/branch per workstream
   (`test/ui-<slice>`) and a single integration branch. Subagents never merge
   each other's work; SA-8 owns the merge and the final full-suite run.
