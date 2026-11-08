# Referee

> **Tagline:** An autonomous research analyst that reads the literature, turns claims into runnable models, and refuses to ship what it can't defend.

**Track:** Brevan Howard — Autonomous Validation Engine · **Event:** WHACK-26

*Alternative names if you prefer: Triage ("your reading queue becomes a validation queue") or Null Hypothesis ("a research engine that tries to reject itself").*

---

## Elevator pitch (200-character field)

Drop in a dataset; Referee finds the relevant papers, turns their claims into runnable models, audits every derivation, and runs a four-layer statistical gauntlet before anything ships.

---

## Inspiration

> "You read a paper, you can't verify the claim, and you've lost the afternoon."
> — a quant developer we met

That sentence is the whole product. Research is not the bottleneck any more; **trust** is. A quant reads dozens of papers, most of which are not worth the time, and the ones that are usually fail to report what is needed to verify them. The same disease exists in model form: across a thousand zero-skill backtests, the expected maximum Sharpe is about 3, so the best result in a search is often the luckiest, not the best.

Existing tools either generate more claims (search, LLM assistants) or grade a finished model (backtest analytics). Nothing sits between the paper and the model and asks whether the claim deserved to exist. That missing layer is what we built.

## What it does

1. **Reads the dataset.** Profiles any pandas-readable table — schema, dtypes, missingness, roles, target/time detection — and classifies it into a dynamically built, general-domain field hierarchy (max depth 4, never pre-seeded).
2. **Reads the literature.** Generates queries and fetches live papers and datasets from free APIs; normalizes, deduplicates, and ranks them. Every research record is cached and snapshotted per run.
3. **Extracts only what can be cited.** Factors must carry evidence IDs that resolve to a specific paper or dataset. No evidence, no factor. Surviving factors are mapped to dataset columns.
4. **Writes runnable hypotheses.** Three Python models, one per method family (linear, group comparison, interaction/derived), AST-validated and executed in an isolated subprocess with repair on failure.
5. **Audits the derivation.** An agentic method critic checks each model against a versioned statistical-pitfalls corpus — leakage, multiplicity, causal overreach, sample size, missing data, time-series dependence — and cites the chunk behind every finding. It is structurally forbidden from judging performance.
6. **Runs the gauntlet.** A separate four-layer validator applies the Deflated Sharpe Ratio for multiple-testing, Monte Carlo permutation against luck, chronological out-of-sample decay, and parameter sensitivity. General regression, classification, and count targets switch to the appropriate primary metric rather than being forced through the strategy engine.
7. **Audits the final result.** An advisory final judge combines nine specialist critics, a devil's advocate, deterministic code checks, the statistical reports, and the selected model evidence into a run-level decision — including `insufficient_evidence` when the audit cannot complete.
8. **Hands off an audit trail.** A versioned `artifact.json`, `preview.md`, `validation.json`, optional model-search/variant/pitfall sidecars, and a UI-ready final-judge fragment.

> **Every claim has an evidence ID. Every model has a code hash. Every verdict has an audit trail.**

## How we built it

- **Stack:** Python 3.12, pandas/NumPy/SciPy, Pydantic, httpx, SQLite (v4 document catalog), DeepSeek for the LLM and Gemini for embeddings, PyQt6 for the optional dashboard, pytest for the suite.
- **Independent layers, by design.** The generator proposes; the method critic audits the derivation; the statistical validator measures the fitted model; the final judge audits the run-level evidence without re-ranking or overriding either. The critic cannot grade performance, the validator never sees the generator's reasoning, and deterministic checks (target leakage, preprocessing before split, temporal leakage) run before the LLM and cannot be overridden.
- **Isolation:** AST allowlist plus `python -I` subprocess, network blocked, explicit timeouts. Best-effort, local, trusted-code isolation — documented as such.
- **Reproducibility:** content-derived IDs, code and data hashes, fixed seeds, cached query sets keyed to the dataset and summary hash.
- **Coverage:** **949 tests passing** (2 skipped), including seeded leakage cases the critic must catch.

## Challenges we ran into

- **Separation of powers.** Making the critic *unable* to bless a model was harder than making it smart. We removed performance metrics from its inputs entirely and required a corpus citation for every issue.
- **Target leakage in generated code.** The generator kept putting the target in its own feature lists. The static validator caught it every time — and the run refused to emit a partial artifact rather than ship a broken set. That failure became a feature of the story.
- **Free-tier quotas.** Provider limits are real. We added model fallbacks, request caching, warm/offline modes, and a mock path so the whole pipeline runs without keys.
- **Making rejection visible.** It is tempting to demo only green checkmarks. The useful output is often `refine` or `insufficient`.

## Accomplishments we're proud of

- Live end-to-end runs on **California Housing** and **Palmer Penguins**: three of three hypotheses across all three families, valid artifacts, and predictions that reproduce exactly (`rtol=1e-9`).
- Real citations: factors resolve to actual OpenAlex records and map to dataset columns.
- A four-layer statistical engine wired in as an independent post-generation stage, with unsupported targets reported as `skipped`, never as failures.
- A verdict vocabulary that includes **`insufficient`** — the system admits what it cannot verify.

## What we learned

Generating hypotheses is the easy part. **Earning the right to believe one is the product.** A validator that can say "insufficient" is worth more than one that always says "pass", and the most credible demo is a rejection.

## What's next

- Full-text paper ingestion (today the pipeline reads metadata and abstracts; semantic scraping is opt-in and never redistributes PDFs).
- A live validation run at quota scale and the full ten-dataset field-classification gate.
- Shared field stores and portfolio-level multiplicity for team deployment.
- A plug-in seam for a general external statistical-validation library.

## Built with

`python` · `pandas` · `numpy` · `scipy` · `pydantic` · `httpx` · `sqlite` · `deepseek` · `google-gemini` · `openalex` · `arxiv` · `huggingface` · `zenodo` · `wikipedia` · `pyqt6` · `pytest`

## Try it out

```powershell
# No keys needed: mock + offline
python -m whack26 run `
  --input data/california_housing.csv `
  --target median_house_value `
  --out runs/demo --mock --offline

python -m whack26 validate-models runs/demo
python -m whack26 validate-artifact runs/demo/artifact.json
python -m whack26 preview runs/demo/artifact.json
python -m whack26 judge runs/demo --mock --offline
```

The project is called Referee; the Python package/CLI keeps the repository's
`whack26` name.

Optional dashboard:

```powershell
pip install -r requirements-ui.txt
python UI_frontEnd.py
```

Repository: `https://github.com/hoisin-sauce/whack-26`

---

## Demo video storyboard (2:30–3:00)

| Time | Shot | On-screen text |
|---|---|---|
| 0:00 | Black screen, the judge quote types out | "You read a paper, you can't verify it, and you've lost the afternoon." |
| 0:15 | Terminal: one `run` command | "One command. One dataset." |
| 0:30 | Pipeline log: classification → research → factors | "It reads the literature." |
| 0:55 | `research.json` / preview: real papers with evidence IDs | "Every factor carries evidence." |
| 1:20 | Three generated hypothesis modules + artifact | "Three runnable models, one code hash each." |
| 1:40 | **A `refine` or `reject` verdict with a corpus citation** | "The critic's job is to say no." |
| 2:05 | Four-layer validation report | "DSR · permutation · OOS decay · sensitivity" |
| 2:30 | `preview.md`: paper → factor → column → code → verdict | "Reading becomes running." |
| 2:50 | Test suite: 949 passed | "949 tests." |

Record the rejection first. If the live run is quota-blocked, record the mock path and label it — then show the recorded live artifacts as evidence.

## Assets to capture before submitting

1. **Hero image (1280×720):** architecture diagram with the three independent layers highlighted — generator / method critic / statistical validator.
2. **Screenshot:** `preview.md` showing paper → factor → column → hypothesis → verdict.
3. **Screenshot:** validation report with a visible `refine`/`reject` and its corpus citation.
4. **GIF (60–90s):** terminal run from command to artifact.
5. **Sanitized demo artifact:** copy `artifact.json`, `preview.md`, and `validation.json` into `demo/` (no keys, no absolute paths) so judges can inspect without running.
6. **UI-compatible CSV:** California Housing with the continuous target last (the dashboard's contract requires it; the raw demo file ends with `ocean_proximity`).
7. **Repo link** plus the mock-mode command above.

---

## Internal notes (do not paste into Devpost)

**Claims that are safe:** 949 passed / 2 skipped; two-dataset live smoke; 3/3 hypothesis families; exact prediction reproduction; real OpenAlex citations; four-layer engine implemented and wired; artifact validation passing.

**Claims to avoid:**
- "All 12 research sources are live" — Kaggle, Semantic Scholar and web search register only when credentials exist.
- "Model search, variants and the agent loop run by default" — they are opt-in flags and default `False`.
- "Live statistical validation is complete" — implemented and mock-tested; the live run is quota-pending.
- "It reads full papers" — metadata/abstracts plus opt-in semantic scraping; no PDF redistribution.
- "Causal discovery" — the critic rejects unsupported causal language; it does not identify causal effects.

**Pre-submission checks:** commit the working tree; generate a fresh demo artifact; verify no `.env`, keys, or absolute paths appear in anything published; re-run the suite once at the final gate.
