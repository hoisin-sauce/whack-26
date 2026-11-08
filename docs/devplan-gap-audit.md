# Gap audit: `devplan.md`

**Bottom line:** it's a strong architecture sketch, not a spec. It names the subsystems and the method menu well, but it is missing the user, the MVP boundary, precise domain language, executable algorithms, data contracts, and any definition of success. A `/to-spec` run today would only restate it.

## What's already strong

- Clear subsystem decomposition: research, field hierarchy, scraping, credibility, hypothesis generation, validation.
- Concrete external-source list (HF Hub, Kaggle, Zenodo, Dataverse, OpenML, data.gov, DataCite, Semantic Scholar, OpenAlex, arXiv).
- Concrete statistical method menu (correlation tests, GLMs, tree models, SINDy).
- Two genuinely good ideas worth preserving: the **jumbled-label leakage check** and the **Python-function + column-applicability** output contract.

## The 10 gaps that block a spec

1. **No user, problem, or job-to-be-done.** The document describes a system, not a product. The spec's Problem Statement, Solution, and User Stories cannot be written.
2. **No MVP boundary.** Research, hierarchy, scraping, credibility, hypothesis generation, and validation are all in scope at once. Nothing says what the first demoable slice is.
3. **Core vocabulary is overloaded.** *field*, *category*, *subfield*, *level*, *classification*, *resource*, *search*, *summary*, *credibility*, *hypothesis*, *model* are used loosely and sometimes interchangeably.
4. **The field-hierarchy algorithm is prose, not invariants.** The rotation/promotion rules at `devplan.md:17` and `devplan.md:43` have ambiguous referents and no thresholds, depth limits, or concurrency story.
5. **The hypothesis output contract is one sentence.** Function signature, applicability labels, dependencies, execution sandbox, and failure behavior are undefined (`devplan.md:72`).
6. **The statistical pipeline contains a likely error.** "Correlation matrix between rows" (`devplan.md:91`) should almost certainly be columns/features; the 30% holdout scheme, multiple testing, missing data, and model selection are unspecified.
7. **The anti-leakage test is underspecified.** It says "two versions" then lists three, and "unscrambles the labels correctly" has no operational definition (`devplan.md:78–79`).
8. **External APIs and scraping have no contracts.** No auth, rate limits, quotas, pagination, normalization, dedup, crawl depth, robots/ToS, or cost.
9. **Credibility and validation are judgment calls without criteria.** Who rates authors/publishers? What formula? What exactly makes a model get thrown out?
10. **There is no evaluation.** No benchmark, metrics, expert review, or success criteria — so "revised spec" has no way to be judged.

## Detailed gap register

Tags: **[DECIDE]** needs a decision, **[RESEARCH]** needs facts gathered, **[CLARIFY]** needs the document made unambiguous.

### 1. Product and scope

- **[DECIDE] P1 — User and problem.** Who is this for: a researcher, an analyst, a hackathon judge? What decision does the system improve?
- **[DECIDE] P2 — MVP boundary.** Is the first slice "upload a dataset, get relevant papers" or "upload a dataset, get ranked hypotheses"? These are very different builds.
- **[DECIDE] P3 — Success criteria.** What counts as a good research result, a good hierarchy, a good hypothesis? Needed for evaluation and acceptance criteria.
- **[DECIDE] P4 — User-facing surface.** No input format, output format, UI/API/CLI, or workflow is described.
- **[CLARIFY] P5 — Constraints.** Time budget, token/cost budget, data sensitivity, deployment target, expected dataset size.
- **[CLARIFY] P6 — Primary workflow.** What happens first, second, and third when a user submits a dataset? The plan is organised by subsystem, not by user journey.

### 2. Domain model and glossary

- **[DECIDE] D1 — Glossary.** Define *resource*, *field*, *category*, *subfield*, *level*, *classification*, *search*, *dataset* (user vs external), *paper*, *summary*, *credibility*, *reliability*, *validity*, *hypothesis*, *model*, *anonymized*, *jumbled labels*, *RAG*.
- **[CLARIFY] D2 — Document types.** The plan defines Dataset, Search, and Paper, but later mentions news articles (`devplan.md:64`) with no type. Is a web result a Search? Is a news article a Paper? What about a GitHub repo or a blog post?
- **[CLARIFY] D3 — "Search" semantics.** Is a Search a query, a result set, a research session, or a web page? Its "to and from link" relationship (`devplan.md:37`) is undefined.
- **[DECIDE] D4 — Relationship semantics.** Cardinality, direction, and edge metadata for dataset↔search, dataset↔paper, search↔search, paper↔paper. Is the link "cites", "mentions", "derived from", or something else?
- **[DECIDE] D5 — Identity and dedup.** The same paper appears in arXiv, Semantic Scholar, and OpenAlex. Dedup by DOI, arXiv ID, title/author matching, or content hash?
- **[CLARIFY] D6 — "RAG".** `devplan.md:15` uses RAG as a similarity mechanism. RAG is a retrieval pattern, not a distance metric. Clarify whether you mean embedding similarity, keyword search, or LLM-generated queries.
- **[CLARIFY] D7 — Level direction.** `devplan.md:42` says level 1 is top-level and level 2 is a subfield, but `devplan.md:17` says a field "increases in level" and things move "down a level" without defining direction.
- **[CLARIFY] D8 — Credibility vs reliability vs validity.** These are used as if interchangeable; they are not. Define each and say which one gates retrieval.

### 3. Research and ingestion

- **[DECIDE] R1 — Input contract.** Accepted formats (CSV, Parquet, Excel, JSON), size limits, schema inference, type detection, and the metadata schema (description, source site, authors, collection method, time period).
- **[DECIDE] R2 — Query construction.** How does a dataset plus optional metadata become search queries and embeddings? Which fields are summarised? Is the query the dataset summary, user description, column names, or an LLM-generated brief?
- **[RESEARCH] R3 — API capability matrix.** For every listed source: auth, free-tier limits, rate limits, pagination, result fields, licensing, and whether a public API actually exists. In particular, verify Google Dataset Search; it has no documented public API.
- **[DECIDE] R4 — Web search provider.** "A free web api" (`devplan.md:25`) is unnamed. Choose one and define its limits and fallback.
- **[DECIDE] R5 — Result normalization.** How are results from different APIs mapped into the three document types? What provenance fields are stored?
- **[DECIDE] R6 — Crawl policy.** Reference/link depth (one hop or recursive?), breadth per resource, cycle handling, PDF/HTML extraction, paywall handling, `robots.txt`, politeness, caching, and legal/ToS constraints.
- **[DECIDE] R7 — Relevance thresholds.** The same/tighter cosine threshold (`devplan.md:54`) needs values, tuning method, and a rule for when to follow all links vs only close links.
- **[DECIDE] R8 — Stopping criteria.** How much research is enough per dataset: result count, token budget, time budget, or marginal-relevance cutoff? "20–50?" (`devplan.md:25`) is unresolved.
- **[DECIDE] R9 — Storage/linkage.** How are research results linked back to the dataset and its field? What triggers research: upload, explicit request, or background job?
- **[CLARIFY] R10 — Freshness.** Do you re-run research later? How do you handle retractions, updated papers, dead links, and changing field assignments?

### 4. Field hierarchy

- **[CLARIFY] H1 — Level semantics.** Define depth direction, maximum depth, and what "generality" means operationally.
- **[DECIDE] H2 — Threshold bounds.** What are the distance bounds per level? Are they fixed, learned, or tuned? Where are they stored?
- **[DECIDE] H3 — Algorithm invariants.** Write pseudocode for: find closest field, ascend to parent, create subfield, promote a subfield above its parent, and reassign resources. State the tree invariants after each step.
- **[CLARIFY] H4 — Ambiguous referents.** `devplan.md:43` says "move the parent and all its subfield down a level and place the parent of the current parent as the new field". Which node is "the parent" at each point?
- **[DECIDE] H5 — Agent judgment.** The "more specific or less specific" call (`devplan.md:17`) needs a rubric, model, determinism policy, audit trail, cost budget, and a tie-breaker.
- **[DECIDE] H6 — Concurrency.** What happens when two resources are added simultaneously and both try to rotate the same branch? Locking, transactions, tree versioning, pgvector index maintenance.
- **[DECIDE] H7 — Field lifecycle.** Rename, merge, split, delete, re-embed, and re-level. What happens to resources under a moved field?
- **[DECIDE] H8 — Seeding.** Who creates top-level fields? The plan preloads statistics only; the finance/sport/shipping examples have no seeding story.
- **[CLARIFY] H9 — Category-creation rule.** "Only create a category if it is part of a category that is not mentioned" (`devplan.md:17`) is not parseable as written.
- **[DECIDE] H10 — Quality limits.** Maximum depth, maximum branching, minimum resources per field, and how to prevent degenerate chains. How is hierarchy quality evaluated?

### 5. Database, embeddings, and summaries

- **[DECIDE] S1 — Schema.** No DDL exists: tables, columns, types, constraints, indexes, timestamps, and provenance. "Create schema ... can be modified later" (`devplan.md:5`) is not a specification.
- **[DECIDE] S2 — Embedding model.** Which model, what dimensions, normalized or not, cosine vs L2 vs inner product, and which pgvector index (HNSW vs IVFFlat). Are fields and summaries embedded with the same model?
- **[DECIDE] S3 — Summary contract.** Length, language, prompt, model, refresh policy, and whether a resource has one summary or multiple representations (abstract, full text, section-level).
- **[DECIDE] S4 — Provenance/versioning.** Store source URL, API, retrieval time, license, content hash, and model/prompt versions. How do you reproduce a past research run?
- **[DECIDE] S5 — Comparison cache.** "Record information about comparisons between resources" (`devplan.md:62`) needs a schema, a cache key, and an invalidation rule.
- **[CLARIFY] S6 — Migrations/backups.** Tooling and policy for schema changes and re-embedding.
- **[DECIDE] S7 — User data isolation.** If multiple users or datasets coexist, how is one user's data kept out of another's retrieval?

### 6. Credibility

- **[DECIDE] C1 — Reliability source.** How are author and publisher reliability assessed: LLM judgment, citation counts, h-index, known publisher lists, retraction databases, or a combination?
- **[DECIDE] C2 — Metric.** The combining formula, weights, scale, threshold, and calibration. Is credibility a hard filter or a ranking signal?
- **[DECIDE] C3 — Overrides.** Does a low credibility score exclude a resource entirely, or only down-rank it? Can the user override?
- **[CLARIFY] C4 — Source-type policy.** Preprints, peer-reviewed papers, news, blogs, and datasets should not share one credibility model without explicit treatment.
- **[DECIDE] C5 — Auditability.** Store the evidence behind each score so a user can see why a source was excluded.
- **[CLARIFY] C6 — Comparison records.** What counts as a "comparison", how long is it valid, and when is it recomputed?

### 7. Hypothesis generation

- **[DECIDE] G1 — Output contract.** Define the Python function signature (input DataFrame? dict? file?), return type (predictions? fitted model? statistic?), applicability predicate, required columns, assumptions, dependencies, serialization, and execution sandbox.
- **[CLARIFY] G2 — "Labelled method".** How is column applicability represented and tested? Is it a predicate over schema, a list of column names, or a natural-language note?
- **[CLARIFY] G3 — Dataset versions.** `devplan.md:78` says two versions then lists three: original, anonymized, and jumbled labels. Define each, which combinations are used, and whether jumbled is also anonymized.
- **[DECIDE] G4 — Jumbling and detection.** How are labels jumbled? What exactly counts as "unscrambling correctly"? How do you avoid false positives and false negatives?
- **[DECIDE] G5 — Context selection.** "All information in parent fields ... and all statistics" (`devplan.md:76`) needs a token budget, retrieval/ranking policy, summarization, and dedup.
- **[DECIDE] G6 — Research-to-model bridge.** When does a paper "present a model that applies" (`devplan.md:81`)? Who judges, against what criteria?
- **[DECIDE] G7 — Agent loop.** Issue taxonomy (runtime errors, diagnostics, validation failures), iteration budget, stopping criteria, same vs different agent, and cost cap.
- **[DECIDE] G8 — Model/provider.** Which LLM, prompts, temperature, seeds, and versioning. How do you reproduce a hypothesis?
- **[CLARIFY] G9 — Statistical correctness.** "Correlation matrix between rows" (`devplan.md:91`) is likely columns/features. The 30% holdout repeated three times (`devplan.md:90`) needs a precise split protocol; multiple testing needs correction; missing data, categorical encoding, outliers, small-n, and class imbalance need policies.
- **[DECIDE] G10 — Causal vs predictive.** The plan generates "hypotheses" and "models" but never says whether these are causal claims. If causal, regressions alone are insufficient; you need an identification strategy.
- **[CLARIFY] G11 — Method applicability.** Log/sqrt transforms "on all sides", Poisson/negative binomial, and SINDy on ordinary tabular data (`devplan.md:92–104`) need applicability checks and fallback rules.
- **[DECIDE] G12 — Ranking and reporting.** How are hypotheses deduplicated, ranked, scored, explained, and presented? What confidence does the user see?

### 8. Validation

- **[DECIDE] V1 — Dataset metadata schema.** The agentic validator examines "metadata the user provided about how the user was gathered" (`devplan.md:112`), but that metadata schema does not exist.
- **[DECIDE] V2 — Agentic rubric.** What biases, confounders, selection effects, and leakage routes are checked? What is the difference between "refine" and "throw out"?
- **[CLARIFY] V3 — "Considered in generation".** The rule at `devplan.md:112` needs an operational definition and an audit trail.
- **[DECIDE] V4 — Statistical validation library.** Which library, which tests, what thresholds, and how multiple comparisons and assumptions are handled.
- **[DECIDE] V5 — Independence.** Are generation and validation performed by separate agents? Are they blind to each other's reasoning? Same data split?
- **[DECIDE] V6 — Validation output.** Pass/fail/uncertain, reasons, effect sizes, confidence intervals, and what the user sees.

### 9. Engineering and operations

- **[DECIDE] E1 — Stack.** Python is implied by pysindy/statsmodels/sklearn, but the framework (FastAPI? Django? CLI?), package manager, and repo structure are unstated.
- **[DECIDE] E2 — Postgres/pgvector setup.** Version, index type, dimensions, distance operator, migration tool, and backup policy.
- **[DECIDE] E3 — API surface.** Endpoints, request/response shapes, auth, async jobs, error codes, and progress reporting.
- **[DECIDE] E4 — Orchestration.** Queue, workers, retries, idempotency, rate limiting, and concurrency.
- **[DECIDE] E5 — Observability.** Logging, metrics, tracing, and cost tracking per research run.
- **[DECIDE] E6 — Security/privacy.** User-data storage, anonymization, PII, secrets, sandboxing generated Python, prompt injection from scraped content, and SSRF.
- **[DECIDE] E7 — Scale targets.** Dataset size, number of resources, expected latency, and throughput.
- **[DECIDE] E8 — Testing.** Unit/integration/e2e strategy, mocked APIs, fixtures, and a regression suite for the hierarchy.
- **[DECIDE] E9 — Deployment/CI.** Environments, CI pipeline, and release process.
- **[DECIDE] E10 — Legal/licensing.** Scraping ToS, `robots.txt`, copyright, API terms, dataset licenses, and attribution.
- **[DECIDE] E11 — Cost budget.** LLM tokens, embeddings, API calls, storage, and a per-dataset ceiling.

## Explicit contradictions and ambiguities

| # | Location | Issue |
|---|---|---|
| 1 | `devplan.md:78` | Says "two versions", then lists three. |
| 2 | `devplan.md:91` | "Correlation matrix between rows" — correlations are normally between columns/features. |
| 3 | `devplan.md:90` | Three runs with "a differing 30% removed" — overlapping or disjoint? Repeated holdout or 3-fold? |
| 4 | `devplan.md:17, 42–43` | Level direction ("increase in level", "down a level") is undefined. |
| 5 | `devplan.md:17` | "Only create a category if it is part of a category that is not mentioned" is unclear. |
| 6 | `devplan.md:15` | "RAG" used as a similarity mechanism; it is a retrieval pattern. |
| 7 | `devplan.md:37` | "Search" is both a document type and an action/API concept. |
| 8 | `devplan.md:64` | News articles are validated but have no document type. |
| 9 | `devplan.md:116` | "Inspection by provided library" — which library, provided by whom? |
| 10 | `devplan.md:43` | "move the parent and all its subfield down a level and place the parent of the current parent as the new field" — referents are ambiguous. |
| 11 | `devplan.md:58` | "effect" should be "affect"; `devplan.md:83` "we the get the agent". |
| 12 | `devplan.md:79` | "If the agent ... unscrambles the labels correctly, we ignore its results" — no definition of detection or why this invalidates the result. |

## Coverage against the `/to-spec` template

| Spec section | Status | What's missing |
|---|---|---|
| Problem Statement | ❌ Missing | No user, problem, or motivation. |
| Solution | ⚠️ Partial | Describes subsystems, not the user-facing solution. |
| User Stories | ❌ Missing | None. |
| Implementation Decisions | ⚠️ Partial | Many architectural ideas, but most are unresolved decisions. |
| Testing Decisions | ❌ Missing | No test strategy or seams. |
| Out of Scope | ❌ Missing | Everything is implicitly in scope. |
| Further Notes | ⚠️ Partial | Research ideas and method menu exist. |

## Suggested first wayfinder tickets

These are the decisions I'd chart first. The first block is unblocked; the second depends on them.

| Ticket | Type | Blocked by |
|---|---|---|
| Define the destination and MVP boundary | grilling | — |
| Build the domain glossary (`GLOSSARY.md`) | grilling | — |
| Verify external API capabilities and limits | research | — |
| Choose the first-slice stack and deployment target | grilling | — |
| Define the core data model and provenance schema | grilling | glossary, stack |
| Specify the field-hierarchy algorithm and invariants | grilling / prototype | data model |
| Define the research and crawl policy | grilling | API research, data model |
| Define the credibility model | grilling / research | data model, research policy |
| Define the hypothesis output contract | grilling / prototype | MVP, stack |
| Define the statistical and validation pipeline | grilling / research | output contract, data model |
| Define the agentic generation/validation loop | grilling | output contract, statistical pipeline |
| Define the evaluation harness and success metrics | grilling | MVP, statistical pipeline |
| Define the preloaded statistics corpus | research | field hierarchy, API research |
| Define the user-facing API/CLI and workflow | grilling / prototype | MVP, output contract |

## Recommended next step

Run **`/wayfinder`** with the destination: *"a revised, buildable spec for the smallest demoable slice of the research-to-hypothesis system."* Feed it this audit as the **Not yet specified** / ticket seed. The first ticket should be the MVP boundary — without it, every other decision stays equally weighted.

If you'd rather settle the whole thing in one sitting, run **`/grill-with-docs`** first, using the top-10 list above as the interview agenda, then **`/to-spec`**.

## Fact-check notes

- Google Dataset Search has no documented public API; R3 needs an explicit fallback-or-drop decision.
- `devplan.md:15` "RAG" should read **embedding similarity / vector search**; RAG is a retrieval-plus-generation framework, not a metric.
- `devplan.md:91` "correlation matrix between rows" should be **columns/features**.
- Citation nits: "down a level" is at `devplan.md:43`, not `:17`; `devplan.md:112` says "how the user was gathered," not "how the data was gathered."
- pgvector (HNSW/IVFFlat), statsmodels formulas, sklearn `PolynomialFeatures(interaction_only=True)`, pysindy, and Poisson/negative-binomial usage all check out.
