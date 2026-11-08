# Referee glossary

**Referee**:
The project in this repository: a local-first autonomous research analyst that
turns a pandas-readable dataset into a dynamically classified field path, cited
research factors, runnable models, independent validation, and an audit trail.
The Python package and CLI keep the historical `whack26` name.
_Avoid_: Whack26 (for the product), pipeline (unqualified)

## Language

### Classification

**Field**:
A topic/domain node in the dynamic hierarchy, identified by an LLM-written name and a one- or two-sentence description. Fields are created from data and are never pre-seeded.
_Avoid_: Category, cluster, tag, label

**Field path**:
The root-to-leaf sequence of fields assigned to a dataset. Each dataset has exactly one field path.
_Avoid_: Classification chain, hierarchy path

**Level**:
The depth of a field in the tree. Roots are level 1 and the maximum is 4; level always equals depth.
_Avoid_: Tier, rank

**Dataset summary**:
The compact text representation of a dataset — schema, dtypes, missingness, cardinality, sample statistics, and a few sample rows — plus the optional user description and source site. It is the classification and embedding input, not the raw dataset.
_Avoid_: Profile, metadata

**Source site**:
The optional URL or site name the user supplies to record where a dataset came from. It is provenance metadata for the summary and artifact, not a research source.
_Avoid_: Source, origin

**Dataset profile**:
The structured object holding the summary inputs and the detected column roles: target, time, id, leakage risk, feature.
_Avoid_: Schema

**Dataset embedding**:
The vector of the dataset summary, used for field similarity and centroid updates.

**Field embedding**:
The pair of vectors representing a field: the embedding of its description and the centroid of its member dataset embeddings. Similarity blends both.
_Avoid_: Category vector

**Classification**:
The act of assigning a dataset to a field path, creating fields when no existing node fits. Classification happens before research; research may refine it once.
_Avoid_: Tagging, labeling, clustering

**Leaf field**:
The most specific field a document is attached to. `documents.leaf_field_id` is the canonical attachment; `document_fields` stores the full root-to-leaf membership, and `datasets.path_json` mirrors it for datasets.
_Avoid_: Terminal field, assigned field

**Field operation**:
An audited hierarchy change recorded in `field_operations`: `create`, `merge`, `promote`, `reassign_leaf`, `recompute_path`, `prune`, or `delete`. Each row records old/new parent and level, affected documents, a reason, and the run that made the change.

**Branch**:
A new child field created under an existing field when a document is specific enough to split the parent. Duplicate siblings are merged instead of duplicated.

**Promotion**:
Moving a field up one level (or swapping it with its parent) when the subtree is mis-levelled, then re-levelling and re-pathing the affected documents.

**Hierarchy version**:
A monotonically increasing counter in `store_meta` bumped whenever the tree changes, so caches can detect stale paths.

**Directly related**:
A document is directly related to field `f` when the similarity of its summary embedding to the field's blended description/centroid score meets the level's threshold `tau[level]`. Placement attaches a document to the shallowest directly related field, not to a merely topical ancestor of a more specific match.
_Avoid_: Matched, relevant

### Research

**Research record**:
A normalized paper, dataset, or web result fetched from a source, identified by `source:source_id`, with title, abstract or summary, authors, year, DOI/URL, license, query, and retrieval time.
_Avoid_: Document, source

**Document**:
A catalogued Dataset, Search, Paper, or Web resource in the SQLite v4 knowledge store. It has a standardized summary, a summary embedding, authors/publisher metadata, a credibility score, a leaf field, optional field assignments, and to/from links.
_Avoid_: Resource (in code and schema), record

**Search document**:
A `kind="search"` document recording one query executed against one source, with its parameters, session, timestamp, result count, and links to its results and follow-up searches.
_Avoid_: Query log, search record

**Catalog query**:
A typed filter over stored documents by kind, source, field subtree, year, credibility, preloaded state, text, or similarity, with deterministic ordering and pagination. Catalog queries are library-only today; `run` records Search documents but does not expose a catalog CLI command.
_Avoid_: SQL query, search API

**Research metrics**:
The optional, key-gated OpenAlex enrichment that resolves document citation counts and author h-index/citation totals after ranking. It is cached, offline-safe, and advisory; failures become warnings.
_Avoid_: Bibliometrics, scoring

**Semantic scraping**:
Target-gated expansion from seed research records through API references and links. A high-similarity parent contributes all its links; a lower-similarity parent contributes only links whose own summaries pass a tighter threshold. Budgets and robots rules bound the crawl.
_Avoid_: Crawling, scraping (unqualified)

**Document link**:
A directed relation between two documents — `references`, `cites`, `linked_from`, `same_as`, `dataset_paper`, or `search_result`.
_Avoid_: Edge, citation

**Credibility score**:
A deterministic, decomposable score over author, venue, source type, recency, and citations, with penalties for retraction, predatory venues, and missing dates. Low-credibility documents are stored but excluded as evidence.
_Avoid_: Trust score, ranking

**Document comparison**:
A cached pairwise judgement between two documents (`dataset_relevance`, `method_overlap`, `finding_agreement`, `population_overlap`, or `contradiction`) with a verdict, similarity, rationale, and evidence. Reused until an embedding or method version changes.
_Avoid_: Diff, match

**Preloaded statistics corpus**:
The layer-1 documents fetched from the 13 statistics/validity seed queries and classified under `Statistics and Validity`. They augment the static pitfalls corpus for method critique and hypothesis context.
_Avoid_: Training set, taxonomy

**Field pitfall**:
A cited, field-scoped caveat retrieved for one node of a dataset's field path. It records the scope (`leaf` or `ancestor`), the category, whether it applies to the dataset (`yes`, `unknown`, or `no`), and a stable `F:<field_id>:<n>` citation. A field pitfall is a candidate caveat, not a finding; `unknown` is background context only.
_Avoid_: Finding, risk, generic caveat

**Factor**:
A candidate determinant or correlate extracted from research records, with a description, direction, claim type, and evidence IDs. A factor may or may not map to a dataset column.
_Avoid_: Feature, variable

**Factor mapping**:
The link from a factor to a dataset column, with confidence and rationale. `in_dataset = false` means the factor is context only and must not be used as a feature.
_Avoid_: Column match

### Hypotheses and validation

**Hypothesis**:
A runnable Python function plus metadata that proposes a relationship or model for the dataset. It has an applicability predicate, required columns, assumptions, a method family, cited factors, and a derivation rationale.
_Avoid_: Model, idea

**Agent loop**:
The optional bounded propose → execute → diagnose → refine/replace/discard cycle that replaces the one-shot hypothesis generator when `WHACK26_AGENT_LOOP_ENABLED=1`. It records every attempt and never exceeds its attempt or time budget.
_Avoid_: Retry loop, auto-repair

**Context pack**:
The bounded, deterministically ordered bundle of field-path/ancestor context, research, factors, model-search candidates, statistics, corpus excerpts, and applicable field pitfalls supplied to generation.
_Avoid_: Prompt, memory

**Applicability predicate**:
The `applies(df) -> bool` function that decides whether a hypothesis can run on a given dataset.
_Avoid_: Filter, guard

**Model search**:
The deterministic search over typed candidate specifications (baseline, linear, logistic, multinomial, Poisson, negative binomial, interaction, group comparison, optional tree/SINDy) across repeated holdouts, followed by screening, selection, and diagnostics.
_Avoid_: AutoML, model training

**Model candidate**:
One typed `ModelSpec` plus its per-fold metrics, baseline deltas, coefficients, diagnostics, AIC/BIC, and rank in the model-search result.

**Dataset variant**:
An anonymized or target-jumbled copy of the dataset used to test PII reliance and label leakage. A jumbled-label model that still performs well is rejected.
_Avoid_: Synthetic data, perturbation

**Research model**:
A model specification reported by a paper or web document, evidence-backed and mapped to dataset columns, stored in `research_models` and optionally applied by a generated hypothesis.
_Avoid_: Extracted model, paper model

**Collection metadata**:
The declared data-generation design: observational/experimental/quasi-experimental/survey/administrative, unit of analysis, sampling, treatment, instrument, running variable, identification strategy, known biases, and measurement notes. Stored in `datasets.collection_json`.
_Avoid_: Data dictionary, provenance

**Generation validation**:
The claim/design review that checks whether a hypothesis's derivation and claim match the collection process. It is separate from statistical validation and never judges model performance.
_Avoid_: Statistical validation, critic

**Method critic**:
The agentic reviewer that checks the derivation method and claim type against the preloaded pitfalls corpus. It does not judge statistical validity or model performance; the independent validator owns that.
_Avoid_: Validator, judge

**Final model judge**:
The advisory run-level audit over the selected model, its statistical validation, and the method-critic results. It reports a decision, code-verified flags, and re-audit steps, and writes a UI-ready HTML fragment. It never re-ranks candidates or blocks artifact emission.
_Avoid_: Method critic, validator, statistical validation

**Statistical validation**:
The independent post-generation pass that runs the four-layer engine over a fitted model: Deflated Sharpe Ratio, Monte Carlo permutation, chronological out-of-sample decay, and parameter sensitivity. It reports `passed`, `failed`, `partial`, `skipped`, or `error` and never blocks artifact emission.
_Avoid_: Critic verdict, performance score

**Statistical backend**:
The engine selected for a validation target: `auto`, `in_slice`, `stats_engine`, `external`, or `none`. `in_slice` is the default for general regression, classification, and count data; `stats_engine` is reserved for continuous, return-like series; `external` delegates to a configured validator. The backend registry exists and is tested, but the current run/validation pipeline uses the target-aware `in_slice`/`stats_engine` path and does not yet consume `WHACK26_STATISTICAL_BACKEND`. When no backend supports the data the verdict is `insufficient`, not `failed`.
_Avoid_: Validator, engine (unqualified)

**Expected calibration error (ECE)**:
The weighted gap between predicted confidence and observed accuracy across probability bins. It is reported for classification validation and can gate a run with `max_ece`.
_Avoid_: Calibration score

**Dispersion**:
The ratio of observed variance to the mean for a count target (Pearson dispersion). Values above 1 indicate overdispersion; count validation reports it and can gate a run with `max_dispersion`.
_Avoid_: Variance

**Run**:
One execution of the pipeline into a run directory. It owns the dataset copy, generated code and predictions, logs, sidecar files, and `artifact.json`.
_Avoid_: Job, experiment

**Artifact**:
The versioned JSON handoff, containing the dataset profile, field path, research records, factors, hypotheses, execution results, critic verdict, and the optional validation, model-search, variant, research-model, collection, field-pitfall, final-judge, and scraping sections.
_Avoid_: Report, output file

**Preview**:
The Markdown rendering generated only from `artifact.json`, used by the report owner and for a quick human-readable audit.
_Avoid_: Report

### Evaluation

**Cold run**:
An evaluation run against an empty field store and cold research cache, so all fields are created from scratch.

**Warm run**:
An evaluation run against the existing field store and cache, measuring assignment stability and duplicate creation.

**Mock mode**:
A development mode that replaces LLM and embedding calls with deterministic fixtures, so the pipeline can be built and tested before real keys are configured.
