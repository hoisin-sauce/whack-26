# Research-to-Hypothesis Slice

A local-first pipeline that turns a pandas-readable dataset into a dynamically classified field path, cited research factors, three runnable hypotheses, and a method-critic verdict.

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
The compact text representation of a dataset — schema, dtypes, missingness, cardinality, sample statistics, and a few sample rows — plus the optional user description. It is the classification and embedding input, not the raw dataset.
_Avoid_: Profile, metadata

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

### Research

**Research record**:
A normalized paper, dataset, or web result fetched from a source, identified by `source:source_id`, with title, abstract or summary, authors, year, DOI/URL, license, query, and retrieval time.
_Avoid_: Document, source

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

**Applicability predicate**:
The `applies(df) -> bool` function that decides whether a hypothesis can run on a given dataset.
_Avoid_: Filter, guard

**Method critic**:
The agentic reviewer that checks the derivation method and claim type against the preloaded pitfalls corpus. It does not judge statistical validity or model performance; the independent validator owns that.
_Avoid_: Validator, judge

**Artifact**:
The versioned JSON handoff plus Markdown preview, containing the dataset profile, field path, research records, factors, hypotheses, execution results, and critic verdict.
_Avoid_: Report, output file

### Evaluation

**Cold run**:
An evaluation run against an empty field store and cold research cache, so all fields are created from scratch.

**Warm run**:
An evaluation run against the existing field store and cache, measuring assignment stability and duplicate creation.

**Mock mode**:
A development mode that replaces LLM and embedding calls with deterministic fixtures, so the pipeline can be built and tested before real keys are configured.
