---
title: "Statistical Modelling Pitfalls — Method-Critic Corpus"
corpus_id: statistical-pitfalls
version: 1.0.1
date: 2026-11-08
ticket: "hoisin-sauce/whack-26#10"
audience: "method critic (LLM agent)"
scope: >
  Retrieve when judging whether the method used to derive a statistical model
  can support the claim being made. The unit of critique is the derivation:
  data provenance, assumptions, leakage, selection, identification, diagnostics,
  sample size, and reporting.
out_of_scope: >
  Recomputing or ranking model performance. A separate statistical validator
  owns discrimination, calibration, error metrics, and benchmark comparisons.
  This corpus checks whether the reported performance estimate is trustworthy
  and whether the method supports the claim.
source_policy: >
  Primary methods papers, reporting guidelines, and authoritative textbooks.
  Every section ends with its own source list so a retrieved chunk is self-contained.
---

# Statistical modelling pitfalls: a method-critic corpus

> **One-line rule.** A model is only as credible as the gap between its claim and its identification or validation strategy. The critic's job is to find that gap.

## 0. How to use this corpus

**Chunk ID:** `S0-how-to-use`

**What this corpus is for.** It is a preloaded reference for an agentic critic that reviews the *method used to derive a model* — in an external paper, a news article, or a hypothesis generated inside this system. It is deliberately about derivation validity, not about whether the fitted model scores well on a benchmark.

**What it is not for.** Do not use it to re-score a model's accuracy, AUC, RMSE, or calibration curve. Another component does that. Use it to decide whether the performance number was obtained honestly (no leakage, correct split, no test-set reuse), whether the model family matches the outcome and estimand, and whether the claim is the kind of claim the design can support.

**Chunking contract.** Each `##` section that declares a **Chunk ID** is a self-contained retrieval chunk; §11 (checklist) and §12 (phrasebook) are appendices for human reference and are not retrieved. Each retrievable chunk begins with a **Chunk ID**, **Triggers**, and **Critic question**. Subheadings are safe to split further. If a retrieved chunk is ambiguous, fall back to the section's "Verdict defaults" and "Red flags".

### 0.1 Claim-type gate (do this first)

Classify the claim before applying any checklist. The same method can be valid for one claim type and invalid for another.

| Claim type | Typical language | What must be validated | Read |
|---|---|---|---|
| **Description** | "X is associated with Y"; "prevalence"; "trend"; "difference between groups" | sampling frame, representativeness, measurement, effect size, uncertainty | §1, §8, §10 |
| **Prediction** | "predicts", "flags", "risk score", "forecast", "classifies" | leakage, split protocol, calibration, discrimination, external validation, distribution shift | §2, §5, §9, §10 |
| **Causal** | "causes", "reduces", "increases", "effect of", "impact", "leads to", "driver" | identification assumptions, design, confounding, positivity, interference, sensitivity analysis | §4 |
| **Explanatory / inferential** | "coefficient", "independent effect", "mediation", "pathway" | model specification, confounding, multiplicity, post-selection inference, effect size | §3, §4, §5 |

If the claim type is unclear, mark the item `INSUFFICIENT` and do not promote it.

### 0.2 Verdict vocabulary

| Verdict | Meaning | Typical evidence |
|---|---|---|
| **PASS** | Method and claim align; key assumptions stated and checked; no fatal leakage; uncertainty and multiplicity handled; enough detail to reproduce. | clean split, diagnostics, pre-specification or correction, adequate n, sensitivity analysis |
| **REFINE** | Fixable gap: missing diagnostics, unclear split, unlabeled exploratory analysis, incomplete reporting. The model may be salvageable. | missing dispersion test, unclear preprocessing order, no calibration report |
| **REJECT** | Fatal flaw: target/temporal leakage; causal claim without identification; model family incompatible with outcome/estimand; sample size grossly inadequate for the claim; selective search with no correction or replication. | test-set tuning, "controlled for everything" causal claim, OLS on counts, 20 predictors with 15 events |
| **INSUFFICIENT** | Cannot verify from the report. Do not promote the claim; request the missing method detail. | no split description, no model specification, no missing-data handling |

**Severity:** `fatal` invalidates the claim; `major` likely changes the conclusion or makes it unreliable; `minor` affects interpretation or reporting but not the core conclusion.

### 0.3 Retrieval map

| If the method mentions… | Read |
|---|---|
| OLS, linear regression, "assumptions", Gauss–Markov, normality, homoscedasticity | §1, §5 |
| train/test split, cross-validation, feature selection, imputation, scaling, SMOTE, hyperparameter tuning | §2, §5 |
| p-values, "significant", many tests, subgroups, stepwise selection, "best model" | §3 |
| "effect", "impact", "causes", propensity score, IV, DiD, RD, synthetic control, mediation | §4 |
| residuals, VIF, outliers, Cook's distance, transformations, AIC/BIC, interactions | §5 |
| Poisson, negative binomial, zero-inflated, hurdle, rates, counts, exposure, offset | §6 |
| sample size, power, events per variable, rare events, external validation | §7 |
| missing data, imputation, MCAR/MAR/MNAR, measurement error, reliability, agreement | §8 |
| time series, seasonality, trend, stationarity, forecasting, panel data | §9 |
| TRIPOD, STROBE, CONSORT, PROBAST, reproducibility, code/data availability | §10 |

### 0.4 Suggested critic output shape

The corpus is format-agnostic, but this schema keeps findings consistent and auditable:

```json
{
  "claim_type": "description | prediction | causal | explanatory",
  "verdict": "pass | refine | reject | insufficient",
  "issues": [
    {
      "chunk": "S2-leakage",
      "severity": "fatal | major | minor",
      "finding": "Preprocessing (imputation and scaling) was fit on the full dataset before the train/test split.",
      "evidence": "Methods, paragraph 3",
      "why": "Test-set information entered training; the reported performance is optimistic.",
      "fix": "Fit all preprocessing inside the training folds and re-evaluate on an untouched test set."
    }
  ],
  "missing_information": ["unit of analysis", "split protocol", "dispersion test"]
}
```

### 0.5 Source hierarchy and citation convention

Prefer, in order: (1) methodological primary papers and reporting guidelines; (2) authoritative textbooks and definitive reviews; (3) applied tutorials from the owning field. Avoid blog posts, vendor documentation, and unverifiable preprints for methodological claims. Inline citations are author–year with DOI links; each section repeats its sources so retrieval does not depend on the bibliography.

---

## 1. Assumptions and estimands

**Chunk ID:** `S1-assumptions`  
**Triggers:** assumptions, estimand, OLS, linear regression, logistic regression, Poisson, negative binomial, Cox, survival, mixed model, multilevel, time series, causal, Gauss–Markov, exogeneity, homoscedasticity, independence, normality, positivity, consistency, SUTVA, exchangeability.  
**Critic question:** Does the model's assumptions match the estimand, the outcome type, and the data-generating process — and did the authors check the assumptions that matter?

### 1.1 The estimand comes first

A regression coefficient can be an association, a prediction input, or a causal effect. Those are different targets with different assumptions. Before checking diagnostics, ask: *what quantity is the paper trying to estimate, and for which population and time point?* A method that is adequate for prediction can be inadequate for a causal claim, and vice versa (Shmueli, 2010; Hernán, Hsu & Healy, 2019).

"All models are wrong" is not a licence to skip assumptions; it is a reason to ask which departures matter for the conclusion (Box, 1976). The critic should require a statement of the assumption, a diagnostic aimed at it, a decision rule, and a sensitivity analysis when the assumption cannot be tested.

### 1.2 Assumption map by model family

| Model family | Key assumptions for the claim | Minimum diagnostics / checks |
|---|---|---|
| **OLS / linear regression** | linearity in parameters; zero conditional mean (no omitted confounding); homoscedasticity; independent errors; no perfect collinearity | residual vs fitted; QQ and scale–location; leverage and Cook's distance; VIF/condition index; RESET and Breusch–Pagan; ACF if ordered |
| **Logistic / probit** | linearity on the link scale; independent observations; no separation; correct outcome definition; adequate events per parameter | component-plus-residual plot; calibration; influence; separation check; events per variable |
| **Poisson / negative binomial** | log-linearity; correct mean–variance function; independence; exposure offset; no unmodeled excess zeros | dispersion test; residual plots; observed vs predicted zeros; offset check; cluster-robust SE |
| **Cox / survival** | proportional hazards; non-informative censoring; correct time origin; no immortal time; competing risks handled | Schoenfeld residuals; martingale residuals; cumulative incidence; proportional-hazards test |
| **Multilevel / mixed** | correct random-effects structure; exchangeable clusters; residual independence; adequate clusters | ICC; caterpillar plots; residual checks; sensitivity to random-effects specification |
| **Time series** | stationarity (or modeled trend); correct lag/seasonality; no structural break; stable error variance | ACF/PACF; unit-root/stationarity tests; Ljung–Box; rolling-origin CV; break tests |
| **Causal (any design)** | consistency; positivity; exchangeability/no unmeasured confounding; SUTVA/no interference; correct treatment and covariate measurement | DAG; balance/overlap; design-specific diagnostics; E-value or negative controls; sensitivity analysis |

Sources for the table: Harrell (2015); McCullagh & Nelder (1989); Cameron & Trivedi (2013); Grambsch & Therneau (1994); Fine & Gray (1999); Hernán & Robins (2020); Westreich & Cole (2010); Hyndman & Athanasopoulos (2021).

### 1.3 Data-shape and provenance checks

- **Unit of analysis.** Are rows independent units (patients, transactions, counties), or repeated measures? If repeated, the effective sample size is closer to the number of clusters than the number of rows.
- **Orientation.** Rows are usually observations and columns are variables. A "correlation matrix between rows" is not a feature correlation matrix for tabular data; if rows really are variables, the text must say so.
- **Time zero.** For survival or time-to-event data, when does follow-up start, when does it end, and could exposure status change during follow-up?
- **Derived variables.** Ratios, change scores, indices, and residuals have their own measurement and denominator problems. Check how they were constructed and whether they were computed before or after the split.
- **Selection.** Who is in the sample, who is excluded, and why? A model for a selected population does not automatically generalize.
- **Measurement.** What instrument produced each variable, when, and with what reliability? A noisy predictor attenuates associations (Frost & Thompson, 2000).
- **Provenance.** Source, collection period, version, and any preprocessing done by a third party.

### 1.4 What "checked" should mean

A credible assumptions section contains four things:

1. **Statement.** The assumption is named in the language of the model, not just "assumptions were met".
2. **Diagnostic.** A plot or test that targets that assumption (see §5).
3. **Decision rule.** What was considered a violation, and what was done about it.
4. **Sensitivity.** If the assumption is untestable (especially causal assumptions), a sensitivity analysis or bounds, not a p-value.

A non-significant diagnostic test does not prove the assumption. Tests have power that depends on sample size, and absence of evidence is not evidence of absence (Greenland et al., 2016; Wasserstein & Lazar, 2016).

### 1.5 Red flags

- "All assumptions were satisfied" with no plots, tests, or decision rules.
- "Normality was confirmed by Shapiro–Wilk" as the only check, while linearity, independence, and heteroscedasticity are ignored.
- "The model fits well, therefore the assumptions hold."
- "p > 0.05, so there is no violation."
- OLS on a binary, ordinal, count, bounded, censored, or time-to-event outcome without justification.
- Poisson on a rate whose denominator is unknown or unmeasured.
- Survival model without a defined time origin, or with treatment defined after follow-up begins (immortal time).
- "We controlled for all variables" without a causal diagram.
- "We used a tree model, so no assumptions are needed" — tree models still assume exchangeable observations and are sensitive to leakage and distribution shift.
- SINDy or another dynamical-systems method applied to cross-sectional tabular data without time-resolved state variables or derivative estimates (Brunton, Proctor & Kutz, 2016).

### 1.6 Verdict defaults

- **REJECT** if the model family cannot represent the outcome or estimand (e.g., OLS on counts presented as a rate model; a survival claim with no time origin).
- **REFINE** if assumptions are testable but unreported, or if a violation is reported without a remedy.
- **PASS** if assumptions are stated, checked with appropriate diagnostics, and violations are addressed or bounded by sensitivity analysis.

### 1.7 Sources

Box (1976); Brunton, Proctor & Kutz (2016); Cameron & Trivedi (2013); Fine & Gray (1999); Frost & Thompson (2000); Grambsch & Therneau (1994); Greenland et al. (2016); Harrell (2015); Hernán, Hsu & Healy (2019); Hernán & Robins (2020); Hyndman & Athanasopoulos (2021); McCullagh & Nelder (1989); Shmueli (2010); Wasserstein & Lazar (2016); Westreich & Cole (2010).

---

## 2. Data leakage and contamination

**Chunk ID:** `S2-leakage`  
**Triggers:** leakage, contamination, train/test split, holdout, cross-validation, preprocessing, imputation before split, feature selection, SMOTE, oversampling, duplicates, group leakage, temporal leakage, look-ahead bias, test-set reuse, nested CV.  
**Critic question:** Did any information from the test set, the future, the outcome, or the same unit enter model training or model selection?

### 2.1 Definition

Data leakage is the introduction of information about the target that should not be legitimately available at prediction time (Kaufman et al., 2012). In its common tabular form it appears as a spurious relationship between predictors and the target that arises as an artefact of data collection, sampling, or preprocessing; it will not be present in the distribution about which the claim is made, so it usually inflates apparent performance (Kapoor & Narayanan, 2023). Leakage is not only an ML problem: it affects any analysis that selects features, tunes choices, or preprocesses before holding data out.

### 2.2 Leakage taxonomy

| Level | Type | What it looks like | Check |
|---|---|---|---|
| **L1** | No clean train/test separation | no test set; preprocessing, imputation, oversampling, or feature selection on the full dataset; duplicates/near-duplicates across splits | all preprocessing inside CV folds; deduplicate before splitting; untouched test set |
| **L2** | Illegitimate features | outcome proxies; post-treatment variables; future information; ID or row order; features unavailable at prediction time | feature timing and availability argument for every feature group |
| **L3.1** | Temporal leakage | random split on time series; future values used to build features; test period before training period | rolling-origin/backtest split; feature timestamps |
| **L3.2** | Non-independence | same patient, site, subject, or cluster in train and test | group/block split at the cluster level |
| **L3.3** | Sampling bias in test distribution | test set from a different population, site, or period than the claim | define target population; external validation; report distribution shift |

The L1/L2/L3 structure follows Kapoor & Narayanan's (2023) eight-type taxonomy; the table merges their four L1 subtypes (no test set; preprocessing on train+test; feature selection on train+test; duplicates) into a single row, and their L2 is a single category. The practical test is the same: *would this information be available, in this form, when the claim is used?*

### 2.3 Pipeline audit

Walk the pipeline in order and ask where each piece of information came from:

1. **Collection.** Is each feature measured before the outcome? Is it a proxy for the outcome or for treatment? Would a deployed model have it at prediction time?
2. **Splitting.** Was the split created before any preprocessing? Is the split unit the same as the claim unit (patient, site, time)? Are duplicates removed before splitting?
3. **Preprocessing.** Were scaling, imputation, encoding, oversampling, dimensionality reduction, and feature selection fit *inside* the training folds only?
4. **Modeling.** Was target encoding fit fold-wise? Were hyperparameters tuned on a validation set, not the test set? Was model selection done on the test set?
5. **Evaluation.** Is the test set used once, at the end? Is the "full-data model" evaluated in-sample? Are all runs reported, or only the best?

### 2.4 Split-protocol audit (including the 30% holdout × 3 design)

The development plan calls for three runs, each removing a different 30% of records, plus a full-data run. Audit it explicitly:

- **Are the three holdouts disjoint?** If not, some observations are tested more than once and the performance estimates are dependent. If disjoint, they form a 3-fold partition covering 90% of the data, with 10% never tested — say which.
- **Was the split made before imputation, scaling, encoding, and feature selection?** If not, test information leaked into training.
- **Is the split stratified by outcome and key groups?** Stratification helps balance, but group/time structure must still be respected.
- **Is the unit of split the unit of analysis?** Split patients, not visits; sites, not rows; time blocks, not random days.
- **Is there a final untouched test set?** If the same holdout is used to choose the model and report its performance, the estimate is optimistic.
- **Is the full-data model's performance labelled in-sample?** A full-data fit can be useful for final inference, but its apparent R²/AUC is not validation.
- **Are all three runs reported with variability, or only the best?** Selecting the best run is a multiplicity problem (see §3).
- **Is the metric appropriate for the claim?** Calibration for risk, discrimination for ranking, error/loss for forecasting, effect size for explanation.

### 2.5 Red flags

- "We normalized the entire dataset before splitting."
- "We selected the top-k features on the full dataset, then cross-validated."
- "We imputed missing values using the full dataset."
- "We used SMOTE/oversampling before splitting."
- "We used a random 80/20 split" for time-series, panel, spatial, or clustered data.
- "We removed duplicates after splitting."
- "We tuned hyperparameters on the test set."
- "We chose the model with the best test performance."
- "We report the full-data model's R²/AUC" without calling it apparent performance.
- "We included a feature measured after the outcome."
- "We used an ID, row index, or file order as a feature."
- "We used future values of the outcome to construct features."
- "The same patients appear in train and test."
- "We report only the best of three holdout runs."

### 2.6 Verdict defaults

- **REJECT** on target leakage, temporal leakage, preprocessing or feature selection outside the CV loop, duplicate leakage, or test-set reuse for model selection.
- **REFINE** on unclear split order, mild full-data scaling without outcome use, or duplicate handling that is described but not demonstrated.
- **PASS** if the split is defined before preprocessing, group/time-aware, the test set is untouched, all preprocessing is inside folds, and all runs are reported.

### 2.7 Sources

Ambroise & McLachlan (2002); Cawley & Talbot (2010); Kapoor & Narayanan (2023); Kaufman et al. (2012); Kriegeskorte et al. (2009); López de Prado (2018); Roberts et al. (2017); Varma & Simon (2006).

---

## 3. Multiplicity, selection, and specification search

**Chunk ID:** `S3-multiplicity`  
**Triggers:** multiple comparisons, multiplicity, p-hacking, garden of forking paths, HARKing, preregistration, family-wise error, false discovery rate, Bonferroni, Holm, Benjamini–Hochberg, subgroup analysis, data snooping, selective inference, specification curve, multiverse analysis, stepwise selection.  
**Critic question:** How many chances did the data have to produce this result, and was that accounted for in the claim?

### 3.1 The denominator problem

Multiplicity is not just the number of p-values printed. It is the number of analytic choices that could have been made differently: outcomes, subgroups, covariates, transformations, model families, thresholds, exclusion rules, splitters, and metrics (Gelman & Loken, 2014; Simmons, Nelson & Simonsohn, 2011; Wicherts et al., 2016). If a paper reports only the significant results, the number of tests is unknown; treat the finding as selectively reported rather than as a confirmed effect (Ioannidis, 2005; Head et al., 2015).

A p-value is not the probability that the null hypothesis is true, nor the probability that the result will replicate, nor a measure of effect size (Wasserstein & Lazar, 2016; Greenland et al., 2016). Report effect sizes with confidence intervals and interpret them in context.

### 3.2 Error-rate control

| Method | Controls | Use when | Caveat |
|---|---|---|---|
| **Bonferroni** | family-wise error rate (FWER) | small, pre-specified family; any dependence | conservative; power loss |
| **Holm** | FWER | same as Bonferroni, uniformly more powerful | still conservative with many tests |
| **Benjamini–Hochberg** | false discovery rate (FDR) | large discovery scans; independent tests (positive dependence under PRDS is covered by Benjamini & Yekutieli, 2001) | does not control FWER |
| **Benjamini–Yekutieli** | FDR | arbitrary dependence | more conservative than BH |
| **Permutation / randomization** | exact error rate under the design | exchangeable units; designed experiments | requires a valid randomization scheme |
| **Hierarchical gatekeeping** | FWER/FDR over pre-specified families | multiple endpoints with a logical order | families must be pre-specified |

Sources: Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001); Bland & Altman (1995); Bender & Lange (2001); Holm (1979).

### 3.3 When adjustment is expected — and when it is not

Adjustment is expected for **confirmatory** claims: a pre-specified primary outcome tested alongside many secondary outcomes, subgroup claims, feature scans (omics, imaging, text), and model selection that is presented as inference. It is not a mechanical requirement for a single pre-specified primary analysis, or for descriptive/exploratory work. Rothman (1990) argues that routine adjustment is inappropriate outside contrived settings; the broader guidance is to label an exploratory scan as exploratory, report the full search, and replicate it rather than pretend a Bonferroni correction turns it into confirmation (Rothman, 1990; Bender & Lange, 2001).

The family to correct is the set of tests that supports the claim, not every test in the paper. State the family and the error rate.

### 3.4 Selection routes to check

- **Outcome switching.** Was the primary outcome pre-specified, or chosen after seeing the data?
- **Subgroup fishing.** Were subgroups pre-specified? Is the claim based on an interaction test, or on "significant in one subgroup, not in another"?
- **Model selection.** Were many models fitted and the best reported? Stepwise selection, AIC/BIC search, and hyperparameter tuning all consume degrees of freedom.
- **Transformation search.** Were log/sqrt/Box–Cox transformations tried until significance?
- **Outlier removal.** Were observations removed because they weakened the result?
- **Threshold search.** Were cut-points chosen to maximize a statistic?
- **Metric search.** Was the metric chosen after seeing which one made the model look best?
- **Split search.** Were multiple splits run and the best reported?

### 3.5 Post-selection inference

If the model was selected using the same data used for inference, conventional p-values and confidence intervals are invalid. Acceptable remedies include sample splitting, selective inference, bootstrap/permutation under the selection procedure, penalization, pre-registration, or independent replication (Berk et al., 2013; Taylor & Tibshirani, 2015; White, 2000). A specification curve or multiverse analysis is a transparent description of the search, not a correction by itself; it becomes evidence when the distribution of estimates is reported and interpreted (Steegen et al., 2016; Simonsohn, Simmons & Nelson, 2020).

### 3.6 Subgroup analyses

Subgroup claims need: a pre-specified subgroup, an interaction test (not two separate significance tests), a report of all subgroups examined, a multiplicity strategy, and a power statement. "Significant in women, not in men" is not evidence of effect modification (Wang et al., 2007; Rothwell, 2005).

### 3.7 Red flags

- "We tested several models and report the best."
- "We found a significant subgroup."
- "We compared responders with non-responders" as a post-hoc analysis.
- "We removed outliers until the result was significant."
- "We tried several transformations and report the significant one."
- "We used stepwise selection, then report p-values."
- "We did not adjust for multiple comparisons" in a discovery scan.
- "p = 0.049, so the effect is real."
- "The result was significant in one of three holdout runs."
- "We report only the significant associations."
- "We changed the primary outcome after seeing the data."
- "We used a one-sided test after seeing the direction."
- "We tested many endpoints but report one."

### 3.8 Verdict defaults

- **REJECT** a confirmatory or causal claim built on an unadjusted selective search with no replication.
- **REFINE** if the analysis is exploratory but not labelled, or if the family of tests is unclear.
- **PASS** if the claim is pre-specified, multiplicity is controlled over a stated family, or the result is replicated and the search is fully reported.

### 3.9 Sources

Bender & Lange (2001); Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001); Berk et al. (2013); Bland & Altman (1995); Gelman & Loken (2014); Greenland et al. (2016); Head et al. (2015); Holm (1979); Ioannidis (2005); Kerr (1998); Nosek et al. (2018); Rothman (1990); Rothwell (2005); Simonsohn, Simmons & Nelson (2020); Steegen et al. (2016); Taylor & Tibshirani (2015); Wang et al. (2007); Wasserstein & Lazar (2016); White (2000); Wicherts et al. (2016).

---

## 4. Causal versus predictive claims

**Chunk ID:** `S4-causal-predictive`  
**Triggers:** causal inference, prediction, explanation, estimand, counterfactual, ATE, ATT, LATE, confounding, collider, mediator, DAG, propensity score, instrumental variable, regression discontinuity, difference-in-differences, synthetic control, target trial, sensitivity analysis, E-value, feature importance.  
**Critic question:** Is the paper making a prediction claim, a descriptive claim, or a causal claim — and does the design identify the quantity it claims?

### 4.1 Classify the task

| Task | Question | What makes it valid | What it does not need |
|---|---|---|---|
| **Description** | What is the distribution, association, or trend in this population? | representative sampling, good measurement, appropriate descriptive statistics | causal identification |
| **Prediction** | What is the best guess for a new unit, under a defined loss function? | no leakage, calibration, discrimination, external validation, distribution-shift check | causal identification |
| **Causal** | What would happen under an intervention, compared with a counterfactual? | identification assumptions and a design that supports them | high predictive accuracy |
| **Explanatory** | Which coefficients describe the conditional associations in this model? | correct specification, confounding control, uncertainty, no over-interpretation | a counterfactual claim |

Source: Hernán, Hsu & Healy (2019); Shmueli (2010).

A model can be excellent at one task and useless at another. High R² or AUC does not support a causal claim; a correctly identified causal model can predict poorly. "Feature importance" is a property of a fitted predictor, not a causal effect (Shmueli, 2010; Strobl et al., 2007).

### 4.2 Causal identification assumptions

A causal claim needs an explicit counterfactual contrast (e.g., ATE, ATT, LATE), a defined treatment and comparison, a time zero, and follow-up. The standard identification assumptions are:

- **Consistency.** The observed outcome under the observed treatment equals the potential outcome under that treatment; treatment versions must be well defined (Cole & Frangakis, 2009).
- **Positivity / overlap.** At every combination of confounders, there is a positive probability of both treatment and control (Westreich & Cole, 2010).
- **Exchangeability / no unmeasured confounding.** Conditional on measured covariates, treatment assignment is independent of potential outcomes. This is an assumption, not a test result.
- **SUTVA / no interference.** One unit's treatment does not affect another's outcome, and there is a single version of treatment (Imbens & Rubin, 2015; Hernán & Robins, 2020).
- **Correct measurement and timing.** Treatment, outcome, confounders, and follow-up must be measured at the right times. Immortal time and post-treatment adjustment are fatal (Suissa, 2008).

This five-item list is a synthesis: Hernán & Robins (2020) present exchangeability, positivity, and consistency as the three core conditions and fold no interference into consistency; Westreich & Cole (2010) add no measurement error and correct model specification.

Because exchangeability is untestable, a causal claim without a sensitivity analysis is incomplete. Report an E-value or another bias analysis, and use negative controls where possible (VanderWeele & Ding, 2017; Lipsitch, Tchetgen Tchetgen & Cohen, 2010). Formal omitted-variable-bias sensitivity analysis is an alternative when a plausible unmeasured confounder can be specified (Cinelli & Hazlett, 2020).

### 4.3 Confounders, mediators, and colliders

| Role | Definition | Adjust? | Failure mode |
|---|---|---|---|
| **Confounder** | common cause of treatment and outcome | yes, if measured and on the backdoor path | residual confounding if poorly measured |
| **Mediator** | lies on the causal pathway from treatment to outcome | not for a total-effect estimate; yes only for a controlled direct effect with explicit assumptions | over-adjustment removes the effect of interest |
| **Collider** | common effect of treatment and outcome (or of their causes) | no | conditioning induces a spurious association (collider bias) |
| **Selection variable** | a variable that determines inclusion in the sample | usually no; model the selection process | selection bias, Berkson's paradox |
| **Proxy confounder** | measured substitute for an unmeasured confounder | maybe; sensitivity analysis required | residual confounding remains |

Sources: Cinelli, Forney & Pearl (2022); Greenland, Pearl & Robins (1999); Hernán, Hernández-Díaz & Robins (2004); Munafò et al. (2018); Westreich & Greenland (2013).

Draw the DAG before choosing covariates. "Control for everything" is not a strategy: it can open collider paths and remove the effect through mediators. Coefficients for confounders in the same model are not causal effects — the Table 2 fallacy (Westreich & Greenland, 2013).

### 4.4 Design-specific checks

| Design | What must be argued | Diagnostics / robustness |
|---|---|---|
| **RCT** | randomization, allocation concealment, blinding, adherence, attrition | baseline balance, CONSORT flow, intention-to-treat analysis |
| **Target trial emulation** | eligibility, treatment strategies, assignment, time zero, follow-up, outcome, causal contrast | explicit target-trial table; positivity; sensitivity analysis (Hernán & Robins, 2016) |
| **Propensity score** | measured confounders only; treatment model; common support | standardized mean differences, overlap, balance plots; not AUC (Rosenbaum & Rubin, 1983; Austin, 2011) |
| **Instrumental variable** | SUTVA, relevance, exclusion restriction, independence, monotonicity (for LATE) | first-stage strength, weak-instrument diagnostics, over-identification tests (Angrist, Imbens & Rubin, 1996; Bound, Jaeger & Baker, 1995) |
| **Regression discontinuity** | continuity of potential outcomes at the cutoff; no manipulation | density test, covariate balance, bandwidth and polynomial sensitivity (Imbens & Lemieux, 2008; McCrary, 2008) |
| **Difference-in-differences** | parallel trends, no anticipation, stable composition | event-study pre-trends; group-time ATTs or TWFE weight diagnostics under staggered adoption (Roth et al., 2023; Goodman-Bacon, 2021; Callaway & Sant'Anna, 2021) |
| **Synthetic control** | donor pool, no interference, pre-treatment fit | placebo tests, pre-period fit (Abadie, Diamond & Hainmueller, 2010); leave-one-out robustness (Abadie, Diamond & Hainmueller, 2015) |
| **Survival / time-to-event** | time zero, non-informative censoring, proportional hazards, competing risks | Schoenfeld residuals, cumulative incidence, immortal-time audit (Grambsch & Therneau, 1994; Fine & Gray, 1999; Suissa, 2008) |

### 4.5 Prediction-specific checks

For prediction claims, the relevant validity questions are leakage (§2), calibration, discrimination, and external validation. Calibration is not optional: a model can rank well and still give badly wrong probabilities (Van Calster et al., 2019). Report the validation population, the prediction horizon, the intended use, and the performance at that horizon. External validation is stronger than a random split (Debray et al., 2015; Collins et al., 2015; Steyerberg, 2019; Wolff et al., 2019).

### 4.6 Causal-language audit

Verbs matter. "Causes", "reduces", "increases", "improves", "leads to", "impact of", "driver of", and "effect of" are causal claims. "Is associated with", "predicts", "correlates with", and "is a marker of" are weaker claims. If the paper uses causal language with observational data, look for an identification strategy. If it has one, check the assumptions; if it does not, downgrade the claim (Haber et al., 2018; Bradford Hill, 1965).

### 4.7 Red flags

- "X causes Y" from observational data without a design.
- "We controlled for all available covariates."
- "The coefficient is significant, so the effect is causal."
- "The model has high R²/AUC, so X is a driver."
- "Feature importance shows the most important cause."
- Propensity-score matching with no balance or overlap diagnostics.
- Adjusting for post-treatment variables or mediators while claiming a total effect.
- IV with no first-stage strength or exclusion argument.
- DiD with staggered adoption and a single TWFE coefficient, with no robustness.
- RD with no manipulation or bandwidth checks.
- Synthetic control with poor pre-treatment fit and no placebo tests.
- "We assumed no unmeasured confounding" with no sensitivity analysis.
- A cross-sectional survey used to estimate a causal effect.
- Conditioning on survival, selection, or another collider.
- Immortal time in the exposure definition.
- An unvalidated surrogate outcome standing in for the outcome of interest.

### 4.8 Verdict defaults

- **REJECT** a causal claim without an identification strategy or with a design assumption that is plainly violated.
- **REFINE** if causal language overreaches but the predictive or descriptive analysis is valid; relabel the claim.
- **PASS** if the task, estimand, identification assumptions, diagnostics, and sensitivity analysis align.

### 4.9 Sources

Abadie, Diamond & Hainmueller (2010, 2015); Angrist, Imbens & Rubin (1996); Austin (2011); Bound, Jaeger & Baker (1995); Bradford Hill (1965); Callaway & Sant'Anna (2021); Cinelli, Forney & Pearl (2022); Cole & Frangakis (2009); Collins et al. (2015); Debray et al. (2015); Fine & Gray (1999); Goodman-Bacon (2021); Grambsch & Therneau (1994); Greenland, Pearl & Robins (1999); Haber et al. (2018); Hernán, Hernández-Díaz & Robins (2004); Hernán, Hsu & Healy (2019); Hernán & Robins (2016, 2020); Imbens & Lemieux (2008); Imbens & Rubin (2015); Lipsitch, Tchetgen Tchetgen & Cohen (2010); McCrary (2008); Munafò et al. (2018); Rosenbaum & Rubin (1983); Roth et al. (2023); Shmueli (2010); Strobl et al. (2007); Suissa (2008); Van Calster et al. (2019); VanderWeele & Ding (2017); Westreich & Cole (2010); Westreich & Greenland (2013); Wolff et al. (2019).

---

## 5. Regression diagnostics and model specification

**Chunk ID:** `S5-diagnostics`  
**Triggers:** regression diagnostics, residuals, QQ plot, heteroscedasticity, autocorrelation, multicollinearity, VIF, Cook's distance, leverage, outliers, influence, linearity, RESET, Breusch–Pagan, White test, transformations, Box–Cox, interactions, AIC, BIC, stepwise, overfitting, cross-validation, nested CV, calibration.  
**Critic question:** Did the authors check the model they actually report, and did they account for the search that produced it?

### 5.1 Diagnostics are not a ritual

A diagnostic is useful only if it targets an assumption that matters for the claim. Reporting a QQ plot while ignoring clustering, or a VIF table while ignoring nonlinearity, is not a diagnostic strategy. The minimum requirement is: the diagnostics match the model family (see §1.2), are reported for the final model, and are acted on.

| Diagnostic target | Standard tools | What to conclude |
|---|---|---|
| **Linearity** | residual vs fitted; component-plus-residual (partial residual) plot; RESET | curvature suggests a missing nonlinear term or a wrong link |
| **Homoscedasticity** | residual vs fitted; Breusch–Pagan; White test | non-constant variance invalidates conventional SEs; use robust SEs or model the variance |
| **Independence** | ACF/PACF; Durbin–Watson; Ljung–Box; cluster-robust SE | dependence invalidates SEs and can bias the model; use time-series, multilevel, or clustered methods |
| **Normality** | QQ plot; histogram | not required for unbiased OLS; matters for exact small-sample tests and some likelihood procedures |
| **Collinearity** | VIF; condition index; correlation matrix | inflates SEs and destabilizes coefficients; thresholds are context-dependent |
| **Influence** | leverage; Cook's distance; DFBETAS | a few points may drive the result; report sensitivity with and without them |
| **Specification** | link function; transformations; interactions; omitted-variable checks | the functional form must match the data-generating process and the estimand |

Sources: Anscombe (1973); Belsley, Kuh & Welsch (1980); Breusch & Pagan (1979); Cook (1977); Harrell (2015); O'Brien (2007); Ramsey (1969); White (1980).

### 5.2 Association screening (correlation, ANOVA, chi-square)

The development plan screens many pairs with Pearson, Spearman, mutual information, ANOVA, and chi-square. Audit the screen itself:

- **Orientation.** Correlations are between columns (variables), not rows (observations), unless rows are variables. A row-wise correlation matrix is meaningless for standard tabular data.
- **Variable types.** Pearson for approximately linear continuous relationships; Spearman for monotonic/ordinal relationships; chi-square for categorical association; ANOVA for group means. Mutual information captures nonlinear dependence but is harder to estimate and interpret.
- **Assumptions.** ANOVA assumes independence, approximate normality, and homoscedasticity; chi-square assumes independent observations and adequate expected cell counts; Pearson is sensitive to outliers and nonlinearity.
- **Multiplicity.** Screening hundreds of pairs and reporting the smallest p-values is a discovery scan. Correct for the family or label it exploratory (§3).
- **Nonlinearity.** A zero Pearson correlation does not mean independence. Anscombe's quartet shows four datasets with nearly identical summary statistics and very different shapes (Anscombe, 1973).
- **Confounding.** Marginal associations can reverse or vanish after adjustment (Simpson's paradox). A correlation matrix is not a causal graph (Simpson, 1951; Greenland, Pearl & Robins, 1999).
- **Effect size and uncertainty.** Report the association size and a confidence interval, not just a p-value.

### 5.3 Transformations

Transformations must be justified by the mean–variance relationship, the scale of the outcome, and interpretability. Log and square-root transforms are not free:

- Logs require positive values; zeros need a separate strategy, and `log(y + 1)` changes the estimand and can distort small counts (O'Hara & Kotze, 2010).
- Box–Cox can help with normality/heteroscedasticity, but the transformed scale must be interpreted back to the original scale (Box & Cox, 1964).
- Retransformation of log-scale predictions needs a smearing correction or another method; exponentiating a mean on the log scale gives a geometric mean — the median only under lognormal or symmetric log-errors — not an arithmetic mean (Duan, 1983).
- Applying log/sqrt to "all sides" mechanically can create impossible values and makes the model hard to falsify. Justify each transform and report the resulting model in original units where possible.

### 5.4 Interactions and nonlinear terms

- Include all lower-order terms when testing an interaction; do not interpret a main effect as an average effect when an interaction is present (Brambor, Clark & Golder, 2006).
- Test the interaction term, not "significant in one group, not in the other" (see §3.6).
- Center or scale variables for interpretability, but centering does not fix confounding or collinearity (Gelman, 2008).
- Pre-specify the interactions that test the hypothesis; data-driven interaction search is a multiplicity problem.

### 5.5 Model selection and overfitting

Stepwise selection, best-subset search, and repeated tuning bias coefficients, standard errors, and p-values, and produce overoptimistic performance (Steyerberg et al., 1999; Harrell et al., 1996). Prefer pre-specified models, penalization (ridge/lasso/elastic net), or nested cross-validation. Report the entire selection process and the uncertainty of the final model.

AIC and BIC are model-selection criteria, not tests of causal truth. They require the same response variable and comparable data across models; comparing AIC across different transformations or different subsets is invalid unless the likelihood is Jacobian-corrected for the transformation. Adjusted R² and bootstrap optimism are useful for internal validation, but they do not replace external validation (Harrell, 2015).

### 5.6 Cross-validation

- Fit all preprocessing and feature selection inside the training fold.
- Use nested CV when hyperparameters are tuned and performance is estimated (Cawley & Talbot, 2010).
- Use group/block CV for clustered data and rolling-origin CV for time series (Roberts et al., 2017).
- Repeat CV and report variability; small-n CV estimates have wide error bars (Varoquaux, 2018; Arlot & Celisse, 2010).
- Do not choose the split that gives the best result.

### 5.7 Survival and ordinal models

- **Proportional hazards.** Test with Schoenfeld residuals; if violated, stratify, add time-varying effects, or use a different model (Grambsch & Therneau, 1994).
- **Competing risks.** Kaplan–Meier overestimates the cumulative incidence of the event of interest when competing events exist; use cause-specific or subdistribution hazards appropriately (Fine & Gray, 1999).
- **Separation.** Logistic MLE can diverge with sparse cells or perfect predictors; use Firth's penalized likelihood and report it (Heinze & Schemper, 2002).
- **Ordinal models.** The proportional-odds assumption should be checked; otherwise use a partial proportional-odds or multinomial model (Brant, 1990).
- **Immortal time.** Define time zero and exposure before follow-up; otherwise the exposure is guaranteed to precede the outcome (Suissa, 2008).

### 5.8 Red flags

- "All assumptions were satisfied" with no plots or tests.
- "We removed outliers to improve fit" without a pre-specified rule.
- "We used stepwise regression" and then report p-values as if the model were pre-specified.
- "We chose the model with the highest R²" or "the lowest AIC" without validation.
- "VIF > 10, so we dropped variables" as a mechanical rule.
- "We interpreted the main effect" in a model with a significant interaction.
- "We used p-values to select variables, then report p-values" without adjustment.
- "We performed cross-validation after feature selection."
- "We tuned on the test set."
- "We report only the final model, not the search."
- "We used 10-fold CV on time-series data."
- "We plotted residuals but did not act on them."
- "We compared AIC across models with different response transformations."
- "We used a tree model and interpreted feature importance causally."

### 5.9 Verdict defaults

- **REJECT** if diagnostics show a material violation and there is no remedy or sensitivity analysis, or if the final model is the product of an unreported search presented as confirmation.
- **REFINE** if diagnostics are missing but the method is otherwise sound, or if the model-selection process is described but not validated.
- **PASS** if diagnostics match the model, the final model is reported with uncertainty, the selection process is transparent, and internal/external validation is appropriate.

### 5.10 Sources

Anscombe (1973); Arlot & Celisse (2010); Belsley, Kuh & Welsch (1980); Box & Cox (1964); Brambor, Clark & Golder (2006); Brant (1990); Breusch & Pagan (1979); Cawley & Talbot (2010); Cook (1977); Duan (1983); Fine & Gray (1999); Gelman (2008); Grambsch & Therneau (1994); Greenland, Pearl & Robins (1999); Harrell (2015); Harrell, Lee & Mark (1996); Heinze & Schemper (2002); O'Brien (2007); O'Hara & Kotze (2010); Ramsey (1969); Roberts et al. (2017); Simpson (1951); Steyerberg et al. (1999); Suissa (2008); Varoquaux (2018); White (1980).

---

## 6. Count and rate models

**Chunk ID:** `S6-count-models`  
**Triggers:** count data, Poisson regression, negative binomial, NB1, NB2, quasi-Poisson, overdispersion, underdispersion, zero-inflated, hurdle, excess zeros, exposure, offset, person-time, rate, incidence, clustering, log(count + 1).  
**Critic question:** Is the count outcome modeled with a variance function and exposure structure that match how the data were generated?

### 6.1 Poisson is a model, not a default

Poisson regression assumes a log-linear conditional mean and a conditional variance equal to the mean (equidispersion). It also assumes independent observations, a correct exposure/offset, and no unmodeled excess zeros (Cameron & Trivedi, 2013; McCullagh & Nelder, 1989). Counts often violate equidispersion; ignoring that produces standard errors that are too small and p-values that are too optimistic.

### 6.2 Symptom → check → remedy

| Symptom | Check | Candidate remedy |
|---|---|---|
| **Variance ≫ mean** | Pearson χ²/df; Cameron–Trivedi regression test; LR test of NB vs Poisson | negative binomial (NB2 default), quasi-Poisson, robust sandwich SEs |
| **Excess zeros** | observed vs predicted zero counts; theoretical plausibility of a structural-zero process | zero-inflated Poisson/NB (Lambert, 1992) or hurdle model (Mullahy, 1986) |
| **Exposure varies** | is there a denominator/person-time? is `log(exposure)` an offset with coefficient 1? | offset, or model rates directly with the correct denominator |
| **Clustering** | ICC; repeated measures within unit | mixed-effects count model or cluster-robust SEs |
| **Underdispersion** | variance < mean | check misspecification/dependence; quasi-Poisson; do not ignore |
| **Truncation/censoring** | are zero counts impossible? are counts censored? | zero-truncated or censored count model |
| **Small counts / rare events** | sparse cells; separation-like behavior | exact/penalized methods; collapse categories; collect more data |

Sources: Cameron & Trivedi (1990, 2013); Coxe, West & Aiken (2009); Hilbe (2011); Lambert (1992); Mullahy (1986); Zeileis, Kleiber & Jackman (2008).

### 6.3 Negative binomial and quasi-Poisson

- **NB2** has variance `μ + αμ²`; **NB1** has variance `φμ`. The choice matters for the mean–variance relationship and for prediction. Report the parameterization and the estimated dispersion.
- **Quasi-Poisson** estimates a dispersion parameter and adjusts SEs, but it is not a full likelihood; it is a quasi-likelihood approach. It does not give an AIC comparable to NB.
- **Robust sandwich SEs** handle heteroscedasticity and some dependence, but they do not fix a misspecified mean function.
- **Zero-inflated models** mix a point mass at zero with a count distribution; **hurdle models** separately model zero vs positive counts. They answer different questions. Choose based on the data-generating process, not solely on AIC or a Vuong test.

### 6.4 Offsets and rates

If the outcome is a count but the question is about a rate, include `log(exposure)` as an offset with its coefficient fixed at 1. Including exposure as an ordinary covariate estimates a different model and can absorb the effect of interest. Check that exposure is measured on the same time scale as the outcome and is not affected by treatment (Cameron & Trivedi, 2013; Zeileis et al., 2008).

### 6.5 Red flags

- "We used Poisson regression" with no dispersion check.
- "We log-transformed counts" or "we used log(count + 1)" without justification (O'Hara & Kotze, 2010).
- "We used OLS on counts" and interpreted coefficients as rates.
- "We used zero-inflated Poisson because there were many zeros" without a structural-zero argument.
- "We included exposure as a covariate" instead of an offset.
- "We fit a negative binomial but report Poisson standard errors."
- "We ignored clustering" in repeated or hierarchical counts.
- "We compared AIC across different response transformations."
- "We used Poisson for person-time data with no offset."
- "We used a rate as the outcome without a denominator."

### 6.6 Verdict defaults

- **REJECT** if a count/rate outcome is modeled with OLS or `log(y + 1)` and the count structure is material to the claim.
- **REFINE** if overdispersion, zero inflation, exposure, or clustering is unaddressed but the model is otherwise interpretable.
- **PASS** if the variance function, offset, zero process, and dependence structure are stated and checked, and the model is compared on a like-for-like basis.

### 6.7 Sources

Cameron & Trivedi (1990, 2013); Coxe, West & Aiken (2009); Hilbe (2011); Lambert (1992); McCullagh & Nelder (1989); Mullahy (1986); O'Hara & Kotze (2010); Zeileis, Kleiber & Jackman (2008).

---

## 7. Sample size, power, and precision

**Chunk ID:** `S7-sample-size`  
**Triggers:** sample size, power, precision, events per variable, EPV, rare events, separation, overfitting, external validation, winner's curse, Type S error, Type M error, class imbalance, SMOTE, small n, p >> n.  
**Critic question:** Is the sample large enough, relative to the model complexity and the claim, to make the reported estimate credible?

### 7.1 Sample size is model-relative

There is no universal minimum n. The relevant question is whether the data can support the number of parameters, the number of events, the number of subgroups, and the precision the claim requires. A large n with a badly specified model is still wrong; a small n with a simple pre-specified model may be defensible.

### 7.2 Rules of thumb and their limits

| Situation | Common check | Important caveat |
|---|---|---|
| **Linear regression, many predictors** | Green's rule: N ≥ 50 + 8m for the overall model; N ≥ 104 + m for individual predictors | a heuristic for medium effects, α = .05 and power = .80, not a power calculation; 50 + 8m is too conservative for m ≥ 7 (Green, 1991) |
| **Logistic / Cox regression** | events per variable (EPV); 10 EPV is widely cited | the 10-EPV convention comes from Peduzzi et al. (1996); later work finds no firm rationale for 1 variable per 10 events and shows the required EPV depends on total n, predictor prevalence, and separation (van Smeden et al., 2016; Vittinghoff & McCulloch, 2007) |
| **Prediction model development** | Riley et al. criteria: shrinkage ≥ 0.9, optimism ≤ 0.05 in Nagelkerke R², precise overall risk | requires an anticipated R²/C-statistic and the number of predictor parameters (Riley et al., 2019) |
| **External validation** | at least 100 events, ideally ≥ 200 events | events, not just participants (Collins, Ogundimu & Altman, 2016) |
| **Rare events** | number of events per parameter; separation diagnostics | use Firth/penalized likelihood; do not rely on resampling alone (King & Zeng, 2001; Heinze & Schemper, 2002) |
| **Small-n CV** | repeated nested CV; report variability | error bars can be large; CV is not a substitute for more data (Varoquaux, 2018) |

### 7.3 Power, Type S, and Type M errors

Underpowered studies do not just fail to detect true effects; the effects they do detect are biased away from zero and noisy in sign and magnitude. Gelman & Carlin (2014) call these Type S (sign) and Type M (magnitude) errors. Selection on statistical significance produces a winner's curse: the reported effect is likely larger than the true effect (Ioannidis, 2005; Button et al., 2013). A significant result in a small sample is therefore not strong evidence of a large effect; it may be evidence of a lucky sample.

### 7.4 Rare events and class imbalance

- **Rare events in logistic regression.** MLE is biased and can separate; use Firth's penalized likelihood, and report the number of events per parameter (King & Zeng, 2001; Heinze & Schemper, 2002).
- **Class imbalance corrections.** Random undersampling, oversampling, and SMOTE can produce poorly calibrated risk models and do not reliably improve AUC. For risk prediction, do not "fix" imbalance by resampling; use a proper loss, penalization, and calibration (van den Goorbergh et al., 2022).
- **Small samples and many predictors.** Pre-specify a small model, use penalization or dimensionality reduction, and validate externally. Stepwise selection in small samples is especially unstable (Steyerberg et al., 1999).

### 7.5 Subgroups and interactions

Subgroup and interaction claims need more data than main effects. A trial powered for the main effect is usually underpowered for interaction tests. Pre-specify subgroups, report all of them, test the interaction, and interpret the result as exploratory unless the study was designed for it (Wang et al., 2007; Rothwell, 2005).

### 7.6 Red flags

- "n = 30, 20 predictors" with no penalization or validation.
- "We used 10 events per variable, so we are fine."
- "We did not perform a power calculation because the data were already collected" — acceptable only if the analysis is explicitly descriptive/exploratory.
- "We found a significant result in a small sample, so the effect must be large."
- "We used SMOTE to balance the classes" for a risk model.
- "We used stepwise selection with 50 events."
- "We validated on 20 patients."
- "We analyzed subgroups post hoc" and present them as confirmatory.
- "We used p < 0.05 to claim an interaction."
- "We used the full dataset for model selection and report in-sample performance."

### 7.7 Verdict defaults

- **REJECT** if the sample is manifestly inadequate for the model complexity and the claim is strong (causal, confirmatory, or a risk model with many parameters).
- **REFINE** if the sample size is not reported but can be inferred, or if the model can be simplified/penalized.
- **PASS** if the sample-size justification matches the model complexity, the events per parameter are adequate for the chosen method, and the precision of the estimate is reported.

### 7.8 Sources

Button et al. (2013); Collins, Ogundimu & Altman (2016); Gelman & Carlin (2014); Green (1991); Heinze & Schemper (2002); Ioannidis (2005); King & Zeng (2001); Peduzzi et al. (1996); Riley et al. (2019); Rothwell (2005); Steyerberg et al. (1999); van den Goorbergh et al. (2022); van Smeden et al. (2016); Varoquaux (2018); Vittinghoff & McCulloch (2007); Wang et al. (2007).

---

## 8. Missing data and measurement error

**Chunk ID:** `S8-missing-measurement`  
**Triggers:** missing data, MCAR, MAR, MNAR, complete-case analysis, listwise deletion, mean imputation, multiple imputation, MICE, chained equations, imputation model, sensitivity analysis, measurement error, regression dilution, misclassification, reliability, agreement, kappa, ICC, Bland–Altman, regression to the mean.  
**Critic question:** Did missingness or measurement error distort the estimate, and did the authors handle it transparently?

### 8.1 Missing-data mechanisms

| Mechanism | Meaning | Consequence |
|---|---|---|
| **MCAR** | missingness is independent of observed and unobserved data | complete-case analysis is unbiased but inefficient |
| **MAR** | missingness depends on observed data (including the outcome or auxiliary variables) | multiple imputation or likelihood-based methods can be valid if the model is correct |
| **MNAR** | missingness depends on unobserved data (e.g., the missing value itself) | requires a sensitivity analysis; no method is guaranteed |

Sources: Rubin (1976); Little & Rubin (2019).

Complete-case analysis is valid in special cases (e.g., MCAR, or when missingness is independent of the outcome given the covariates included in the model), but it is inefficient and can be biased if the model is misspecified — in particular, when missingness depends on the outcome, complete-case analysis is biased toward the null. "We excluded missing cases" is not a justification by itself. Report the amount and pattern of missingness by variable and group, compare completers with non-completers, and state the assumed mechanism.

### 8.2 Multiple imputation checklist

- Impute **m > 1** datasets; analyze each; pool estimates with Rubin's rules (Rubin, 1987).
- Include the outcome and auxiliary variables in the imputation model; otherwise the imputation is uncongenial to the analysis.
- Preserve interactions, nonlinear terms, and transformations that are in the analysis model (White, Royston & Wood, 2011).
- Impute separately by treatment group if the analysis is subgroup-specific.
- Choose the number of imputations based on the fraction of missing information; 5–10 is often too few for high missingness (Graham, Olchowski & Gilreath, 2007).
- Do not use single mean imputation, last-observation-carried-forward, or a missing indicator as a substitute for a proper imputation model.
- For MNAR, report a sensitivity analysis (delta adjustment, pattern-mixture, or tipping-point) (National Research Council, 2010).

### 8.3 Measurement error and reliability

- **Classical measurement error in a predictor** attenuates the observed association toward the null (regression dilution); this is exact for a single predictor, while with multiple correlated predictors the bias can go in either direction. Correct with validation/reliability data, regression calibration, or SIMEX (Cook & Stefanski, 1994), and report the reliability (Frost & Thompson, 2000).
- **Misclassification of a binary outcome** can bias associations; the direction depends on whether the error is differential with respect to exposure. Do not assume it is harmless.
- **Agreement is not correlation.** Use Bland–Altman limits of agreement for continuous measurements, ICC for reliability (not for agreement between two measurement methods), and kappa for categorical agreement; a high correlation can coexist with poor agreement (Bland & Altman, 1999; Shrout & Fleiss, 1979; Sim & Wright, 2005).
- **Regression to the mean.** Extreme baseline values tend to move toward the mean on remeasurement. Uncontrolled pre–post comparisons overstate change; use a control group or model the phenomenon (Barnett, van der Pols & Dobson, 2005).
- **Derived variables.** Ratios, change scores, and indices inherit the measurement error of their components; reliability can be much worse than the inputs suggest.

### 8.4 Red flags

- "We excluded missing data" with no comparison or mechanism statement.
- "We imputed the mean" or "we used last observation carried forward."
- "We used a missing indicator" as the only handling.
- "We assumed data were missing at random" without justification or sensitivity analysis.
- "We used complete-case analysis because it is conservative" — not necessarily.
- "The exposure was measured once with a noisy instrument" and no reliability estimate.
- "We used correlation to assess agreement."
- "We used change scores" without accounting for regression to the mean.
- "We validated the instrument on the same data used to develop it."

### 8.5 Verdict defaults

- **REJECT** if missingness is substantial and handled by complete-case analysis or single imputation with no justification, or if measurement error is ignored for a key exposure.
- **REFINE** if missingness is reported but the mechanism/imputation model is underspecified, or if reliability is not reported.
- **PASS** if missingness is quantified, the mechanism is stated, multiple imputation or a likelihood method is used appropriately, and measurement error/reliability is addressed.

### 8.6 Sources

Barnett, van der Pols & Dobson (2005); Bland & Altman (1999); Cook & Stefanski (1994); Frost & Thompson (2000); Graham, Olchowski & Gilreath (2007); Little & Rubin (2019); National Research Council (2010); Rubin (1976, 1987); Shrout & Fleiss (1979); Sim & Wright (2005); Sterne et al. (2009); White, Royston & Wood (2011).

---

## 9. Time series and dependent data

**Chunk ID:** `S9-time-series`  
**Triggers:** time series, temporal dependence, autocorrelation, stationarity, unit root, trend, seasonality, cyclical, structural break, spurious regression, look-ahead bias, rolling-origin CV, backtesting, panel data, clustered data.  
**Critic question:** Did the analysis respect the temporal order and dependence structure of the data?

### 9.1 Stationarity and spurious regression

Regressing two independent non-stationary series can produce a high R² and apparently significant coefficients — spurious regression (Yule, 1926; Granger & Newbold, 1974). Check stationarity (unit-root tests such as Dickey–Fuller), difference or detrend as appropriate, and consider cointegration if the series are integrated. Do not interpret a trend correlation as evidence of a causal relationship.

### 9.2 Dependence and seasonality

- Check ACF/PACF and residual autocorrelation (Ljung–Box).
- Model seasonality explicitly (seasonal terms, Fourier terms, seasonal differencing, or a state-space model) and check residual seasonality.
- Use HAC/Newey–West or cluster-robust SEs when dependence remains.
- Test for structural breaks; a model fitted across a regime change may not apply to the current regime.
- For panel data, account for within-unit correlation and time-varying confounding.

Sources: Dickey & Fuller (1979); Granger & Newbold (1974); Ljung & Box (1978); Hyndman & Athanasopoulos (2021).

### 9.3 Validation for time series

Random cross-validation leaks future information into training. Use rolling-origin/backtesting: train on the past, test on the future, and roll forward. Report performance across horizons and periods, not a single random split (Roberts et al., 2017; Hyndman & Athanasopoulos, 2021). Beware look-ahead bias in feature construction: a feature must be computable at the forecast origin (López de Prado, 2018).

### 9.4 Red flags

- "We used random cross-validation" on time-series data.
- "We detrended the series" with no stationarity test or justification.
- "We found a strong correlation between two trending series."
- "We used future values to construct a feature."
- "We ignored seasonality" in a series with a clear seasonal pattern.
- "We fitted one model across a structural break."
- "We used a standard train/test split" for a forecasting problem with a temporal deployment.
- "We validated on a random sample of time points."

### 9.5 Verdict defaults

- **REJECT** if random CV or future information contaminates a forecasting claim, or if a spurious regression is presented as evidence.
- **REFINE** if stationarity/seasonality/dependence is unaddressed but the data are otherwise suitable.
- **PASS** if the temporal structure is modeled, validation is rolling-origin, and look-ahead bias is excluded.

### 9.6 Sources

Dickey & Fuller (1979); Granger & Newbold (1974); Hyndman & Athanasopoulos (2021); Ljung & Box (1978); López de Prado (2018); Roberts et al. (2017); Yule (1926).

---

## 10. Reporting, reproducibility, and critical appraisal

**Chunk ID:** `S10-reporting`  
**Triggers:** TRIPOD, TRIPOD+AI, STROBE, CONSORT, PROBAST, reporting guideline, reproducibility, code availability, data availability, preregistration, selective reporting, publication bias.  
**Critic question:** Is the method reported in enough detail to reproduce and audit, and does the report follow the relevant guideline?

### 10.1 Reporting guidelines by study type

| Study type | Guideline | What it forces the author to report |
|---|---|---|
| **Prediction model** | TRIPOD+AI for regression or ML (Collins et al., 2024), which supersedes TRIPOD 2015 (Collins et al., 2015) | source data, predictors, outcome, sample size, missing data, model specification, performance, calibration, validation |
| **Observational study** | STROBE (von Elm et al., 2007) | design, setting, participants, variables, bias, sample size, statistical methods, limitations |
| **Randomized trial** | CONSORT (Schulz, Altman & Moher, 2010) | randomization, blinding, flow, outcomes, analysis populations |
| **Risk of bias in prediction studies** | PROBAST (Wolff et al., 2019) | participants, predictors, outcome, analysis; risk of bias and applicability |
| **Systematic review** | PRISMA 2020 (Page et al., 2021) | search, selection, data extraction, synthesis, bias |

Use the guideline as an audit checklist, not as a certificate. A paper can check the boxes and still have a fatal design flaw.

### 10.2 Reproducibility checklist

- Are the data, code, and computing environment available (or is there a credible controlled-access route)?
- Are package versions, random seeds, and preprocessing steps reported?
- Is the model specification complete enough to refit (formula, link, penalties, hyperparameters)?
- Are all outcomes, subgroups, and model runs reported, not just the significant ones?
- Are the analysis choices pre-registered or clearly labelled post hoc?
- Are negative results and failed models reported?
- Is the performance estimate accompanied by uncertainty (CIs, bootstrap, repeated CV)?

Sources: Nosek et al. (2015, 2018); Munafò et al. (2017); Sandve et al. (2013).

### 10.3 Selective reporting and publication bias

If only significant outcomes or models are reported, the literature overstates effects. Check for outcome switching, unpublished outcomes, and the file-drawer problem (Chan et al., 2004; Rosenthal, 1979; Ioannidis, 2005). A funnel plot or registry comparison can help in a meta-analysis; funnel asymmetry detects small-study effects generally — publication bias and other biases — not publication bias alone (Egger et al., 1997).

### 10.4 Red flags

- "Data available upon request" with no repository or accession number.
- No code, no package versions, no seed.
- Only the final model is reported; the search is invisible.
- The primary outcome differs between the registry, the methods, and the results.
- "We used standard methods" with no specification.
- Performance reported without uncertainty.
- Negative results or failed models omitted.
- A reporting guideline is claimed but the corresponding items are missing.

### 10.5 Verdict defaults

- **REJECT** if the method cannot be reconstructed and the claim is strong; irreproducibility is a validity problem, not a formatting problem.
- **REFINE** if the method is mostly described but key details (split, model, missing data) are absent.
- **PASS** if the report contains enough detail to reproduce the analysis and follows the relevant guideline.

### 10.6 Sources

Chan et al. (2004); Collins et al. (2015, 2024); Egger et al. (1997); Ioannidis (2005); Munafò et al. (2017); Nosek et al. (2015, 2018); Page et al. (2021); Rosenthal (1979); Sandve et al. (2013); Schulz, Altman & Moher (2010); von Elm et al. (2007); Wolff et al. (2019).

---

## 11. One-page critic checklist

Use this after classifying the claim (§0.1). A "no" is not automatically fatal; an unexplained "no" is a finding.

**Claim and design**
1. Is the claim type (description / prediction / causal / explanatory) explicit?
2. Is the estimand, target population, and time horizon stated?
3. Is the unit of analysis stated and consistent with the model?
4. Is the sampling/selection process described?
5. For causal claims, is there an identification strategy with explicit assumptions?

**Data and leakage**
6. Is every feature available at prediction time?
7. Was the split created before any preprocessing, imputation, or feature selection?
8. Are scaling, imputation, encoding, and feature selection fit inside training folds?
9. Are groups/time blocks respected in the split?
10. Are duplicates/near-duplicates removed before splitting?
11. Is there an untouched test set, used only once?
12. Are all runs reported, or only the best?

**Model and assumptions**
13. Is the model family appropriate for the outcome type?
14. Are the model's key assumptions stated?
15. Are diagnostics reported for the final model?
16. Are influential points/outliers handled with a pre-specified rule?
17. Is overfitting addressed (shrinkage, penalization, nested CV, external validation)?
18. For prediction, is calibration reported alongside discrimination?
19. For survival/count/ordinal data, are the special assumptions checked?

**Inference and selection**
20. Is the primary analysis pre-specified, or is the result exploratory?
21. Is multiplicity controlled over a stated family, or is the analysis labelled exploratory?
22. Is post-selection inference valid (sample splitting, selective inference, replication)?
23. Are subgroup claims based on interaction tests and pre-specification?
24. Are effect sizes and confidence intervals reported, not just p-values?
25. For causal claims, are confounders/mediators/colliders distinguished?
26. Is there a sensitivity analysis for unmeasured confounding?

**Data quality and reporting**
27. Is missingness quantified and handled with an appropriate method?
28. Is measurement error/reliability addressed for key variables?
29. Is temporal dependence handled (stationarity, seasonality, rolling CV)?
30. Is the method reproducible (code, data, versions, seeds, full results)?
31. Does the conclusion match the evidence and the claim type?

---

## 12. Red-flag phrasebook

| Phrase in the method | Likely issue | Chunk |
|---|---|---|
| "We normalized the entire dataset before splitting." | preprocessing leakage | §2 |
| "We selected the top 20 features, then ran 10-fold CV." | feature-selection leakage | §2 |
| "We imputed missing values using the full dataset." | imputation leakage | §2 |
| "We used SMOTE before splitting." | oversampling leakage; calibration harm | §2, §7 |
| "We used a random 80/20 split." (time series/panel) | temporal/group leakage | §2, §9 |
| "We removed duplicates after splitting." | duplicate leakage | §2 |
| "We tuned hyperparameters on the test set." | test-set reuse | §2 |
| "We report the full-data model's R²/AUC." | in-sample performance | §2 |
| "We tested several models and report the best." | multiplicity / selection | §3 |
| "We found a significant subgroup." | subgroup multiplicity | §3 |
| "We removed outliers until the result was significant." | p-hacking / influence | §3, §5 |
| "We tried several transformations and report the significant one." | multiplicity | §3 |
| "We used stepwise regression, then report p-values." | post-selection inference | §3, §5 |
| "We did not adjust for multiple comparisons." | multiplicity (context-dependent) | §3 |
| "X causes Y." (observational data) | causal overreach | §4 |
| "We controlled for all available variables." | collider/mediator bias | §4 |
| "The model has high R²/AUC, so X is a driver." | predictive vs causal | §4 |
| "Feature importance shows the cause." | predictive vs causal | §4 |
| "We used propensity score matching." (no balance) | incomplete confounding control | §4 |
| "We adjusted for post-treatment variables." | over-adjustment / collider bias | §4 |
| "We assumed no unmeasured confounding." | untestable assumption, no sensitivity | §4 |
| "We used IV." (no first stage/exclusion) | weak instrument / exclusion violation | §4 |
| "We used DiD." (staggered adoption) | TWFE bias / parallel trends | §4 |
| "We used RD." (no density/bandwidth checks) | manipulation / bandwidth sensitivity | §4 |
| "We used Poisson regression." (no dispersion check) | overdispersion | §6 |
| "We log-transformed counts." | wrong estimand | §6 |
| "We included exposure as a covariate." | offset misspecification | §6 |
| "We used zero-inflated Poisson because there were many zeros." | unjustified zero model | §6 |
| "We used 10 events per variable." | EPV heuristic | §7 |
| "We used SMOTE to balance the classes." | calibration harm | §7 |
| "We validated on 20 patients." | inadequate external validation | §7 |
| "We excluded missing cases." | complete-case bias | §8 |
| "We imputed the mean." | single imputation | §8 |
| "We assumed data were missing at random." | untested MAR | §8 |
| "We used correlation to assess agreement." | wrong agreement metric | §8 |
| "We used random cross-validation." (time series) | temporal leakage | §9 |
| "We found a strong correlation between two trending series." | spurious regression | §9 |
| "All assumptions were satisfied." | no evidence | §1, §5 |
| "We chose the model with the highest R²." | overfitting / selection | §5 |
| "We compared AIC across transformations." | incomparable likelihoods | §5 |
| "Data available upon request." | reproducibility | §10 |

---

## References

Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program. *Journal of the American Statistical Association, 105*(490), 493–505. https://doi.org/10.1198/jasa.2009.ap08746

Abadie, A., Diamond, A., & Hainmueller, J. (2015). Comparative politics and the synthetic control method. *American Journal of Political Science, 59*(2), 495–510. https://doi.org/10.1111/ajps.12116

Ambroise, C., & McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. *Proceedings of the National Academy of Sciences, 99*(10), 6562–6566. https://doi.org/10.1073/pnas.102102699

Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of causal effects using instrumental variables. *Journal of the American Statistical Association, 91*(434), 444–455. https://doi.org/10.1080/01621459.1996.10476902

Anscombe, F. J. (1973). Graphs in statistical analysis. *The American Statistician, 27*(1), 17–21. https://doi.org/10.1080/00031305.1973.10478966

Arlot, S., & Celisse, A. (2010). A survey of cross-validation procedures for model selection. *Statistics Surveys, 4*, 40–79. https://doi.org/10.1214/09-SS054

Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research, 46*(3), 399–424. https://doi.org/10.1080/00273171.2011.568786

Barnett, A. G., van der Pols, J. C., & Dobson, A. J. (2005). Regression to the mean: What it is and how to deal with it. *International Journal of Epidemiology, 34*(1), 215–220. https://doi.org/10.1093/ije/dyh299

Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression diagnostics: Identifying influential data and sources of collinearity*. Wiley. https://doi.org/10.1002/0471725153

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x

Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics, 29*(4), 1165–1188. https://doi.org/10.1214/aos/1013699998

Bender, R., & Lange, S. (2001). Adjusting for multiple testing—when and how? *Journal of Clinical Epidemiology, 54*(4), 343–349. https://doi.org/10.1016/S0895-4356(00)00314-0

Berk, R., Brown, L., Buja, A., Zhang, K., & Zhao, L. (2013). Valid post-selection inference. *The Annals of Statistics, 41*(2), 802–837. https://doi.org/10.1214/12-AOS1077

Bland, J. M., & Altman, D. G. (1995). Multiple significance tests: The Bonferroni method. *BMJ, 310*(6973), 170. https://doi.org/10.1136/bmj.310.6973.170

Bland, J. M., & Altman, D. G. (1999). Measuring agreement in method comparison studies. *Statistical Methods in Medical Research, 8*(2), 135–160. https://doi.org/10.1191/096228099673819272

Box, G. E. P. (1976). Science and statistics. *Journal of the American Statistical Association, 71*(356), 791–799. https://doi.org/10.1080/01621459.1976.10480949

Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B, 26*(2), 211–243. https://doi.org/10.1111/j.2517-6161.1964.tb00553.x

Bradford Hill, A. (1965). The environment and disease: Association or causation? *Proceedings of the Royal Society of Medicine, 58*(5), 295–300. https://doi.org/10.1177/003591576505800503

Brambor, T., Clark, W. R., & Golder, M. (2006). Understanding interaction models: Improving empirical analyses. *Political Analysis, 14*(1), 63–82. https://doi.org/10.1093/pan/mpi014

Brant, R. (1990). Assessing proportionality in the proportional odds model for ordinal logistic regression. *Biometrics, 46*(4), 1171–1178. https://doi.org/10.2307/2532457

Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica, 47*(5), 1287–1294. https://doi.org/10.2307/1911963

Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences, 113*(15), 3932–3937. https://doi.org/10.1073/pnas.1517384113

Button, K. S., Ioannidis, J. P. A., Mokrysz, C., Nosek, B. A., Flint, J., Robinson, E. S. J., & Munafò, M. R. (2013). Power failure: Why small sample size undermines the reliability of neuroscience. *Nature Reviews Neuroscience, 14*(5), 365–376. https://doi.org/10.1038/nrn3475

Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics, 225*(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001

Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. *Journal of Econometrics, 46*(3), 347–364. https://doi.org/10.1016/0304-4076(90)90014-K

Cameron, A. C., & Trivedi, P. K. (2013). *Regression analysis of count data* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9781139013567

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. https://jmlr.org/papers/v11/cawley10a.html

Chan, A.-W., Hróbjartsson, A., Haahr, M. T., Gøtzsche, P. C., & Altman, D. G. (2004). Empirical evidence for selective reporting of outcomes in randomized trials. *JAMA, 291*(20), 2457–2465. https://doi.org/10.1001/jama.291.20.2457

Cinelli, C., Forney, A., & Pearl, J. (2022). A crash course in good and bad controls. *Sociological Methods & Research, 53*(3), 1071–1104. https://doi.org/10.1177/00491241221099552

Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B, 82*(1), 39–67. https://doi.org/10.1111/rssb.12348

Cole, S. R., & Frangakis, C. E. (2009). The consistency statement in causal inference: A definition or an assumption? *Epidemiology, 20*(1), 3–5. https://doi.org/10.1097/EDE.0b013e31818ef366

Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., et al. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. https://doi.org/10.1136/bmj-2023-078378

Collins, G. S., Ogundimu, E. O., & Altman, D. G. (2016). Sample size considerations for the external validation of a multivariable prognostic model: A resampling study. *Statistics in Medicine, 35*(2), 214–226. https://doi.org/10.1002/sim.6787

Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. M. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): The TRIPOD statement. *Annals of Internal Medicine, 162*(1), 55–63. https://doi.org/10.7326/M14-0697

Cook, J. R., & Stefanski, L. A. (1994). Simulation-extrapolation estimation in parametric measurement error models. *Journal of the American Statistical Association, 89*(428), 1314–1328. https://doi.org/10.1080/01621459.1994.10476871

Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics, 19*(1), 15–18. https://doi.org/10.1080/00401706.1977.10489493

Coxe, S., West, S. G., & Aiken, L. S. (2009). The analysis of count data: A gentle introduction to Poisson regression and its alternatives. *Journal of Personality Assessment, 91*(2), 121–136. https://doi.org/10.1080/00223890802634175

Debray, T. P. A., Vergouwe, Y., Koffijberg, H., Nieboer, D., Steyerberg, E. W., & Moons, K. G. M. (2015). A new framework to enhance the interpretation of external validation studies of clinical prediction models. *Journal of Clinical Epidemiology, 68*(3), 279–289. https://doi.org/10.1016/j.jclinepi.2014.06.018

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association, 74*(366), 427–431. https://doi.org/10.2307/2286348

Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association, 78*(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017

Egger, M., Davey Smith, G., Schneider, M., & Minder, C. (1997). Bias in meta-analysis detected by a simple, graphical test. *BMJ, 315*(7109), 629–634. https://doi.org/10.1136/bmj.315.7109.629

Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association, 94*(446), 496–509. https://doi.org/10.1080/01621459.1999.10474144

Frost, C., & Thompson, S. G. (2000). Correcting for regression dilution bias: Comparison of methods for a single predictor variable. *Journal of the Royal Statistical Society: Series A, 163*(2), 173–189. https://doi.org/10.1111/1467-985X.00164

Gelman, A. (2008). Scaling regression inputs by dividing by two standard deviations. *Statistics in Medicine, 27*(15), 2865–2873. https://doi.org/10.1002/sim.3107

Gelman, A., & Carlin, J. (2014). Beyond power calculations: Assessing Type S (sign) and Type M (magnitude) errors. *Perspectives on Psychological Science, 9*(6), 641–651. https://doi.org/10.1177/1745691614551642

Gelman, A., & Loken, E. (2014). The statistical crisis in science. *American Scientist, 102*(6), 460. https://doi.org/10.1511/2014.111.460

Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics, 225*(2), 254–277. https://doi.org/10.1016/j.jeconom.2021.03.014

Graham, J. W., Olchowski, A. E., & Gilreath, T. D. (2007). How many imputations are really needed? Some practical clarifications of multiple imputation theory. *Prevention Science, 8*(3), 206–213. https://doi.org/10.1007/s11121-007-0070-9

Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika, 81*(3), 515–526. https://doi.org/10.1093/biomet/81.3.515

Granger, C. W. J., & Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics, 2*(2), 111–120. https://doi.org/10.1016/0304-4076(74)90034-7

Green, S. B. (1991). How many subjects does it take to do a regression analysis? *Multivariate Behavioral Research, 26*(3), 499–510. https://doi.org/10.1207/s15327906mbr2603_7

Greenland, S., Pearl, J., & Robins, J. M. (1999). Causal diagrams for epidemiologic research. *Epidemiology, 10*(1), 37–48. https://doi.org/10.1097/00001648-199901000-00008

Greenland, S., Senn, S. J., Rothman, K. J., Carlin, J. B., Poole, C., Goodman, S. N., & Altman, D. G. (2016). Statistical tests, P values, confidence intervals, and power: A guide to misinterpretations. *European Journal of Epidemiology, 31*(4), 337–350. https://doi.org/10.1007/s10654-016-0149-3

Haber, N., Smith, E. R., Moscoe, E., Andrews, K., Audy, R., Bell, W., et al. (2018). Causal language and strength of inference in academic and media articles shared in social media (CLAIMS): A systematic review. *PLOS ONE, 13*(5), e0196346. https://doi.org/10.1371/journal.pone.0196346

Harrell, F. E. (2015). *Regression modeling strategies: With applications to linear models, logistic and ordinal regression, and survival analysis* (2nd ed.). Springer. https://doi.org/10.1007/978-3-319-19425-7

Harrell, F. E., Lee, K. L., & Mark, D. B. (1996). Multivariable prognostic models: Issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. *Statistics in Medicine, 15*(4), 361–387. https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4

Head, M. L., Holman, L., Lanfear, R., Kahn, A. T., & Jennions, M. D. (2015). The extent and consequences of p-hacking in science. *PLOS Biology, 13*(3), e1002106. https://doi.org/10.1371/journal.pbio.1002106

Heinze, G., & Schemper, M. (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine, 21*(16), 2409–2419. https://doi.org/10.1002/sim.1047

Hernán, M. A., Hernández-Díaz, S., & Robins, J. M. (2004). A structural approach to selection bias. *Epidemiology, 15*(5), 615–625. https://doi.org/10.1097/01.ede.0000135174.63482.43

Hernán, M. A., Hsu, J., & Healy, B. (2019). A second chance to get causal inference right: A classification of data science tasks. *CHANCE, 32*(1), 42–49. https://doi.org/10.1080/09332480.2019.1579578

Hernán, M. A., & Robins, J. M. (2016). Using big data to emulate a target trial when a randomized trial is not available. *American Journal of Epidemiology, 183*(8), 758–764. https://doi.org/10.1093/aje/kwv254

Hernán, M. A., & Robins, J. M. (2020). *Causal inference: What if*. Chapman & Hall/CRC. https://miguelhernan.org/whatifbook

Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511973420

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Imbens, G. W., & Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics, 142*(2), 615–635. https://doi.org/10.1016/j.jeconom.2007.05.001

Imbens, G. W., & Rubin, D. B. (2015). *Causal inference for statistics, social, and biomedical sciences: An introduction*. Cambridge University Press. https://doi.org/10.1017/CBO9781139025751

Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine, 2*(8), e124. https://doi.org/10.1371/journal.pmed.0020124

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804. https://doi.org/10.1016/j.patter.2023.100804

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 1–21. https://doi.org/10.1145/2382577.2382579

Kerr, N. L. (1998). HARKing: Hypothesizing after the results are known. *Personality and Social Psychology Review, 2*(3), 196–217. https://doi.org/10.1207/s15327957pspr0203_4

King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis, 9*(2), 137–163. https://doi.org/10.1093/oxfordjournals.pan.a004868

Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S. F., & Baker, C. I. (2009). Circular analysis in systems neuroscience: The dangers of double dipping. *Nature Neuroscience, 12*(5), 535–540. https://doi.org/10.1038/nn.2303

Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics, 34*(1), 1–14. https://doi.org/10.2307/1269547

Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative controls: A tool for detecting confounding and bias in observational studies. *Epidemiology, 21*(3), 383–388. https://doi.org/10.1097/EDE.0b013e3181d61eeb

Little, R. J. A., & Rubin, D. B. (2019). *Statistical analysis with missing data* (3rd ed.). Wiley. https://doi.org/10.1002/9781119482260

Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. *Biometrika, 65*(2), 297–303. https://doi.org/10.1093/biomet/65.2.297

López de Prado, M. (2018). *Advances in financial machine learning*. Wiley. ISBN 978-1119482086.

McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics, 142*(2), 698–714. https://doi.org/10.1016/j.jeconom.2007.05.005

McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall. https://doi.org/10.1007/978-1-4899-3242-6

Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of Econometrics, 33*(3), 341–365. https://doi.org/10.1016/0304-4076(86)90002-3

Munafò, M. R., Tilling, K., Taylor, A. E., Evans, D. M., & Davey Smith, G. (2018). Collider scope: When selection bias can substantially influence observed associations. *International Journal of Epidemiology, 47*(1), 226–235. https://doi.org/10.1093/ije/dyx206

Munafò, M. R., Nosek, B. A., Bishop, D. V. M., Button, K. S., Chambers, C. D., Percie du Sert, N., et al. (2017). A manifesto for reproducible science. *Nature Human Behaviour, 1*, 0021. https://doi.org/10.1038/s41562-016-0021

National Research Council. (2010). *The prevention and treatment of missing data in clinical trials*. National Academies Press. https://doi.org/10.17226/12955

Nosek, B. A., Alter, G., Banks, G. C., Borsboom, D., Bowman, S. D., Breckler, S. J., et al. (2015). Promoting an open research culture. *Science, 348*(6242), 1422–1425. https://doi.org/10.1126/science.aab2374

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. https://doi.org/10.1073/pnas.1708274114

O'Brien, R. M. (2007). A caution regarding rules of thumb for variance inflation factors. *Quality & Quantity, 41*(5), 673–690. https://doi.org/10.1007/s11135-006-9018-6

O'Hara, R. B., & Kotze, D. J. (2010). Do not log-transform count data. *Methods in Ecology and Evolution, 1*(2), 118–122. https://doi.org/10.1111/j.2041-210X.2010.00021.x

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71. https://doi.org/10.1136/bmj.n71

Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1996). A simulation study of the number of events per variable in logistic regression analysis. *Journal of Clinical Epidemiology, 49*(12), 1373–1379. https://doi.org/10.1016/S0895-4356(96)00236-3

Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. *Journal of the Royal Statistical Society: Series B, 31*(2), 350–371. https://doi.org/10.1111/j.2517-6161.1969.tb00796.x

Riley, R. D., Snell, K. I. E., Ensor, J., Burke, D. L., Harrell, F. E., Moons, K. G. M., & Collins, G. S. (2019). Minimum sample size for developing a multivariable prediction model: Part II — Binary and time-to-event outcomes. *Statistics in Medicine, 38*(7), 1276–1296. https://doi.org/10.1002/sim.7992

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography, 40*(8), 913–929. https://doi.org/10.1111/ecog.02881

Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika, 70*(1), 41–55. https://doi.org/10.1093/biomet/70.1.41

Rosenthal, R. (1979). The file drawer problem and tolerance for null results. *Psychological Bulletin, 86*(3), 638–641. https://doi.org/10.1037/0033-2909.86.3.638

Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023). What's trending in difference-in-differences? A synthesis of the recent econometrics literature. *Journal of Econometrics, 235*(2), 2218–2244. https://doi.org/10.1016/j.jeconom.2023.03.008

Rothman, K. J. (1990). No adjustments are needed for multiple comparisons. *Epidemiology, 1*(1), 43–46. https://doi.org/10.1097/00001648-199001000-00010

Rothwell, P. M. (2005). Subgroup analysis in randomised controlled trials: Importance, indications, and interpretation. *The Lancet, 365*(9454), 176–186. https://doi.org/10.1016/S0140-6736(05)17709-5

Rubin, D. B. (1976). Inference and missing data. *Biometrika, 63*(3), 581–592. https://doi.org/10.1093/biomet/63.3.581

Rubin, D. B. (1987). *Multiple imputation for nonresponse in surveys*. Wiley. https://doi.org/10.1002/9780470316696

Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten simple rules for reproducible computational research. *PLOS Computational Biology, 9*(10), e1003285. https://doi.org/10.1371/journal.pcbi.1003285

Schulz, K. F., Altman, D. G., & Moher, D. (2010). CONSORT 2010 statement: Updated guidelines for reporting parallel group randomised trials. *BMJ, 340*, c332. https://doi.org/10.1136/bmj.c332

Shmueli, G. (2010). To explain or to predict? *Statistical Science, 25*(3), 289–310. https://doi.org/10.1214/10-STS330

Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin, 86*(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420

Sim, J., & Wright, C. C. (2005). The kappa statistic in reliability studies: Use, interpretation, and sample size requirements. *Physical Therapy, 85*(3), 257–268. https://doi.org/10.1093/ptj/85.3.257

Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science, 22*(11), 1359–1366. https://doi.org/10.1177/0956797611417632

Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour, 4*(11), 1208–1214. https://doi.org/10.1038/s41562-020-0912-z

Simpson, E. H. (1951). The interpretation of interaction in contingency tables. *Journal of the Royal Statistical Society: Series B, 13*(2), 238–241. https://doi.org/10.1111/j.2517-6161.1951.tb00088.x

Steegen, S., Tuerlinckx, F., Gelman, A., & Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science, 11*(5), 702–712. https://doi.org/10.1177/1745691616658637

Sterne, J. A. C., White, I. R., Carlin, J. B., Spratt, M., Royston, P., Kenward, M. G., et al. (2009). Multiple imputation for missing data in epidemiological and clinical research: Potential and pitfalls. *BMJ, 338*, b2393. https://doi.org/10.1136/bmj.b2393

Steyerberg, E. W. (2019). *Clinical prediction models: A practical approach to development, validation, and updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Steyerberg, E. W., Eijkemans, M. J. C., Harrell, F. E., & Habbema, J. D. F. (1999). Stepwise selection in small data sets: A simulation study of bias in logistic regression analysis. *Journal of Clinical Epidemiology, 52*(10), 935–942. https://doi.org/10.1016/S0895-4356(99)00103-1

Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics, 8*, 25. https://doi.org/10.1186/1471-2105-8-25

Suissa, S. (2008). Immortal time bias in pharmacoepidemiology. *American Journal of Epidemiology, 167*(4), 492–499. https://doi.org/10.1093/aje/kwm324

Taylor, J., & Tibshirani, R. J. (2015). Statistical learning and selective inference. *Proceedings of the National Academy of Sciences, 112*(25), 7629–7634. https://doi.org/10.1073/pnas.1507583112

Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: The Achilles heel of predictive analytics. *BMC Medicine, 17*, 230. https://doi.org/10.1186/s12916-019-1466-7

van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association, 29*(9), 1525–1534. https://doi.org/10.1093/jamia/ocac093

van Smeden, M., de Groot, J. A. H., Moons, K. G. M., Collins, G. S., Altman, D. G., Eijkemans, M. J. C., & Reitsma, J. B. (2016). No rationale for 1 variable per 10 events criterion for binary logistic regression analysis. *BMC Medical Research Methodology, 16*, 163. https://doi.org/10.1186/s12874-016-0267-3

VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine, 167*(4), 268–274. https://doi.org/10.7326/M16-2607

Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics, 7*, 91. https://doi.org/10.1186/1471-2105-7-91

Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage, 180*, 68–77. https://doi.org/10.1016/j.neuroimage.2017.06.061

Vittinghoff, E., & McCulloch, C. E. (2007). Relaxing the rule of ten events per variable in logistic and Cox regression. *American Journal of Epidemiology, 165*(6), 710–718. https://doi.org/10.1093/aje/kwk052

von Elm, E., Altman, D. G., Egger, M., Pocock, S. J., Gøtzsche, P. C., & Vandenbroucke, J. P. (2007). The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: Guidelines for reporting observational studies. *PLoS Medicine, 4*(10), e296. https://doi.org/10.1371/journal.pmed.0040296

Wang, R., Lagakos, S. W., Ware, J. H., Hunter, D. J., & Drazen, J. M. (2007). Statistics in medicine — Reporting of subgroup analyses in clinical trials. *New England Journal of Medicine, 357*(21), 2189–2194. https://doi.org/10.1056/NEJMsr077003

Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician, 70*(2), 129–133. https://doi.org/10.1080/00031305.2016.1154108

Westreich, D., & Cole, S. R. (2010). Invited commentary: Positivity in practice. *American Journal of Epidemiology, 171*(6), 674–677. https://doi.org/10.1093/aje/kwp436

Westreich, D., & Greenland, S. (2013). The table 2 fallacy: Presenting and interpreting confounder and modifier coefficients. *American Journal of Epidemiology, 177*(4), 292–298. https://doi.org/10.1093/aje/kws412

White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica, 48*(4), 817–838. https://doi.org/10.2307/1912934

White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152

White, I. R., Royston, P., & Wood, A. M. (2011). Multiple imputation using chained equations: Issues and guidance for practice. *Statistics in Medicine, 30*(4), 377–399. https://doi.org/10.1002/sim.4067

Wicherts, J. M., Veldkamp, C. L. S., Augusteijn, H. E. M., Bakker, M., van Aert, R. C. M., & van Assen, M. A. L. M. (2016). Degrees of freedom in planning, running, analyzing, and reporting psychological studies: A checklist to avoid p-hacking. *Frontiers in Psychology, 7*, 1832. https://doi.org/10.3389/fpsyg.2016.01832

Wolff, R. F., Moons, K. G. M., Riley, R. D., Whiting, P. F., Westwood, M., Collins, G. S., et al. (2019). PROBAST: A tool to assess the risk of bias and applicability of prediction model studies. *Annals of Internal Medicine, 170*(1), 51–58. https://doi.org/10.7326/M18-1376

Yule, G. U. (1926). Why do we sometimes get nonsense-correlations between time-series? A study in sampling and the nature of time-series. *Journal of the Royal Statistical Society, 89*(1), 1–63. https://doi.org/10.2307/2341482

Zeileis, A., Kleiber, C., & Jackman, S. (2008). Regression models for count data in R. *Journal of Statistical Software, 27*(8), 1–25. https://doi.org/10.18637/jss.v027.i08
