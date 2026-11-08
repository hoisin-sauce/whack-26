# Field hierarchy evaluation

Generated: 2026-11-08T05:57:32Z
Dataset: `synthetic-field-hierarchy-v1`
Strategy: `keyword`
Cases: 30
Overall status: **PASS**

## Metrics

| Metric | Count | Rate |
|---|---:|---:|
| Exact field match | 30/30 | 100.0% |
| Ancestor match (strict) | 0/30 | 0.0% |
| Exact or ancestor match | 30/30 | 100.0% |
| Over-specificity | 0/30 | 0.0% |
| Under-specificity | 0/30 | 0.0% |
| Path consistency | 30/30 | 100.0% |
| Branch precision | 1/1 | 100.0% |
| Promotion precision | 0/0 | 100.0% |

## Gates

| Gate | Status | Detail |
|---|---|---|
| Over-specific attachment | pass | 0/30 (0.0%); allowed 0% |
| Exact or ancestor match | pass | placements at or above the labelled field: 30/30 (100.0%); required 80% |
| Path consistency | pass | predicted root-to-leaf paths: 30/30 (100.0%); required 100% |
| Branch precision | pass | accepted new fields: 1/1 (100.0%); required 80% |
| Promotion precision | skip | promotions that improve the hierarchy: no predictions |
| Under-specificity (diagnostic) | pass | placements no broader than the labelled field: 30/30 (100.0%); required 50% |

## Placements

| Document | Expected | Predicted | Relation | Path consistent |
|---|---|---|---|---|
| doc_penguin_ecology | field_ecology | field_ecology | exact | yes |
| doc_coral_restoration | field_conservation | field_conservation | exact | yes |
| doc_maize_genetics | field_genetics | field_genetics | exact | yes |
| doc_mammal_biology | field_biology | field_biology | exact | yes |
| doc_minimum_wage | field_labor | field_labor | exact | yes |
| doc_mortgage_rates | field_housing | field_housing | exact | yes |
| doc_economic_growth | field_economics | field_economics | exact | yes |
| doc_neural_classifier | field_ml | field_ml | exact | yes |
| doc_database_networking | field_systems | field_systems | exact | yes |
| doc_algorithm_design | field_cs | field_cs | exact | yes |
| doc_outbreak_surveillance | field_epi | field_epi | exact | yes |
| doc_vaccination_mortality | field_health | field_health | exact | yes |
| doc_science_method | field_science | field_science | exact | yes |
| doc_ecosystem_services | field_ecology | field_ecology | exact | yes |
| doc_genetic_variation | field_genetics | field_genetics | exact | yes |
| doc_unemployment | field_labor | field_labor | exact | yes |
| doc_real_estate | field_housing | field_housing | exact | yes |
| doc_model_training | field_ml | field_ml | exact | yes |
| doc_compiler_systems | field_systems | field_systems | exact | yes |
| doc_epidemic_models | field_epi | field_epi | exact | yes |
| doc_health_disparities | field_health | field_health | exact | yes |
| doc_biodiversity | field_ecology | field_ecology | exact | yes |
| doc_agroecology | field_agroecology | field_agroecology | exact | yes |
| doc_public_health_systems | field_health | field_health | exact | yes |
| doc_software_testing | field_cs | field_cs | exact | yes |
| doc_protected_areas | field_conservation | field_conservation | exact | yes |
| doc_macro_economics | field_economics | field_economics | exact | yes |
| doc_housing_markets | field_housing | field_housing | exact | yes |
| doc_deep_learning | field_ml | field_ml | exact | yes |
| doc_species_richness | field_biology | field_biology | exact | yes |

## Lifecycle events

| Predicted branch | Parent | Accepted |
|---|---|---|
| field_agroecology (Agroecology) | field_ecology | yes |

- No promotions predicted.

## Warnings

- None.

## Method

- Offline and deterministic: no network calls and no live LLM.
- `keyword` is the default baseline; `perfect`, `over-specific`, and `under-specific` are diagnostic strategies used to exercise the metrics.
- The blocking gate is no over-specific attachment, at least 80% exact/ancestor match, and path consistency. Branch and promotion precision are blocking when lifecycle changes are predicted.
- Under-specificity is reported as a diagnostic because the spec's exact/ancestor gate intentionally accepts broader attachments.
