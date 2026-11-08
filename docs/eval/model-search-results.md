# Automated model-search evaluation

> **Historical snapshot.** This report was generated with the deterministic
> local fallback before the optional backends were installed. The current
> development environment has `scikit-learn`, `pysindy`, and `statsmodels`
> available; rerun `eval/model_search.py` to refresh the optional-backend table.

Mode: local
Folds: 3; seed: 0
Top-family recovery: **100%**
Baseline improvement: **100%**
Leakage clean: **yes**

## Gates

| Gate | Status | Detail |
|---|---|---|
| Top-family recovery | pass | 100% of datasets have the known family or a better alternative in the top two |
| Baseline improvement | pass | 100% of datasets beat the baseline beyond the configured margin |
| Target leakage | pass | no candidate uses a target or target-derived column |

## Datasets

| Dataset | Target | Kind | Metric | Top families | Improvement | Leakage | Holdout |
|---|---|---|---|---|---|---|---|
| linear | y | continuous | r2 | linear, linear | +0.9574 | ok | - |
| interaction | y | continuous | r2 | interaction, linear | +0.9366 | ok | - |
| binary_logistic | y_binary | binary | accuracy | logistic, logistic | +0.1528 | ok | 0.7500 |
| multiclass | y_class | multiclass | accuracy | multinomial, multinomial | +0.2284 | ok | - |
| count_overdispersed | y_count | count | poisson_deviance | negative_binomial, poisson | +1.9255 | ok | - |
| time_series | y | time_series | r2 | linear, sindy | +4.2192 | ok | - |
| group_means | y | continuous | r2 | group_comparison, group_comparison | +0.9197 | ok | - |
| missing_collinear | y | continuous | r2 | linear, linear | +0.9327 | ok | - |
| small_n_high_cardinality | y | continuous | r2 | linear, linear | +0.6865 | ok | - |

## Optional backends

| Backend | Available | Version |
|---|---|---|
| sklearn | no | - |
| statsmodels | no | - |
| pysindy | no | - |

## Model-search modules

| Module | Available |
|---|---|
| data.py | yes |
| splits.py | yes |
| screen.py | yes |
| candidates.py | yes |
| fit.py | yes |
| metrics.py | yes |
| select.py | yes |
| diagnostics.py | yes |
| trees.py | yes |
| sindy.py | yes |
| report.py | yes |
| pipeline.py | yes |
