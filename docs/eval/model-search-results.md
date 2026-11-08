# Automated model-search evaluation

Mode: auto
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
| linear | y | continuous | r2 | interaction, linear | +0.9438 | ok | - |
| interaction | y | continuous | r2 | interaction, interaction | +0.8960 | ok | - |
| binary_logistic | y_binary | binary | accuracy | logistic, interaction | +0.1736 | ok | 0.7750 |
| multiclass | y_class | multiclass | accuracy | multinomial, multinomial | +0.2593 | ok | - |
| count_overdispersed | y_count | count | poisson_deviance | negative_binomial, poisson | +2.3102 | ok | - |
| time_series | y | time_series | r2 | linear, linear | +3.8313 | ok | - |
| group_means | y | continuous | r2 | group_comparison, linear | +0.0764 | ok | - |
| missing_collinear | y | continuous | r2 | interaction, interaction | +0.9531 | ok | - |
| small_n_high_cardinality | y | continuous | r2 | linear, linear | +0.3292 | ok | - |

## Optional backends

| Backend | Available | Version |
|---|---|---|
| sklearn | yes | 1.9.1 |
| statsmodels | yes | 0.15.0 |
| pysindy | yes | 2.1.0 |

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
