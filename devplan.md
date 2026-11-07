# Development plan

Setup postgres pgvector database

Create schema for database - can be modified later on 

Create key systems

## Research system

Take provided dataset and additional provided information - e.g. a description, a source site  (all metadata will be optional)

Do a vector lookup on the database on the field or any papers that might be related - allow for filtering with different parameters on the database

Fields are organised as so, there is a set of top level fields, i.e. finance, sport, shipping. We compare the semantic similarity with RAG which we can then use to do further search

Once we have the closest top-level fields, we do a search on level 2 fields to try and find the closest second-level fields related to those top-level fields and repeat until we are at a deeper level. When we have reached the final level we need a judgement call to tell if something is more specific or less specific and we swap orders in the tree with the current category until the closest layers are ordered according to generality. If an object is classed as under a field and that field increases in level then all objects under its classification should also have their classification increase accordingly so that the hierarchy still stands as so. We only create a category if it is part of a category that is not mentioned -- rephrase

Once we have identified the classifications that this dataset that we are provided with falls under, we will attempt to research about the field and related areas

We will use search apis to find general results on the web and results in academia.

For finding similar datasets, where there is a free api we can use , use Hugging Face Hub, Kaggle, Zenodo, Dataverse, OpenML, data.gov, DataCite, Google Dataset Search
For papers regarding the subject, we use a free api for Semantic Scholar, OpenAlex, arXiv 
We additionally use a free web api to find the base 20-50? search results

From there we use our semantic scraping methodology for the papers (by references) and for the web results (by links)

### Semantic scraping methodology

If the resource has already been catalogued, we can ignore it/only record the link to it from the other resource - verify this by comparing the link used to access the resource with any access information removed e.g. from="....." and with a sha512 hash of the data

When given a resource, we encode a summary of what it covers
We are also given a target that we are searching for, or its semantic embedding

If the summary of the resource has a semantic embedding close enough by some metric, we add all of its links to be added to our results, otherwise we include only the links with summaries that are close enough by the same (or a tighter) metric (e.g. comparing the cosine distance by some given threshold). We add the resource to the dataset in a similar way to how we classify the user's data, and associate it with the category in the hierarchy that it is closest to.

### Resource processing on adding

When a resource is added. We generate a summary of it (by asking an agent) which can be used to get a standardised embedding. We make note of the authors and publishers and assess the reliability of them. We use a metric combining these to determine whether or not a resource is credible, and as such we keep it stored, but do not consider it if it is not credible.

We also need a way to consider what resources to compare this to, to make possible findings and note areas of research as a comparison piecewise between each dataset may be too cumbersome. 

we do a direct comparison with how it might effect the dataset we are currently processing itself and we record information about comparisons between resources in the database to avoid the need for recalculation.

As part of the verification of papers and news articles we apply a critique based on the statistics methodologies preloaded into the system

### Preloaded data

We preload the dataset with a large amount of layer 1 data under the category of statistics and validity of statistical measures, we need to figure out what search terms to use for the apis to do this

## Hypothesis generation system

All hypotheses should be returned in the end in the form of a python function and a labelled method of determining which columns it applies to in the dataset.

### agentic

For this we work with two versions of the dataset, one where the data is anonymized and one where it is not and one where the labels are jumbled around.
If the agent working with labels that are jumbled around ever unscrambles the labels correctly, we ignore its results.

When generating hypotheses based on a dataset, we consider those found during the research of the dataset itself. If when considering a piece of research we think that it presents a model that applies to our data, then we add that to our generated models.

Considering all of this we the get the agent to attempt to build a model. The agent needs to be capable of receiving issues that have occurred with the model and either restructuring it to solve these issues or throwing out the model. This can either be the same agent or a different one. If the model is thrown out, we may then attempt to produce new models with the additional information provided to the context.


### statistical

Attempt to classify the data, both agentically and statically

When performing static analysis on the dataset, we perform it three times, each with a differing 30% of records removed from the dataset at random and left aside as testing data, and once with the full dataset included.
Calculate a correlation matrix between rows in the dataset. Pearson & spearman, mutual information and anova/chi-squared tests
Where a high likelihood of correlation is found, attempt to find regressions (with log/sqrt transforms on all sides).
For binary or categorical data, use logistic or multinomial regressions (GLMs)
Use poisson or negative binomial for counting data.
When using time data attempt to check for seasonal or cyclical trends

#### Multi-column

Tree-based models (decision trees, random forests, gradient boosting): capture interactions and thresholds without you specifying them. Feature importances and partial dependence plots then show you what structure to model explicitly.
Multiple regression with interactions: this is the quickest. statsmodels formulas (y ~ x1 * x2) or sklearn’s PolynomialFeatures(interaction_only=True) do it in a few lines. Automated search over term sets is a short loop: generate candidate term sets, fit each, rank by AIC/BIC.

#### Pure equations

SINDy with pysindy: the quickest route to an explicit equation. Build a library of candidate terms (polynomials, optionally sin/cos/exp), call fit, and sparse regression picks the few terms that matter. It’s designed for time-series derivatives, but you can use the same idea on ordinary tabular data by treating your target as the thing to fit and the library as your candidate features.

## Hypothesis validation

### agentic

Inspection of validation before we pass off to statistical library to verify

In this validation we examine the metadata the user provided about how the user was gathered and track through the methodology used to generate the model on the data by the agent. If anything that the data generation process would introduce is considered in this generation then we throw out the model, or attempt to refine it.

### statistical

Inspection by provided library.