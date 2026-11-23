# Final Results

## Scope

The analytical dataset contains 20,094 posts from `MachineLearning`, `artificial`, and `datascience`, created from 1 January through 31 March 2024. Comments through 3 April provide complete 72-hour outcomes.

## Descriptive Findings

- Mean replies within 72 hours: 4.96
- Median replies within 72 hours: 1
- Standard deviation: 19.84
- Maximum replies: 584
- Posts receiving at least one reply: 14,338 of 20,094
- Bootstrap 95% confidence interval for mean replies: 4.68 to 5.25

The mean remains much larger than the median, confirming a strongly right-skewed distribution. Regression therefore uses `log(1 + replies)`.

## Probability And Inference

- High-engagement prevalence: 10.4%
- Simulated Type I error at `alpha = 0.05`: 5.0%
- Simulated power for a standardized effect of 0.5 with 30 observations per group: 49.9%
- Weekend versus weekday Welch test: `p = 0.341`, negligible effect (`d = 0.016`)
- Topic versus high engagement: `p < 0.001`, small association (`Cramer's V = 0.097`)
- Community ANOVA: `p < 0.001`, small-to-moderate effect (`eta squared = 0.039`)
- Equal variances fail, but the Kruskal-Wallis sensitivity test also gives `p < 0.001`

These results indicate association, not causation.

## Topics

Five NMF topics represent:

1. Data-science careers and analytics
2. Generative AI, video, and future applications
3. Machine learning, deep learning, and education
4. Model training and applied ML
5. Help, projects, advice, and resumes

## Predictive Modelling

The multi-signal target recognizes discussion volume, vote quality (`score × upvote ratio`), unique participants, conversation depth, and crossposts. Quantile thresholds are learned from training data only; any crosspost also qualifies. Reddit score is a fuzzed archive snapshot, not exact upvotes, and shares are unavailable. Historical score velocity cannot be reconstructed from one snapshot, so the live workflow records it between repeated checks instead. Data is split chronologically into 14,065 training, 3,014 validation, and 3,015 test posts. Test prevalence is 10.38%.

### Publication-time model

Selected model: Random Forest.

| Metric | Test result |
|---|---:|
| Accuracy | 0.710 |
| Precision | 0.203 |
| Recall | 0.613 |
| F1 | 0.305 |
| ROC-AUC | 0.728 |
| PR-AUC | 0.242 |

This is a clear improvement over the 0.104 prevalence baseline, but publication-time prediction remains difficult because platform exposure and ranking are unavailable.

### One-hour early model

Selected model: Random Forest. It adds first-hour reply velocity, unique repliers, and maximum conversation depth.

| Metric | Test result |
|---|---:|
| Accuracy | 0.946 |
| Precision | 0.779 |
| Recall | 0.674 |
| F1 | 0.723 |
| ROC-AUC | 0.918 |
| PR-AUC | 0.727 |

First-hour reply velocity is the most important early feature, followed by unique repliers and conversation depth. The decision cutoff `0.579` was selected by maximizing F1 on validation data and then frozen before test evaluation. This model answers a different and explicitly declared question: prediction after one hour, not at publication.

### Regression

Random Forest was selected for `log(1 + replies)`, with test `R² = 0.054`. Exact reply counts remain difficult to predict, so classification is the stronger task.

Archived body text remains excluded from all predictive models because later moderation can replace it with `[removed]`, causing retrieval-time leakage.

## PCA And Clustering

- The first two PCA components explain 44.9% of standardized numerical variance.
- Three K-Means clusters give the best tested silhouette score: 0.321.
- Hierarchical clustering uses a deterministic 1,000-post sample.

Clusters are exploratory behavioural groupings, not objective categories.

## Reply Network

The pseudonymous reply network contains 100,341 reply events, 84,096 unique directed edges, and 29,668 users. The largest weakly connected component contains 29,538 users. Only aggregate network results are exposed.

## Time Series

Daily and weekly activity, engagement, sentiment, and four-week rolling activity are generated. Across a three-week holdout, the previous-week baseline has MAE 79.7 posts and a three-week moving average has MAE 50.6 posts. Thirteen weekly periods remain insufficient for annual seasonality claims.
