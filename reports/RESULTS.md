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

The hybrid target is high engagement when a post reaches either the subreddit-specific top 10% of replies within 72 hours or top 5% of observed Reddit score. Thresholds are learned from training data only. Reddit score is a fuzzed archive snapshot, not exact upvotes, and shares are unavailable. Data is split chronologically into 14,065 training, 3,014 validation, and 3,015 test posts. Test prevalence is 9.92%.

### Publication-time model

Selected model: Random Forest.

| Metric | Test result |
|---|---:|
| Accuracy | 0.761 |
| Precision | 0.208 |
| Recall | 0.502 |
| F1 | 0.294 |
| ROC-AUC | 0.726 |
| PR-AUC | 0.252 |

This is a clear improvement over the 0.099 prevalence baseline, but publication-time prediction remains difficult because platform exposure and ranking are unavailable.

### One-hour early model

Selected model: Random Forest. It adds only reply count and unique repliers observed during the first hour.

| Metric | Test result |
|---|---:|
| Accuracy | 0.946 |
| Precision | 0.740 |
| Recall | 0.696 |
| F1 | 0.717 |
| ROC-AUC | 0.915 |
| PR-AUC | 0.719 |

Unique repliers during the first hour is the most important early feature. The decision cutoff `0.524` was selected by maximizing F1 on validation data and then frozen before test evaluation. This model answers a different and explicitly declared question: prediction after one hour, not at publication.

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
