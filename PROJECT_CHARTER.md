# Project Charter

**Status:** Approved 22 November 2026

## Project Title

**Statistical Analysis and Machine Learning for Engagement and Behaviour in Online Reddit Communities**

## 1. Project Purpose

This academic project will study how discussions develop in selected Reddit communities concerned with artificial intelligence and technology. It will combine probability, statistics, Python data engineering, exploratory analysis, classical machine learning, model optimization, visualization, and basic deployment in one reproducible workflow.

The project is intended to demonstrate the methods in the academic syllabus. It is not intended to profile individual users, make causal claims, or build a commercial engagement-ranking system.

## 2. Fixed Scope

### Data source

- Development source: Arctic Shift public Reddit archive API
- Access method: timestamp-bounded public API requests
- Platform: Reddit only
- External datasets: none in the minimum viable project

The project originally preferred Reddit for Researchers, but no BigQuery access is configured. On 22 November 2026, execution was changed to the publicly accessible Arctic Shift archive so development could start immediately. The archive is an unofficial source and its completeness, collection method, score snapshots, and unspecified redistribution rights will be reported as limitations. Raw records will not be redistributed.

### Communities

The initial data request will cover these six communities:

1. `r/artificial`
2. `r/MachineLearning`
3. `r/datascience`
4. `r/technology`
5. `r/ChatGPT`
6. `r/LocalLLaMA`

These communities provide a mixture of general technology discussion, academic machine learning, professional data science, and generative-AI discussion. Their availability and fields will be confirmed during archive extraction and schema validation.

### Study period

- Root posts: 1 January 2024 through 31 March 2024
- Comment outcome buffer: through 3 April 2024
- Final working size: 20,094 analysis posts and 106,003 matched comments
- Pilot extraction: approximately 1,000 records
- Development extraction: approximately 20,000 to 50,000 records

The outcome buffer allows every post created during the study period to receive a complete 72-hour observation window.

### Units of analysis

- Root post: descriptive statistics, statistical tests, regression, and classification
- Comment: sentiment and reply behaviour
- Complete discussion thread: propagation and network features
- Pseudonymous user: aggregate activity and reply-network analysis only
- Community-week: time-series analysis

## 3. Research Questions

### Primary questions

1. What are the distributions and statistical characteristics of engagement across the selected communities?
2. How are posting time, text characteristics, sentiment, topic, author history, and recent community activity associated with engagement?
3. Are observed differences in engagement between communities, topics, sentiment groups, and posting periods statistically significant and practically meaningful?
4. Can publication-time information predict whether a post will receive high engagement within 72 hours?

### Supporting questions

5. Can posts be grouped into interpretable behavioural clusters using numerical features, PCA, K-Means, and hierarchical clustering?
6. How do activity, sentiment, topics, and engagement change from week to week?
7. What do reply depth, response time, and user-to-user reply networks reveal about discussion propagation?

## 4. Outcome Definitions

### Engagement

The main engagement measure is:

```text
reply_count_72h = number of comments in a post's discussion tree
                  created within 72 hours of the root post
```

Secondary engagement measures are:

```text
unique_repliers_72h
maximum_thread_depth_72h
median_first_level_response_time
```

Score is a secondary engagement outcome. It is an archive snapshot, commonly refreshed around 36 hours, and is treated as Reddit's fuzzed net score rather than an exact upvote count. Shares are unavailable.

### Regression target

```text
regression_target = log1p(reply_count_72h)
```

The transformation reduces the influence of the expected heavy right tail while retaining zero-engagement posts.

### Classification target

```text
high_engagement = 1 when reply_count_72h is at or above the
                  subreddit-specific 90th percentile OR observed Reddit
                  score is at or above the 95th percentile, with both
                  thresholds calculated from training-period posts only;
                  otherwise 0
```

Training thresholds will be reused unchanged for validation and test data. This avoids deriving target thresholds from future observations and creates a meaningful imbalanced-learning task.

### Prediction time

The primary prediction is made at publication time. Only information available at or before `created_at` may be used as a feature. Final scores, future replies, eventual thread depth, and future author statistics are prohibited predictors.

## 5. Planned Features

### Publication-time features

- Community
- Hour of day, weekday, month, and weekend indicator
- Title and body character and word counts
- URL indicator and URL domain category
- Sentiment score and label
- Topic probabilities and dominant topic
- Author activity calculated strictly before publication
- Community activity calculated strictly before publication

### Outcome and analytical features

- Replies and unique repliers within 72 hours
- Response time
- Thread depth and breadth
- Weekly activity and engagement aggregates
- Reply-network centrality measures

Outcome and network features will not be used as publication-time predictors when they occur after the prediction time.

## 6. Statistical Hypotheses

All confirmatory tests will use a predeclared significance level of `alpha = 0.05`. Confidence intervals and effect sizes will accompany p-values. Assumptions will be checked before interpreting a test.

### H1: URL and high engagement

```text
H0: Posts with and without URLs have equal high-engagement proportions.
H1: Their high-engagement proportions differ.
```

Planned method: two-proportion Z-test when its assumptions are satisfied.

### H2: Weekend and weekday engagement

```text
H0: Weekend and weekday posts have equal mean log1p engagement.
H1: Their mean log1p engagement differs.
```

Planned method: Welch's independent-samples t-test, with a non-parametric sensitivity analysis if needed.

### H3: Topic and engagement class

```text
H0: Dominant topic and high-engagement class are independent.
H1: They are associated.
```

Planned method: Chi-square test of independence and Cramer's V.

### H4: Community engagement

```text
H0: Mean log1p engagement is equal across the selected communities.
H1: At least one community mean differs.
```

Planned method: one-way ANOVA after checking assumptions, followed by an appropriate post-hoc analysis. A robust or non-parametric alternative will be reported if assumptions are materially violated.

### H5: Predictive value

```text
H0: Publication-time behavioural and text features do not improve
    high-engagement prediction over a dummy baseline.
H1: At least one fitted model improves out-of-time performance over
    the dummy baseline.
```

Planned evidence: chronological test-set F1 and precision-recall AUC, with uncertainty where practical.

These analyses measure association and predictive value, not causation.

## 7. Required Academic Coverage

| Syllabus area | Minimum project evidence |
|---|---|
| Probability and statistics | Descriptive statistics, conditional probability, sampling simulation, Normal/Binomial/Poisson/Uniform demonstrations, estimates, and confidence intervals |
| Python engineering | NumPy operations, Pandas loading/filtering/aggregation, cleaning, feature engineering, preprocessing, EDA, and outlier analysis |
| Computing techniques | Scikit-learn pipeline, Linear and Logistic Regression, KNN, Decision Tree, K-Means, hierarchical clustering, PCA, evaluation, and weekly time-series analysis |
| Advanced techniques | Imbalance treatment, SMOTE inside training only, Grid Search, Random Search, Random Forest, Bagging, Boosting, cross-validation, SHAP or LIME, and a limited AutoML comparison |
| Statistical modelling | Z-test, Welch's t-test, Chi-square, ANOVA, p-values, effect sizes, Type I/II error simulation, bias-variance analysis, and model metrics |
| Visualization | Static plots, interactive Streamlit views, clear visual design, and a connected data story |
| Production application | Versioned training pipeline, FastAPI prediction endpoint, Streamlit client, basic monitoring, and documented retraining procedure |

## 8. Evaluation Design

### Temporal split

Data will be ordered by root-post creation time:

- First 70%: training
- Next 15%: validation
- Final 15%: testing

Complete threads will remain within the split of their root post. Preprocessing, topic models, feature selection, resampling, and tuning will be fitted using training data only.

### Regression metrics

- Mean Absolute Error
- Root Mean Squared Error
- R-squared

### Classification metrics

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-recall AUC
- Confusion matrix

Accuracy will be reported but will not be the primary measure because high engagement is intentionally uncommon.

## 9. Ethics and Privacy

- Use only records returned by the documented, timestamp-bounded archive queries.
- Keep source URLs, extraction times, checksums, and observed coverage in a local manifest.
- Keep Reddit-provided identifiers hashed or replace them with project-specific pseudonyms.
- Do not attempt to identify, deanonymize, contact, or link users across platforms.
- Do not infer sensitive personal traits.
- Do not publish usernames, user leaderboards, raw searchable quotations, or raw text downloads.
- Restrict raw data to approved project members and approved storage.
- Use aggregate examples in the report and dashboard.
- Suppress dashboard groups containing fewer than 10 observations.
- Present findings as behaviour in the observed communities, not as representative of Reddit users or the general public.
- Document missing, removed, private, spam, automated, and moderation-affected content as possible sources of bias.

The project will still apply data minimization, pseudonymization, aggregate reporting, and non-identification practices even though the development source is publicly accessible.

## 10. Boundaries

The minimum viable project will not include:

- Real-time Reddit ingestion
- Cross-platform user matching
- Individual-level psychological or demographic profiling
- Causal claims from observational associations
- Transformer training or large language model fine-tuning
- A public raw-text search tool
- Collection of every available subreddit or year
- Production use of the engagement model for ranking or moderation

NLP and network analysis will support the statistical and machine-learning study rather than replace the syllabus-focused work.

## 11. Success Criteria

### Data

- A documented and reproducible extraction query
- A data dictionary for every retained source and derived field
- No duplicate primary keys in final tables
- Valid post-comment relationships for retained analytical records
- Missingness and exclusions quantified rather than hidden
- A structured Parquet dataset within the target working size

### Academic analysis

- Every syllabus area in Section 7 has at least one correctly implemented, interpreted artifact
- Statistical tests include assumptions, uncertainty, effect sizes, and practical interpretation
- Probability-distribution demonstrations distinguish simulation from empirical model fit
- All predictive features pass a publication-time leakage review

### Modelling

- Dummy baselines are compared with every main predictive task
- Final performance is measured once on an untouched chronological test set
- Imbalance methods are evaluated without altering validation or test distributions
- Results are considered valid even if advanced models do not outperform simple models, provided the evaluation is correct and limitations are explained

### Application

- Streamlit dashboard presents the main statistical story without exposing users
- FastAPI returns validated predictions from the versioned final pipeline
- Training metadata, model version, metrics, and monitoring checks are recorded

### Communication

- The final report answers each research question
- Charts have titles, units, readable labels, and explanatory captions
- Associations, predictions, limitations, and non-causal conclusions are clearly separated

## 12. Stage 1 Approval Gate

Before starting the Reddit for Researchers application, confirm or revise:

- [x] Project title
- [x] The six selected communities
- [x] The 2024 study period
- [x] The 72-hour engagement definition
- [x] The top-10% classification target
- [x] The research questions and hypotheses
- [x] Public archive selected as the development source

The analytical scope was approved on 22 November 2026. The development source was changed from Reddit for Researchers to the Arctic Shift public API on the same date so pilot acquisition could begin without an access dependency.
