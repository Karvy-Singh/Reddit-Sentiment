"""Topic modelling and inferential statistics for the pilot dataset."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def add_topics(
    frame: pd.DataFrame, n_topics: int = 5, random_state: int = 42
) -> tuple[pd.DataFrame, list[list[str]]]:
    if len(frame) < n_topics * 10:
        raise ValueError("too few records for the requested topic count")

    text = (frame["title"].fillna("") + " " + frame["body"].fillna(""))
    text = text.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    text = text.str.replace(r"\[(removed|deleted)\]", " ", regex=True, case=False).str.strip()
    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=5, max_df=0.90, max_features=1500, ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform(text)
    if matrix.shape[1] < n_topics:
        raise ValueError("too few text features for topic modelling")

    model = NMF(n_components=n_topics, init="nndsvda", random_state=random_state)
    weights = model.fit_transform(matrix)
    terms = vectorizer.get_feature_names_out()
    keywords = [terms[row.argsort()[-8:][::-1]].tolist() for row in model.components_]

    result = frame.copy()
    result["topic_id"] = weights.argmax(axis=1)
    result["topic_strength"] = weights.max(axis=1)
    return result, keywords


def add_high_engagement(
    frame: pd.DataFrame, quantile: float = 0.90, score_quantile: float = 0.95
) -> pd.DataFrame:
    if not 0 < quantile < 1 or not 0 < score_quantile < 1:
        raise ValueError("quantile must be between zero and one")
    result = frame.copy()
    reply_thresholds = result.groupby("subreddit")["reply_count_72h"].transform(
        lambda values: values.quantile(quantile, interpolation="higher")
    )
    score_thresholds = result.groupby("subreddit")["score_observed"].transform(
        lambda values: values.quantile(score_quantile, interpolation="higher")
    )
    result["high_engagement_threshold"] = reply_thresholds.astype("int64")
    result["high_engagement_score_threshold"] = score_thresholds.astype("int64")
    result["high_engagement"] = result["reply_count_72h"].ge(
        reply_thresholds
    ) | result["score_observed"].ge(score_thresholds)
    return result


def bootstrap_mean_ci(
    values: pd.Series, samples: int = 2000, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size < 2:
        raise ValueError("at least two values are required for a confidence interval")
    generator = np.random.default_rng(seed)
    means = generator.choice(clean, size=(samples, clean.size), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    return tuple(np.quantile(means, [tail, 1 - tail]).tolist())


def _float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def statistical_tests(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"has_url", "high_engagement", "is_weekend", "log_replies_72h", "topic_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing statistical fields: {sorted(missing)}")

    url_table = pd.crosstab(frame["has_url"], frame["high_engagement"])
    if url_table.shape != (2, 2):
        raise ValueError("URL proportion test requires both binary groups and outcomes")
    successes = url_table[True].to_numpy()
    totals = url_table.sum(axis=1).to_numpy()
    pooled = successes.sum() / totals.sum()
    standard_error = math.sqrt(pooled * (1 - pooled) * np.sum(1 / totals))
    z_score = (successes[1] / totals[1] - successes[0] / totals[0]) / standard_error
    z_p = 2 * stats.norm.sf(abs(z_score))

    weekend = frame.loc[frame["is_weekend"], "log_replies_72h"]
    weekday = frame.loc[~frame["is_weekend"], "log_replies_72h"]
    t_result = stats.ttest_ind(weekend, weekday, equal_var=False)
    pooled_sd = math.sqrt((weekend.var(ddof=1) + weekday.var(ddof=1)) / 2)
    cohen_d = (weekend.mean() - weekday.mean()) / pooled_sd

    topic_table = pd.crosstab(frame["topic_id"], frame["high_engagement"])
    chi2, chi_p, _, _ = stats.chi2_contingency(topic_table)
    cramers_v = math.sqrt(chi2 / (topic_table.to_numpy().sum() * (min(topic_table.shape) - 1)))

    groups = [group["log_replies_72h"].to_numpy() for _, group in frame.groupby("subreddit")]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    grand_mean = frame["log_replies_72h"].mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    total = ((frame["log_replies_72h"] - grand_mean) ** 2).sum()
    levene = stats.levene(*groups, center="median")

    return {
        "url_proportion_z_test": {
            "statistic": _float(z_score),
            "p_value": _float(z_p),
            "risk_difference": _float(successes[1] / totals[1] - successes[0] / totals[0]),
        },
        "weekend_welch_t_test": {
            "statistic": _float(t_result.statistic),
            "p_value": _float(t_result.pvalue),
            "cohen_d": _float(cohen_d),
        },
        "topic_chi_square": {
            "statistic": _float(chi2),
            "p_value": _float(chi_p),
            "cramers_v": _float(cramers_v),
        },
        "subreddit_anova": {
            "statistic": _float(anova.statistic),
            "p_value": _float(anova.pvalue),
            "eta_squared": _float(between / total),
            "levene_p_value": _float(levene.pvalue),
            "kruskal_statistic": _float(kruskal.statistic),
            "kruskal_p_value": _float(kruskal.pvalue),
        },
    }


def probability_summary(frame: pd.DataFrame, seed: int = 42) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    engagement = frame["reply_count_72h"].to_numpy()
    high_rate = frame["high_engagement"].mean()
    hourly = frame.set_index("created_at").resample("1h").size()
    sample_means = generator.choice(engagement, size=(2000, 30), replace=True).mean(axis=1)

    null_a = generator.normal(0, 1, size=(2000, 30))
    null_b = generator.normal(0, 1, size=(2000, 30))
    null_p = stats.ttest_ind(null_a, null_b, axis=1, equal_var=False).pvalue
    effect_b = generator.normal(0.5, 1, size=(2000, 30))
    effect_p = stats.ttest_ind(null_a, effect_b, axis=1, equal_var=False).pvalue

    return {
        "conditional_probabilities": {
            "p_high": _float(high_rate),
            "p_high_given_weekend": _float(frame.loc[frame.is_weekend, "high_engagement"].mean()),
            "p_high_given_url": _float(frame.loc[frame.has_url, "high_engagement"].mean()),
            "p_reply_given_positive_sentiment": _float(
                frame.loc[frame.sentiment > 0.05, "has_reply_72h"].mean()
            ),
        },
        "distribution_parameters": {
            "normal_log_engagement_mean": _float(frame["log_replies_72h"].mean()),
            "normal_log_engagement_sd": _float(frame["log_replies_72h"].std()),
            "binomial_n": 100,
            "binomial_p": _float(high_rate),
            "poisson_hourly_rate": _float(hourly.mean()),
            "uniform_range": [0.0, 1.0],
        },
        "sampling": {
            "population_mean_replies": _float(engagement.mean()),
            "sample_mean_mean": _float(sample_means.mean()),
            "sample_mean_sd": _float(sample_means.std(ddof=1)),
            "sample_mean_95_interval": [
                _float(value) for value in np.quantile(sample_means, [0.025, 0.975])
            ],
            "bootstrap_mean_95_ci": [
                _float(value) for value in bootstrap_mean_ci(frame["reply_count_72h"])
            ],
        },
        "decision_errors": {
            "simulated_type_i_rate_at_0_05": _float((null_p < 0.05).mean()),
            "simulated_power_for_d_0_5": _float((effect_p < 0.05).mean()),
            "simulated_type_ii_rate_for_d_0_5": _float((effect_p >= 0.05).mean()),
        },
    }
