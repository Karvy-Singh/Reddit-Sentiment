#!/usr/bin/env python3
"""Generate the pilot feature table, statistics, and concise EDA outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasuper.analysis import add_high_engagement, add_topics, probability_summary, statistical_tests
from datasuper.data import build_features, load_analysis_posts


ROOT = Path(__file__).resolve().parents[1]


def save_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(frame["reply_count_72h"], bins=40, ax=axes[0])
    axes[0].set(xlim=(0, frame["reply_count_72h"].quantile(0.99)), title="Replies within 72 hours")
    sns.histplot(frame["log_replies_72h"], bins=30, ax=axes[1])
    axes[1].set(title="Log-transformed engagement")
    figure.tight_layout()
    figure.savefig(output_dir / "engagement_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=frame, x="subreddit", y="log_replies_72h", ax=axis)
    axis.set(title="Engagement by community", xlabel="Community", ylabel="log(1 + replies in 72h)")
    figure.tight_layout()
    figure.savefig(output_dir / "community_engagement.png", dpi=160)
    plt.close(figure)

    correlation_fields = [
        "title_words",
        "body_words",
        "sentiment",
        "author_prior_posts",
        "community_posts_prior_24h",
        "log_replies_72h",
    ]
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(frame[correlation_fields].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set(title="Feature correlations")
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(figure)

    daily = frame.set_index("created_at").groupby("subreddit").resample("1D").size().rename("posts").reset_index()
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=daily, x="created_at", y="posts", hue="subreddit", marker="o", ax=axis)
    axis.set(title="Daily post activity", xlabel="Date", ylabel="Posts")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output_dir / "daily_activity.png", dpi=160)
    plt.close(figure)

    generator = np.random.default_rng(42)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    distributions = [
        (generator.normal(0, 1, 2000), "Normal"),
        (generator.binomial(100, frame.high_engagement.mean(), 2000), "Binomial"),
        (generator.poisson(frame.set_index("created_at").resample("1h").size().mean(), 2000), "Poisson"),
        (generator.uniform(0, 1, 2000), "Uniform"),
    ]
    for axis, (values, title) in zip(axes.flat, distributions):
        sns.histplot(values, bins=30, ax=axis)
        axis.set_title(title)
    figure.suptitle("Probability distribution simulations")
    figure.tight_layout()
    figure.savefig(output_dir / "probability_distributions.png", dpi=160)
    plt.close(figure)


def main() -> None:
    processed = ROOT / "data" / "processed"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    frame = build_features(load_analysis_posts(processed / "analysis_posts.parquet"))
    frame, topic_keywords = add_topics(frame)
    frame = add_high_engagement(frame)
    frame.to_parquet(processed / "features.parquet", index=False, compression="zstd")

    descriptive = frame[
        ["reply_count_72h", "unique_repliers_72h", "first_response_hours", "title_words", "body_words"]
    ].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    descriptive.to_csv(reports / "descriptive_statistics.csv")

    daily = (
        frame.set_index("created_at")
        .groupby("subreddit")
        .resample("1D")
        .agg(posts=("post_id", "size"), mean_replies=("reply_count_72h", "mean"), mean_sentiment=("sentiment", "mean"))
        .reset_index()
    )
    daily.to_csv(reports / "daily_metrics.csv", index=False)

    weekly = (
        frame.set_index("created_at")
        .groupby("subreddit")
        .resample("1W")
        .agg(posts=("post_id", "size"), mean_replies=("reply_count_72h", "mean"), mean_sentiment=("sentiment", "mean"))
        .reset_index()
    )
    weekly["posts_rolling_4w"] = weekly.groupby("subreddit")["posts"].transform(
        lambda values: values.rolling(4, min_periods=1).mean()
    )
    weekly.to_csv(reports / "weekly_metrics.csv", index=False)

    weekly_total = frame.set_index("created_at").resample("1W").size().astype(float)
    holdout = min(3, len(weekly_total) - 3)
    test_actual = weekly_total.iloc[-holdout:]
    naive = weekly_total.shift(1).iloc[-holdout:]
    moving_average = weekly_total.shift(1).rolling(3).mean().iloc[-holdout:]
    time_series = {
        "weekly_periods": len(weekly_total),
        "holdout_periods": holdout,
        "naive_previous_week_mae": float((test_actual - naive).abs().mean()),
        "three_week_moving_average_mae": float(
            (test_actual - moving_average).abs().mean()
        ),
    }

    summary = {
        "rows": len(frame),
        "communities": sorted(frame["subreddit"].unique().tolist()),
        "topic_keywords": {str(index): words for index, words in enumerate(topic_keywords)},
        "statistical_tests": statistical_tests(frame),
        "probability": probability_summary(frame),
        "time_series": time_series,
    }
    (reports / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(frame, reports / "figures")
    print(f"Wrote {len(frame):,} feature rows and analysis outputs to {reports}")


if __name__ == "__main__":
    main()
