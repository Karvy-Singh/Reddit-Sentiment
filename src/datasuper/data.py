"""Validated loading and leakage-safe feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


REQUIRED_COLUMNS = {
    "post_id",
    "author_id",
    "subreddit",
    "created_at",
    "title",
    "body",
    "url",
    "reply_count_72h",
}


def validate_posts(posts: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(posts.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if posts.empty:
        raise ValueError("analysis dataset is empty")
    if posts["post_id"].isna().any() or posts["post_id"].duplicated().any():
        raise ValueError("post_id must be present and unique")
    if posts["created_at"].isna().any():
        raise ValueError("created_at contains missing values")
    if not isinstance(posts["created_at"].dtype, pd.DatetimeTZDtype):
        raise ValueError("created_at must be timezone-aware")
    if (posts["reply_count_72h"] < 0).any():
        raise ValueError("reply_count_72h cannot be negative")


def load_analysis_posts(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"analysis dataset not found: {path}")
    posts = pd.read_parquet(path)
    validate_posts(posts)
    return posts


def _previous_24h_count(times: pd.Series) -> np.ndarray:
    timestamps = times.astype("int64").to_numpy()
    day = pd.Timedelta(hours=24).value // 1_000
    starts = np.searchsorted(timestamps, timestamps - day, side="left")
    return np.arange(len(timestamps)) - starts


def build_features(posts: pd.DataFrame) -> pd.DataFrame:
    validate_posts(posts)
    result = posts.sort_values("created_at").reset_index(drop=True).copy()
    text = (result["title"].fillna("") + " " + result["body"].fillna("")).str.strip()
    analyzer = SentimentIntensityAnalyzer()

    result["title_words"] = result["title"].fillna("").str.split().str.len()
    result["body_words"] = result["body"].fillna("").str.split().str.len()
    result["text_chars"] = text.str.len()
    result["has_url"] = result["url"].notna() & result["url"].ne("")
    result["hour"] = result["created_at"].dt.hour
    result["day_of_week"] = result["created_at"].dt.day_of_week
    result["is_weekend"] = result["day_of_week"].ge(5)
    result["sentiment"] = [
        analyzer.polarity_scores(value)["compound"]
        for value in result["title"].fillna("")
    ]
    result["author_prior_posts"] = result.groupby("author_id", dropna=False).cumcount()
    result["community_posts_prior_24h"] = (
        result.groupby("subreddit", group_keys=False)["created_at"]
        .transform(_previous_24h_count)
        .astype("int64")
    )
    result["log_replies_72h"] = np.log1p(result["reply_count_72h"])
    return result


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")

    ordered = frame.sort_values("created_at").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("dataset is too small for the requested split")
    if train["created_at"].max() > validation["created_at"].min():
        raise AssertionError("training and validation periods overlap")
    if validation["created_at"].max() > test["created_at"].min():
        raise AssertionError("validation and test periods overlap")
    return train, validation, test
