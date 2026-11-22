"""Leakage-safe model construction and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


NUMERIC_FEATURES = [
    "title_words",
    "sentiment",
    "author_prior_posts",
    "community_posts_prior_24h",
    "hour",
    "day_of_week",
    "has_url",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["subreddit"]
TEXT_FEATURE = "title"
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]
EARLY_NUMERIC_FEATURES = NUMERIC_FEATURES + ["reply_count_1h", "unique_repliers_1h"]
EARLY_MODEL_FEATURES = EARLY_NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]


def engagement_thresholds(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    required = {"subreddit", "reply_count_72h", "score_observed"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"missing engagement columns: {sorted(missing)}")
    grouped = train.groupby("subreddit")
    reply_thresholds = grouped["reply_count_72h"].quantile(
        0.90, interpolation="higher"
    )
    score_thresholds = grouped["score_observed"].quantile(
        0.95, interpolation="higher"
    )
    if reply_thresholds.empty:
        raise ValueError("cannot calculate engagement thresholds from empty data")
    return {
        str(key): {
            "reply_count_72h": int(reply_thresholds[key]),
            "score_observed": int(score_thresholds[key]),
        }
        for key in reply_thresholds.index
    }


def engagement_labels(
    frame: pd.DataFrame, thresholds: dict[str, dict[str, int]]
) -> pd.Series:
    unknown_mask = ~frame["subreddit"].isin(thresholds)
    if unknown_mask.any():
        unknown = sorted(frame.loc[unknown_mask, "subreddit"].unique())
        raise ValueError(f"missing engagement thresholds for: {unknown}")
    reply_thresholds = frame["subreddit"].map(
        {key: value["reply_count_72h"] for key, value in thresholds.items()}
    )
    score_thresholds = frame["subreddit"].map(
        {key: value["score_observed"] for key, value in thresholds.items()}
    )
    return (
        frame["reply_count_72h"].ge(reply_thresholds)
        | frame["score_observed"].ge(score_thresholds)
    ).astype("int8")


def preprocessor(numeric_features: list[str] | None = None) -> ColumnTransformer:
    numeric_features = numeric_features or NUMERIC_FEATURES
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES),
            (
                "title",
                TfidfVectorizer(
                    stop_words="english", min_df=3, max_features=1000, ngram_range=(1, 2)
                ),
                TEXT_FEATURE,
            ),
        ],
        verbose_feature_names_out=True,
    )


def classification_models(
    random_state: int = 42, early: bool = False
) -> dict[str, Pipeline]:
    numeric_features = EARLY_NUMERIC_FEATURES if early else NUMERIC_FEATURES

    def pipeline(model: Any) -> Pipeline:
        return Pipeline(
            [("preprocess", preprocessor(numeric_features)), ("model", model)]
        )

    return {
        "dummy": pipeline(DummyClassifier(strategy="prior")),
        "logistic": pipeline(LogisticRegression(max_iter=2000)),
        "logistic_weighted": pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced")
        ),
        "knn": pipeline(KNeighborsClassifier(n_neighbors=15)),
        "decision_tree": pipeline(
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=random_state)
        ),
        "random_forest": pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "bagging": pipeline(
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=5),
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": pipeline(
            GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        ),
        "smote_logistic": ImbalancedPipeline(
            [
                ("preprocess", preprocessor(numeric_features)),
                ("smote", SMOTE(random_state=random_state)),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
    }


def regression_models(random_state: int = 42) -> dict[str, Pipeline]:
    def pipeline(model: Any) -> Pipeline:
        return Pipeline([("preprocess", preprocessor()), ("model", model)])

    return {
        "dummy": pipeline(DummyRegressor(strategy="mean")),
        "linear": pipeline(LinearRegression()),
        "knn": pipeline(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "decision_tree": pipeline(
            DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=random_state)
        ),
        "random_forest": pipeline(
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }


def f1_optimal_threshold(target: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if thresholds.size == 0:
        raise ValueError("threshold optimization requires both target classes")
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    probability = model.predict_proba(features)[:, 1]
    prediction = probability >= threshold
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
    }


def regression_metrics(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    prediction = model.predict(features)
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }
