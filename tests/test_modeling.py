import pandas as pd
import pytest

from datasuper.modeling import (
    EARLY_MODEL_FEATURES,
    MODEL_FEATURES,
    engagement_labels,
    engagement_thresholds,
    f1_optimal_threshold,
)


def test_thresholds_are_learned_per_community() -> None:
    train = pd.DataFrame(
        {
            "subreddit": ["a"] * 10 + ["b"] * 10,
            "reply_count_72h": list(range(10)) + list(range(0, 100, 10)),
            "score_observed": list(range(10)) + list(range(0, 100, 10)),
            "upvote_ratio_observed": [1.0] * 20,
            "unique_repliers_72h": list(range(10)) + list(range(0, 100, 10)),
            "max_thread_depth_72h": list(range(10)) + list(range(0, 100, 10)),
            "num_crossposts_observed": [0] * 20,
        }
    )
    thresholds = engagement_thresholds(train)
    assert thresholds == {
        "a": {
            "reply_count_72h": 9,
            "score_observed": 9,
            "vote_quality": 9.0,
            "unique_repliers_72h": 9,
            "max_thread_depth_72h": 9,
            "num_crossposts_observed": 1,
        },
        "b": {
            "reply_count_72h": 90,
            "score_observed": 90,
            "vote_quality": 90.0,
            "unique_repliers_72h": 90,
            "max_thread_depth_72h": 90,
            "num_crossposts_observed": 1,
        },
    }
    assert engagement_labels(train, thresholds).sum() == 2

    score_only = pd.DataFrame(
        {
            "subreddit": ["a"],
            "reply_count_72h": [0],
            "score_observed": [136],
            "upvote_ratio_observed": [0.95],
            "unique_repliers_72h": [0],
            "max_thread_depth_72h": [0],
            "num_crossposts_observed": [0],
        }
    )
    assert engagement_labels(score_only, thresholds).item() == 1


def test_unknown_community_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "subreddit": ["unknown"],
            "reply_count_72h": [1],
            "score_observed": [1],
            "upvote_ratio_observed": [1.0],
            "unique_repliers_72h": [1],
            "max_thread_depth_72h": [1],
            "num_crossposts_observed": [0],
        }
    )
    with pytest.raises(ValueError, match="missing engagement thresholds"):
        engagement_labels(
            frame,
            {
                "known": {
                    "reply_count_72h": 1,
                    "score_observed": 1,
                    "vote_quality": 1.0,
                    "unique_repliers_72h": 1,
                    "max_thread_depth_72h": 1,
                    "num_crossposts_observed": 1,
                }
            },
        )


def test_model_features_exclude_retrieval_time_body_state() -> None:
    assert "body" not in MODEL_FEATURES
    assert "body_words" not in MODEL_FEATURES
    assert "title" in MODEL_FEATURES
    assert {"reply_count_1h", "unique_repliers_1h"}.issubset(EARLY_MODEL_FEATURES)
    assert "max_thread_depth_1h" in EARLY_MODEL_FEATURES


def test_f1_threshold_is_learned_from_probabilities() -> None:
    target = pd.Series([0, 0, 1, 1])
    threshold = f1_optimal_threshold(target, [0.1, 0.4, 0.6, 0.9])
    assert 0.4 < threshold <= 0.6
