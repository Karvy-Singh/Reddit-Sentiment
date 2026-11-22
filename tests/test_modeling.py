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
        }
    )
    thresholds = engagement_thresholds(train)
    assert thresholds == {"a": 9, "b": 90}
    assert engagement_labels(train, thresholds).sum() == 2


def test_unknown_community_fails_closed() -> None:
    frame = pd.DataFrame({"subreddit": ["unknown"], "reply_count_72h": [1]})
    with pytest.raises(ValueError, match="missing engagement thresholds"):
        engagement_labels(frame, {"known": 1})


def test_model_features_exclude_retrieval_time_body_state() -> None:
    assert "body" not in MODEL_FEATURES
    assert "body_words" not in MODEL_FEATURES
    assert "title" in MODEL_FEATURES
    assert {"reply_count_1h", "unique_repliers_1h"}.issubset(EARLY_MODEL_FEATURES)


def test_f1_threshold_is_learned_from_probabilities() -> None:
    target = pd.Series([0, 0, 1, 1])
    threshold = f1_optimal_threshold(target, [0.1, 0.4, 0.6, 0.9])
    assert 0.4 < threshold <= 0.6
