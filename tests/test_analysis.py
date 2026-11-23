import pandas as pd
import pytest

from datasuper.analysis import add_high_engagement, bootstrap_mean_ci


def test_high_engagement_uses_group_thresholds() -> None:
    frame = pd.DataFrame(
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
    result = add_high_engagement(frame)
    assert result.groupby("subreddit")["high_engagement"].sum().eq(1).all()
    assert result.groupby("subreddit")["high_engagement_threshold"].first().to_dict() == {"a": 9, "b": 90}
    assert result.groupby("subreddit")["high_engagement_score_threshold"].first().to_dict() == {
        "a": 9,
        "b": 90,
    }


def test_bootstrap_interval_is_ordered() -> None:
    low, high = bootstrap_mean_ci(pd.Series([1, 2, 3, 4]), samples=200)
    assert low < high


def test_invalid_quantile_fails() -> None:
    with pytest.raises(ValueError, match="quantile"):
        add_high_engagement(pd.DataFrame(), quantile=1)
