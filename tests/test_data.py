import pandas as pd
import pytest

from datasuper.data import build_features, chronological_split, validate_posts


def sample_posts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "post_id": [f"p{i}" for i in range(10)],
            "author_id": ["a", "a"] + [f"a{i}" for i in range(8)],
            "subreddit": ["one"] * 5 + ["two"] * 5,
            "created_at": pd.date_range("2024-01-01", periods=10, freq="12h", tz="UTC"),
            "title": ["good result"] * 10,
            "body": [""] * 10,
            "url": [None] * 10,
            "reply_count_72h": list(range(10)),
        }
    )


def test_feature_history_uses_only_prior_rows() -> None:
    features = build_features(sample_posts())
    assert features.loc[0, "author_prior_posts"] == 0
    assert features.loc[1, "author_prior_posts"] == 1
    assert features.loc[0, "community_posts_prior_24h"] == 0
    assert features.loc[1, "community_posts_prior_24h"] == 1
    assert features["sentiment"].gt(0).all()


def test_chronological_split_is_ordered_and_complete() -> None:
    train, validation, test = chronological_split(sample_posts())
    assert len(train) + len(validation) + len(test) == 10
    assert train.created_at.max() <= validation.created_at.min()
    assert validation.created_at.max() <= test.created_at.min()


def test_validation_rejects_duplicate_ids() -> None:
    posts = sample_posts()
    posts.loc[1, "post_id"] = posts.loc[0, "post_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_posts(posts)
