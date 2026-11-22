from datetime import datetime, timezone

import pytest

from datasuper.reddit_validation import (
    comment_metrics,
    evaluate_prediction,
    extract_post_id,
    make_payload,
)


def test_extract_post_id_from_url_or_bare_id() -> None:
    post_id = "1bsovs1"
    assert extract_post_id(post_id) == post_id
    assert (
        extract_post_id(
            "https://www.reddit.com/r/datascience/comments/1bsovs1/example/"
        )
        == post_id
    )
    with pytest.raises(ValueError, match="Reddit post URL"):
        extract_post_id("https://www.reddit.com/r/datascience/")


def test_comment_metrics_observe_fixed_horizons() -> None:
    created = datetime(2024, 1, 1, tzinfo=timezone.utc)
    comments = [
        {"created_utc": created.timestamp() + 60, "author": "one"},
        {"created_utc": created.timestamp() + 3600, "author": "one"},
        {"created_utc": created.timestamp() + 7200, "author": "two"},
        {"created_utc": created.timestamp() + 72 * 3600 + 1, "author": "three"},
    ]
    assert comment_metrics(comments, created) == {
        "reply_count_1h": 2,
        "unique_repliers_1h": 1,
        "reply_count_72h": 3,
        "unique_repliers_72h": 2,
    }


def test_real_post_payload_and_evaluation() -> None:
    post = {
        "created_utc": datetime(2024, 1, 6, 12, tzinfo=timezone.utc).timestamp(),
        "subreddit": "datascience",
        "title": "A useful model result",
        "url": "https://example.com",
    }
    metrics = {
        "reply_count_1h": 2,
        "unique_repliers_1h": 2,
        "reply_count_72h": 12,
    }
    payload = make_payload(
        post,
        metrics,
        author_prior_posts=3,
        community_posts_prior_24h=50,
    )
    assert payload.title_words == 4
    assert payload.sentiment > 0
    assert payload.is_weekend
    assert payload.reply_count_1h == 2

    prediction = {"label": "high_engagement"}
    assert evaluate_prediction(prediction, 12, 11) == {
        "actual_reply_count_72h": 12,
        "engagement_reply_threshold": 11,
        "actual_label": "high_engagement",
        "prediction_correct": True,
    }
