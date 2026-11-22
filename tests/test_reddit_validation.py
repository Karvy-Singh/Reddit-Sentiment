from datetime import datetime, timezone

import pandas as pd
import pytest

from datasuper.reddit_validation import (
    ArcticShiftClient,
    comment_metrics,
    evaluate_prediction,
    extract_post_id,
    make_payload,
    validate_reddit_post,
)


def test_archive_422_after_valid_page_ends_pagination() -> None:
    class FakeResponse:
        def __init__(self, status_code: int, data: list[dict] | None = None) -> None:
            self.status_code = status_code
            self.data = data or []

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError("unexpected HTTP error")

        def json(self) -> dict:
            return {"data": self.data}

    class FakeSession:
        def __init__(self) -> None:
            self.responses = [
                FakeResponse(200, [{"id": "p1", "created_utc": 1_704_067_200}]),
                FakeResponse(422),
            ]

        def get(self, *args, **kwargs) -> FakeResponse:
            return self.responses.pop(0)

    records = ArcticShiftClient(FakeSession()).search(
        "posts",
        after=datetime(2024, 1, 1, tzinfo=timezone.utc),
        before=datetime(2024, 1, 2, tzinfo=timezone.utc),
        subreddit="datascience",
    )
    assert [record["id"] for record in records] == ["p1"]


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


def test_saved_prediction_is_later_evaluated(tmp_path) -> None:
    created = datetime(2024, 4, 1, 12, tzinfo=timezone.utc)

    class FakeClient:
        def post(self, post_id: str) -> dict:
            return {
                "id": post_id,
                "created_utc": created.timestamp(),
                "subreddit": "datascience",
                "title": "A practical machine learning question",
                "url": f"https://www.reddit.com/comments/{post_id}/",
                "author": "[deleted]",
            }

        def search(self, kind: str, **kwargs) -> list[dict]:
            if kind == "comments":
                return [
                    {"id": "c1", "created_utc": created.timestamp() + 1800, "author": "a"},
                    {"id": "c2", "created_utc": created.timestamp() + 7200, "author": "b"},
                ]
            return []

    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {"post_id": ["training-post"], "created_at": [datetime(2024, 1, 1, tzinfo=timezone.utc)]}
    ).to_parquet(features_path, index=False)
    output_dir = tmp_path / "predictions"

    prospective = validate_reddit_post(
        "newpost",
        features_path=features_path,
        output_dir=output_dir,
        client=FakeClient(),
        now=created.replace(day=2, hour=10),
    )
    assert prospective["test_type"] == "prospective"
    assert prospective["features_at_1h"]["reply_count_1h"] == 1
    assert prospective["latest_observation"]["replies_observed_so_far"] == 2
    assert "evaluation" not in prospective

    evaluated = validate_reddit_post(
        "newpost",
        features_path=features_path,
        output_dir=output_dir,
        client=FakeClient(),
        now=created.replace(day=4, hour=13),
    )
    assert evaluated["prediction"] == prospective["prediction"]
    assert evaluated["evaluation"]["actual_reply_count_72h"] == 2
