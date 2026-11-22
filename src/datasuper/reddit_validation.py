"""Real-post prediction and outcome validation using the Arctic Shift archive."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from datasuper.service import PredictionRequest


BASE_URL = "https://arctic-shift.photon-reddit.com/api"
POST_ID_PATTERN = re.compile(r"(?:/comments/|^)([a-z0-9]+)(?:/|$)", re.IGNORECASE)


def extract_post_id(value: str) -> str:
    """Extract a Reddit base36 post ID from a post URL or accept a bare ID."""
    match = POST_ID_PATTERN.search(value.strip())
    if not match:
        raise ValueError("expected a Reddit post URL or post ID")
    return match.group(1).lower()


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = "DataSuper-real-post-validation/1.0"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class ArcticShiftClient:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or build_session()

    def post(self, post_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{BASE_URL}/posts/ids", params={"ids": post_id}, timeout=(15, 180)
        )
        response.raise_for_status()
        records = response.json().get("data", [])
        if not records:
            raise ValueError(f"Reddit post not found in Arctic Shift: {post_id}")
        return records[0]

    def search(
        self,
        kind: str,
        *,
        after: datetime,
        before: datetime,
        **filters: str,
    ) -> list[dict[str, Any]]:
        """Fetch a complete ascending timestamp range, de-duplicated by ID."""
        after_ms = int(after.timestamp() * 1000)
        before_ms = int(before.timestamp() * 1000)
        cursor = after_ms - 1
        records_by_id: dict[str, dict[str, Any]] = {}

        while cursor < before_ms:
            params = {
                **filters,
                "after": cursor,
                "before": before_ms,
                "limit": "auto",
                "sort": "asc",
                "meta-app": "DataSuper-real-post-validation",
            }
            response = self.session.get(
                f"{BASE_URL}/{kind}/search", params=params, timeout=(15, 180)
            )
            response.raise_for_status()
            page = response.json().get("data", [])
            if not page:
                break

            for record in page:
                created_ms = int(float(record["created_utc"]) * 1000)
                if after_ms <= created_ms < before_ms and record.get("id"):
                    records_by_id[str(record["id"])] = record

            next_cursor = int(float(page[-1]["created_utc"]) * 1000)
            cursor = next_cursor if next_cursor > cursor else cursor + 1000

        return sorted(records_by_id.values(), key=lambda item: item["created_utc"])


def comment_metrics(
    comments: list[dict[str, Any]], created_at: datetime
) -> dict[str, int]:
    one_hour = created_at + timedelta(hours=1)
    seventy_two_hours = created_at + timedelta(hours=72)
    early = [
        item
        for item in comments
        if created_at <= timestamp(item["created_utc"]) <= one_hour
    ]
    outcome = [
        item
        for item in comments
        if created_at <= timestamp(item["created_utc"]) <= seventy_two_hours
    ]
    return {
        "reply_count_1h": len(early),
        "unique_repliers_1h": len(
            {
                item.get("author")
                for item in early
                if item.get("author") not in {None, "[deleted]", "[removed]"}
            }
        ),
        "reply_count_72h": len(outcome),
        "unique_repliers_72h": len(
            {
                item.get("author")
                for item in outcome
                if item.get("author") not in {None, "[deleted]", "[removed]"}
            }
        ),
    }


def timestamp(value: int | float) -> datetime:
    return datetime.fromtimestamp(float(value), tz=timezone.utc)


def make_payload(
    post: dict[str, Any],
    metrics: dict[str, int],
    *,
    author_prior_posts: int,
    community_posts_prior_24h: int,
) -> PredictionRequest:
    created_at = timestamp(post["created_utc"])
    title = str(post.get("title") or "").strip()
    return PredictionRequest(
        subreddit=str(post["subreddit"]),
        title=title,
        title_words=len(title.split()),
        sentiment=SentimentIntensityAnalyzer().polarity_scores(title)["compound"],
        author_prior_posts=author_prior_posts,
        community_posts_prior_24h=community_posts_prior_24h,
        reply_count_1h=metrics["reply_count_1h"],
        unique_repliers_1h=metrics["unique_repliers_1h"],
        hour=created_at.hour,
        day_of_week=created_at.weekday(),
        has_url=bool(post.get("url")),
        is_weekend=created_at.weekday() >= 5,
    )


def history_start(features_path: Path) -> datetime:
    frame = pd.read_parquet(features_path, columns=["created_at"])
    return frame["created_at"].min().to_pydatetime()


def post_was_in_local_dataset(features_path: Path, post_id: str) -> bool:
    frame = pd.read_parquet(features_path, columns=["post_id"])
    return bool(frame["post_id"].astype(str).eq(post_id).any())


def collect_post_data(
    client: ArcticShiftClient,
    post: dict[str, Any],
    supported_subreddits: list[str],
    dataset_start: datetime,
) -> tuple[dict[str, int], int, int]:
    created_at = timestamp(post["created_utc"])
    comments = client.search(
        "comments",
        after=created_at,
        before=created_at + timedelta(hours=72, milliseconds=1),
        link_id=str(post["id"]),
    )
    metrics = comment_metrics(comments, created_at)
    community_posts = client.search(
        "posts",
        after=created_at - timedelta(hours=24),
        before=created_at,
        subreddit=str(post["subreddit"]),
    )

    author_prior_posts = 0
    author = post.get("author")
    if author not in {None, "[deleted]", "[removed]"}:
        author_posts = client.search(
            "posts",
            after=dataset_start,
            before=created_at,
            author=str(author),
        )
        author_prior_posts = sum(
            str(item.get("subreddit")) in supported_subreddits
            for item in author_posts
        )
    return metrics, author_prior_posts, len(community_posts)


def evaluate_prediction(
    prediction: dict[str, Any], actual_replies: int, engagement_threshold: int
) -> dict[str, Any]:
    actual_label = (
        "high_engagement"
        if actual_replies >= engagement_threshold
        else "ordinary_engagement"
    )
    return {
        "actual_reply_count_72h": actual_replies,
        "engagement_reply_threshold": engagement_threshold,
        "actual_label": actual_label,
        "prediction_correct": prediction["label"] == actual_label,
    }
