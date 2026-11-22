#!/usr/bin/env python3
"""Predict and validate 72-hour engagement for a real Reddit post URL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from datasuper.reddit_validation import (
    ArcticShiftClient,
    collect_post_data,
    evaluate_prediction,
    extract_post_id,
    history_start,
    make_payload,
    post_was_in_local_dataset,
    timestamp,
)
from datasuper.service import load_artifact, predict


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
OUTPUT_DIR = ROOT / "reports" / "reddit_predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reddit_url", help="Reddit post URL or base36 post ID")
    args = parser.parse_args()

    artifact = load_artifact()
    post_id = extract_post_id(args.reddit_url)
    client = ArcticShiftClient()
    post = client.post(post_id)
    subreddit = str(post["subreddit"])
    if subreddit not in artifact["thresholds"]:
        supported = ", ".join(f"r/{name}" for name in artifact["thresholds"])
        parser.error(f"model does not support r/{subreddit}; supported: {supported}")

    created_at = timestamp(post["created_utc"])
    now = datetime.now(timezone.utc)
    age = now - created_at
    if age < timedelta(hours=1):
        ready_at = created_at + timedelta(hours=1)
        parser.error(f"one-hour prediction features are available at {ready_at.isoformat()}")

    dataset_start = history_start(FEATURES_PATH)
    if created_at < dataset_start:
        parser.error(
            f"post predates model history, which starts at {dataset_start.isoformat()}"
        )

    metrics, author_prior, community_prior = collect_post_data(
        client,
        post,
        list(artifact["thresholds"]),
        dataset_start,
    )
    output_path = OUTPUT_DIR / f"{post_id}.json"
    if output_path.exists():
        record = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        payload = make_payload(
            post,
            metrics,
            author_prior_posts=author_prior,
            community_posts_prior_24h=community_prior,
        )
        result = predict(payload, log=False)
        record = {
            "post_id": post_id,
            "reddit_url": post.get("url") or args.reddit_url,
            "subreddit": subreddit,
            "post_created_at_utc": created_at.isoformat(),
            "prediction_recorded_at_utc": now.isoformat(),
            "test_type": "prospective" if age < timedelta(hours=72) else "retrospective",
            "was_in_local_dataset": post_was_in_local_dataset(FEATURES_PATH, post_id),
            "features_at_1h": payload.model_dump(exclude={"title"}),
            "prediction": result,
        }

    if age >= timedelta(hours=72):
        record.pop("evaluation_available_at_utc", None)
        record["evaluation"] = evaluate_prediction(
            record["prediction"],
            metrics["reply_count_72h"],
            int(artifact["thresholds"][subreddit]),
        )
        record["evaluated_at_utc"] = now.isoformat()
    else:
        record["evaluation_available_at_utc"] = (
            created_at + timedelta(hours=72)
        ).isoformat()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
