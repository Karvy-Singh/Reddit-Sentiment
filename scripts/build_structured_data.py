#!/usr/bin/env python3
"""Convert raw Reddit JSONL files into pseudonymized Parquet tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


POST_COLUMNS = [
    "post_id",
    "author_id",
    "subreddit",
    "created_at",
    "title",
    "body",
    "url",
    "domain",
    "score_observed",
    "num_comments_observed",
    "upvote_ratio_observed",
    "is_self",
    "is_over_18",
    "is_locked",
    "is_stickied",
    "retrieved_at",
    "source_file",
]

COMMENT_COLUMNS = [
    "comment_id",
    "post_id",
    "parent_id",
    "parent_type",
    "author_id",
    "subreddit",
    "created_at",
    "body",
    "score_observed",
    "controversiality",
    "is_submitter",
    "is_stickied",
    "retrieved_at",
    "source_file",
]


def iter_jsonl(paths: Iterable[Path]) -> Iterable[tuple[dict[str, Any], Path]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    yield json.loads(line), path
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def get_local_salt(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return bytes.fromhex(path.read_text(encoding="ascii").strip())
    salt = secrets.token_bytes(32)
    path.write_text(salt.hex() + "\n", encoding="ascii")
    path.chmod(0o600)
    return salt


def pseudonymize(author: Any, salt: bytes) -> str | None:
    if not author or author in {"[deleted]", "[removed]"}:
        return None
    digest = hashlib.sha256(salt + str(author).encode("utf-8")).hexdigest()
    return f"u_{digest[:20]}"


def strip_fullname(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return text.split("_", maxsplit=1)[-1]


def created_datetime(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="s", utc=True)


def retrieved_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, unit="s", utc=True)


def build_posts(paths: list[Path], salt: bytes) -> pd.DataFrame:
    rows = []
    for record, path in iter_jsonl(paths):
        rows.append(
            {
                "post_id": record.get("id"),
                "author_id": pseudonymize(record.get("author"), salt),
                "subreddit": record.get("subreddit"),
                "created_at": created_datetime(record.get("created_utc")),
                "title": record.get("title") or "",
                "body": record.get("selftext") or "",
                "url": record.get("url"),
                "domain": record.get("domain"),
                "score_observed": record.get("score"),
                "num_comments_observed": record.get("num_comments"),
                "upvote_ratio_observed": record.get("upvote_ratio"),
                "is_self": record.get("is_self"),
                "is_over_18": record.get("over_18"),
                "is_locked": record.get("locked"),
                "is_stickied": record.get("stickied"),
                "retrieved_at": retrieved_datetime(record.get("retrieved_on")),
                "source_file": path.name,
            }
        )
    return pd.DataFrame(rows, columns=POST_COLUMNS)


def build_comments(paths: list[Path], salt: bytes) -> pd.DataFrame:
    rows = []
    for record, path in iter_jsonl(paths):
        parent_fullname = record.get("parent_id")
        rows.append(
            {
                "comment_id": record.get("id"),
                "post_id": strip_fullname(record.get("link_id")),
                "parent_id": strip_fullname(parent_fullname),
                "parent_type": (
                    str(parent_fullname).split("_", maxsplit=1)[0]
                    if parent_fullname
                    else None
                ),
                "author_id": pseudonymize(record.get("author"), salt),
                "subreddit": record.get("subreddit"),
                "created_at": created_datetime(record.get("created_utc")),
                "body": record.get("body") or "",
                "score_observed": record.get("score"),
                "controversiality": record.get("controversiality"),
                "is_submitter": record.get("is_submitter"),
                "is_stickied": record.get("stickied"),
                "retrieved_at": retrieved_datetime(record.get("retrieved_on")),
                "source_file": path.name,
            }
        )
    return pd.DataFrame(rows, columns=COMMENT_COLUMNS)


def table_report(frame: pd.DataFrame, id_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "unique_ids": 0, "duplicate_ids_removed": 0}
    duplicate_count = int(frame.duplicated(id_column).sum())
    return {
        "rows": int(len(frame)),
        "unique_ids": int(frame[id_column].nunique(dropna=True)),
        "duplicate_ids_removed": duplicate_count,
        "missing_id": int(frame[id_column].isna().sum()),
        "missing_author": int(frame["author_id"].isna().sum()),
        "empty_text": int(
            ((frame["title"].eq("") & frame["body"].eq(""))).sum()
            if "title" in frame
            else frame["body"].eq("").sum()
        ),
        "created_at_min": frame["created_at"].min().isoformat(),
        "created_at_max": frame["created_at"].max().isoformat(),
        "subreddits": frame["subreddit"].value_counts().sort_index().to_dict(),
    }


def build_analysis_posts(
    posts: pd.DataFrame, comments: pd.DataFrame, horizon_hours: int
) -> pd.DataFrame:
    comment_subreddits = set(comments["subreddit"].dropna().unique())
    analysis_posts = posts[posts["subreddit"].isin(comment_subreddits)].copy()
    root_times = analysis_posts.set_index("post_id")["created_at"]
    eligible_comments = comments[comments["post_id"].isin(root_times.index)].copy()
    eligible_comments["hours_after_post"] = (
        eligible_comments["created_at"]
        - eligible_comments["post_id"].map(root_times)
    ).dt.total_seconds() / 3600
    eligible_comments = eligible_comments[
        eligible_comments["hours_after_post"].between(
            0, horizon_hours, inclusive="both"
        )
    ]

    metrics = eligible_comments.groupby("post_id").agg(
        reply_count_72h=("comment_id", "size"),
        unique_repliers_72h=("author_id", "nunique"),
        first_response_hours=("hours_after_post", "min"),
    )
    early_metrics = (
        eligible_comments[eligible_comments["hours_after_post"].le(1)]
        .groupby("post_id")
        .agg(
            reply_count_1h=("comment_id", "size"),
            unique_repliers_1h=("author_id", "nunique"),
        )
    )
    analysis_posts = analysis_posts.merge(
        metrics, how="left", left_on="post_id", right_index=True
    )
    analysis_posts = analysis_posts.merge(
        early_metrics, how="left", left_on="post_id", right_index=True
    )
    analysis_posts["reply_count_72h"] = (
        analysis_posts["reply_count_72h"].fillna(0).astype("int64")
    )
    analysis_posts["unique_repliers_72h"] = (
        analysis_posts["unique_repliers_72h"].fillna(0).astype("int64")
    )
    analysis_posts["reply_count_1h"] = (
        analysis_posts["reply_count_1h"].fillna(0).astype("int64")
    )
    analysis_posts["unique_repliers_1h"] = (
        analysis_posts["unique_repliers_1h"].fillna(0).astype("int64")
    )
    analysis_posts["has_reply_72h"] = analysis_posts["reply_count_72h"] > 0
    analysis_posts["has_reply_1h"] = analysis_posts["reply_count_1h"] > 0
    if (analysis_posts["reply_count_1h"] > analysis_posts["reply_count_72h"]).any():
        raise AssertionError("one-hour replies cannot exceed 72-hour replies")
    return analysis_posts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--metadata-dir", type=Path, default=Path("data/metadata")
    )
    args = parser.parse_args()

    post_paths = sorted(args.raw_dir.glob("*_posts_*.jsonl"))
    comment_paths = sorted(args.raw_dir.glob("*_comments_*.jsonl"))
    if not post_paths and not comment_paths:
        parser.error(f"no Reddit JSONL files found in {args.raw_dir}")

    salt = get_local_salt(args.metadata_dir / ".author_salt")
    posts = build_posts(post_paths, salt)
    comments = build_comments(comment_paths, salt)

    post_report = table_report(posts, "post_id")
    comment_report = table_report(comments, "comment_id")
    posts = posts.drop_duplicates("post_id", keep="last")
    comments = comments.drop_duplicates("comment_id", keep="last")

    known_post_ids = set(posts["post_id"].dropna())
    matched_comments = int(comments["post_id"].isin(known_post_ids).sum())
    comments_in_window = comments[comments["post_id"].isin(known_post_ids)].copy()
    analysis_posts = build_analysis_posts(posts, comments_in_window, horizon_hours=72)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    posts_path = args.output_dir / "posts.parquet"
    comments_path = args.output_dir / "comments.parquet"
    analysis_posts_path = args.output_dir / "analysis_posts.parquet"
    posts.to_parquet(posts_path, index=False, compression="zstd")
    comments_in_window.to_parquet(comments_path, index=False, compression="zstd")
    analysis_posts.to_parquet(analysis_posts_path, index=False, compression="zstd")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_post_files": [path.name for path in post_paths],
        "raw_comment_files": [path.name for path in comment_paths],
        "posts": post_report,
        "comments": comment_report,
        "comments_matching_downloaded_posts": matched_comments,
        "comments_excluded_without_downloaded_post": int(len(comments) - matched_comments),
        "processed_posts": int(len(posts)),
        "processed_comments": int(len(comments_in_window)),
        "analysis_posts": int(len(analysis_posts)),
        "analysis_post_subreddits": sorted(
            analysis_posts["subreddit"].dropna().unique().tolist()
        ),
        "analysis_posts_with_replies_72h": int(
            analysis_posts["has_reply_72h"].sum()
        ),
        "outputs": {
            "posts": str(posts_path),
            "comments": str(comments_path),
            "analysis_posts": str(analysis_posts_path),
        },
    }
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.metadata_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(posts):,} posts to {posts_path}")
    print(f"Wrote {len(comments_in_window):,} matching comments to {comments_path}")
    print(f"Wrote {len(analysis_posts):,} analysis posts to {analysis_posts_path}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()
