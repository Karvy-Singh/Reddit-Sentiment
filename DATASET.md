# Dataset Guide

## Current Status

The workspace contains the final academic Reddit dataset acquired from the public Arctic Shift API. This is an unofficial archive, not the Reddit for Researchers BigQuery dataset.

### Analysis coverage

- Root posts: `2024-01-01 00:00:00 UTC` to `2024-04-01 00:00:00 UTC` exclusive
- Comment outcome window: through `2024-04-04 00:00:00 UTC` exclusive
- Post communities: `artificial`, `MachineLearning`, `datascience`, `technology`, `ChatGPT`, and `LocalLLaMA`
- Complete comment files currently used: `artificial`, `MachineLearning`, and `datascience`
- Engagement horizon: 72 hours after each root post

The longer comment window supplies a full 72-hour outcome horizon for every analysis post. Incomplete `.part` downloads are resumable but are intentionally ignored by the structured-data builder.

### Validated size

- 23,794 downloaded posts across all available pilot and final files
- 114,795 downloaded comments in the three analysis communities
- 106,003 comments matched to downloaded root posts
- 20,094 analysis posts with complete outcome windows

## Files

### Raw data

`data/raw/*.jsonl` contains one JSON object per source record. Raw files retain archive fields and are excluded from version control.

### Structured data

`data/processed/posts.parquet` contains all downloaded pseudonymized posts.

`data/processed/comments.parquet` contains pseudonymized comments whose `post_id` exists in the downloaded post table.

`data/processed/analysis_posts.parquet` contains only posts from communities with completed comment-window files. It adds:

```text
reply_count_72h
unique_repliers_72h
reply_count_1h
unique_repliers_1h
first_response_hours
has_reply_1h
has_reply_72h
```

Use `analysis_posts.parquet` for engagement statistics. Do not treat absent comments in communities with incomplete comment downloads as zero engagement.

### Metadata

- `data/metadata/download_manifest.json`: source URLs, ranges, record counts, checksums, and extraction times
- `data/metadata/validation_report.json`: row counts, missingness, duplicates, joins, exclusions, and output paths
- `data/metadata/.author_salt`: local secret used for stable author pseudonyms; never commit or share it

## Reproduction

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Download the final analysis posts:

```bash
.venv/bin/python scripts/download_reddit.py \
  --subreddits artificial MachineLearning datascience \
  --start 2024-01-01 \
  --end 2024-04-01 \
  --kinds posts
```

Download the complete comment outcome window:

```bash
.venv/bin/python scripts/download_reddit.py \
  --subreddits artificial MachineLearning datascience \
  --start 2024-01-01 \
  --end 2024-04-04 \
  --kinds comments
```

Build the Parquet tables:

```bash
.venv/bin/python scripts/build_structured_data.py
```

The downloader skips completed files and resumes `.part` files from their last valid record.

## Data Limitations

- The archive may contain collection gaps or records later edited or deleted on Reddit.
- Scores, comment totals, and upvote ratios are observation-time snapshots, not controlled 72-hour outcomes.
- The API supplies records in pages of up to 100, so large comment ranges are slow to acquire.
- Three months support basic trends and holdout forecasting, but not annual seasonality or broad generalization claims.
- Comment files include replies to posts outside the selected root-post window; these are excluded from processed comments.
- Deleted authors are represented as missing and all retained authors are pseudonymized.
- The archive does not specify an open redistribution license. Do not publish raw records or raw text.

## Modelling Windows

- Publication-time model: title, time, community, sentiment, and prior activity available when the post is created
- Early model: publication features plus reply count and unique repliers observed during the first hour
- Outcome: subreddit-specific top-decile replies accumulated within 72 hours
