#!/usr/bin/env python3
"""Predict and validate 72-hour engagement for a real Reddit post URL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasuper.reddit_validation import validate_reddit_post


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
OUTPUT_DIR = ROOT / "reports" / "reddit_predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reddit_url", help="Reddit post URL or base36 post ID")
    args = parser.parse_args()

    try:
        record = validate_reddit_post(
            args.reddit_url,
            features_path=FEATURES_PATH,
            output_dir=OUTPUT_DIR,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(record, indent=2))
    output_path = OUTPUT_DIR / f"{record['post_id']}.json"
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
