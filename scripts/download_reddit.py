#!/usr/bin/env python3
"""Download timestamp-bounded Reddit records from the Arctic Shift API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://arctic-shift.photon-reddit.com/api"
KINDS = ("posts", "comments")


def parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("dates must use YYYY-MM-DD") from exc


def utc_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = "DataSuper-academic-project/1.0"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_records(
    session: requests.Session,
    subreddit: str,
    kind: str,
    start: datetime,
    end: datetime,
    output_path: Path,
) -> dict[str, Any]:
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    after = utc_milliseconds(start) - 1
    before = utc_milliseconds(end)
    record_count = 0
    page_count = 0
    resumed_records = 0

    if part_path.exists():
        with part_path.open("r", encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Cannot resume invalid partial file {part_path}:{line_number}"
                    ) from exc
                after = max(after, int(record["created_utc"] * 1000))
                record_count += 1
        resumed_records = record_count
        print(f"Resuming {part_path} after {record_count:,} records", flush=True)

    with part_path.open("a", encoding="utf-8") as handle:
        while True:
            params = {
                "subreddit": subreddit,
                "after": after,
                "before": before,
                "limit": "auto",
                "sort": "asc",
                "meta-app": "DataSuper-academic-project",
            }
            for attempt in range(1, 6):
                response = session.get(
                    f"{BASE_URL}/{kind}/search", params=params, timeout=(15, 180)
                )
                if response.status_code != 422 or attempt == 5:
                    break
                time.sleep(attempt * 2)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(f"Arctic Shift API error: {payload['error']}")

            records = payload.get("data", [])
            if not records:
                break

            for record in records:
                created_ms = int(record["created_utc"] * 1000)
                if utc_milliseconds(start) <= created_ms < before:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    record_count += 1

            page_count += 1
            next_after = int(records[-1]["created_utc"] * 1000)
            if next_after <= after:
                next_after = after + 1000
            after = next_after
            if page_count == 1 or page_count % 10 == 0:
                print(
                    f"{subreddit}/{kind}: {record_count:,} records "
                    f"({page_count} API page{'s' if page_count != 1 else ''})",
                    flush=True,
                )
            time.sleep(0.25)

    part_path.replace(output_path)
    return {
        "subreddit": subreddit,
        "kind": kind,
        "start_utc": start.isoformat(),
        "end_exclusive_utc": end.isoformat(),
        "records": record_count,
        "api_pages": page_count,
        "resumed_records": resumed_records,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": f"{BASE_URL}/{kind}/search",
    }


def existing_file_entry(
    path: Path, subreddit: str, kind: str, start: datetime, end: datetime
) -> dict[str, Any]:
    with path.open("rb") as handle:
        record_count = sum(1 for _ in handle)
    return {
        "subreddit": subreddit,
        "kind": kind,
        "start_utc": start.isoformat(),
        "end_exclusive_utc": end.isoformat(),
        "records": record_count,
        "api_pages": None,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "downloaded_at_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "source_url": f"{BASE_URL}/{kind}/search",
    }


def update_manifest(manifest_path: Path, entries: list[dict[str, Any]]) -> None:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "source": "Arctic Shift public API",
            "source_home": "https://arctic-shift.photon-reddit.com/download-tool",
            "downloads": [],
        }

    existing = {entry["path"]: entry for entry in manifest["downloads"]}
    existing.update({entry["path"]: entry for entry in entries})
    manifest["downloads"] = sorted(existing.values(), key=lambda item: item["path"])
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subreddits", nargs="+", required=True)
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/metadata/download_manifest.json"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.end <= args.start:
        parser.error("--end must be after --start")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()
    manifest_entries: list[dict[str, Any]] = []
    start_label = args.start.date().isoformat()
    end_label = args.end.date().isoformat()

    for subreddit in args.subreddits:
        for kind in args.kinds:
            output_path = args.output_dir / (
                f"{subreddit}_{kind}_{start_label}_{end_label}_exclusive.jsonl"
            )
            if output_path.exists() and not args.force:
                print(f"Skipping existing file: {output_path}")
                entry = existing_file_entry(
                    output_path, subreddit, kind, args.start, args.end
                )
                manifest_entries.append(entry)
                update_manifest(args.manifest, [entry])
                continue
            entry = download_records(
                session, subreddit, kind, args.start, args.end, output_path
            )
            manifest_entries.append(entry)
            update_manifest(args.manifest, [entry])

    if manifest_entries:
        total = sum(entry["records"] for entry in manifest_entries)
        print(f"Downloaded {total:,} records across {len(manifest_entries)} files.")
    else:
        print("No files downloaded.")


if __name__ == "__main__":
    main()
