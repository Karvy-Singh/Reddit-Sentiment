#!/usr/bin/env python3
"""Compare logged prediction inputs with the pilot training distribution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from datasuper.modeling import EARLY_NUMERIC_FEATURES
from datasuper.monitoring import population_stability_index


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    log_path = ROOT / "logs" / "predictions.jsonl"
    output_path = ROOT / "reports" / "monitoring_report.json"
    if not log_path.is_file() or log_path.stat().st_size == 0:
        report = {"status": "no_prediction_data", "records": 0}
    else:
        baseline = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
        current = pd.read_json(log_path, lines=True)
        enough_data = len(current) >= 30
        psi = {}
        if enough_data:
            for field in EARLY_NUMERIC_FEATURES:
                if field in {"has_url", "is_weekend"} or field not in current:
                    continue
                observed = current[field].dropna().to_numpy()
                if len(observed) >= 30:
                    psi[field] = population_stability_index(
                        baseline[field].to_numpy(), observed
                    )
        report = {
            "status": "ok" if enough_data else "insufficient_prediction_data",
            "records": len(current),
            "positive_prediction_rate": float(current["label"].eq("high_engagement").mean()),
            "mean_probability": float(current["probability"].mean()),
            "numeric_psi": psi,
            "warning_fields": sorted(field for field, value in psi.items() if value >= 0.2),
        }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Monitoring status: {report['status']} ({report['records']} records)")


if __name__ == "__main__":
    main()
