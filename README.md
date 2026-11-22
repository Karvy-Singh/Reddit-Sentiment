# Reddit-Sentiment
Data science pipeline for analyzing and predicting engagement patterns, discussion topics, and reply networks across selected Reddit communities.

## Current Dataset

- 23,794 downloaded posts
- 106,003 comments matched to downloaded root posts
- 20,094 analysis posts from three communities with complete 72-hour comment windows
- 72-hour reply outcomes derived from comment timestamps rather than snapshot comment totals
- Stable local pseudonyms; raw usernames are absent from processed tables

See `DATASET.md` for provenance, coverage, and limitations.

## Pipeline

```text
Arctic Shift JSONL
    -> schema validation and pseudonymization
    -> Parquet post/comment tables
    -> leakage-safe publication and one-hour features
    -> statistics, probability, topics, and EDA
    -> chronological ML evaluation
    -> clustering, PCA, and reply network
    -> versioned model, API, dashboard, and monitoring
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Commands use the local source package:

```bash
export PYTHONPATH=src
```

## Run

Build the structured tables after acquisition:

```bash
.venv/bin/python scripts/build_structured_data.py
```

Generate features, statistics, topics, figures, and daily metrics:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_analysis.py
```

Train, tune, evaluate, explain, and version the models:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_models.py
```

Build the reply network and monitoring report:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_network.py
PYTHONPATH=src .venv/bin/python scripts/monitor_model.py
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Start the API:

```bash
PYTHONPATH=src .venv/bin/uvicorn api:app --reload
```

Start the dashboard:

```bash
PYTHONPATH=src .venv/bin/streamlit run dashboard.py
```

## Outputs

- `data/processed/features.parquet`: analytical feature table
- `data/processed/clusters.parquet`: PCA and cluster assignments
- `data/processed/network_*.parquet`: pseudonymous network tables
- `reports/analysis_summary.json`: probability and statistical results
- `reports/model_report.json`: validation, test, tuning, interpretation, PCA, and clustering results
- `reports/network_report.json`: aggregate propagation-network measures
- `reports/figures/`: static report figures
- `models/publication_classifier.joblib`: publication-time baseline
- `models/engagement_classifier.joblib`: versioned one-hour prediction pipeline

## Methodological Guardrails

- Engagement thresholds are learned from training-period data only.
- Splits are chronological: 70% training, 15% validation, and 15% testing.
- Preprocessing, TF-IDF, SMOTE, and tuning are fitted without test data.
- Retrieval-time body text is excluded from prediction because moderation can replace it after publication.
- The one-hour model uses only replies already observed by the declared one-hour prediction cutoff.
- Model selection uses validation PR-AUC; the final test period is evaluated after selection.
- Decision thresholds maximize F1 on validation data only and are then frozen for testing and API inference.
- Dashboard outputs are aggregate and do not expose user identities or raw quotations.
- The three-month sample supports basic trend analysis but not annual seasonality.

See `reports/RESULTS.md` for current findings.
