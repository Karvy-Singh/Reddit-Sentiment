from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from datasuper.monitoring import population_stability_index
from datasuper.service import ROOT, app


def valid_payload() -> dict:
    return {
        "subreddit": "datascience",
        "title": "A practical machine learning question",
        "title_words": 5,
        "sentiment": 0.0,
        "author_prior_posts": 0,
        "community_posts_prior_24h": 50,
        "reply_count_1h": 1,
        "unique_repliers_1h": 1,
        "max_thread_depth_1h": 1,
        "hour": 12,
        "day_of_week": 2,
        "has_url": False,
        "is_weekend": False,
    }


def test_api_health_and_prediction() -> None:
    assert (ROOT / "models" / "engagement_classifier.joblib").is_file()
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    response = client.post("/predict-engagement", json=valid_payload())
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


def test_api_rejects_inconsistent_calendar_fields() -> None:
    payload = valid_payload() | {"day_of_week": 6, "is_weekend": False}
    assert TestClient(app).post("/predict-engagement", json=payload).status_code == 422


def test_api_rejects_impossible_early_counts() -> None:
    payload = valid_payload() | {"reply_count_1h": 1, "unique_repliers_1h": 2}
    assert TestClient(app).post("/predict-engagement", json=payload).status_code == 422


def test_psi_detects_shift() -> None:
    expected = np.linspace(0, 1, 1000)
    assert population_stability_index(expected, expected) == 0
    assert population_stability_index(expected, expected + 1) > 0.2
