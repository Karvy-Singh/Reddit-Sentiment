"""Validated local inference service for the engagement classifier."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "engagement_classifier.joblib"
LOG_PATH = ROOT / "logs" / "predictions.jsonl"


class PredictionRequest(BaseModel):
    subreddit: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=500)
    title_words: int = Field(ge=1, le=100)
    sentiment: float = Field(ge=-1, le=1)
    author_prior_posts: int = Field(ge=0, le=1_000_000)
    community_posts_prior_24h: int = Field(ge=0, le=10_000_000)
    reply_count_1h: int = Field(ge=0, le=1_000_000)
    unique_repliers_1h: int = Field(ge=0, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    has_url: bool
    is_weekend: bool

    @model_validator(mode="after")
    def validate_consistency(self) -> "PredictionRequest":
        if self.title_words != len(self.title.split()):
            raise ValueError("title_words must match the supplied title")
        if self.is_weekend != (self.day_of_week >= 5):
            raise ValueError("is_weekend must agree with day_of_week")
        if self.unique_repliers_1h > self.reply_count_1h:
            raise ValueError("unique_repliers_1h cannot exceed reply_count_1h")
        return self


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"model artifact not found: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    required = {
        "model",
        "model_version",
        "features",
        "thresholds",
        "target_definition",
        "decision_threshold",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(f"invalid model artifact; missing {sorted(missing)}")
    return artifact


def predict(payload: PredictionRequest, log: bool = True) -> dict:
    artifact = load_artifact()
    if payload.subreddit not in artifact["thresholds"]:
        raise ValueError(f"unsupported subreddit: {payload.subreddit}")
    frame = pd.DataFrame([payload.model_dump()])
    probability = float(artifact["model"].predict_proba(frame[artifact["features"]])[:, 1][0])
    result = {
        "label": (
            "high_engagement"
            if probability >= artifact["decision_threshold"]
            else "ordinary_engagement"
        ),
        "probability": probability,
        "decision_threshold": artifact["decision_threshold"],
        "model_version": artifact["model_version"],
        "target_definition": artifact["target_definition"],
        "engagement_thresholds": artifact["thresholds"][payload.subreddit],
    }
    if log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = payload.model_dump(exclude={"title"}) | result
        record["predicted_at_utc"] = datetime.now(timezone.utc).isoformat()
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return result


app = FastAPI(title="DataSuper Engagement API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    artifact = load_artifact()
    return {"status": "ok", "model_version": artifact["model_version"]}


@app.post("/predict-engagement")
def predict_engagement(payload: PredictionRequest) -> dict:
    try:
        return predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
