#!/usr/bin/env python3
"""Train, tune, explain, and persist the pilot models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from datasuper.data import chronological_split
from datasuper.modeling import (
    EARLY_MODEL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    classification_metrics,
    classification_models,
    engagement_labels,
    engagement_thresholds,
    f1_optimal_threshold,
    regression_metrics,
    regression_models,
)


ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def tune_models(train: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    splitter = TimeSeriesSplit(n_splits=3)
    tree = classification_models()["decision_tree"]
    forest = classification_models()["random_forest"]
    grid = GridSearchCV(
        tree,
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [2, 5, 10]},
        scoring="average_precision",
        cv=splitter,
        n_jobs=-1,
    )
    random = RandomizedSearchCV(
        forest,
        {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", 0.7, 1.0],
        },
        n_iter=5,
        scoring="average_precision",
        cv=splitter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    grid.fit(train[MODEL_FEATURES], target)
    random.fit(train[MODEL_FEATURES], target)
    return {"grid_tree": grid.best_estimator_, "randomized_forest": random.best_estimator_}


def explain_model(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    features: list[str],
) -> dict:
    permutation = permutation_importance(
        model,
        test[features],
        target,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
    )
    importance = dict(
        sorted(
            zip(features, permutation.importances_mean.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    explanation_train = train.iloc[-min(5000, len(train)) :]
    transformed_train = model.named_steps["preprocess"].transform(
        explanation_train[features]
    )
    transformed_test = model.named_steps["preprocess"].transform(test[features])
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()
        transformed_test = transformed_test.toarray()
    names = model.named_steps["preprocess"].get_feature_names_out().tolist()
    estimator = model.named_steps["model"]
    explainer = LimeTabularExplainer(
        transformed_train,
        feature_names=names,
        class_names=["ordinary", "high"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    explanation = explainer.explain_instance(
        transformed_test[0], estimator.predict_proba, num_features=min(8, len(names))
    )
    return {"permutation_importance": importance, "lime_first_test_row": explanation.as_list()}


def fit_classifier_track(
    models: dict[str, object],
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_y: pd.Series,
    validation_y: pd.Series,
    test_y: pd.Series,
) -> tuple[object, dict]:
    validation_scores = {}
    for name, model in models.items():
        model.fit(train[features], train_y)
        validation_scores[name] = classification_metrics(
            model, validation[features], validation_y
        )
    best_name = max(validation_scores, key=lambda name: validation_scores[name]["pr_auc"])
    validation_probability = models[best_name].predict_proba(validation[features])[:, 1]
    decision_threshold = f1_optimal_threshold(validation_y, validation_probability)
    train_scores = classification_metrics(
        models[best_name], train[features], train_y, decision_threshold
    )
    validation_tuned = classification_metrics(
        models[best_name], validation[features], validation_y, decision_threshold
    )
    best_model = clone(models[best_name])
    development = pd.concat([train, validation], ignore_index=True)
    development_y = pd.concat([train_y, validation_y], ignore_index=True)
    best_model.fit(development[features], development_y)
    return best_model, {
        "validation": validation_scores,
        "selected": best_name,
        "decision_threshold": decision_threshold,
        "train_selected": train_scores,
        "validation_selected_tuned": validation_tuned,
        "test": classification_metrics(
            best_model, test[features], test_y, decision_threshold
        ),
        "interpretation": explain_model(best_model, development, test, test_y, features),
    }


def clustering(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    values = frame[NUMERIC_FEATURES].astype(float)
    scaled = StandardScaler().fit_transform(values)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)

    scores = {}
    for clusters in range(2, 7):
        labels = KMeans(n_clusters=clusters, n_init=20, random_state=RANDOM_STATE).fit_predict(scaled)
        scores[clusters] = silhouette_score(scaled, labels)
    best_clusters = max(scores, key=scores.get)
    kmeans_labels = KMeans(
        n_clusters=best_clusters, n_init=20, random_state=RANDOM_STATE
    ).fit_predict(scaled)

    sample_size = min(1000, len(frame))
    sample_indices = np.linspace(0, len(frame) - 1, sample_size, dtype=int)
    hierarchical = np.full(len(frame), -1)
    hierarchical[sample_indices] = AgglomerativeClustering(n_clusters=best_clusters).fit_predict(
        scaled[sample_indices]
    )

    assignments = frame[["post_id", "subreddit"]].copy()
    assignments["pca_1"] = components[:, 0]
    assignments["pca_2"] = components[:, 1]
    assignments["kmeans_cluster"] = kmeans_labels
    assignments["hierarchical_cluster"] = hierarchical
    report = {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "silhouette_scores": {str(key): float(value) for key, value in scores.items()},
        "selected_clusters": int(best_clusters),
        "hierarchical_sample_size": sample_size,
    }
    return assignments, report


def main() -> None:
    frame = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    train, validation, test = chronological_split(frame)
    thresholds = engagement_thresholds(train)
    train_y = engagement_labels(train, thresholds)
    validation_y = engagement_labels(validation, thresholds)
    test_y = engagement_labels(test, thresholds)

    development = pd.concat([train, validation], ignore_index=True)
    publication_models = classification_models()
    publication_models.update(tune_models(train, train_y))
    publication_classifier, publication_report = fit_classifier_track(
        publication_models,
        MODEL_FEATURES,
        train,
        validation,
        test,
        train_y,
        validation_y,
        test_y,
    )
    early_classifier, early_report = fit_classifier_track(
        classification_models(early=True),
        EARLY_MODEL_FEATURES,
        train,
        validation,
        test,
        train_y,
        validation_y,
        test_y,
    )

    regressors = regression_models()
    regression_validation = {}
    for name, model in regressors.items():
        model.fit(train[MODEL_FEATURES], train["log_replies_72h"])
        regression_validation[name] = regression_metrics(
            model, validation[MODEL_FEATURES], validation["log_replies_72h"]
        )
    best_regression_name = min(
        regression_validation, key=lambda name: regression_validation[name]["rmse"]
    )
    regression_train = regression_metrics(
        regressors[best_regression_name], train[MODEL_FEATURES], train["log_replies_72h"]
    )
    best_regressor = clone(regressors[best_regression_name])
    best_regressor.fit(development[MODEL_FEATURES], development["log_replies_72h"])
    regression_test = regression_metrics(
        best_regressor, test[MODEL_FEATURES], test["log_replies_72h"]
    )

    assignments, cluster_report = clustering(frame)
    assignments.to_parquet(
        ROOT / "data" / "processed" / "clusters.parquet", index=False, compression="zstd"
    )

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    publication_artifact = {
        "model": publication_classifier,
        "model_version": "2.0.0-hybrid",
        "features": MODEL_FEATURES,
        "thresholds": thresholds,
        "target_definition": "top-10% replies within 72h or top-5% observed Reddit score",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(development),
        "decision_threshold": publication_report["decision_threshold"],
    }
    early_artifact = {
        "model": early_classifier,
        "model_version": "3.0.0-hybrid-early",
        "features": EARLY_MODEL_FEATURES,
        "thresholds": thresholds,
        "target_definition": "top-10% replies within 72h or top-5% observed Reddit score",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(development),
        "decision_threshold": early_report["decision_threshold"],
    }
    joblib.dump(publication_artifact, model_dir / "publication_classifier.joblib")
    joblib.dump(early_artifact, model_dir / "engagement_classifier.joblib")

    report = {
        "target_definition": "top-10% replies within 72h or top-5% observed Reddit score",
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "class_rates": {
            "train": float(train_y.mean()),
            "validation": float(validation_y.mean()),
            "test": float(test_y.mean()),
        },
        "engagement_thresholds": thresholds,
        "classification_validation": publication_report["validation"],
        "selected_classifier": publication_report["selected"],
        "classification_train_selected": publication_report["train_selected"],
        "classification_validation_selected_tuned": publication_report[
            "validation_selected_tuned"
        ],
        "classification_decision_threshold": publication_report[
            "decision_threshold"
        ],
        "classification_test": publication_report["test"],
        "early_classification_validation": early_report["validation"],
        "selected_early_classifier": early_report["selected"],
        "early_classification_train_selected": early_report["train_selected"],
        "early_classification_validation_selected_tuned": early_report[
            "validation_selected_tuned"
        ],
        "early_classification_decision_threshold": early_report[
            "decision_threshold"
        ],
        "early_classification_test": early_report["test"],
        "regression_validation": regression_validation,
        "selected_regressor": best_regression_name,
        "regression_train_selected": regression_train,
        "regression_test": regression_test,
        "clustering": cluster_report,
        "interpretation": publication_report["interpretation"],
        "early_interpretation": early_report["interpretation"],
    }
    (ROOT / "reports" / "model_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Publication {publication_report['selected']}: "
        f"PR-AUC={publication_report['test']['pr_auc']:.3f}; "
        f"early {early_report['selected']}: PR-AUC={early_report['test']['pr_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
