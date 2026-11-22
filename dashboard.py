"""Streamlit dashboard for aggregate pilot results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from datasuper.service import PredictionRequest, predict


ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_data() -> tuple[pd.DataFrame, dict, dict, dict]:
    features = pd.read_parquet(ROOT / "data" / "processed" / "features.parquet")
    analysis = json.loads((ROOT / "reports" / "analysis_summary.json").read_text())
    models = json.loads((ROOT / "reports" / "model_report.json").read_text())
    network = json.loads((ROOT / "reports" / "network_report.json").read_text())
    return features, analysis, models, network


st.set_page_config(page_title="DataSuper", layout="wide")
st.title("Online Community Behaviour")
st.caption("A one-week Reddit pilot. Results describe the observed sample, not Reddit generally.")

data, analysis, models, network = load_data()
overview, statistics, topics, network_tab, modelling, prediction = st.tabs(
    ["Overview", "Statistics", "Topics", "Network", "Models", "Prediction"]
)

with overview:
    columns = st.columns(4)
    columns[0].metric("Posts", f"{len(data):,}")
    columns[1].metric("Communities", data.subreddit.nunique())
    columns[2].metric("Median replies", f"{data.reply_count_72h.median():.0f}")
    columns[3].metric("Posts with replies", f"{data.has_reply_72h.mean():.1%}")
    daily = data.set_index("created_at").groupby("subreddit").resample("1D").size().rename("posts").reset_index()
    st.plotly_chart(
        px.line(daily, x="created_at", y="posts", color="subreddit", markers=True),
        width="stretch",
    )

with statistics:
    st.plotly_chart(
        px.box(data, x="subreddit", y="log_replies_72h", color="subreddit"),
        width="stretch",
    )
    rows = []
    for name, result in analysis["statistical_tests"].items():
        rows.append({"test": name, "p_value": result["p_value"], "statistic": result["statistic"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("ANOVA has unequal variances; the report includes a Kruskal-Wallis sensitivity test.")

with topics:
    topic_counts = data.topic_id.value_counts().sort_index().rename_axis("topic_id").reset_index(name="posts")
    topic_counts["keywords"] = topic_counts.topic_id.astype(str).map(
        lambda key: ", ".join(analysis["topic_keywords"][key][:5])
    )
    st.plotly_chart(px.bar(topic_counts, x="keywords", y="posts"), width="stretch")

with network_tab:
    columns = st.columns(4)
    columns[0].metric("Users", f"{network['nodes']:,}")
    columns[1].metric("Reply events", f"{network['reply_events']:,}")
    columns[2].metric("Unique edges", f"{network['unique_edges']:,}")
    columns[3].metric("Largest component", f"{network['largest_component_nodes']:,}")
    nodes = pd.read_parquet(ROOT / "data" / "processed" / "network_nodes.parquet")
    st.plotly_chart(
        px.histogram(nodes, x="in_degree", nbins=50, log_y=True, title="Incoming reply distribution"),
        width="stretch",
    )
    st.caption("Only aggregate pseudonymous network statistics are displayed.")

with modelling:
    publication = models["classification_test"]
    early = models["early_classification_test"]
    comparison_metrics = pd.DataFrame(
        {
            "model": ["Publication time", "After one hour"],
            "precision": [publication["precision"], early["precision"]],
            "recall": [publication["recall"], early["recall"]],
            "f1": [publication["f1"], early["f1"]],
            "pr_auc": [publication["pr_auc"], early["pr_auc"]],
        }
    )
    st.dataframe(comparison_metrics, hide_index=True, width="stretch")
    st.caption(
        "Validation-selected cutoffs: "
        f"publication {models['classification_decision_threshold']:.3f}; "
        f"one-hour {models['early_classification_decision_threshold']:.3f}."
    )
    comparison = pd.DataFrame(models["classification_validation"]).T.reset_index(names="model")
    st.plotly_chart(px.bar(comparison, x="model", y="pr_auc"), width="stretch")
    st.info("The deployed model predicts after one hour; the publication-time model remains the harder baseline.")

with prediction:
    title = st.text_input("Post title", "A practical machine learning question")
    subreddit = st.selectbox("Community", sorted(models["engagement_thresholds"]))
    hour = st.slider("Hour (UTC)", 0, 23, 12)
    day = st.slider("Day of week (Monday = 0)", 0, 6, 2)
    replies_1h = st.number_input("Replies in first hour", min_value=0, value=1)
    unique_1h = st.number_input(
        "Unique repliers in first hour", min_value=0, max_value=int(replies_1h), value=0
    )
    if st.button("Estimate engagement"):
        payload = PredictionRequest(
            subreddit=subreddit,
            title=title,
            title_words=len(title.split()),
            sentiment=0.0,
            author_prior_posts=0,
            community_posts_prior_24h=50,
            reply_count_1h=int(replies_1h),
            unique_repliers_1h=int(unique_1h),
            hour=hour,
            day_of_week=day,
            has_url=False,
            is_weekend=day >= 5,
        )
        result = predict(payload, log=False)
        st.metric("High-engagement probability", f"{result['probability']:.1%}")
