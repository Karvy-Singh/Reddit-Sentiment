"""Streamlit dashboard for aggregate pilot results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from datasuper.reddit_validation import validate_reddit_post
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
    st.subheader("Test a real Reddit post")
    st.caption(
        "Supports r/datascience, r/MachineLearning, and r/artificial. "
        "The post must be at least one hour old."
    )
    reddit_url = st.text_input(
        "Reddit post URL",
        placeholder="https://www.reddit.com/r/datascience/comments/...",
    )
    if st.button("Fetch post and predict", type="primary"):
        if not reddit_url.strip():
            st.warning("Enter a Reddit post URL.")
        else:
            try:
                with st.spinner("Reading the post and timestamped comments..."):
                    record = validate_reddit_post(
                        reddit_url,
                        features_path=ROOT / "data" / "processed" / "features.parquet",
                        output_dir=ROOT / "reports" / "reddit_predictions",
                    )
            except Exception as exc:
                st.error(f"Could not test this post: {exc}")
            else:
                result = record["prediction"]
                observation = record["latest_observation"]
                columns = st.columns(4)
                columns[0].metric("Model prediction", result["label"].replace("_", " ").title())
                columns[1].metric("High-engagement probability", f"{result['probability']:.1%}")
                columns[2].metric("Replies observed so far", observation["replies_observed_so_far"])
                columns[3].metric("Reddit score observed", observation["score_observed"])
                signal_columns = st.columns(5)
                signal_columns[0].metric(
                    "Upvote ratio", f"{observation['upvote_ratio_observed']:.0%}"
                )
                signal_columns[1].metric(
                    "Unique repliers", observation["unique_repliers_observed"]
                )
                signal_columns[2].metric(
                    "Thread depth", observation["max_thread_depth_observed"]
                )
                signal_columns[3].metric(
                    "Crossposts", observation["crossposts_observed"]
                )
                signal_columns[4].metric(
                    "Comments/hour", f"{observation['comment_velocity_per_hour']:.2f}"
                )
                score_velocity = observation["score_velocity_per_hour"]
                st.caption(
                    "Score velocity since the previous check: "
                    + (f"{score_velocity:+.2f}/hour" if score_velocity is not None else "check again later")
                )
                st.caption(
                    f"Prediction used {record['features_at_1h']['reply_count_1h']} replies "
                    "from the first hour only. Later replies were not model inputs."
                )
                thresholds = result["engagement_thresholds"]
                st.caption(
                    "High engagement requires any one of: "
                    f"{thresholds['reply_count_72h']} replies, vote quality "
                    f"{thresholds['vote_quality']:.1f}, "
                    f"{thresholds['unique_repliers_72h']} unique repliers, thread depth "
                    f"{thresholds['max_thread_depth_72h']}, or a crosspost."
                )
                st.caption(
                    "Reddit score is the platform's fuzzed net score, not an exact upvote or share count."
                )
                if "evaluation" in record:
                    evaluation = record["evaluation"]
                    if evaluation["prediction_correct"]:
                        st.success("The model prediction matched the actual 72-hour outcome.")
                    else:
                        st.error("The model prediction did not match the actual 72-hour outcome.")
                    st.write(
                        f"Actual: **{evaluation['actual_reply_count_72h']} replies** in 72 hours "
                        f"from **{evaluation['actual_unique_repliers_72h']} people**, maximum "
                        f"depth **{evaluation['actual_max_thread_depth_72h']}**, Reddit score "
                        f"**{evaluation['actual_score_observed']}** at "
                        f"**{evaluation['actual_upvote_ratio_observed']:.0%}** approval, and "
                        f"**{evaluation['actual_crossposts_observed']} crossposts**."
                    )
                else:
                    st.info(
                        "The final comparison will be available at "
                        f"{record['evaluation_available_at_utc']}. Submit this URL again after that time."
                    )
                if record["was_in_local_dataset"]:
                    st.warning("This post was part of the local dataset, so it is not a fresh real-world test.")

    with st.expander("Manual feature test"):
        title = st.text_input("Post title", "A practical machine learning question")
        subreddit = st.selectbox("Community", sorted(models["engagement_thresholds"]))
        hour = st.slider("Hour (UTC)", 0, 23, 12)
        day = st.slider("Day of week (Monday = 0)", 0, 6, 2)
        replies_1h = st.number_input("Replies in first hour", min_value=0, value=1)
        unique_1h = st.number_input(
            "Unique repliers in first hour", min_value=0, max_value=int(replies_1h), value=0
        )
        depth_1h = st.number_input(
            "Maximum reply depth in first hour",
            min_value=0,
            max_value=int(replies_1h),
            value=0,
        )
        if st.button("Estimate manually"):
            payload = PredictionRequest(
                subreddit=subreddit,
                title=title,
                title_words=len(title.split()),
                sentiment=0.0,
                author_prior_posts=0,
                community_posts_prior_24h=50,
                reply_count_1h=int(replies_1h),
                unique_repliers_1h=int(unique_1h),
                max_thread_depth_1h=int(depth_1h),
                hour=hour,
                day_of_week=day,
                has_url=False,
                is_weekend=day >= 5,
            )
            result = predict(payload, log=False)
            st.metric("High-engagement probability", f"{result['probability']:.1%}")
