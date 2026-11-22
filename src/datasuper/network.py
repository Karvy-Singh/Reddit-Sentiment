"""Reply-edge construction for pseudonymous Reddit records."""

from __future__ import annotations

import pandas as pd


def reply_edges(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    required_posts = {"post_id", "author_id"}
    required_comments = {"comment_id", "post_id", "parent_id", "parent_type", "author_id"}
    if missing := required_posts.difference(posts.columns):
        raise ValueError(f"missing post columns: {sorted(missing)}")
    if missing := required_comments.difference(comments.columns):
        raise ValueError(f"missing comment columns: {sorted(missing)}")

    post_authors = posts.set_index("post_id")["author_id"]
    comment_authors = comments.set_index("comment_id")["author_id"]
    result = comments[["comment_id", "post_id", "parent_id", "parent_type", "author_id", "created_at"]].copy()
    result["target_author_id"] = result["parent_id"].map(comment_authors)
    root_parent = result["parent_type"].eq("t3")
    result.loc[root_parent, "target_author_id"] = result.loc[root_parent, "post_id"].map(post_authors)
    result = result.rename(columns={"author_id": "source_author_id"})
    result = result.dropna(subset=["source_author_id", "target_author_id"])
    result = result[result["source_author_id"] != result["target_author_id"]]
    return result.reset_index(drop=True)
