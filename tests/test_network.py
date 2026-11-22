import pandas as pd

from datasuper.network import reply_edges


def test_reply_edges_resolve_post_and_comment_parents() -> None:
    posts = pd.DataFrame({"post_id": ["p1"], "author_id": ["root"]})
    comments = pd.DataFrame(
        {
            "comment_id": ["c1", "c2"],
            "post_id": ["p1", "p1"],
            "parent_id": ["p1", "c1"],
            "parent_type": ["t3", "t1"],
            "author_id": ["user1", "user2"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        }
    )
    edges = reply_edges(posts, comments)
    assert edges[["source_author_id", "target_author_id"]].values.tolist() == [
        ["user1", "root"],
        ["user2", "user1"],
    ]
