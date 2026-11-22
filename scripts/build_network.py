#!/usr/bin/env python3
"""Build aggregate user reply-network outputs."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from datasuper.network import reply_edges


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    processed = ROOT / "data" / "processed"
    posts = pd.read_parquet(processed / "posts.parquet")
    comments = pd.read_parquet(processed / "comments.parquet")
    edges = reply_edges(posts, comments)
    weighted = (
        edges.groupby(["source_author_id", "target_author_id"])
        .size()
        .rename("weight")
        .reset_index()
    )
    graph = nx.from_pandas_edgelist(
        weighted,
        source="source_author_id",
        target="target_author_id",
        edge_attr="weight",
        create_using=nx.DiGraph,
    )
    pagerank = nx.pagerank(graph, weight="weight") if graph else {}
    nodes = pd.DataFrame(
        {
            "author_id": list(graph.nodes),
            "in_degree": [graph.in_degree(node, weight="weight") for node in graph],
            "out_degree": [graph.out_degree(node, weight="weight") for node in graph],
            "pagerank": [pagerank[node] for node in graph],
        }
    )
    weighted.to_parquet(processed / "network_edges.parquet", index=False, compression="zstd")
    nodes.to_parquet(processed / "network_nodes.parquet", index=False, compression="zstd")

    undirected = graph.to_undirected()
    report = {
        "reply_events": len(edges),
        "unique_edges": len(weighted),
        "nodes": graph.number_of_nodes(),
        "density": nx.density(graph),
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "largest_component_nodes": max(
            (len(component) for component in nx.weakly_connected_components(graph)), default=0
        ),
        "average_clustering_undirected": nx.average_clustering(undirected, weight="weight")
        if undirected
        else 0.0,
    }
    (ROOT / "reports" / "network_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(weighted):,} reply edges across {len(nodes):,} users")


if __name__ == "__main__":
    main()
