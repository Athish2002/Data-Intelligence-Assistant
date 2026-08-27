"""
dia/graph_engine.py
───────────────────
Graph Intelligence & Relational Network Analytics Engine (NetworkX).
Constructs entity-interaction graphs, calculates topological centrality metrics
(PageRank, Betweenness, Degree), and isolates collusion rings & community clusters.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd

log = logging.getLogger("dia.graph")


def construct_and_analyze_entity_graph(
    df: pd.DataFrame,
    source_node_col: str,
    target_node_col: str,
    weight_col: str | None = None,
    max_edges: int = 2000,
) -> dict[str, Any]:
    """
    Constructs a relational entity graph from tabular pairs and computes
    topological network properties, PageRank influence, and community clusters.
    """
    df_clean = df[[source_node_col, target_node_col]].dropna().copy()
    if weight_col and weight_col in df.columns:
        df_clean["weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)
    else:
        df_clean["weight"] = 1.0

    if len(df_clean) > max_edges:
        df_clean = df_clean.head(max_edges)

    # Initialize Graph
    G = nx.Graph()
    for _, row in df_clean.iterrows():
        u = str(row[source_node_col])
        v = str(row[target_node_col])
        w = float(row["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < 2:
        return {
            "status": "insufficient_nodes",
            "message": "At least 2 unique nodes are required to build a network graph.",
        }

    # 1. Topological Metrics
    density = float(nx.density(G))
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)

    # 2. Centrality Scores
    pagerank_scores = nx.pagerank(G, weight="weight")
    degree_centrality = nx.degree_centrality(G)

    # Betweenness on sampled nodes if graph is large
    if n_nodes < 500:
        betweenness = nx.betweenness_centrality(G, weight="weight")
    else:
        betweenness = nx.betweenness_centrality(G, k=min(100, n_nodes), weight="weight")

    # 3. Community Detection (Connected Components / Modularity)
    components = list(nx.connected_components(G))
    top_communities = [
        {"community_id": idx + 1, "member_count": len(c), "sample_members": list(c)[:5]}
        for idx, c in enumerate(sorted(components, key=len, reverse=True)[:5])
    ]

    # Top Central Influential Nodes (PageRank)
    top_influencers = [
        {
            "node_id": node,
            "pagerank": round(float(score), 5),
            "degree": int(G.degree(node)),
            "degree_centrality": round(float(degree_centrality.get(node, 0.0)), 5),
            "betweenness": round(float(betweenness.get(node, 0.0)), 5),
        }
        for node, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    ]

    # Collusion / Dense Ring Anomaly Detection (High clustering + high degree nodes)
    clustering_coeff = nx.clustering(G)
    dense_anomalies = [
        {"node_id": node, "clustering_score": round(float(clustering_coeff[node]), 3), "degree": int(G.degree(node))}
        for node in sorted(clustering_coeff, key=lambda n: clustering_coeff[n] * G.degree(n), reverse=True)[:10]
        if G.degree(node) >= 2
    ]

    return {
        "status": "success",
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "graph_density": round(density, 6),
        "is_fully_connected": is_connected,
        "connected_components_count": n_components,
        "top_influential_nodes": top_influencers,
        "top_communities": top_communities,
        "dense_collusion_anomalies": dense_anomalies,
    }
