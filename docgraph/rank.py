"""PageRank over the call graph for relevance ranking."""
from __future__ import annotations

import networkx as nx

from docgraph.db import GraphDB


def compute_pagerank(db: GraphDB) -> dict[int, float]:
    """Build directed graph from CALLS + REFERENCES_ + INHERITS, run PageRank."""
    g = nx.DiGraph()
    edges = db.fetch_all(
        "MATCH (a)-[:CALLS|INHERITS|REFERENCES_|INSTANTIATES]->(b) "
        "RETURN a.id AS src, b.id AS dst"
    )
    for row in edges:
        g.add_edge(row["src"], row["dst"])
    if not g.nodes:
        return {}
    return nx.pagerank(g, alpha=0.85, max_iter=50, tol=1e-4)


def write_pagerank(db: GraphDB, scores: dict[int, float]) -> None:
    """Update pagerank property on Function and Class nodes."""
    rows = [{"id": nid, "pr": pr} for nid, pr in scores.items()]
    if not rows:
        return
    for label in ("Function", "Class", "File"):
        db.execute(
            f"UNWIND $rows AS row MATCH (n:{label} {{id: row.id}}) SET n.pagerank = row.pr",
            {"rows": rows},
        )
