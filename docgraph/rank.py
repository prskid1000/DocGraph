"""PageRank over the call graph for relevance ranking.

Personalized PageRank lets a query say "rank by proximity to *this* file or
*this* function". networkx supports it via the `personalization` arg —
we cache the constructed graph between calls so per-query PR is fast (~10ms
on a 5k-node graph)."""
from __future__ import annotations

from threading import Lock

import networkx as nx

from docgraph.db import GraphDB


def _build_graph(db: GraphDB) -> nx.DiGraph:
    g = nx.DiGraph()
    edges = db.fetch_all(
        "MATCH (a)-[:CALLS|INHERITS|REFERENCES_|INSTANTIATES]->(b) "
        "RETURN a.id AS src, b.id AS dst"
    )
    for row in edges:
        g.add_edge(row["src"], row["dst"])
    return g


def compute_pagerank(db: GraphDB) -> dict[int, float]:
    g = _build_graph(db)
    if not g.nodes:
        return {}
    return nx.pagerank(g, alpha=0.85, max_iter=50, tol=1e-4)


def write_pagerank(db: GraphDB, scores: dict[int, float]) -> None:
    rows = [{"id": nid, "pr": pr} for nid, pr in scores.items()]
    if not rows:
        return
    for label in ("Function", "Class", "File"):
        db.execute(
            f"UNWIND $rows AS row MATCH (n:{label} {{id: row.id}}) SET n.pagerank = row.pr",
            {"rows": rows},
        )


# --- Personalized PageRank ------------------------------------------------


class PersonalizedRanker:
    """Caches the call-graph between queries. Build once, run PR many times.

    Thread-safe. The cache is invalidated by replacing the instance — callers
    that index need to drop their PersonalizedRanker reference.
    """

    def __init__(self, db: GraphDB):
        self._db = db
        self._lock = Lock()
        self._graph: nx.DiGraph | None = None

    def _ensure_graph(self) -> nx.DiGraph:
        with self._lock:
            if self._graph is None:
                self._graph = _build_graph(self._db)
            return self._graph

    def personalized(self, focus_ids: list[int], alpha: float = 0.85) -> dict[int, float]:
        """Run PR with mass concentrated on focus_ids. Returns {id: score}."""
        g = self._ensure_graph()
        if not g.nodes:
            return {}
        focus = [i for i in focus_ids if i in g.nodes]
        if not focus:
            return nx.pagerank(g, alpha=alpha, max_iter=50, tol=1e-4)
        personalization = {n: 0.0 for n in g.nodes}
        for fid in focus:
            personalization[fid] = 1.0 / len(focus)
        return nx.pagerank(
            g, alpha=alpha, personalization=personalization,
            max_iter=50, tol=1e-4,
        )
