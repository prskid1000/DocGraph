"""PageRank over the call graph for relevance ranking.

Sparse power iteration (scipy CSR) over edge arrays read through Arrow --
bounded memory (O(E) ints) and a few seconds on a 1M-edge graph, instead of
a networkx DiGraph with a Python object per edge. Semantics match
networkx.pagerank: duplicate edges collapse, dangling mass is spread by
the personalization vector, L1 convergence at N * tol.

Personalized PageRank lets a query say "rank by proximity to *this* file or
*this* function"; `PersonalizedRanker` builds the CSR once per Retriever
(a Retriever lives until the next reindex) and runs PPR per query.
"""
from __future__ import annotations

from threading import Lock
from typing import Callable

import numpy as np
import scipy.sparse as sp

from docgraph.db import GraphDB

RANK_RELS = ("CALLS", "INHERITS", "REFERENCES_", "INSTANTIATES")


class ScoreMap:
    """Read-only {node_id: score} backed by sorted arrays (a 250k-entry
    dict per PPR query would cost more than the iteration)."""

    __slots__ = ("ids", "scores")

    def __init__(self, ids: np.ndarray, scores: np.ndarray):
        self.ids = ids
        self.scores = scores

    def get(self, key, default=0.0):
        i = int(np.searchsorted(self.ids, int(key)))
        if i < len(self.ids) and int(self.ids[i]) == int(key):
            return float(self.scores[i])
        return default

    def __contains__(self, key) -> bool:
        i = int(np.searchsorted(self.ids, int(key)))
        return i < len(self.ids) and int(self.ids[i]) == int(key)

    def __getitem__(self, key):
        v = self.get(key, None)
        if v is None:
            raise KeyError(key)
        return v

    def __len__(self) -> int:
        return len(self.ids)

    def __bool__(self) -> bool:
        return len(self.ids) > 0

    def items(self):
        return zip(self.ids.tolist(), self.scores.tolist())

    def keys(self):
        return self.ids.tolist()


class _Graph:
    def __init__(self, src: np.ndarray, dst: np.ndarray):
        nodes = np.unique(np.concatenate([src, dst])) if len(src) else np.zeros(0, np.int64)
        self.nodes = nodes
        n = len(nodes)
        self.n = n
        if n == 0:
            self.M = None
            self.dangling = np.zeros(0, bool)
            return
        si = np.searchsorted(nodes, src)
        di = np.searchsorted(nodes, dst)
        key = np.unique(si.astype(np.int64) * n + di)          # DiGraph: no multi-edges
        si, di = key // n, key % n
        out_deg = np.bincount(si, minlength=n).astype(np.float64)
        w = 1.0 / out_deg[si]
        # column-stochastic transition: x_new[dst] += x[src] / outdeg[src]
        self.M = sp.csr_matrix((w, (di, si)), shape=(n, n))
        self.dangling = out_deg == 0

    def pagerank(self, alpha: float = 0.85, personalization: np.ndarray | None = None,
                 max_iter: int = 100, tol: float = 1e-6,
                 progress: Callable[[int], None] | None = None) -> np.ndarray:
        n = self.n
        if n == 0:
            return np.zeros(0)
        p = np.full(n, 1.0 / n) if personalization is None else personalization / personalization.sum()
        x = np.full(n, 1.0 / n)
        for it in range(max_iter):
            last = x
            dmass = float(last[self.dangling].sum())
            x = alpha * (self.M @ last + dmass * p) + (1.0 - alpha) * p
            if progress is not None:
                progress(it)
            if np.abs(x - last).sum() < n * tol:
                break
        return x


def compute_pagerank(db: GraphDB, progress: Callable[[int], None] | None = None) -> ScoreMap:
    src, dst, _c, _k = db.edge_endpoints(RANK_RELS)
    g = _Graph(src, dst)
    return ScoreMap(g.nodes, g.pagerank(progress=progress))


def write_pagerank(db: GraphDB, scores, file_rollup: bool = True) -> None:
    """Write scores to every Function / Class (0.0 for nodes without rank
    edges) and, with `file_rollup`, each File's rank as the sum of its
    symbols' ranks (files are never endpoints of the rank edges)."""
    if isinstance(scores, dict):
        ids = np.array(sorted(scores), dtype=np.int64)
        scores = ScoreMap(ids, np.array([scores[i] for i in ids.tolist()], dtype=np.float64))
    per_file: dict[str, float] = {}
    for label in ("Function", "Class"):
        d = db.node_columns(label, {"id": "n.id", "file": "n.file"})
        ids = d.get("id")
        if ids is None or len(ids) == 0:
            continue
        ids = ids.astype(np.int64)
        pos = np.searchsorted(scores.ids, ids) if len(scores.ids) else np.zeros(len(ids), np.int64)
        pos = np.clip(pos, 0, max(0, len(scores.ids) - 1))
        hit = (len(scores.ids) > 0) & (scores.ids[pos] == ids) if len(scores.ids) else np.zeros(len(ids), bool)
        vals = np.where(hit, scores.scores[pos] if len(scores.ids) else 0.0, 0.0)
        db.set_node_values(label, ids, {"pagerank": vals})
        if file_rollup:
            for f, v in zip(d["file"].tolist(), vals.tolist()):
                if v:
                    per_file[f] = per_file.get(f, 0.0) + v
    if file_rollup:
        fd = db.node_columns("File", {"id": "n.id", "path": "n.path"})
        if fd.get("id") is not None and len(fd["id"]):
            vals = np.array([per_file.get(p, 0.0) for p in fd["path"].tolist()])
            db.set_node_values("File", fd["id"].astype(np.int64), {"pagerank": vals})


# --- Personalized PageRank ------------------------------------------------


class PersonalizedRanker:
    """Caches the rank graph between queries. Build once, run PPR many times.

    Thread-safe. Invalidated by replacing the instance (a new Retriever is
    created after every reindex)."""

    def __init__(self, db: GraphDB):
        self._db = db
        self._lock = Lock()
        self._graph: _Graph | None = None

    def _ensure_graph(self) -> _Graph:
        with self._lock:
            if self._graph is None:
                src, dst, _c, _k = self._db.edge_endpoints(RANK_RELS)
                self._graph = _Graph(src, dst)
            return self._graph

    def personalized(self, focus_ids: list[int], alpha: float = 0.85) -> ScoreMap:
        """PPR with mass on focus_ids (global PR when none is in the graph)."""
        g = self._ensure_graph()
        if g.n == 0:
            return ScoreMap(np.zeros(0, np.int64), np.zeros(0))
        f = np.asarray(sorted({int(i) for i in focus_ids}), dtype=np.int64)
        hits = np.intersect1d(g.nodes, f)
        if len(hits) == 0:
            return ScoreMap(g.nodes, g.pagerank(alpha=alpha, max_iter=50, tol=1e-4))
        p = np.zeros(g.n)
        p[np.searchsorted(g.nodes, hits)] = 1.0 / len(hits)
        return ScoreMap(g.nodes, g.pagerank(alpha=alpha, personalization=p, max_iter=50, tol=1e-4))
