"""Graph algorithms for health() on numpy / scipy arrays.

networkx needs a Python object per node and edge; on a 200k-function repo
building the graph and walking it costs seconds. These work on CSR arrays:

* sampled_betweenness  Brandes (1 BFS per sampled source, level-synchronous
                       and vectorised), same sampling and normalisation as
                       networkx.betweenness_centrality(G, k, seed, normalized)
* short_cycles         strongly connected components via scipy, then the
                       shortest cycle through the nodes of each non-trivial
                       component (BFS, length-bounded) -- a bounded sample of
                       real cycles, shortest first
"""
from __future__ import annotations

import random

import numpy as np


def _csr(n: int, a: np.ndarray, b: np.ndarray):
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, a + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, b.astype(np.int64)


def _gather(indptr: np.ndarray, indices: np.ndarray, frontier: np.ndarray):
    """(source per neighbour, neighbour) of every edge out of `frontier`."""
    starts = indptr[frontier]
    counts = indptr[frontier + 1] - starts
    total = int(counts.sum())
    if total == 0:
        z = np.zeros(0, np.int64)
        return z, z
    src = np.repeat(frontier, counts)
    offs = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    idx = offs + np.arange(total, dtype=np.int64)
    return src, indices[idx]


def sampled_betweenness(nodes: list[int], ua: np.ndarray, ub: np.ndarray, k: int, seed: int = 7) -> dict[int, float]:
    """Betweenness of an undirected graph (edges ua-ub over node ids),
    estimated from k sampled sources exactly like networkx (random.Random(
    seed).sample over the node list in first-appearance order)."""
    n = len(nodes)
    if n <= 2 or k <= 0:
        return {}
    pos = {v: i for i, v in enumerate(nodes)}
    a = np.fromiter((pos[int(x)] for x in ua.tolist()), dtype=np.int64, count=len(ua))
    b = np.fromiter((pos[int(x)] for x in ub.tolist()), dtype=np.int64, count=len(ub))
    keep = a != b
    a, b = a[keep], b[keep]
    # networkx.Graph collapses parallel edges
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * n + hi)
    lo, hi = key // n, key % n
    indptr, indices = _csr(n, np.concatenate([lo, hi]), np.concatenate([hi, lo]))
    rng = random.Random(seed)
    sources = rng.sample(list(range(n)), min(k, n))
    bc = np.zeros(n, dtype=np.float64)
    for s in sources:
        dist = np.full(n, -1, dtype=np.int64)
        sigma = np.zeros(n, dtype=np.float64)
        dist[s] = 0
        sigma[s] = 1.0
        frontier = np.array([s], dtype=np.int64)
        levels = []          # per level: (src, dst) of the shortest-path DAG edges
        d = 0
        while len(frontier):
            src, nb = _gather(indptr, indices, frontier)
            if not len(nb):
                break
            new = dist[nb] == -1
            if new.any():
                dist[np.unique(nb[new])] = d + 1
            on = dist[nb] == d + 1
            src, nb = src[on], nb[on]
            np.add.at(sigma, nb, sigma[src])
            levels.append((src, nb))
            frontier = np.unique(nb)
            d += 1
        delta = np.zeros(n, dtype=np.float64)
        for src, nb in reversed(levels):
            np.add.at(delta, src, sigma[src] / sigma[nb] * (1.0 + delta[nb]))
        delta[s] = 0.0
        bc += delta
    # networkx._rescale for normalized, undirected, sampled (k)
    scale = 1.0 / ((n - 1) * (n - 2))
    scale *= n / len(sources)
    bc *= scale
    return {nodes[i]: float(bc[i]) for i in range(n)}


def short_cycles(a: list[str], b: list[str], limit: int, max_len: int = 6) -> list[list[str]]:
    """Up to `limit` distinct directed cycles of length <= max_len,
    shortest first: the shortest cycle through each node of each
    non-trivial strongly connected component."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    names = sorted(set(a) | set(b))
    if not names:
        return []
    ix = {v: i for i, v in enumerate(names)}
    n = len(names)
    ai = np.fromiter((ix[x] for x in a), dtype=np.int64, count=len(a))
    bi = np.fromiter((ix[x] for x in b), dtype=np.int64, count=len(b))
    m = csr_matrix((np.ones(len(ai), dtype=np.int8), (ai, bi)), shape=(n, n))
    _nc, lab = connected_components(m, directed=True, connection="strong")
    sizes = np.bincount(lab)
    big = np.nonzero(sizes > 1)[0]
    if not len(big):
        return []
    indptr, indices = m.indptr.astype(np.int64), m.indices.astype(np.int64)
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    comps = sorted(big.tolist(), key=lambda c: (int(sizes[c]), c))
    for c in comps:
        members = np.nonzero(lab == c)[0]
        in_comp = lab == c
        for s in members.tolist():
            # BFS inside the component for the shortest path back to s
            parent = {s: -1}
            frontier = [s]
            found = None
            for _depth in range(max_len):
                nxt = []
                for u in frontier:
                    for v in indices[indptr[u]:indptr[u + 1]].tolist():
                        if not in_comp[v]:
                            continue
                        if v == s:
                            found = u
                            break
                        if v not in parent:
                            parent[v] = u
                            nxt.append(v)
                    if found is not None:
                        break
                if found is not None or not nxt:
                    break
                frontier = nxt
            if found is None:
                continue
            path = []
            u = found
            while u != -1:
                path.append(u)
                u = parent[u]
            path.reverse()
            cyc = [names[i] for i in path]
            j = cyc.index(min(cyc))
            key = tuple(cyc[j:] + cyc[:j])
            if key in seen:
                continue
            seen.add(key)
            out.append(list(key))
            if len(out) >= limit:
                return sorted(out, key=lambda x: (len(x), x))
    return sorted(out, key=lambda x: (len(x), x))
