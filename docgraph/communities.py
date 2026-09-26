"""Server-side community detection.

Louvain modularity (networkx's built-in `louvain_communities`, pure
Python -- no new dependency) over an undirected weighted graph of
Files, Classes and Functions:

    CALLS / INSTANTIATES / INHERITS   weight = edge confidence (default 1)
    CONTAINS (File->entity, Class->method)   weight 0.5
    IMPORTS (File->File)                     weight 0.3

The File/CONTAINS links keep clusters file-coherent (two helpers in one
module with no call between them still land together), and Files get a
membership of their own so the UI can colour a files-only view.

Each community gets an auto-generated name (common directory + its top
PageRank members), a cohesion score (internal weight / (internal + cut)),
and its top members / files. Deterministic: fixed seed, sorted output.
"""
from __future__ import annotations

import os
import posixpath
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

# Louvain in pure Python is roughly linear but not free; beyond this many
# nodes the index pass would stall for minutes, so we coarsen instead
# (functions fold into their file). Tuned on a ~40k-node repo (~20 s).
MAX_FINE_NODES = 60_000


@dataclass
class Community:
    index: int
    name: str
    members: list[int]
    size: int
    cohesion: float
    top_members: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    pagerank: float = 0.0


def _common_dir(files: list[str]) -> str:
    dirs = [posixpath.dirname(f) for f in files if f]
    if not dirs:
        return ""
    try:
        common = os.path.commonpath(dirs) if all(dirs) else ""
    except ValueError:
        common = ""
    return common.replace("\\", "/")


def _stem(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _name_for(members: list[dict], file_counts: Counter | None = None
              ) -> tuple[str, list[str], list[str]]:
    from docgraph.resolve import COMMON_METHOD_NAMES
    syms = sorted((m for m in members if m["label"] in ("Function", "Class")),
                  key=lambda m: (m["label"] != "Class",
                                 m["name"] in COMMON_METHOD_NAMES or len(m["name"]) < 4
                                 or m["name"].startswith("__"),
                                 -(m.get("pagerank") or 0.0), m["name"]))
    if file_counts is None:
        file_counts = Counter(m["file"] for m in members if m.get("file"))
    n_files_total = sum(file_counts.values())
    top_files = [f for f, _ in file_counts.most_common(5)]
    top = [m["name"] for m in syms[:5]]
    if file_counts:
        main_file, n_main = file_counts.most_common(1)[0]
        if n_main >= 0.6 * n_files_total or len(file_counts) == 1:
            area = _stem(main_file)
        else:
            area = _common_dir(list(file_counts)) or _stem(main_file)
            area = area.rsplit("/", 1)[-1] if area else _stem(main_file)
    else:
        area = "misc"
    label = ", ".join(top[:2]) if top else ""
    name = f"{area}: {label}" if label else area
    return name, top, top_files


# Above MAX_FINE_NODES active nodes, symbols fold into their File; above
# MAX_FOLD_NODES folded nodes, files fold into directories (the deepest
# directory level that fits). networkx's Louvain then runs on at most
# MAX_FOLD_NODES nodes, and all aggregation is numpy.
MAX_FOLD_NODES = 60_000


def detect(
    nodes: dict[int, dict],
    edges: list[tuple[int, int, float]],
    min_size: int = 2,
    resolution: float = 1.0,
    seed: int = 42,
) -> list[Community]:
    """nodes: {id: {label, name, file, pagerank}}; edges: (a, b, weight).
    Returns communities sorted by size (largest first)."""
    ids = np.array(sorted(nodes), dtype=np.int64)
    info = [nodes[int(i)] for i in ids.tolist()]
    ea = np.array([e[0] for e in edges], dtype=np.int64)
    eb = np.array([e[1] for e in edges], dtype=np.int64)
    ew = np.array([e[2] for e in edges], dtype=np.float64)
    return detect_arrays(ids, [m["label"] for m in info], [m.get("name") or "" for m in info],
                         [m.get("file") or "" for m in info],
                         np.array([float(m.get("pagerank") or 0.0) for m in info]),
                         ea, eb, ew, min_size=min_size, resolution=resolution, seed=seed)


def detect_arrays(ids: np.ndarray, labels: list[str], names: list[str], files: list[str],
                  pagerank: np.ndarray, ea: np.ndarray, eb: np.ndarray, ew: np.ndarray,
                  min_size: int = 2, resolution: float = 1.0, seed: int = 42,
                  progress=None) -> list[Community]:
    """Array form of `detect`: `ids` sorted ascending; per-node label, name,
    file and pagerank; undirected weighted edges given by node id."""
    n = len(ids)
    if n == 0 or len(ea) == 0:
        return []
    ia = np.searchsorted(ids, ea)
    ib = np.searchsorted(ids, eb)
    ok = (ia < n) & (ib < n)
    ia, ib, w, ea2, eb2 = ia[ok], ib[ok], ew[ok], ea[ok], eb[ok]
    ok = (ids[ia] == ea2) & (ids[ib] == eb2) & (ia != ib)
    ia, ib, w = ia[ok], ib[ok], w[ok]
    if len(ia) == 0:
        return []
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    key, inv = np.unique(lo * n + hi, return_inverse=True)
    w = np.bincount(inv, weights=w)
    lo, hi = key // n, key % n
    active = np.zeros(n, dtype=bool)
    active[lo] = True
    active[hi] = True
    # ---- folding tiers ----
    group = np.arange(n, dtype=np.int64)          # fine node -> work node
    if int(active.sum()) > MAX_FINE_NODES:
        file_node = {f: i for i, (lab, f) in enumerate(zip(labels, files)) if lab == "File"}
        for i in range(n):
            if labels[i] != "File":
                fi = file_node.get(files[i])
                if fi is not None:
                    group[i] = fi
        if len(np.unique(group[active])) > MAX_FOLD_NODES:
            depth = max((f.count("/") for f in files), default=0)
            while True:
                dir_slot: dict[str, int] = {}
                g2 = np.empty(n, dtype=np.int64)
                for i in range(n):
                    parts_ = files[i].split("/")[:-1]
                    d = "/".join(parts_[:depth])
                    g2[i] = dir_slot.setdefault(d, len(dir_slot))
                if len(np.unique(g2[active])) <= MAX_FOLD_NODES or depth == 0:
                    group = g2
                    break
                depth -= 1
    if progress:
        progress("communities: louvain")
    ga, gb = group[lo], group[hi]
    keep = ga != gb
    g = nx.Graph()
    g.add_nodes_from(np.unique(group[active]).tolist())
    if keep.any():
        base = int(group.max()) + 1
        k2, inv2 = np.unique(np.minimum(ga[keep], gb[keep]) * base + np.maximum(ga[keep], gb[keep]),
                             return_inverse=True)
        w2 = np.bincount(inv2, weights=w[keep])
        g.add_weighted_edges_from(zip((k2 // base).tolist(), (k2 % base).tolist(), w2.tolist()))
    g.remove_nodes_from([v for v in list(g.nodes) if g.degree(v) == 0])
    if g.number_of_nodes() == 0:
        return []
    parts = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
    part_of = np.full(int(group.max()) + 1, -1, dtype=np.int64)
    for pi, part in enumerate(parts):
        for v in part:
            part_of[int(v)] = pi
    comm = np.where(active, part_of[group], -1)
    # ---- cohesion over the fine graph ----
    ca, cb = comm[lo], comm[hi]
    valid = (ca >= 0) & (cb >= 0)
    npart = len(parts)
    same = valid & (ca == cb)
    internal = np.bincount(ca[same], weights=w[same], minlength=npart)
    cross = valid & (ca != cb)
    cut = (np.bincount(ca[cross], weights=w[cross], minlength=npart)
           + np.bincount(cb[cross], weights=w[cross], minlength=npart))
    order = np.lexsort((-pagerank, comm))
    comm_sorted = comm[order]
    bounds = np.searchsorted(comm_sorted, np.arange(npart + 1))
    out: list[Community] = []
    for pi in range(npart):
        mem = order[bounds[pi]:bounds[pi + 1]]
        if len(mem) < min_size:
            continue
        fc = Counter(files[int(i)] for i in mem.tolist() if files[int(i)])
        head = [{"label": labels[int(i)], "name": names[int(i)], "file": files[int(i)],
                 "pagerank": float(pagerank[int(i)])} for i in mem[:300].tolist()]
        name, top, top_files = _name_for(head, fc)
        tot = internal[pi] + cut[pi]
        coh = (internal[pi] / tot) if tot > 0 else 0.0
        out.append(Community(
            index=pi, name=name, members=sorted(int(ids[i]) for i in mem.tolist()),
            size=len(mem), cohesion=round(float(coh), 4), top_members=top, files=top_files,
            pagerank=float(pagerank[mem].sum()),
        ))
    out.sort(key=lambda c: (-c.size, c.name))
    seen: Counter = Counter()
    for c in out:
        seen[c.name] += 1
        if seen[c.name] > 1:
            c.name = f"{c.name} #{seen[c.name]}"
    return out
