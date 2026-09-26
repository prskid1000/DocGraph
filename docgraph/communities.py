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


def _name_for(members: list[dict]) -> tuple[str, list[str], list[str]]:
    from docgraph.resolve import COMMON_METHOD_NAMES
    syms = sorted((m for m in members if m["label"] in ("Function", "Class")),
                  key=lambda m: (m["label"] != "Class",
                                 m["name"] in COMMON_METHOD_NAMES or len(m["name"]) < 4
                                 or m["name"].startswith("__"),
                                 -(m.get("pagerank") or 0.0), m["name"]))
    files = [m["file"] for m in members if m.get("file")]
    file_counts = Counter(files)
    top_files = [f for f, _ in file_counts.most_common(5)]
    top = [m["name"] for m in syms[:5]]
    if file_counts:
        main_file, n_main = file_counts.most_common(1)[0]
        if n_main >= 0.6 * len(files) or len(file_counts) == 1:
            area = _stem(main_file)
        else:
            area = _common_dir(list(file_counts)) or _stem(main_file)
            area = area.rsplit("/", 1)[-1] if area else _stem(main_file)
    else:
        area = "misc"
    label = ", ".join(top[:2]) if top else ""
    name = f"{area}: {label}" if label else area
    return name, top, top_files


def detect(
    nodes: dict[int, dict],
    edges: list[tuple[int, int, float]],
    min_size: int = 2,
    resolution: float = 1.0,
    seed: int = 42,
) -> list[Community]:
    """nodes: {id: {label, name, file, pagerank}}; edges: (a, b, weight).
    Returns communities sorted by size (largest first)."""
    g = nx.Graph()
    for nid in nodes:
        g.add_node(nid)
    for a, b, w in edges:
        if a == b or a not in nodes or b not in nodes:
            continue
        if g.has_edge(a, b):
            g[a][b]["weight"] += w
        else:
            g.add_edge(a, b, weight=w)
    # Drop isolated nodes before Louvain (they would each be a singleton).
    isolated = [n for n in g.nodes if g.degree(n) == 0]
    g.remove_nodes_from(isolated)
    if g.number_of_nodes() == 0:
        return []

    fold: dict[int, int] = {}
    work = g
    if g.number_of_nodes() > MAX_FINE_NODES:
        # Coarsen: fold every symbol into its File node, cluster files.
        file_id_by_path = {v["file"]: k for k, v in nodes.items() if v["label"] == "File"}
        for nid in g.nodes:
            v = nodes[nid]
            if v["label"] != "File":
                fid = file_id_by_path.get(v.get("file") or "")
                if fid is not None:
                    fold[nid] = fid
        work = nx.Graph()
        for a, b, d in g.edges(data=True):
            fa, fb = fold.get(a, a), fold.get(b, b)
            if fa == fb:
                continue
            if work.has_edge(fa, fb):
                work[fa][fb]["weight"] += d["weight"]
            else:
                work.add_edge(fa, fb, weight=d["weight"])
        if work.number_of_nodes() == 0:
            return []

    parts = nx.community.louvain_communities(work, weight="weight",
                                             resolution=resolution, seed=seed)
    if fold:
        expanded: list[set[int]] = []
        members_of: dict[int, set[int]] = defaultdict(set)
        for sym, fid in fold.items():
            members_of[fid].add(sym)
        for part in parts:
            s = set(part)
            for fid in part:
                s |= members_of.get(fid, set())
            expanded.append(s)
        parts = expanded

    # Cohesion needs per-node community membership over the fine graph.
    comm_of: dict[int, int] = {}
    for i, part in enumerate(parts):
        for n in part:
            comm_of[n] = i
    internal = defaultdict(float)
    cut = defaultdict(float)
    for a, b, d in g.edges(data=True):
        ca, cb = comm_of.get(a), comm_of.get(b)
        w = d.get("weight", 1.0)
        if ca is None or cb is None:
            continue
        if ca == cb:
            internal[ca] += w
        else:
            cut[ca] += w
            cut[cb] += w

    out: list[Community] = []
    for i, part in enumerate(parts):
        if len(part) < min_size:
            continue
        members = [dict(nodes[n], id=n) for n in part if n in nodes]
        name, top, files = _name_for(members)
        tot = internal[i] + cut[i]
        coh = (internal[i] / tot) if tot > 0 else 0.0
        pr = sum((m.get("pagerank") or 0.0) for m in members)
        out.append(Community(
            index=i, name=name, members=sorted(n for n in part if n in nodes),
            size=len(members), cohesion=round(coh, 4),
            top_members=top, files=files, pagerank=pr,
        ))
    out.sort(key=lambda c: (-c.size, c.name))
    # Disambiguate duplicate names
    seen: Counter = Counter()
    for c in out:
        seen[c.name] += 1
        if seen[c.name] > 1:
            c.name = f"{c.name} #{seen[c.name]}"
    return out
