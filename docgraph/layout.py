"""Deterministic world layout for the graph UI (computed at index time).

Every node gets a fixed (x, y) in one global world -- the "terrain" the
tile server cuts into chunks. Hierarchical, so it stays bounded on big
graphs instead of simulating every node:

1. **Symbols inside their file.** Each File is a disc; its symbols sit on a
   golden-angle spiral around the file node, in line order (a class and its
   methods stay adjacent). Disc radius grows with sqrt(#symbols).
2. **Files inside their cluster.** Cluster = the file's Louvain community;
   files without one are grouped by top-level directory. Small clusters run
   a short collision-aware force layout over the file graph (O(F^2) per
   step, F <= FORCE_MAX_FILES); big ones place files on a spiral in BFS
   order of the file graph so connected files stay neighbours.
3. **Clusters in the world.** Same force layout over the cluster graph
   (edge weight = cross-cluster symbol edges), or a size-ordered spiral when
   there are more than FORCE_MAX_CLUSTERS clusters.

No randomness: the same inputs give the same positions. Incremental
passes do not call `compute_layout` -- surviving nodes keep their stored
positions and only new nodes are placed (`spiral_point`, `place_file`).

Pure numpy; the indexer feeds it arrays read from Kuzu.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

SPACING = 10.0          # distance between neighbouring symbol centres
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))
FORCE_MAX_FILES = 300
FORCE_MAX_CLUSTERS = 600
FILE_GAP = 1.5 * SPACING
CLUSTER_GAP = 6.0 * SPACING

KIND_FILE, KIND_CLASS, KIND_FUNCTION, KIND_VARIABLE, KIND_CLUSTER = 0, 1, 2, 3, 4


def spiral_point(i: int, c: float = SPACING * 0.62) -> tuple[float, float]:
    """Offset of the i-th point (0-based) on the golden-angle spiral."""
    r = c * math.sqrt(i + 1.0)
    a = (i + 1) * GOLDEN
    return r * math.cos(a), r * math.sin(a)


def spiral_offsets(n: int, c: float = SPACING * 0.62) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    r = c * np.sqrt(i + 1.0)
    a = (i + 1.0) * GOLDEN
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def file_radius(n_symbols: int) -> float:
    return SPACING * 0.62 * math.sqrt(n_symbols + 1.0) + SPACING * 0.8


def _hash01(s: str) -> float:
    return int(hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:8], 16) / 0xFFFFFFFF


def pseudo_cluster_key(path: str) -> str:
    parts = path.split("/")
    return parts[0] if len(parts) > 1 else "(root)"


def _pack_spiral(radii: np.ndarray, order: np.ndarray, gap: float) -> np.ndarray:
    """Place discs in `order` on an outward spiral without overlaps (greedy:
    each disc walks the spiral until it clears the placed ones). Grid-hashed
    so it stays ~linear."""
    n = len(radii)
    pos = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return pos
    cell = float(max(radii.max() * 2 + gap, 1.0))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    rmax = float(radii.max())
    t = 0.0
    step_c = max(float(np.median(radii)) + gap, 1.0) * 0.35
    for rank, i in enumerate(order.tolist()):
        ri = float(radii[i])
        if rank == 0:
            x = y = 0.0
        else:
            while True:
                t += 0.5
                rr = step_c * math.sqrt(t)
                x, y = rr * math.cos(t * GOLDEN), rr * math.sin(t * GOLDEN)
                gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
                reach = int(math.ceil((ri + rmax + gap) / cell))
                ok = True
                for dx in range(-reach, reach + 1):
                    for dy in range(-reach, reach + 1):
                        for j in grid.get((gx + dx, gy + dy), ()):
                            need = ri + float(radii[j]) + gap
                            if (pos[j, 0] - x) ** 2 + (pos[j, 1] - y) ** 2 < need * need:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    break
        pos[i] = (x, y)
        grid[(int(math.floor(x / cell)), int(math.floor(y / cell)))].append(i)
    return pos


def _force_layout(radii: np.ndarray, ei: np.ndarray, ej: np.ndarray, w: np.ndarray,
                  gap: float, iters: int = 120) -> np.ndarray:
    """Collision-aware force layout of discs (O(n^2) per step, n small).
    Deterministic: starts from the non-overlapping spiral packing."""
    n = len(radii)
    order = np.argsort(-radii, kind="stable")
    pos = _pack_spiral(radii, order, gap)
    if n <= 2:
        return pos
    wn = w / (w.max() if len(w) and w.max() > 0 else 1.0)
    need = radii[:, None] + radii[None, :] + gap
    scale = float(np.mean(radii) + gap)
    for it in range(iters):
        cool = 1.0 - it / iters
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
        np.fill_diagonal(dist, np.inf)
        # short-range repulsion that turns into hard separation on overlap
        over = np.clip(need - dist, 0.0, None)
        rep = (scale * scale / (dist * dist)) * 0.05 + over * 0.5
        f = (d / dist[..., None] * rep[..., None]).sum(1)
        if len(ei):
            dv = pos[ej] - pos[ei]
            dl = np.sqrt((dv ** 2).sum(-1)) + 1e-6
            pull = (np.clip(dl - need[ei, ej], 0.0, None) * 0.05 * wn)[:, None] * dv / dl[:, None]
            np.add.at(f, ei, pull)
            np.add.at(f, ej, -pull)
        f -= pos * 0.002  # gravity
        mag = np.sqrt((f ** 2).sum(-1)) + 1e-9
        cap = scale * 0.5 * cool + 0.05 * scale
        f *= np.minimum(1.0, cap / mag)[:, None]
        pos += f
    # final overlap removal (few passes)
    for _ in range(60):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
        np.fill_diagonal(dist, np.inf)
        over = np.clip(need - dist, 0.0, None)
        if over.max() <= 1e-3:
            break
        push = (d / dist[..., None] * (over * 0.51)[..., None]).sum(1)
        pos += push
    return pos - pos.mean(axis=0)


def _bfs_order(n: int, ei: np.ndarray, ej: np.ndarray, w: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Visit order: BFS from the heaviest node of each component, strongest
    edges first, so neighbours in the file graph land near each other."""
    adj: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for a, b, ww in zip(ei.tolist(), ej.tolist(), w.tolist()):
        adj[a].append((-ww, b))
        adj[b].append((-ww, a))
    for lst in adj:
        lst.sort()
    seen = np.zeros(n, dtype=bool)
    out: list[int] = []
    for start in np.argsort(-weight, kind="stable").tolist():
        if seen[start]:
            continue
        seen[start] = True
        dq = deque([start])
        while dq:
            u = dq.popleft()
            out.append(u)
            for _w, v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    dq.append(v)
    return np.asarray(out, dtype=np.int64)


@dataclass
class LayoutResult:
    x: np.ndarray                       # per input node
    y: np.ndarray
    cluster_keys: list = field(default_factory=list)   # layout cluster key per cluster slot
    cluster_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cluster_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cluster_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    file_cluster: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


def compute_layout(kinds: np.ndarray, file_of: np.ndarray, order_key: np.ndarray,
                   file_paths: dict[int, str], file_community: dict[int, int],
                   fe_a: np.ndarray, fe_b: np.ndarray, fe_w: np.ndarray,
                   pagerank: np.ndarray | None = None,
                   progress=None) -> LayoutResult:
    """Positions for every node.

    kinds[i]      KIND_* of node i
    file_of[i]    index of node i's File node (i itself for a File; -1 = none)
    order_key[i]  in-file order (line number)
    file_paths    {file node index: path}
    file_community{file node index: community id} (absent = no community)
    fe_a/fe_b/fe_w  aggregated file-file edges (file node indices, weight)
    """
    n = len(kinds)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    files = np.nonzero(kinds == KIND_FILE)[0]
    if len(files) == 0:
        # symbols without files (should not happen): one spiral
        off = spiral_offsets(n)
        return LayoutResult(x=off[:, 0].copy(), y=off[:, 1].copy())
    # -- 1. symbols per file ------------------------------------------------
    sym = np.nonzero((kinds != KIND_FILE) & (file_of >= 0))[0]
    order = np.lexsort((order_key[sym], file_of[sym]))
    sym = sym[order]
    fo = file_of[sym]
    starts = np.searchsorted(fo, files, side="left")
    ends = np.searchsorted(fo, files, side="right")
    counts = (ends - starts)
    radius = np.array([file_radius(int(c)) for c in counts], dtype=np.float64)
    local = np.zeros((len(sym), 2), dtype=np.float64)
    for fi in range(len(files)):
        s, e = int(starts[fi]), int(ends[fi])
        if e > s:
            local[s:e] = spiral_offsets(e - s)
    if progress:
        progress("layout: files")
    # -- 2. layout clusters --------------------------------------------------
    fidx_of = {int(f): k for k, f in enumerate(files.tolist())}
    keys: list = []
    key_slot: dict = {}
    fclu = np.zeros(len(files), dtype=np.int64)
    for k, f in enumerate(files.tolist()):
        c = file_community.get(int(f))
        key = ("c", int(c)) if c is not None else ("d", pseudo_cluster_key(file_paths.get(int(f), "")))
        if key not in key_slot:
            key_slot[key] = len(keys)
            keys.append(key)
        fclu[k] = key_slot[key]
    # file-file edges in file-slot space
    if len(fe_a):
        fa = np.array([fidx_of.get(int(a), -1) for a in fe_a.tolist()], dtype=np.int64)
        fb = np.array([fidx_of.get(int(b), -1) for b in fe_b.tolist()], dtype=np.int64)
        ok = (fa >= 0) & (fb >= 0) & (fa != fb)
        fa, fb, fw = fa[ok], fb[ok], np.asarray(fe_w, dtype=np.float64)[ok]
    else:
        fa = fb = np.zeros(0, np.int64)
        fw = np.zeros(0)
    fpos = np.zeros((len(files), 2), dtype=np.float64)
    c_r = np.zeros(len(keys), dtype=np.float64)
    by_c: dict[int, list[int]] = defaultdict(list)
    for k in range(len(files)):
        by_c[int(fclu[k])].append(k)
    same = fclu[fa] == fclu[fb] if len(fa) else np.zeros(0, bool)
    ia, ib, iw = fa[same], fb[same], fw[same]
    intra: dict[int, list[int]] = defaultdict(list)
    for e_i, c in enumerate(fclu[ia].tolist() if len(ia) else []):
        intra[c].append(e_i)
    pr = pagerank if pagerank is not None else np.zeros(n)
    # deterministic in-cluster member order: by path
    for c in range(len(keys)):
        mem = sorted(by_c[c], key=lambda k: file_paths.get(int(files[k]), ""))
        m = np.asarray(mem, dtype=np.int64)
        loc = {int(k): j for j, k in enumerate(m.tolist())}
        es = intra.get(c, [])
        ea = np.array([loc[int(ia[e])] for e in es], dtype=np.int64)
        eb = np.array([loc[int(ib[e])] for e in es], dtype=np.int64)
        ew = np.array([float(iw[e]) for e in es], dtype=np.float64)
        rr = radius[m]
        if len(m) <= FORCE_MAX_FILES:
            p = _force_layout(rr, ea, eb, ew, FILE_GAP, iters=80 if len(m) > 40 else 120)
        else:
            weight = counts[m].astype(np.float64) + np.asarray([pr[int(files[k])] for k in m])
            bfs = _bfs_order(len(m), ea, eb, ew, weight)
            p = _pack_spiral(rr, bfs, FILE_GAP)
            p -= p.mean(axis=0)
        fpos[m] = p
        c_r[c] = float(np.max(np.sqrt((p ** 2).sum(1)) + rr)) if len(m) else SPACING
    if progress:
        progress("layout: clusters")
    # -- 3. clusters in the world -------------------------------------------
    if len(fa):
        ca, cb = fclu[fa], fclu[fb]
        cross = ca != cb
        ca, cb, cw = ca[cross], cb[cross], fw[cross]
        lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
        if len(lo):
            key_arr = lo * len(keys) + hi
            uk, inv = np.unique(key_arr, return_inverse=True)
            ww = np.bincount(inv, weights=cw)
            ca, cb, cw = uk // len(keys), uk % len(keys), ww
        else:
            ca = cb = np.zeros(0, np.int64)
            cw = np.zeros(0)
    else:
        ca = cb = np.zeros(0, np.int64)
        cw = np.zeros(0)
    if len(keys) <= FORCE_MAX_CLUSTERS:
        cpos = _force_layout(c_r, ca, cb, cw, CLUSTER_GAP, iters=150)
    else:
        cpos = _pack_spiral(c_r, np.argsort(-c_r, kind="stable"), CLUSTER_GAP)
    # -- compose ------------------------------------------------------------
    world_f = fpos + cpos[fclu]
    x[files] = world_f[:, 0]
    y[files] = world_f[:, 1]
    fslot = np.array([fidx_of[int(f)] for f in fo.tolist()], dtype=np.int64) if len(fo) else np.zeros(0, np.int64)
    if len(sym):
        x[sym] = world_f[fslot, 0] + local[:, 0]
        y[sym] = world_f[fslot, 1] + local[:, 1]
    # orphans (no file): spiral around the origin, outside the world
    orphan = np.nonzero((kinds != KIND_FILE) & (file_of < 0))[0]
    if len(orphan):
        ext = float(np.max(np.sqrt(x ** 2 + y ** 2))) + CLUSTER_GAP * 4
        off = spiral_offsets(len(orphan))
        x[orphan] = off[:, 0] + ext
        y[orphan] = off[:, 1]
    return LayoutResult(x=x, y=y, cluster_keys=keys, cluster_x=cpos[:, 0].copy(),
                        cluster_y=cpos[:, 1].copy(), cluster_r=c_r, file_cluster=fclu)


def place_file(anchor: tuple[float, float] | None, path: str,
               extent: float) -> tuple[float, float]:
    """World position for a file that did not exist at the last layout:
    near `anchor` (its neighbours' / cluster's centre) or on the world rim,
    offset deterministically by the path hash."""
    h = _hash01(path)
    a = h * 2 * math.pi
    if anchor is not None:
        r = SPACING * (6.0 + 10.0 * _hash01(path + "#r"))
        return anchor[0] + r * math.cos(a), anchor[1] + r * math.sin(a)
    r = extent + CLUSTER_GAP * (1.0 + 2.0 * _hash01(path + "#r"))
    return r * math.cos(a), r * math.sin(a)
