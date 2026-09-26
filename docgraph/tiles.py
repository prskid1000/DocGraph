"""Streamed graph tiles ("chunks") for the web UI.

The world layout (x, y per node, stored in Kuzu) is cut into a quadtree of
square tiles, level 0 = the whole world ... level MAX_LEVEL = finest. Three
levels of detail share one encoding:

    sym   real File / Class / Function / Variable nodes and real edges
    file  File nodes (size = #symbols) with file-file edges aggregated per kind
    clu   cluster super-nodes (Louvain communities + directory groups) with
          cluster-cluster edges aggregated per kind

Each LOD is stored once, sorted by the Morton (Z-order) code of the node's
quantised position, so ANY quadtree tile is one contiguous code range -- no
per-level copies. A tile request picks, per tile, the finest LOD whose node
count fits the budget (the finest level always serves `sym`, whatever the
count: nothing is ever truncated).

Edges are stored once, keyed by the tile of their lower-id endpoint (the
"owner"); a second index keyed by the other endpoint lets a tile also serve
edges it only receives, so an edge is drawn when either endpoint's tile is
loaded. The client de-duplicates by edge id.

The sidecar (.docgraph/tiles/) is derived data: rebuilt from Kuzu at index
time (`build`), loaded by the host (`TileStore`). Kuzu stays the source of
truth for positions.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MAX_LEVEL = 16            # 16 bits per axis
GRID = 1 << MAX_LEVEL
TILE_BUDGET = 1500        # nodes per tile before a coarser LOD is served
NAME_BUDGET = 400         # labelled nodes per symbol tile (top PageRank)
MAGIC = 0x31544744        # "DGT1"
VERSION = 3                # wire format of a tile (the npz may carry extra arrays)
# File / cluster views aggregate edges; a tile carries at most this many per
# node, heaviest first (the header says how many there were). Symbol tiles
# always carry every edge.
AGG_EDGES_PER_NODE = 6
AGG_EDGES_MIN = 2000
LODS = ("sym", "file", "clu")
NODE_KINDS = ["File", "Class", "Function", "Variable", "Cluster"]
EDGE_KINDS = ["CONTAINS", "CALLS", "IMPORTS", "IMPORTS_SYMBOL", "INHERITS",
              "IMPLEMENTS", "OVERRIDES", "REFERENCES_", "INSTANTIATES",
              "DECORATED_BY", "RETURNS", "SIMILAR_TO", "TESTS",
              "CO_CHANGED_WITH", "LINKS_TO"]
EDGE_KIND_ID = {k: i for i, k in enumerate(EDGE_KINDS)}
KIND_ID = {k: i for i, k in enumerate(NODE_KINDS)}
FLAG_TEST = 1


# ---- Morton codes -------------------------------------------------------------

def _part1by1(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint32) & np.uint32(0x0000FFFF)
    v = (v | (v << np.uint32(8))) & np.uint32(0x00FF00FF)
    v = (v | (v << np.uint32(4))) & np.uint32(0x0F0F0F0F)
    v = (v | (v << np.uint32(2))) & np.uint32(0x33333333)
    v = (v | (v << np.uint32(1))) & np.uint32(0x55555555)
    return v


def morton(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return (_part1by1(gx) | (_part1by1(gy) << np.uint32(1))).astype(np.uint32)


def tile_range(level: int, tx: int, ty: int) -> tuple[int, int]:
    """[lo, hi) Morton-code range of tile (tx, ty) at `level`."""
    shift = MAX_LEVEL - level
    lo = int(morton(np.array([tx << shift]), np.array([ty << shift]))[0])
    return lo, lo + (1 << (2 * shift))


# ---- building ------------------------------------------------------------------

@dataclass
class TileSource:
    """Arrays read from Kuzu (db.tile_source)."""
    ids: np.ndarray          # int64, every File/Class/Function/Variable
    kinds: np.ndarray        # uint8 KIND_ID
    x: np.ndarray
    y: np.ndarray
    pagerank: np.ndarray
    file_row: np.ndarray     # row of the node's File (-1 none); a File points at itself
    community: np.ndarray    # int64 community id (-1 none)
    names: list[str]
    flags: np.ndarray        # uint8
    edge_a: np.ndarray       # node ids
    edge_b: np.ndarray
    edge_kind: np.ndarray    # uint8 EDGE_KIND_ID
    edge_conf: np.ndarray    # float32 (1.0 when unknown)
    clusters: dict           # community id -> {name, x, y, r}


def _quantise(x: np.ndarray, y: np.ndarray, bbox: tuple[float, float, float]) -> np.ndarray:
    minx, miny, side = bbox
    gx = np.clip(((x - minx) / side * GRID).astype(np.int64), 0, GRID - 1)
    gy = np.clip(((y - miny) / side * GRID).astype(np.int64), 0, GRID - 1)
    return morton(gx, gy)


def _name_blob(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = [n.encode("utf-8", "replace") for n in names]
    off = np.zeros(len(enc) + 1, dtype=np.uint32)
    if enc:
        off[1:] = np.cumsum([len(e) for e in enc], dtype=np.uint64).astype(np.uint32)
    blob = np.frombuffer(b"".join(enc), dtype=np.uint8) if enc else np.zeros(0, np.uint8)
    return off, blob


def _lod_edges(prefix: str, ids_sorted, code, ra, rb, ek, ec, ew) -> dict[str, np.ndarray]:
    """Edge arrays of one LOD from endpoint rows in the Morton-sorted node
    order: owner = lower-id endpoint, sorted by owner code, plus the
    receiver index sorted by the other endpoint's code."""
    out: dict[str, np.ndarray] = {}
    ra = np.asarray(ra, dtype=np.int64)
    rb = np.asarray(rb, dtype=np.int64)
    ida, idb = ids_sorted[ra], ids_sorted[rb]
    own_a = ida <= idb
    owner = np.where(own_a, ra, rb)
    other = np.where(own_a, rb, ra)
    ocode = code[owner]
    eo = np.argsort(ocode, kind="stable")
    out[f"{prefix}_ea"] = ra[eo].astype(np.uint32)       # edge source row (direction kept)
    out[f"{prefix}_eb"] = rb[eo].astype(np.uint32)       # edge target row
    out[f"{prefix}_ek"] = np.asarray(ek, dtype=np.uint8)[eo]
    ecf = np.asarray(ec)
    out[f"{prefix}_ec"] = (ecf[eo] if ecf.dtype == np.uint8 else
                           np.clip(ecf.astype(np.float32)[eo] * 255.0, 0, 255).astype(np.uint8))
    out[f"{prefix}_ew"] = np.asarray(ew, dtype=np.uint32)[eo]
    out[f"{prefix}_ecode"] = ocode[eo]
    rcode = code[other[eo]]
    ro = np.argsort(rcode, kind="stable")
    out[f"{prefix}_rcode"] = rcode[ro]
    out[f"{prefix}_ridx"] = ro.astype(np.uint32)
    return out


def _lod_arrays(prefix: str, ids, kinds, x, y, pr, size, cluster, flags, names,
                ea, eb, ek, ec, ew, bbox, file_ids=None) -> dict[str, np.ndarray]:
    """Sort one LOD by Morton code and lay out its node + edge arrays.
    ea/eb are node ROWS (pre-sort); returned edges are in sorted-row space."""
    code = _quantise(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), bbox)
    order = np.argsort(code, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out: dict[str, np.ndarray] = {
        f"{prefix}_id": np.asarray(ids, dtype=np.int64)[order],
        f"{prefix}_x": np.asarray(x, dtype=np.float32)[order],
        f"{prefix}_y": np.asarray(y, dtype=np.float32)[order],
        f"{prefix}_kind": np.asarray(kinds, dtype=np.uint8)[order],
        f"{prefix}_pr": np.asarray(pr, dtype=np.float32)[order],
        f"{prefix}_size": np.asarray(size, dtype=np.uint32)[order],
        f"{prefix}_cluster": np.asarray(cluster, dtype=np.int64)[order],
        f"{prefix}_flags": np.asarray(flags, dtype=np.uint8)[order],
        f"{prefix}_code": code[order],
    }
    if file_ids is not None:
        out[f"{prefix}_file"] = np.asarray(file_ids, dtype=np.int64)[order]
    if isinstance(names, tuple):                      # (off, blob) already encoded
        off, blob = _gather_names(names[0], names[1], order)
    else:
        off, blob = _name_blob([names[i] for i in order.tolist()])
    out[f"{prefix}_name_off"], out[f"{prefix}_name_blob"] = off, blob
    # id lookup (for locate)
    ids_sorted = out[f"{prefix}_id"]
    lk = np.argsort(ids_sorted, kind="stable")
    out[f"{prefix}_id_sorted"] = ids_sorted[lk]
    out[f"{prefix}_id_row"] = lk.astype(np.int64)
    if len(order):
        ra = inv[np.asarray(ea, dtype=np.int64)]
        rb = inv[np.asarray(eb, dtype=np.int64)]
    else:
        ra = rb = np.zeros(0, np.int64)
    out.update(_lod_edges(prefix, ids_sorted, out[f"{prefix}_code"], ra, rb, ek, ec, ew))
    return out


def _gather_names(off: np.ndarray, blob: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, blob) of the names of `rows`, in that order (vectorised)."""
    rows = np.asarray(rows, dtype=np.int64)
    starts = off[rows].astype(np.int64)
    lens = (off[rows + 1].astype(np.int64) - starts)
    noff = np.zeros(len(rows) + 1, dtype=np.uint32)
    if len(rows):
        noff[1:] = np.cumsum(lens).astype(np.uint32)
    total = int(noff[-1]) if len(rows) else 0
    if total == 0:
        return noff, np.zeros(0, np.uint8)
    idx = np.repeat(starts - noff[:-1].astype(np.int64), lens) + np.arange(total, dtype=np.int64)
    return noff, blob[idx]


def _aggregate(ga: np.ndarray, gb: np.ndarray, kind: np.ndarray, conf: np.ndarray,
               weight: np.ndarray, n_groups: int) -> tuple[np.ndarray, ...]:
    """Collapse (group_a, group_b, kind) triples: weight summed, conf max.
    Self-loops (ga == gb) are dropped. Direction is kept."""
    keep = ga != gb
    ga, gb, kind, conf, weight = ga[keep], gb[keep], kind[keep], conf[keep], weight[keep]
    if len(ga) == 0:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.uint8), np.zeros(0, np.float32), np.zeros(0, np.uint32)
    key = (ga.astype(np.int64) * n_groups + gb.astype(np.int64)) * 32 + kind.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=weight.astype(np.float64)).astype(np.uint32)
    cmax = np.zeros(len(uk), dtype=np.float32)
    np.maximum.at(cmax, inv, conf.astype(np.float32))
    k = (uk % 32).astype(np.uint8)
    pair = uk // 32
    return pair // n_groups, pair % n_groups, k, cmax, w


def _world_bbox(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) == 0:
        return (-1.0, -1.0, 2.0)
    minx, maxx = float(x.min()), float(x.max())
    miny, maxy = float(y.min()), float(y.max())
    side = max(maxx - minx, maxy - miny, 1.0)
    pad = side * 0.125 + 50.0      # room for nodes placed by later incrementals
    return (minx - pad, miny - pad, side + 2 * pad)


def _coarse_lods(arrays: dict, ftab: dict, fedges: tuple, cluster_names: dict[int, str], bbox) -> dict:
    """file + clu LODs from the file table ({id, x, y, pr, size, cluster,
    names(off, blob)}) and the aggregated file edges (rows a, b, kind,
    conf 0..1, weight)."""
    ga, gb, gk, gc, gw = fedges
    nf = len(ftab["id"])
    arrays.update(_lod_arrays("file", ftab["id"], np.zeros(nf, np.uint8), ftab["x"], ftab["y"],
                              ftab["pr"], ftab["size"], ftab["cluster"], np.zeros(nf, np.uint8),
                              ftab["names"], ga, gb, gk, gc, gw, bbox))
    cids = np.unique(ftab["cluster"]) if nf else np.zeros(0, np.int64)
    fclu = np.searchsorted(cids, ftab["cluster"]).astype(np.int64) if nf else np.zeros(0, np.int64)
    size = np.asarray(ftab["size"], dtype=np.float64)
    fx = np.asarray(ftab["x"], dtype=np.float64)
    fy = np.asarray(ftab["y"], dtype=np.float64)
    ccount = np.bincount(fclu, weights=size, minlength=len(cids))
    cx = np.bincount(fclu, weights=fx * size, minlength=len(cids)) / np.maximum(ccount, 1)
    cy = np.bincount(fclu, weights=fy * size, minlength=len(cids)) / np.maximum(ccount, 1)
    cpr = np.bincount(fclu, weights=np.asarray(ftab["pr"], dtype=np.float64), minlength=len(cids))
    cnames = [str(cluster_names.get(int(c)) or f"#{c}") for c in cids.tolist()]
    cr = np.zeros(len(cids))
    if nf:
        dx = fx - cx[fclu]
        dy = fy - cy[fclu]
        np.maximum.at(cr, fclu, np.sqrt(dx * dx + dy * dy))
    ca, cb, ck, cc, cw = _aggregate(fclu[ga] if len(ga) else np.zeros(0, np.int64),
                                    fclu[gb] if len(gb) else np.zeros(0, np.int64),
                                    gk, gc, gw, max(1, len(cids)))
    arrays.update(_lod_arrays("clu", cids, np.full(len(cids), KIND_ID["Cluster"], np.uint8),
                              cx, cy, cpr, ccount.astype(np.uint32), cids,
                              np.zeros(len(cids), np.uint8), cnames, ca, cb, ck, cc, cw, bbox))
    arrays["clu_r"] = cr[np.argsort(_quantise(cx, cy, bbox), kind="stable")].astype(np.float32)
    return [{"id": int(c), "name": cnames[i], "x": float(cx[i]), "y": float(cy[i]),
             "r": float(cr[i]), "size": int(ccount[i])} for i, c in enumerate(cids.tolist())]


def _file_table_from_sym(arrays: dict) -> tuple[dict, np.ndarray]:
    """File table rows (in sym order of File nodes) + sym row -> file-table
    row (-1 none)."""
    kind = arrays["sym_kind"]
    frows = np.nonzero(kind == KIND_ID["File"])[0]
    fids = arrays["sym_id"][frows]
    lk = np.argsort(fids, kind="stable")
    sf = arrays["sym_file"]
    if len(fids):
        srt = fids[lk]
        pc = np.clip(np.searchsorted(srt, sf), 0, len(fids) - 1)
        node_file = np.where(srt[pc] == sf, lk[pc], -1).astype(np.int64)
    else:
        node_file = np.full(len(sf), -1, dtype=np.int64)
    pr = arrays["sym_pr"].astype(np.float64)
    fpr = np.zeros(len(frows), dtype=np.float64)
    ok = node_file >= 0
    np.add.at(fpr, node_file[ok], pr[ok])
    ftab = {"id": fids, "x": arrays["sym_x"][frows], "y": arrays["sym_y"][frows], "pr": fpr,
            "size": arrays["sym_size"][frows], "cluster": arrays["sym_cluster"][frows],
            "names": _gather_names(arrays["sym_name_off"], arrays["sym_name_blob"], frows)}
    return ftab, node_file


def _file_edges_from_sym(arrays: dict, node_file: np.ndarray, n_files: int) -> tuple:
    ea = arrays["sym_ea"].astype(np.int64)
    eb = arrays["sym_eb"].astype(np.int64)
    fa, fb = node_file[ea], node_file[eb]
    ok = (fa >= 0) & (fb >= 0)
    return _aggregate(fa[ok], fb[ok], arrays["sym_ek"][ok],
                      arrays["sym_ec"][ok].astype(np.float32) / 255.0,
                      arrays["sym_ew"][ok], max(1, n_files))


def _manifest(arrays: dict, bbox, generation: int, clusters: list[dict], t0: float, name: str) -> dict:
    kinds = arrays["sym_kind"]
    ek = arrays["sym_ek"]
    ec = arrays["sym_ec"]
    x, y = arrays["sym_x"], arrays["sym_y"]
    n = len(kinds)
    nonfile = kinds != KIND_ID["File"]
    return {
        "version": VERSION, "generation": int(generation), "file": name,
        "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[2]],
        "content_bbox": ([float(x.min()), float(y.min()), float(x.max()), float(y.max())]
                         if n else [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[2]]),
        "max_level": MAX_LEVEL, "tile_budget": TILE_BUDGET,
        "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS, "lods": list(LODS),
        "counts": {"sym_nodes": int(len(arrays["sym_id"])), "sym_edges": int(len(arrays["sym_ea"])),
                   "file_nodes": int(len(arrays["file_id"])), "file_edges": int(len(arrays["file_ea"])),
                   "clu_nodes": int(len(arrays["clu_id"])), "clu_edges": int(len(arrays["clu_ea"]))},
        "kind_counts": {NODE_KINDS[k]: int(c) for k, c in
                        enumerate(np.bincount(kinds, minlength=4)[:4].tolist())},
        "edge_kind_counts": {EDGE_KINDS[k]: int(c) for k, c in
                             enumerate(np.bincount(ek, minlength=len(EDGE_KINDS)).tolist()) if c},
        "pr_max": float(arrays["sym_pr"][nonfile].max()) if nonfile.any() else 0.0,
        "has_conf": bool(((ek == EDGE_KIND_ID["CALLS"]) & (ec < 255)).any()),
        "clusters": sorted(clusters, key=lambda c: -c["size"]),
        "built_at": time.time(), "build_seconds": round(time.perf_counter() - t0, 3),
    }


def build_arrays(src: TileSource, generation: int,
                 progress: Callable[[str], None] | None = None) -> tuple[dict, dict]:
    """(arrays, manifest) of a full build, in memory."""
    t0 = time.perf_counter()
    n = len(src.ids)
    bbox = _world_bbox(np.asarray(src.x, dtype=np.float64), np.asarray(src.y, dtype=np.float64))
    order_id = np.argsort(src.ids, kind="stable")
    sid = src.ids[order_id]

    def rows_of(q: np.ndarray) -> np.ndarray:
        if n == 0:
            return np.full(len(q), -1, np.int64)
        p = np.clip(np.searchsorted(sid, q), 0, n - 1)
        return np.where(sid[p] == q, order_id[p], -1)
    ea = rows_of(np.asarray(src.edge_a, dtype=np.int64))
    eb = rows_of(np.asarray(src.edge_b, dtype=np.int64))
    ok = (ea >= 0) & (eb >= 0) & (ea != eb)
    ea, eb = ea[ok], eb[ok]
    ek = np.asarray(src.edge_kind)[ok]
    ec = np.asarray(src.edge_conf, dtype=np.float32)[ok]
    size = np.ones(n, dtype=np.uint32)
    files = np.nonzero(src.kinds == KIND_ID["File"])[0]
    fr = src.file_row
    has_file = fr >= 0
    sym_per_file = np.bincount(fr[has_file & (src.kinds != KIND_ID["File"])], minlength=n)
    size[files] = sym_per_file[files] + 1
    # layout cluster per node: community, else the file's community, else a
    # directory pseudo-cluster (negative id)
    comm = src.community.copy()
    file_comm = np.where(has_file, comm[np.clip(fr, 0, None)], -1)
    node_clu = np.where(comm >= 0, comm, file_comm)
    names: dict[int, str] = {int(k): str((v or {}).get("name") or "") for k, v in src.clusters.items()}
    need = np.nonzero(node_clu < 0)[0]
    if len(need):
        slot: dict[str, int] = {}
        for i in need.tolist():
            f = int(fr[i]) if fr[i] >= 0 else i
            path = src.names[f] if src.kinds[f] == KIND_ID["File"] else ""
            d = path.split("/")[0] if "/" in path else "(root)"
            if d not in slot:
                slot[d] = -(len(slot) + 1)
                names[slot[d]] = d + "/"
            node_clu[i] = slot[d]
    file_id = np.where(has_file, src.ids[np.clip(fr, 0, None)], -1)
    file_id[files] = src.ids[files]
    if progress:
        progress("tiles: symbols")
    arrays: dict[str, np.ndarray] = {}
    arrays.update(_lod_arrays("sym", src.ids, src.kinds, src.x, src.y, src.pagerank, size,
                              node_clu, src.flags, src.names, ea, eb, ek, ec,
                              np.ones(len(ea), np.uint32), bbox, file_ids=file_id))
    if progress:
        progress("tiles: files")
    ftab, node_file = _file_table_from_sym(arrays)
    fedges = _file_edges_from_sym(arrays, node_file, len(ftab["id"]))
    clusters = _coarse_lods(arrays, ftab, fedges, names, bbox)
    name = f"tiles-{generation}.npz"
    return arrays, _manifest(arrays, bbox, generation, clusters, t0, name)


def write(out_dir: Path, arrays: dict, manifest: dict) -> None:
    """Persist one generation (npz, then manifest; older npz removed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = manifest["file"]
    tmp = out_dir / (name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, out_dir / name)
    tmpm = out_dir / "manifest.json.tmp"
    tmpm.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmpm, out_dir / "manifest.json")
    for old in out_dir.glob("tiles-*.npz"):
        if old.name != name:
            try:
                old.unlink()
            except OSError:
                pass


def load(out_dir: Path) -> tuple[dict, dict] | None:
    """(manifest, arrays) of the persisted generation, or None. The arrays
    are read-only memory maps of the (uncompressed) npz: tiles nobody looks
    at cost no resident memory."""
    try:
        man = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
        arrays = _read_arrays(Path(out_dir) / man["file"])
    except Exception:
        return None
    return man, arrays


def _read_arrays(path: Path) -> dict:
    from docgraph.procmem import mmap_npz
    arrays = mmap_npz(path)
    if arrays is None:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    return arrays


def build(src: TileSource, out_dir: Path, generation: int,
          progress: Callable[[str], None] | None = None) -> dict:
    """Write tiles-<generation>.npz + manifest.json under `out_dir`.
    Returns the manifest."""
    arrays, manifest = build_arrays(src, generation, progress)
    if progress:
        progress("tiles: writing")
    write(out_dir, arrays, manifest)
    return manifest


# ---- incremental patch ---------------------------------------------------------

@dataclass
class TileDelta:
    """One incremental pass, for `patch`: nodes removed (ids), nodes added
    (parallel arrays), and the edge journal (db.EdgeJournal ops)."""
    removed: np.ndarray                  # int64 node ids
    ids: np.ndarray                      # new nodes ...
    kinds: np.ndarray                    # uint8 KIND_ID
    x: np.ndarray
    y: np.ndarray
    pagerank: np.ndarray
    file_id: np.ndarray                  # int64 (a File: itself)
    community: np.ndarray                # int64 (-1 none)
    names: list[str]                     # File: path; else the symbol name
    flags: np.ndarray                    # uint8
    ops: list[tuple]                     # EdgeJournal.ops


def _cluster_names(man: dict) -> dict[int, str]:
    return {int(c["id"]): str(c.get("name") or "") for c in man.get("clusters") or []}


def _rows_in(q, srt, row) -> np.ndarray:
    """Row of each id in `q` via a (sorted ids, row) lookup; -1 if absent."""
    q = np.asarray(q, dtype=np.int64)
    if len(srt) == 0 or len(q) == 0:
        return np.full(len(q), -1, np.int64)
    p = np.clip(np.searchsorted(srt, q), 0, len(srt) - 1)
    return np.where(srt[p] == q, row[p], -1)


_NODE_COLS = (("id", np.int64), ("x", np.float32), ("y", np.float32), ("kind", np.uint8),
              ("pr", np.float32), ("size", np.uint32), ("cluster", np.int64), ("flags", np.uint8))


def _patch_lod(a: dict, prefix: str, bbox, drop: np.ndarray, new: dict, names: list[str],
               edge_keep: np.ndarray, edge_w: np.ndarray | None, edge_c: np.ndarray | None,
               add_a: np.ndarray, add_b: np.ndarray, add_k: np.ndarray, add_c: np.ndarray,
               add_w: np.ndarray) -> dict:
    """One LOD with nodes dropped / merged in by Morton code and edges
    dropped / re-weighted / merged in by owner code -- linear merges, no
    re-sort of the big arrays. New edges are given by endpoint ids."""
    P = prefix
    n0 = len(a[f"{P}_id"])
    keep = ~drop
    ko = np.nonzero(keep)[0]
    kcode = a[f"{P}_code"][ko]
    nx = np.asarray(new["x"], dtype=np.float64)
    ncode = _quantise(nx, np.asarray(new["y"], dtype=np.float64), bbox) if len(nx) else np.zeros(0, np.uint32)
    no = np.argsort(ncode, kind="stable")
    ncode_s = ncode[no]
    ins = np.searchsorted(kcode, ncode_s, side="right")
    total = len(ko) + len(no)
    pos_new = ins + np.arange(len(no))
    is_new = np.zeros(total, dtype=bool)
    is_new[pos_new] = True
    pos_kept = np.nonzero(~is_new)[0]
    old2new = np.full(n0, -1, dtype=np.int64)
    old2new[ko] = pos_kept
    out: dict[str, np.ndarray] = {}
    cols = list(_NODE_COLS) + ([("file", np.int64)] if f"{P}_file" in a else [])
    for col, dt in cols:
        o = np.empty(total, dtype=dt)
        o[pos_kept] = a[f"{P}_{col}"][ko]
        if len(no):
            o[pos_new] = np.asarray(new[col], dtype=dt)[no]
        out[f"{P}_{col}"] = o
    code = np.empty(total, dtype=np.uint32)
    code[pos_kept] = kcode
    code[pos_new] = ncode_s
    out[f"{P}_code"] = code
    # names: gather from the old blob + the new names
    noff, nblob = _name_blob(list(names))
    off0, blob0 = a[f"{P}_name_off"].astype(np.int64), a[f"{P}_name_blob"]
    starts_all = np.concatenate([off0[:-1], noff[:-1].astype(np.int64) + len(blob0)])
    lens_all = np.concatenate([np.diff(off0), np.diff(noff.astype(np.int64))])
    src_rows = np.empty(total, dtype=np.int64)
    src_rows[pos_kept] = ko
    src_rows[pos_new] = n0 + no
    blob_all = np.concatenate([blob0, nblob]) if len(nblob) else blob0
    starts, lens = starts_all[src_rows], lens_all[src_rows]
    off = np.zeros(total + 1, dtype=np.uint32)
    if total:
        off[1:] = np.cumsum(lens).astype(np.uint32)
    ntot = int(off[-1])
    idx = np.repeat(starts - off[:-1].astype(np.int64), lens) + np.arange(ntot, dtype=np.int64)
    out[f"{P}_name_off"] = off
    out[f"{P}_name_blob"] = blob_all[idx] if ntot else np.zeros(0, np.uint8)
    ids = out[f"{P}_id"]
    lk = np.argsort(ids, kind="stable")
    out[f"{P}_id_sorted"] = ids[lk]
    out[f"{P}_id_row"] = lk.astype(np.int64)
    # ---- edges
    ea0 = a[f"{P}_ea"].astype(np.int64)
    eb0 = a[f"{P}_eb"].astype(np.int64)
    E0 = len(ea0)
    emask = edge_keep & (old2new[ea0] >= 0) & (old2new[eb0] >= 0) if E0 else np.zeros(0, bool)
    keep_e = np.nonzero(emask)[0]
    kea, keb = old2new[ea0[keep_e]], old2new[eb0[keep_e]]
    k_ecode = a[f"{P}_ecode"][keep_e]
    ra_ = _rows_in(add_a, out[f"{P}_id_sorted"], out[f"{P}_id_row"])
    rb_ = _rows_in(add_b, out[f"{P}_id_sorted"], out[f"{P}_id_row"])
    okn = (ra_ >= 0) & (rb_ >= 0) & (ra_ != rb_)
    ra_, rb_ = ra_[okn], rb_[okn]
    pk_ = np.asarray(add_k, dtype=np.uint8)[okn]
    pc_ = np.asarray(add_c, dtype=np.uint8)[okn]
    pw_ = np.asarray(add_w, dtype=np.uint32)[okn]
    own_a = ids[ra_] <= ids[rb_]
    owner = np.where(own_a, ra_, rb_)
    other = np.where(own_a, rb_, ra_)
    n_ecode = code[owner]
    eo = np.argsort(n_ecode, kind="stable")
    ra_, rb_, pk_, pc_, pw_, n_ecode, other = (ra_[eo], rb_[eo], pk_[eo], pc_[eo], pw_[eo],
                                               n_ecode[eo], other[eo])
    eins = np.searchsorted(k_ecode, n_ecode, side="right")
    E = len(keep_e) + len(eo)
    epos_new = eins + np.arange(len(eo))
    e_is_new = np.zeros(E, dtype=bool)
    e_is_new[epos_new] = True
    epos_kept = np.nonzero(~e_is_new)[0]

    def emerge(oldvals, newvals, dtype):
        o = np.empty(E, dtype=dtype)
        o[epos_kept] = oldvals
        o[epos_new] = newvals
        return o
    w_old = (edge_w if edge_w is not None else a[f"{P}_ew"])[keep_e]
    c_old = (edge_c if edge_c is not None else a[f"{P}_ec"])[keep_e]
    out[f"{P}_ea"] = emerge(kea, ra_, np.uint32)
    out[f"{P}_eb"] = emerge(keb, rb_, np.uint32)
    out[f"{P}_ek"] = emerge(a[f"{P}_ek"][keep_e], pk_, np.uint8)
    out[f"{P}_ec"] = emerge(c_old, pc_, np.uint8)
    out[f"{P}_ew"] = emerge(w_old, pw_, np.uint32)
    out[f"{P}_ecode"] = emerge(k_ecode, n_ecode, np.uint32)
    old_e2new = np.full(E0, -1, dtype=np.int64)
    old_e2new[keep_e] = epos_kept
    r_old = old_e2new[a[f"{P}_ridx"].astype(np.int64)] if E0 else np.zeros(0, np.int64)
    rsel = r_old >= 0
    r_kept = r_old[rsel]
    r_kcode = a[f"{P}_rcode"][rsel]
    n_rcode = code[other]
    ro = np.argsort(n_rcode, kind="stable")
    rins = np.searchsorted(r_kcode, n_rcode[ro], side="right")
    R = len(r_kept) + len(ro)
    rpos_new = rins + np.arange(len(ro))
    r_is_new = np.zeros(R, dtype=bool)
    r_is_new[rpos_new] = True
    rpos_kept = np.nonzero(~r_is_new)[0]
    ridx = np.empty(R, dtype=np.int64)
    ridx[rpos_kept] = r_kept
    ridx[rpos_new] = epos_new[ro]
    rcode = np.empty(R, dtype=np.uint32)
    rcode[rpos_kept] = r_kcode
    rcode[rpos_new] = n_rcode[ro]
    out[f"{P}_ridx"] = ridx.astype(np.uint32)
    out[f"{P}_rcode"] = rcode
    out["_old2new"] = old2new
    out["_keep_e"] = keep_e
    return out


def _conf_u8(c) -> np.ndarray:
    return np.clip(np.asarray(c, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def patch(arrays: dict, man: dict, delta: TileDelta, generation: int) -> tuple[dict, dict]:
    """New (arrays, manifest) with `delta` applied. Every LOD is patched,
    not rebuilt: nodes and edges are dropped / merged in by Morton code
    (linear merges of already-sorted arrays), file edges are re-weighted by
    the delta of the symbol edges between their files, cluster edges by the
    delta of the file edges. O(changed) apart from a few vectorised passes
    over the edge arrays. Raises KeyError for a sidecar without `sym_file`."""
    t0 = time.perf_counter()
    a = arrays
    b0 = man["bbox"]
    bbox = (float(b0[0]), float(b0[1]), float(b0[2] - b0[0]))
    kid = EDGE_KIND_ID
    FILE = KIND_ID["File"]
    ids0 = a["sym_id"]
    n0 = len(ids0)
    sym_file0 = a["sym_file"]
    id_sorted, id_row = a["sym_id_sorted"], a["sym_id_row"]

    # ---- symbol nodes
    removed = np.unique(np.asarray(delta.removed, dtype=np.int64))
    for op in delta.ops:
        if op[0] == "del_nodes":
            removed = np.union1d(removed, np.asarray(op[1], dtype=np.int64))
    nid = np.asarray(delta.ids, dtype=np.int64)
    drop = np.zeros(n0, dtype=bool)
    rr = _rows_in(np.union1d(removed, nid), id_sorted, id_row)
    drop[rr[rr >= 0]] = True
    nkind = np.asarray(delta.kinds, dtype=np.uint8)
    nfile = np.asarray(delta.file_id, dtype=np.int64)
    npr = np.asarray(delta.pagerank, dtype=np.float64)
    nsize = np.ones(len(nid), dtype=np.uint32)
    is_f = nkind == FILE
    if len(nid):
        cnt: dict[int, int] = {}
        for f in nfile[~is_f].tolist():
            cnt[f] = cnt.get(f, 0) + 1
        nsize[is_f] = np.array([cnt.get(int(i), 0) + 1 for i in nid[is_f].tolist()], dtype=np.uint32)
    # cluster of new nodes: community, else the file's (new) community, else
    # the directory pseudo cluster
    cnames = _cluster_names(man)
    ncomm = np.asarray(delta.community, dtype=np.int64).copy()
    if len(nid):
        fcomm = {int(i): int(c) for i, c, k in zip(nid.tolist(), ncomm.tolist(), nkind.tolist())
                 if k == FILE and c >= 0}
        pseudo = {v[:-1]: k for k, v in cnames.items() if k < 0 and v.endswith("/")}
        next_pseudo = min([k for k in cnames if k < 0], default=0) - 1
        path_of = {int(i): nm for i, nm, k in zip(nid.tolist(), delta.names, nkind.tolist()) if k == FILE}
        for j in np.nonzero(ncomm < 0)[0].tolist():
            c = fcomm.get(int(nfile[j]))
            if c is None:
                path = path_of.get(int(nfile[j]), "")
                d = path.split("/")[0] if "/" in path else "(root)"
                if d not in pseudo:
                    pseudo[d] = next_pseudo
                    cnames[next_pseudo] = d + "/"
                    next_pseudo -= 1
                c = pseudo[d]
            ncomm[j] = c

    # ---- symbol edges: journal ops, in order, over (existing, pending adds)
    ea0 = a["sym_ea"].astype(np.int64)
    eb0 = a["sym_eb"].astype(np.int64)
    ek0 = a["sym_ek"]
    ekeep = np.ones(len(ea0), dtype=bool)
    pend: list[list[np.ndarray]] = []            # [ids_a, ids_b, kind, conf]

    def filt(m_of):
        return [[p[0][m], p[1][m], p[2][m], p[3][m]] for p in pend for m in [m_of(p)]]
    for op in delta.ops:
        kind = op[0]
        if kind == "del_nodes":
            nd = np.asarray(op[1], dtype=np.int64)
            pend = filt(lambda p: ~(np.isin(p[0], nd) | np.isin(p[1], nd)))
            continue
        k = kid.get(op[1])
        if k is None:
            continue
        if kind == "add":
            ia = np.asarray(op[2], dtype=np.int64)
            ib = np.asarray(op[3], dtype=np.int64)
            cf = np.asarray(op[4], dtype=np.float32) if op[4] is not None else np.ones(len(ia), np.float32)
            pend.append([ia, ib, np.full(len(ia), k, np.uint8), cf])
        elif kind == "del_kind":
            ekeep &= ek0 != k
            pend = filt(lambda p: p[2] != k)
        elif kind == "del_out":
            src = np.asarray(op[2], dtype=np.int64)
            srows = _rows_in(src, id_sorted, id_row)
            srows = srows[srows >= 0]
            if len(srows):
                flag = np.zeros(n0, dtype=bool)
                flag[srows] = True
                ekeep &= ~((ek0 == k) & flag[ea0])
            pend = filt(lambda p: ~((p[2] == k) & np.isin(p[0], src)))
        elif kind == "del_pairs":
            pa_ = np.asarray(op[2], dtype=np.int64)
            pb_ = np.asarray(op[3], dtype=np.int64)
            ra_ = _rows_in(pa_, id_sorted, id_row)
            rb_ = _rows_in(pb_, id_sorted, id_row)
            okp = (ra_ >= 0) & (rb_ >= 0)
            if okp.any():
                flag = np.zeros(n0, dtype=bool)
                flag[ra_[okp]] = True
                sel = np.nonzero((ek0 == k) & flag[ea0])[0]
                hit = np.isin(ea0[sel] * n0 + eb0[sel], ra_[okp] * n0 + rb_[okp])
                ekeep[sel[hit]] = False
            keys2 = (pa_ << 32) | pb_          # ids are < 2**31
            pend = filt(lambda p: ~((p[2] == k) & np.isin((p[0] << 32) | p[1], keys2)))
    if len(removed) and pend:
        pend = filt(lambda p: ~(np.isin(p[0], removed) | np.isin(p[1], removed)))
    if pend:
        add_a = np.concatenate([p[0] for p in pend])
        add_b = np.concatenate([p[1] for p in pend])
        add_k = np.concatenate([p[2] for p in pend])
        add_c = np.concatenate([p[3] for p in pend])
    else:
        add_a = add_b = np.zeros(0, np.int64)
        add_k = np.zeros(0, np.uint8)
        add_c = np.zeros(0, np.float32)
    new_sym = {"id": nid, "x": delta.x, "y": delta.y, "kind": nkind, "pr": npr, "size": nsize,
               "cluster": ncomm, "flags": delta.flags, "file": nfile}
    out = _patch_lod(a, "sym", bbox, drop, new_sym, list(delta.names), ekeep, None, None,
                     add_a, add_b, add_k, _conf_u8(add_c), np.ones(len(add_a), np.uint32))
    old2new = out.pop("_old2new")
    keep_e = out.pop("_keep_e")

    # ---- file LOD: file-edge weight delta from the symbol-edge delta
    fid_old = a["file_id"]
    Fn0 = len(fid_old)
    frm_drop = np.zeros(Fn0, dtype=bool)
    gone_files = ids0[drop & (a["sym_kind"] == FILE)]
    fr_ = _rows_in(gone_files, a["file_id_sorted"], a["file_id_row"])
    frm_drop[fr_[fr_ >= 0]] = True
    gone_set = gone_files
    # removed symbol edges between two surviving files: -1
    rem = np.ones(len(ea0), dtype=bool)
    rem[keep_e] = False
    rfa, rfb = sym_file0[ea0[rem]], sym_file0[eb0[rem]]
    rk = ek0[rem]
    live_ = ~(np.isin(rfa, gone_set) | np.isin(rfb, gone_set)) & (rfa != rfb)
    rfa, rfb, rk = rfa[live_], rfb[live_], rk[live_]
    # added symbol edges (as written into the new arrays): +1
    sid, srow = out["sym_id_sorted"], out["sym_id_row"]
    ra_ = _rows_in(add_a, sid, srow)
    rb_ = _rows_in(add_b, sid, srow)
    ok = (ra_ >= 0) & (rb_ >= 0) & (ra_ != rb_)
    afa, afb = out["sym_file"][ra_[ok]], out["sym_file"][rb_[ok]]
    ak, ac = add_k[ok], _conf_u8(add_c[ok])
    ok2 = afa != afb
    afa, afb, ak, ac = afa[ok2], afb[ok2], ak[ok2], ac[ok2]
    dfa = np.concatenate([rfa, afa])
    dfb = np.concatenate([rfb, afb])
    dk = np.concatenate([rk, ak]).astype(np.int64)
    dw = np.concatenate([np.full(len(rfa), -1, np.int64), np.ones(len(afa), np.int64)])
    dc = np.concatenate([np.zeros(len(rfa), np.uint8), ac])
    touched = np.unique(np.concatenate([dfa, dfb])) if len(dfa) else np.zeros(0, np.int64)
    T_ = max(1, len(touched))
    fw = a["file_ew"].astype(np.int64).copy()
    fc = a["file_ec"].copy()
    fkeep = np.ones(len(a["file_ea"]), dtype=bool)
    new_fa = new_fb = np.zeros(0, np.int64)
    new_fk = np.zeros(0, np.uint8)
    new_fc = np.zeros(0, np.uint8)
    new_fw = np.zeros(0, np.uint32)
    if len(dfa):
        dkey = (np.searchsorted(touched, dfa) * T_ + np.searchsorted(touched, dfb)) * 32 + dk
        uk, inv = np.unique(dkey, return_inverse=True)
        uw = np.bincount(inv, weights=dw.astype(np.float64)).astype(np.int64)
        uc = np.zeros(len(uk), dtype=np.uint8)
        np.maximum.at(uc, inv, dc)
        # existing file edges between two touched files
        fea, feb = a["file_ea"].astype(np.int64), a["file_eb"].astype(np.int64)
        tflag = np.isin(fid_old, touched)
        cand = np.nonzero(tflag[fea] & tflag[feb])[0]
        ckey = ((np.searchsorted(touched, fid_old[fea[cand]]) * T_
                 + np.searchsorted(touched, fid_old[feb[cand]])) * 32 + a["file_ek"][cand].astype(np.int64))
        pos = np.clip(np.searchsorted(uk, ckey), 0, max(0, len(uk) - 1))
        hit = (len(uk) > 0) & (uk[pos] == ckey) if len(uk) else np.zeros(len(cand), bool)
        hc, hp = cand[hit], pos[hit]
        fw[hc] += uw[hp]
        fc[hc] = np.maximum(fc[hc], uc[hp])
        fkeep[hc[fw[hc] <= 0]] = False
        matched = np.zeros(len(uk), dtype=bool)
        matched[hp] = True
        nm = ~matched & (uw > 0)
        pair = uk[nm] // 32
        new_fa = touched[pair // T_]
        new_fb = touched[pair % T_]
        new_fk = (uk[nm] % 32).astype(np.uint8)
        new_fc = uc[nm]
        new_fw = uw[nm].astype(np.uint32)
    # new File nodes
    nf_ids = nid[is_f]
    fpr_new = np.zeros(len(nf_ids), dtype=np.float64)
    if len(nf_ids):
        slot_f = {int(i): j for j, i in enumerate(nf_ids.tolist())}
        tgt = np.array([slot_f.get(int(f), -1) for f in nfile.tolist()], dtype=np.int64)
        okf = tgt >= 0
        np.add.at(fpr_new, tgt[okf], npr[okf])
    new_file = {"id": nf_ids, "x": np.asarray(delta.x, dtype=np.float64)[is_f],
                "y": np.asarray(delta.y, dtype=np.float64)[is_f], "kind": np.zeros(len(nf_ids), np.uint8),
                "pr": fpr_new, "size": nsize[is_f], "cluster": ncomm[is_f],
                "flags": np.zeros(len(nf_ids), np.uint8)}
    fnames = [n for n, f in zip(delta.names, is_f.tolist()) if f]
    fout = _patch_lod(a, "file", bbox, frm_drop, new_file, fnames, fkeep,
                      np.clip(fw, 0, None).astype(np.uint32), fc,
                      new_fa, new_fb, new_fk, new_fc, new_fw)
    fold2new = fout.pop("_old2new")
    fkeep_e = fout.pop("_keep_e")
    out.update(fout)

    # ---- cluster LOD: nodes from the file table (small), edges re-weighted
    # by the file-edge delta
    fcl = out["file_cluster"]
    cids = np.unique(fcl) if len(fcl) else np.zeros(0, np.int64)
    fclu = np.searchsorted(cids, fcl).astype(np.int64)
    size = out["file_size"].astype(np.float64)
    fx, fy = out["file_x"].astype(np.float64), out["file_y"].astype(np.float64)
    ccount = np.bincount(fclu, weights=size, minlength=len(cids))
    cx = np.bincount(fclu, weights=fx * size, minlength=len(cids)) / np.maximum(ccount, 1)
    cy = np.bincount(fclu, weights=fy * size, minlength=len(cids)) / np.maximum(ccount, 1)
    cpr = np.bincount(fclu, weights=out["file_pr"].astype(np.float64), minlength=len(cids))
    cr = np.zeros(len(cids))
    if len(fcl):
        dx = fx - cx[fclu]
        dy = fy - cy[fclu]
        np.maximum.at(cr, fclu, np.sqrt(dx * dx + dy * dy))
    cl_names = [str(cnames.get(int(c)) or f"#{c}") for c in cids.tolist()]
    # old cluster edges in cluster-id space, plus the file-edge weight delta
    cid_old = a["clu_id"]
    ca0 = cid_old[a["clu_ea"].astype(np.int64)]
    cb0 = cid_old[a["clu_eb"].astype(np.int64)]
    ck0 = a["clu_ek"].astype(np.int64)
    cw0 = a["clu_ew"].astype(np.int64)
    cc0 = a["clu_ec"]
    fcl_old = a["file_cluster"]
    fea0 = a["file_ea"].astype(np.int64)
    feb0 = a["file_eb"].astype(np.int64)
    w_before = a["file_ew"].astype(np.int64)
    w_after = np.zeros(len(fea0), dtype=np.int64)
    w_after[fkeep_e] = np.clip(fw[fkeep_e], 0, None)
    chg = np.nonzero(w_after != w_before)[0]
    xa = [ca0, fcl_old[fea0[chg]]]
    xb = [cb0, fcl_old[feb0[chg]]]
    xk = [ck0, a["file_ek"][chg].astype(np.int64)]
    xw = [cw0, (w_after[chg] - w_before[chg])]
    xc = [cc0, fc[chg]]
    if len(new_fa):
        fsid, fsrow = out["file_id_sorted"], out["file_id_row"]
        ra2 = _rows_in(new_fa, fsid, fsrow)
        rb2 = _rows_in(new_fb, fsid, fsrow)
        ok3 = (ra2 >= 0) & (rb2 >= 0)
        xa.append(fcl[ra2[ok3]])
        xb.append(fcl[rb2[ok3]])
        xk.append(new_fk[ok3].astype(np.int64))
        xw.append(new_fw[ok3].astype(np.int64))
        xc.append(new_fc[ok3])
    xa_, xb_ = np.concatenate(xa), np.concatenate(xb)
    xk_, xw_, xc_ = np.concatenate(xk), np.concatenate(xw), np.concatenate(xc)
    okc = np.isin(xa_, cids) & np.isin(xb_, cids) & (xa_ != xb_)
    xa_, xb_, xk_, xw_, xc_ = xa_[okc], xb_[okc], xk_[okc], xw_[okc], xc_[okc]
    C = max(1, len(cids))
    ckey = (np.searchsorted(cids, xa_) * C + np.searchsorted(cids, xb_)) * 32 + xk_
    uk, inv = np.unique(ckey, return_inverse=True)
    uw = np.bincount(inv, weights=xw_.astype(np.float64)).astype(np.int64)
    uc = np.zeros(len(uk), dtype=np.uint8)
    np.maximum.at(uc, inv, xc_)
    live_c = uw > 0
    pair = uk[live_c] // 32
    arrays_c = _lod_arrays("clu", cids, np.full(len(cids), KIND_ID["Cluster"], np.uint8), cx, cy, cpr,
                           ccount.astype(np.uint32), cids, np.zeros(len(cids), np.uint8), cl_names,
                           pair // C, pair % C, (uk[live_c] % 32).astype(np.uint8), uc[live_c],
                           uw[live_c].astype(np.uint32), bbox)
    out.update(arrays_c)
    out["clu_r"] = cr[np.argsort(_quantise(cx, cy, bbox), kind="stable")].astype(np.float32)
    clusters = [{"id": int(c), "name": cl_names[i], "x": float(cx[i]), "y": float(cy[i]),
                 "r": float(cr[i]), "size": int(ccount[i])} for i, c in enumerate(cids.tolist())]
    del fold2new, old2new
    name = f"tiles-{generation}.npz"
    return out, _manifest(out, bbox, generation, clusters, t0, name)


# ---- serving -------------------------------------------------------------------

def _pad4(b: bytes) -> bytes:
    r = (-len(b)) % 4
    return b + b"\0" * r


class TileStore:
    """Loaded sidecar for one root. Reloads when manifest.json changes."""

    def __init__(self, tiles_dir: Path):
        self.dir = Path(tiles_dir)
        self._lock = threading.Lock()
        self._mtime = -1.0
        self.manifest: dict | None = None
        self.a: dict[str, np.ndarray] = {}

    def _refresh(self) -> None:
        mpath = self.dir / "manifest.json"
        try:
            mt = mpath.stat().st_mtime
        except OSError:
            self.manifest, self.a, self._mtime = None, {}, -1.0
            return
        if mt == self._mtime and self.manifest is not None:
            return
        with self._lock:
            if mt == self._mtime and self.manifest is not None:
                return
            try:
                man = json.loads(mpath.read_text(encoding="utf-8"))
                if (self.manifest is not None
                        and int(man.get("generation", -1)) <= int(self.manifest.get("generation", -1))):
                    # the persisted copy of what install() already serves
                    self._mtime = mt
                    return
                arrays = _read_arrays(self.dir / man["file"])
            except Exception:
                return
            self.manifest, self.a, self._mtime = man, arrays, mt

    def install(self, manifest: dict, arrays: dict) -> None:
        """Serve an in-memory generation right away (the indexer's patch);
        the npz written behind it is not re-read."""
        with self._lock:
            if self.manifest is not None and \
                    int(manifest.get("generation", 0)) < int(self.manifest.get("generation", 0)):
                return
            self.manifest, self.a = manifest, arrays
            try:
                self._mtime = (self.dir / "manifest.json").stat().st_mtime
            except OSError:
                self._mtime = -1.0

    def unload(self) -> None:
        """Drop the served arrays (idle unload); the next request reloads
        the persisted generation (memory-mapped)."""
        with self._lock:
            self.manifest, self.a, self._mtime = None, {}, -1.0

    def ready(self) -> bool:
        self._refresh()
        return self.manifest is not None

    def get_manifest(self) -> dict | None:
        self._refresh()
        return self.manifest

    def _count(self, lod: str, lo: int, hi: int) -> int:
        code = self.a[f"{lod}_code"]
        return int(np.searchsorted(code, hi, "left") - np.searchsorted(code, lo, "left"))

    def pick_lod(self, level: int, lo: int, hi: int, budget: int) -> str:
        if level >= MAX_LEVEL:
            return "sym"
        for lod in LODS:
            if self._count(lod, lo, hi) <= budget:
                return lod
        return "clu"

    def tile(self, level: int, tx: int, ty: int, *, lod: str | None = None,
             budget: int = TILE_BUDGET, edge_kinds: set[int] | None = None,
             min_conf: float = 0.0, node_kinds: set[int] | None = None) -> tuple[bytes, str]:
        """Packed tile bytes + etag (content hash)."""
        self._refresh()
        if self.manifest is None:
            raise FileNotFoundError("no tiles")
        level = max(0, min(MAX_LEVEL, int(level)))
        n_t = 1 << level
        tx = max(0, min(n_t - 1, int(tx)))
        ty = max(0, min(n_t - 1, int(ty)))
        lo, hi = tile_range(level, tx, ty)
        if lod not in LODS:
            lod = self.pick_lod(level, lo, hi, budget)
        a = self.a
        code = a[f"{lod}_code"]
        s, e = int(np.searchsorted(code, lo, "left")), int(np.searchsorted(code, hi, "left"))
        rows = np.arange(s, e, dtype=np.int64)
        if node_kinds is not None and len(rows):
            rows = rows[np.isin(a[f"{lod}_kind"][rows], np.fromiter(node_kinds, np.uint8))]
        # edges: owned by this tile + received by this tile
        ecode = a[f"{lod}_ecode"]
        es, ee = int(np.searchsorted(ecode, lo, "left")), int(np.searchsorted(ecode, hi, "left"))
        own = np.arange(es, ee, dtype=np.int64)
        rcode = a[f"{lod}_rcode"]
        rs, re_ = int(np.searchsorted(rcode, lo, "left")), int(np.searchsorted(rcode, hi, "left"))
        recv = a[f"{lod}_ridx"][rs:re_].astype(np.int64)
        if len(recv):
            rc = ecode[recv]
            recv = recv[(rc < lo) | (rc >= hi)]
        eidx = np.concatenate([own, recv]) if len(recv) else own
        if len(eidx) and edge_kinds is not None:
            eidx = eidx[np.isin(a[f"{lod}_ek"][eidx], np.fromiter(edge_kinds, np.uint8))]
        if len(eidx) and min_conf > 0:
            ek = a[f"{lod}_ek"][eidx]
            conf = a[f"{lod}_ec"][eidx].astype(np.float32) / 255.0
            eidx = eidx[(ek != EDGE_KIND_ID["CALLS"]) | (conf >= min_conf)]
        m_total = len(eidx)
        if lod != "sym" and len(eidx):
            cap = max(AGG_EDGES_MIN, AGG_EDGES_PER_NODE * max(1, e - s))
            if len(eidx) > cap:
                w = a[f"{lod}_ew"][eidx]
                keep = np.argpartition(-w.astype(np.int64), cap - 1)[:cap]
                eidx = np.sort(eidx[keep])
        ea = a[f"{lod}_ea"][eidx].astype(np.int64)
        eb = a[f"{lod}_eb"][eidx].astype(np.int64)
        K = a[f"{lod}_kind"]
        if node_kinds is not None and len(eidx):
            nk = np.fromiter(node_kinds, np.uint8)
            keep = np.isin(K[ea], nk) & np.isin(K[eb], nk)
            eidx, ea, eb = eidx[keep], ea[keep], eb[keep]
        X, Y, ID, C = a[f"{lod}_x"], a[f"{lod}_y"], a[f"{lod}_id"], a[f"{lod}_cluster"]
        n = len(rows)

        # endpoint -> local index: tile node, else a ghost endpoint listed once
        def local_of(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p = np.searchsorted(rows, r)
            pc = np.clip(p, 0, max(0, n - 1))
            inside = (n > 0) & (rows[pc] == r) if n else np.zeros(len(r), bool)
            return np.where(inside, pc, -1), inside

        la, ia = local_of(ea)
        lb, ib = local_of(eb)
        ghost_rows = np.unique(np.concatenate([ea[~ia], eb[~ib]])) if len(eidx) else np.zeros(0, np.int64)
        g = len(ghost_rows)
        if g:
            la = np.where(ia, la, n + np.searchsorted(ghost_rows, ea))
            lb = np.where(ib, lb, n + np.searchsorted(ghost_rows, eb))
        # names: all for file / clu, top-PageRank for sym
        name_rows = rows
        if lod == "sym" and n > NAME_BUDGET:
            pr = a["sym_pr"][rows]
            top = np.argpartition(-pr, NAME_BUDGET - 1)[:NAME_BUDGET]
            name_rows = np.sort(rows[top])
        local = np.searchsorted(rows, name_rows)
        off, blob = a[f"{lod}_name_off"], a[f"{lod}_name_blob"]
        pieces = [bytes(blob[int(off[r]):int(off[r + 1])]) for r in name_rows.tolist()]
        noff = np.zeros(len(pieces) + 1, dtype=np.uint32)
        if pieces:
            noff[1:] = np.cumsum([len(p) for p in pieces])
        nblob = b"".join(pieces)
        # The header's generation slot stays 0: a tile's bytes (and so its
        # ETag) depend only on its content, so after an incremental reindex
        # every tile the change did not touch revalidates as unchanged.
        head = struct.pack("<IHBBIIIIIIIII", MAGIC, VERSION, LODS.index(lod), level, tx, ty,
                           n, len(eidx), g, len(pieces), len(nblob), 0, m_total)
        parts = [head,
                 ID[rows].astype(np.int32).tobytes(),
                 X[rows].tobytes(), Y[rows].tobytes(),
                 a[f"{lod}_pr"][rows].tobytes(),
                 a[f"{lod}_size"][rows].tobytes(),
                 C[rows].astype(np.int32).tobytes(),
                 _pad4(K[rows].tobytes()),
                 _pad4(a[f"{lod}_flags"][rows].tobytes()),
                 ID[ghost_rows].astype(np.int32).tobytes(),
                 X[ghost_rows].tobytes(), Y[ghost_rows].tobytes(),
                 C[ghost_rows].astype(np.int32).tobytes(),
                 _pad4(K[ghost_rows].tobytes()),
                 eidx.astype(np.uint32).tobytes(),
                 la.astype(np.uint32).tobytes(), lb.astype(np.uint32).tobytes(),
                 a[f"{lod}_ew"][eidx].tobytes(),
                 _pad4(a[f"{lod}_ek"][eidx].tobytes()),
                 _pad4(a[f"{lod}_ec"][eidx].tobytes()),
                 local.astype(np.uint32).tobytes(),
                 noff.tobytes(),
                 _pad4(nblob)]
        body = b"".join(parts)
        etag = hashlib.blake2b(body, digest_size=10).hexdigest()
        return body, etag

    def locate(self, ids: list[int]) -> list[dict]:
        """World position + kind of symbol-level nodes by id."""
        self._refresh()
        if self.manifest is None:
            return []
        srt, row = self.a["sym_id_sorted"], self.a["sym_id_row"]
        out: list[dict] = []
        q = np.asarray([int(i) for i in ids], dtype=np.int64)
        pos = np.searchsorted(srt, q)
        for i, p in zip(q.tolist(), pos.tolist()):
            if p < len(srt) and int(srt[p]) == i:
                r = int(row[p])
                off, blob = self.a["sym_name_off"], self.a["sym_name_blob"]
                out.append({
                    "id": i, "x": float(self.a["sym_x"][r]), "y": float(self.a["sym_y"][r]),
                    "kind": NODE_KINDS[int(self.a["sym_kind"][r])],
                    "cluster": int(self.a["sym_cluster"][r]),
                    "pagerank": float(self.a["sym_pr"][r]),
                    "name": bytes(blob[int(off[r]):int(off[r + 1])]).decode("utf-8", "replace"),
                })
        return out


def frame_batch(items: list[tuple[int, int, int, bytes, str]]) -> bytes:
    """Batch payload: u32 count, then per tile u32 level, x, y, byte length,
    10-byte etag (+2 pad), bytes (padded to 4). Length 0 = "unchanged": the
    client already holds the tile with that ETag."""
    out = [struct.pack("<I", len(items))]
    for level, tx, ty, body, etag in items:
        out.append(struct.pack("<IIII", level, tx, ty, len(body)))
        out.append(bytes.fromhex(etag).ljust(12, b"\0"))
        out.append(_pad4(body))
    return b"".join(out)
