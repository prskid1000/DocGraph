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
VERSION = 2
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


def _lod_arrays(prefix: str, ids, kinds, x, y, pr, size, cluster, flags, names,
                ea, eb, ek, ec, ew, bbox) -> dict[str, np.ndarray]:
    """Sort one LOD by Morton code and lay out its node + edge arrays.
    ea/eb are node ROWS (pre-sort); returned edges are in sorted-row space."""
    code = _quantise(x, y, bbox)
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
    off, blob = _name_blob([names[i] for i in order.tolist()])
    out[f"{prefix}_name_off"], out[f"{prefix}_name_blob"] = off, blob
    # id lookup (for locate)
    ids_sorted = out[f"{prefix}_id"]
    lk = np.argsort(ids_sorted, kind="stable")
    out[f"{prefix}_id_sorted"] = ids_sorted[lk]
    out[f"{prefix}_id_row"] = lk.astype(np.int64)
    # edges: owner = lower-id endpoint
    ra, rb = inv[np.asarray(ea, dtype=np.int64)], inv[np.asarray(eb, dtype=np.int64)]
    ida, idb = ids_sorted[ra], ids_sorted[rb]
    own_a = ida <= idb
    owner = np.where(own_a, ra, rb)
    other = np.where(own_a, rb, ra)
    ocode = out[f"{prefix}_code"][owner]
    eo = np.argsort(ocode, kind="stable")
    out[f"{prefix}_ea"] = ra[eo].astype(np.uint32)       # edge source row (direction kept)
    out[f"{prefix}_eb"] = rb[eo].astype(np.uint32)       # edge target row
    out[f"{prefix}_ek"] = np.asarray(ek, dtype=np.uint8)[eo]
    out[f"{prefix}_ec"] = np.clip(np.asarray(ec, dtype=np.float32)[eo] * 255.0, 0, 255).astype(np.uint8)
    out[f"{prefix}_ew"] = np.asarray(ew, dtype=np.uint32)[eo]
    out[f"{prefix}_ecode"] = ocode[eo]
    rcode = out[f"{prefix}_code"][other[eo]]
    ro = np.argsort(rcode, kind="stable")
    out[f"{prefix}_rcode"] = rcode[ro]
    out[f"{prefix}_ridx"] = ro.astype(np.uint32)
    return out


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


def build(src: TileSource, out_dir: Path, generation: int,
          progress: Callable[[str], None] | None = None) -> dict:
    """Write tiles-<generation>.npz + manifest.json under `out_dir`.
    Returns the manifest."""
    t0 = time.perf_counter()
    n = len(src.ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n == 0:
        bbox = (-1.0, -1.0, 2.0)
    else:
        minx, maxx = float(src.x.min()), float(src.x.max())
        miny, maxy = float(src.y.min()), float(src.y.max())
        side = max(maxx - minx, maxy - miny, 1.0)
        pad = side * 0.125 + 50.0      # room for nodes placed by later incrementals
        bbox = (minx - pad, miny - pad, side + 2 * pad)
    id_row = {int(v): i for i, v in enumerate(src.ids.tolist())}
    ea = np.array([id_row.get(int(a), -1) for a in src.edge_a.tolist()], dtype=np.int64)
    eb = np.array([id_row.get(int(b), -1) for b in src.edge_b.tolist()], dtype=np.int64)
    ok = (ea >= 0) & (eb >= 0) & (ea != eb)
    ea, eb = ea[ok], eb[ok]
    ek = src.edge_kind[ok]
    ec = src.edge_conf[ok]
    arrays: dict[str, np.ndarray] = {}
    # ---- sym ----
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
    pseudo_names: dict[int, str] = {}
    need = np.nonzero(node_clu < 0)[0]
    if len(need):
        slot: dict[str, int] = {}
        for i in need.tolist():
            f = int(fr[i]) if fr[i] >= 0 else i
            path = src.names[f] if src.kinds[f] == KIND_ID["File"] else ""
            d = path.split("/")[0] if "/" in path else "(root)"
            if d not in slot:
                slot[d] = -(len(slot) + 1)
                pseudo_names[slot[d]] = d + "/"
            node_clu[i] = slot[d]
    if progress:
        progress("tiles: symbols")
    arrays.update(_lod_arrays("sym", src.ids, src.kinds, src.x, src.y, src.pagerank, size,
                              node_clu, src.flags, src.names, ea, eb, ek, ec,
                              np.ones(len(ea), np.uint32), bbox))
    # ---- file ----
    frow_of = np.full(n, -1, dtype=np.int64)
    frow_of[files] = np.arange(len(files))
    node_file = np.where(has_file, frow_of[np.clip(fr, 0, None)], -1)
    node_file[files] = np.arange(len(files))
    fa_, fb_ = node_file[ea], node_file[eb]
    okf = (fa_ >= 0) & (fb_ >= 0)
    ga, gb, gk, gc, gw = _aggregate(fa_[okf], fb_[okf], ek[okf], ec[okf],
                                    np.ones(int(okf.sum()), np.uint32), max(1, len(files)))
    fpr = np.zeros(len(files), dtype=np.float64)
    np.add.at(fpr, node_file[node_file >= 0], src.pagerank[node_file >= 0])
    if progress:
        progress("tiles: files")
    arrays.update(_lod_arrays("file", src.ids[files], np.zeros(len(files), np.uint8),
                              src.x[files], src.y[files], fpr, size[files], node_clu[files],
                              np.zeros(len(files), np.uint8), [src.names[i] for i in files.tolist()],
                              ga, gb, gk, gc, gw, bbox))
    # ---- clu ----
    cids = np.unique(node_clu[files]) if len(files) else np.zeros(0, np.int64)
    crow = {int(c): i for i, c in enumerate(cids.tolist())}
    fclu = np.array([crow[int(c)] for c in node_clu[files].tolist()], dtype=np.int64)
    ccount = np.bincount(fclu, weights=size[files].astype(np.float64), minlength=len(cids))
    cx = np.bincount(fclu, weights=src.x[files] * size[files], minlength=len(cids)) / np.maximum(ccount, 1)
    cy = np.bincount(fclu, weights=src.y[files] * size[files], minlength=len(cids)) / np.maximum(ccount, 1)
    cpr = np.bincount(fclu, weights=fpr, minlength=len(cids))
    cnames: list[str] = []
    cr = np.zeros(len(cids))
    for i, c in enumerate(cids.tolist()):
        info = src.clusters.get(int(c)) or {}
        cnames.append(str(info.get("name") or pseudo_names.get(int(c)) or f"#{c}"))
    # radius: farthest member file from the centre
    dx = src.x[files] - cx[fclu]
    dy = src.y[files] - cy[fclu]
    np.maximum.at(cr, fclu, np.sqrt(dx * dx + dy * dy))
    ca, cb, ck, cc, cw = _aggregate(fclu[ga] if len(ga) else ga, fclu[gb] if len(gb) else gb,
                                    gk, gc, gw, max(1, len(cids)))
    arrays.update(_lod_arrays("clu", cids, np.full(len(cids), KIND_ID["Cluster"], np.uint8),
                              cx, cy, cpr, ccount.astype(np.uint32), cids,
                              np.zeros(len(cids), np.uint8), cnames, ca, cb, ck, cc, cw, bbox))
    arrays["clu_r"] = cr[np.argsort(_quantise(cx, cy, bbox), kind="stable")].astype(np.float32)
    if progress:
        progress("tiles: writing")
    name = f"tiles-{generation}.npz"
    tmp = out_dir / (name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, out_dir / name)
    clusters = [{"id": int(c), "name": cnames[i], "x": float(cx[i]), "y": float(cy[i]),
                 "r": float(cr[i]), "size": int(ccount[i])} for i, c in enumerate(cids.tolist())]
    manifest = {
        "version": VERSION, "generation": int(generation), "file": name,
        "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[2]],
        "content_bbox": ([float(src.x.min()), float(src.y.min()), float(src.x.max()), float(src.y.max())]
                         if n else [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[2]]),
        "max_level": MAX_LEVEL, "tile_budget": TILE_BUDGET,
        "node_kinds": NODE_KINDS, "edge_kinds": EDGE_KINDS, "lods": list(LODS),
        "counts": {"sym_nodes": int(len(arrays["sym_id"])), "sym_edges": int(len(arrays["sym_ea"])),
                   "file_nodes": int(len(arrays["file_id"])), "file_edges": int(len(arrays["file_ea"])),
                   "clu_nodes": int(len(arrays["clu_id"])), "clu_edges": int(len(arrays["clu_ea"]))},
        "kind_counts": {NODE_KINDS[k]: int(c) for k, c in
                        enumerate(np.bincount(src.kinds, minlength=4)[:4].tolist())},
        "edge_kind_counts": {EDGE_KINDS[k]: int(c) for k, c in
                             enumerate(np.bincount(ek, minlength=len(EDGE_KINDS)).tolist()) if c},
        "pr_max": float(src.pagerank[src.kinds != KIND_ID["File"]].max())
        if (src.kinds != KIND_ID["File"]).any() else 0.0,
        "has_conf": bool(((ek == EDGE_KIND_ID["CALLS"]) & (ec < 0.999)).any()),
        "clusters": sorted(clusters, key=lambda c: -c["size"]),
        "built_at": time.time(), "build_seconds": round(time.perf_counter() - t0, 3),
    }
    tmpm = out_dir / "manifest.json.tmp"
    tmpm.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmpm, out_dir / "manifest.json")
    for old in out_dir.glob("tiles-*.npz"):
        if old.name != name:
            try:
                old.unlink()
            except OSError:
                pass
    return manifest


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
                with np.load(self.dir / man["file"], allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
            except Exception:
                return
            self.manifest, self.a, self._mtime = man, arrays, mt

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
        gen = int(self.manifest.get("generation", 0))
        head = struct.pack("<IHBBIIIIIIII", MAGIC, VERSION, LODS.index(lod), level, tx, ty,
                           n, len(eidx), g, len(pieces), len(nblob), gen & 0xFFFFFFFF)
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
    10-byte etag (+2 pad), bytes (padded to 4)."""
    out = [struct.pack("<I", len(items))]
    for level, tx, ty, body, etag in items:
        out.append(struct.pack("<IIII", level, tx, ty, len(body)))
        out.append(bytes.fromhex(etag).ljust(12, b"\0"))
        out.append(_pad4(body))
    return b"".join(out)
