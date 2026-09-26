"""Schema v4: plain-text fallback, notebooks, indexed search (+ brute
fallback), bounded similarity / PageRank / communities, the world layout,
the LOD tile sidecar + endpoints, and the incremental layout / resolution
scope."""
from __future__ import annotations

import gc
import json
import shutil
import struct
import textwrap
from pathlib import Path

import numpy as np
import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.parse import classify_file, notebook_text, parse_file
from docgraph.retrieve import Retriever
from docgraph.summary import text_chunks


# ---------------------------------------------------------------- parsing ---

def test_classify_file_kinds(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\nRUN pip install x\n")
    (tmp_path / "LICENSE").write_text("MIT License\n\nPermission is hereby granted.\n")
    (tmp_path / "notes.zzz").write_text("just some words\n")
    (tmp_path / "blob.zzz").write_bytes(b"\x00\x01\x02binary")
    (tmp_path / "pic.png").write_bytes(b"not really a png")
    (tmp_path / "a.py").write_text("x = 1\n")
    assert classify_file(tmp_path / "a.py") == "python"
    assert classify_file(tmp_path / "Dockerfile") in ("text:dockerfile", "dockerfile")
    assert classify_file(tmp_path / "LICENSE") == "text:txt"
    assert classify_file(tmp_path / "notes.zzz") == "text:text"
    assert classify_file(tmp_path / "blob.zzz") is None
    assert classify_file(tmp_path / "pic.png") is None


def test_text_chunks_line_numbers():
    body = "\n\n".join(("para %d " % i) * 30 for i in range(10))
    lines = body.splitlines()
    chunks = text_chunks(body)
    assert len(chunks) >= 2
    for s, e, b in chunks:
        assert "\n".join(lines[s - 1:e]).strip() == b.strip()
    # a single huge line (minified data) is cut into windows on that line
    big = text_chunks("x" * 5000)
    assert len(big) == 3 and all(s == e == 1 for s, e, _b in big)


def test_parse_text_fallback_encodings(tmp_path: Path):
    p = tmp_path / "README"
    p.write_bytes(b"\xef\xbb\xbfTitle\n\nSome text about tiles.\n")
    fp = parse_file(p, tmp_path)
    assert fp is not None and fp.language == "txt" and fp.entities == []
    assert fp.chunks and fp.chunks[0]["body"].startswith("Title")
    c = tmp_path / "legacy.txt"
    c.write_bytes("caf\xe9 cr\xe8me".encode("cp1252"))
    fp2 = parse_file(c, tmp_path)
    assert fp2 is not None and "caf" in fp2.chunks[0]["body"]
    assert parse_file(p, tmp_path, text_fallback=False) is None


def test_symbolless_grammar_file_gets_chunks(tmp_path: Path):
    j = tmp_path / "data.json"
    j.write_text(json.dumps({"alpha": 1, "beta": [1, 2, 3]}, indent=2))
    fp = parse_file(j, tmp_path)
    assert fp is not None and fp.language == "json" and fp.chunks
    py = tmp_path / "m.py"
    py.write_text("def f():\n    return 1\n")
    assert parse_file(py, tmp_path).chunks == []


def _notebook() -> str:
    return json.dumps({
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# Analysis\n", "\n", "We load the data here.\n"]},
            {"cell_type": "code", "source": ["def load_frame(path):\n", "    return open(path).read()\n"]},
            {"cell_type": "code", "source": "x = load_frame('a.csv')\n"},
        ],
    })


def test_notebook_parse(tmp_path: Path):
    nb = tmp_path / "an.ipynb"
    nb.write_text(_notebook())
    fp = parse_file(nb, tmp_path)
    assert fp is not None and fp.language == "python"
    ent = {e.name: e for e in fp.entities}
    assert "load_frame" in ent
    text, lang, spans = notebook_text(nb.read_bytes())
    assert lang == "python"
    lines = text.split("\n")
    e = ent["load_frame"]
    assert lines[e.line_start - 1].startswith("def load_frame")
    assert fp.chunks and "load the data" in fp.chunks[0]["body"]
    s = fp.chunks[0]["line_start"]
    assert lines[s - 1].startswith("# Analysis")
    assert any(r.kind == "CALLS" and r.target_name == "load_frame" for r in fp.edges)


# ---------------------------------------------------------- bounded math ---

def test_similarity_blocked_and_ivf_agree():
    from docgraph.similar import top_similar
    rng = np.random.default_rng(0)
    base = rng.normal(size=(120, 32)).astype(np.float32)
    x = np.repeat(base, 30, axis=0) + 0.25 * rng.normal(size=(3600, 32)).astype(np.float32)
    s1, d1, v1 = top_similar(x, 5, 0.5)                       # exact blocks
    s2, d2, v2 = top_similar(x, 5, 0.5, exact_max=100)        # IVF path
    assert not (s1 == d1).any() and not (s2 == d2).any()
    assert (v1 >= 0.5).all() and (v2 >= 0.5).all()
    a, b = set(zip(s1.tolist(), d1.tolist())), set(zip(s2.tolist(), d2.tolist()))
    assert len(a & b) / len(a) > 0.95


def test_pagerank_matches_networkx():
    import networkx as nx
    from docgraph.rank import ScoreMap, _Graph
    rng = np.random.default_rng(3)
    src, dst = rng.integers(0, 300, 1500), rng.integers(0, 300, 1500)
    g = _Graph(src.astype(np.int64), dst.astype(np.int64))
    x = g.pagerank(tol=1e-12, max_iter=500)
    ref = nx.pagerank(nx.DiGraph(list(zip(src.tolist(), dst.tolist()))), tol=1e-12, max_iter=1000)
    assert max(abs(x[i] - ref[int(nd)]) for i, nd in enumerate(g.nodes)) < 1e-8
    sm = ScoreMap(g.nodes, x)
    k = int(g.nodes[5])
    assert sm.get(k) == pytest.approx(ref[k], abs=1e-8) and k in sm and sm.get(10**9, 0.0) == 0.0


def test_communities_fold_tiers(monkeypatch):
    from docgraph import communities as C
    ids, labels, names, files, pr = [], [], [], [], []
    ea, eb, ew = [], [], []
    nid = 1
    for d in range(4):
        for f in range(6):
            path = f"pkg{d}/sub/m{f}.py"
            fid = nid
            nid += 1
            ids.append(fid); labels.append("File"); names.append(path); files.append(path); pr.append(0.0)
            for s in range(5):
                ids.append(nid); labels.append("Function"); names.append(f"f{nid}"); files.append(path)
                pr.append(0.01)
                ea.append(fid); eb.append(nid); ew.append(0.5)
                if s:
                    ea.append(nid - 1); eb.append(nid); ew.append(1.0)
                nid += 1
            if f:
                ea.append(fid - 6); eb.append(fid); ew.append(0.3)
        if d:   # one import between neighbouring packages
            ea.append(ids[files.index(f"pkg{d - 1}/sub/m0.py")]); eb.append(ids[files.index(f"pkg{d}/sub/m0.py")])
            ew.append(0.3)
    order = np.argsort(ids)
    args = (np.asarray(ids)[order], [labels[i] for i in order], [names[i] for i in order],
            [files[i] for i in order], np.asarray(pr)[order], np.asarray(ea), np.asarray(eb), np.asarray(ew))
    full = C.detect_arrays(*args)
    assert full and sum(c.size for c in full) <= len(ids)
    monkeypatch.setattr(C, "MAX_FINE_NODES", 10)       # fold symbols -> files
    folded = C.detect_arrays(*args)
    by = {}
    for c in folded:
        for m in c.members:
            by[m] = c.index
    for i, lab, f in zip(ids, labels, files):          # a file and its symbols stay together
        if lab == "Function":
            fid = ids[files.index(f)]
            assert by.get(i) == by.get(fid)
    monkeypatch.setattr(C, "MAX_FOLD_NODES", 4)        # ... and files -> directories
    dirs = C.detect_arrays(*args)
    assert dirs
    by = {m: c.index for c in dirs for m in c.members}
    for i, f in zip(ids, files):                       # a directory never splits
        j = ids[files.index(f.rsplit("/", 1)[0] + "/m0.py")]
        assert by.get(i) == by.get(j)


def test_layout_deterministic_and_non_overlapping():
    from docgraph import layout as L
    rng = np.random.default_rng(1)
    F, S = 60, 600
    kinds = np.concatenate([np.zeros(F, np.uint8), np.full(S, 2, np.uint8)])
    file_of = np.concatenate([np.arange(F), rng.integers(0, F, S)])
    order = np.concatenate([np.zeros(F), np.arange(S)]).astype(float)
    paths = {i: f"d{i % 5}/m{i}.py" for i in range(F)}
    comm = {i: i // 12 for i in range(F)}
    fa, fb = rng.integers(0, F, 200), rng.integers(0, F, 200)
    fw = np.ones(200)
    r1 = L.compute_layout(kinds, file_of, order, paths, comm, fa, fb, fw)
    r2 = L.compute_layout(kinds, file_of, order, paths, comm, fa, fb, fw)
    assert np.array_equal(r1.x, r2.x) and np.array_equal(r1.y, r2.y)
    counts = np.bincount(file_of[F:], minlength=F)
    rad = np.array([L.file_radius(int(c)) for c in counts])
    for a in range(F):
        for b in range(a + 1, F):
            d = np.hypot(r1.x[a] - r1.x[b], r1.y[a] - r1.y[b])
            assert d >= (rad[a] + rad[b]) * 0.98, (a, b)
    # every symbol sits inside its file's disc
    for i in range(F, F + S):
        f = file_of[i]
        assert np.hypot(r1.x[i] - r1.x[f], r1.y[i] - r1.y[f]) <= rad[f] + 1e-6


# ------------------------------------------------------------------ tiles ---

def _decode(body: bytes) -> dict:
    """Python mirror of the UI's tile decoder (docgraph/ui/index.html parseTile)."""
    magic, ver, lod, level, tx, ty, n, m, g, k, nb, gen = struct.unpack_from("<IHBBIIIIIIII", body, 0)
    assert magic == 0x31544744
    o = 40

    def arr(dt, c, item=4):
        nonlocal o
        a = np.frombuffer(body, dtype=dt, count=c, offset=o)
        o += c * item
        return a

    def u8(c):
        nonlocal o
        a = np.frombuffer(body, dtype=np.uint8, count=c, offset=o)
        o += (c + 3) & ~3
        return a
    t = {"lod": lod, "level": level, "tx": tx, "ty": ty}
    t["ids"] = arr(np.int32, n); t["x"] = arr(np.float32, n); t["y"] = arr(np.float32, n)
    t["pr"] = arr(np.float32, n); t["size"] = arr(np.uint32, n); t["clu"] = arr(np.int32, n)
    t["kind"] = u8(n); t["flags"] = u8(n)
    t["gids"] = arr(np.int32, g); t["gx"] = arr(np.float32, g); t["gy"] = arr(np.float32, g)
    t["gclu"] = arr(np.int32, g); t["gkind"] = u8(g)
    t["eid"] = arr(np.uint32, m); t["ea"] = arr(np.uint32, m); t["eb"] = arr(np.uint32, m)
    t["ew"] = arr(np.uint32, m); t["ek"] = u8(m); t["ec"] = u8(m)
    nl = arr(np.uint32, k); off = arr(np.uint32, k + 1)
    blob = body[o:o + nb]
    t["names"] = {int(nl[i]): blob[off[i]:off[i + 1]].decode() for i in range(k)}
    return t


def _synthetic_source(n_files: int = 40, per_file: int = 25):
    from docgraph import tiles as T
    from docgraph import layout as L
    ids, kinds, file_row, names = [], [], [], []
    for f in range(n_files):
        fr = len(ids)
        ids.append(1000 + fr); kinds.append(0); file_row.append(fr); names.append(f"d{f % 4}/m{f}.py")
        for s in range(per_file):
            ids.append(1000 + len(ids)); kinds.append(2); file_row.append(fr); names.append(f"fn_{f}_{s}")
    n = len(ids)
    kinds_a = np.asarray(kinds, np.uint8)
    fr_a = np.asarray(file_row, np.int64)
    res = L.compute_layout(kinds_a, fr_a, np.arange(n, dtype=float),
                           {i: names[i] for i in range(n) if kinds[i] == 0},
                           {i: (i // 150) for i in range(n) if kinds[i] == 0},
                           np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
    rng = np.random.default_rng(2)
    ea = rng.integers(0, n, 3000)
    eb = rng.integers(0, n, 3000)
    comm = np.array([100 + (fr_a[i] // 150) for i in range(n)], np.int64)
    src = T.TileSource(
        ids=np.asarray(ids, np.int64), kinds=kinds_a, x=res.x, y=res.y,
        pagerank=rng.random(n).astype(np.float32), file_row=fr_a, community=comm, names=names,
        flags=np.zeros(n, np.uint8), edge_a=np.asarray(ids)[ea], edge_b=np.asarray(ids)[eb],
        edge_kind=np.full(3000, T.EDGE_KIND_ID["CALLS"], np.uint8),
        edge_conf=rng.random(3000).astype(np.float32), clusters={100: {"name": "c0"}})
    return src


def test_tiles_partition_and_edges(tmp_path: Path):
    from docgraph import tiles as T
    src = _synthetic_source()
    man = T.build(src, tmp_path, generation=7)
    assert man["generation"] == 7 and man["counts"]["sym_nodes"] == len(src.ids)
    st = T.TileStore(tmp_path)
    level = 4
    seen: list[int] = []
    edges: set[int] = set()
    for tx in range(1 << level):
        for ty in range(1 << level):
            body, etag = st.tile(level, tx, ty, lod="sym")
            t = _decode(body)
            seen += t["ids"].tolist()
            edges |= set(t["eid"].tolist())
            assert st.tile(level, tx, ty, lod="sym")[1] == etag     # stable ETag
            if len(t["ids"]):  # nodes lie inside their tile
                b = man["bbox"]
                w = (b[2] - b[0]) / (1 << level)
                assert (t["x"] >= b[0] + tx * w - 1e-3).all() and (t["x"] <= b[0] + (tx + 1) * w + 1e-3).all()
            for a_, b_ in zip(t["ea"].tolist(), t["eb"].tolist()):   # every endpoint decodable
                assert a_ < len(t["ids"]) + len(t["gids"]) and b_ < len(t["ids"]) + len(t["gids"])
    assert sorted(seen) == sorted(src.ids.tolist())            # every node exactly once per level
    assert len(edges) == man["counts"]["sym_edges"]            # every edge served by some tile
    # LOD by budget: the whole world at level 0 is clusters, the finest level always symbols
    assert _decode(st.tile(0, 0, 0, budget=50)[0])["lod"] == 1      # 40 files fit, 1040 symbols do not
    assert _decode(st.tile(0, 0, 0, budget=20)[0])["lod"] == 2      # neither: cluster super-nodes
    assert _decode(st.tile(T.MAX_LEVEL, 0, 0, budget=1)[0])["lod"] == 0
    # locate + batch framing
    loc = st.locate([int(src.ids[3]), 999999])
    assert len(loc) == 1 and loc[0]["name"] == src.names[3]
    fr = T.frame_batch([(1, 0, 0, b"abcd", "00" * 10)])
    assert struct.unpack_from("<I", fr, 0)[0] == 1 and fr[-4:] == b"abcd"


def test_tile_filters(tmp_path: Path):
    from docgraph import tiles as T
    src = _synthetic_source(10, 10)
    T.build(src, tmp_path, generation=1)
    st = T.TileStore(tmp_path)
    full = _decode(st.tile(0, 0, 0, lod="sym")[0])
    only_files = _decode(st.tile(0, 0, 0, lod="sym", node_kinds={T.KIND_ID["File"]})[0])
    assert len(only_files["ids"]) == 10 < len(full["ids"])
    hi = _decode(st.tile(0, 0, 0, lod="sym", min_conf=0.9)[0])
    assert len(hi["eid"]) < len(full["eid"])


# ------------------------------------------------- indexing integration ---

TEXT_REPO = {
    "app/core.py": '''
        """Core module."""


        def orbit_planner(n):
            """Plan the satellite orbit."""
            return helper_step(n) + 1


        def helper_step(n):
            return n * 2
        ''',
    "app/use.py": '''
        from app.core import orbit_planner


        def run_it():
            return orbit_planner(3)
        ''',
    "README.md": "# Zephyrine\n\nThe zephyrine subsystem calibrates telescopes at dawn.\n",
    "Dockerfile": "FROM python:3.12-slim\nRUN pip install zephyrine\n",
    "LICENSE": "Permission is hereby granted, free of charge, to any person.\n",
    "docs/guide.rst": "Guide\n=====\n\nMagnetometer drift compensation is described here.\n",
    "config/app.cfg": "[server]\nport = 8080\nquokka_mode = enabled\n",
    "nb/explore.ipynb": _notebook(),
    "assets/blob.bin2": None,
}


@pytest.fixture(scope="module")
def text_indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("text_repo")
    for rel, content in TEXT_REPO.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            p.write_bytes(b"\x00\x00binary\x00stuff")
        else:
            p.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    cfg = load_config(root, history=False)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    emb = Embedder(cfg.embedding_model)
    ix = Indexer(cfg, db, embedder=emb)
    stats = ix.index_all(incremental=False)
    ix.db.close()
    db.close()
    ro = GraphDB(cfg.db_path, embedding_dim=384, read_only=True)
    r = Retriever(ro, emb, cfg=cfg)
    yield cfg, ro, r, emb, stats
    ro.close()
    gc.collect()


def test_text_files_become_file_nodes(text_indexed):
    cfg, db, r, _e, stats = text_indexed
    langs = {x["p"]: x["l"] for x in db.fetch_all("MATCH (f:File) RETURN f.path AS p, f.language AS l")}
    assert langs["Dockerfile"] in ("dockerfile",)
    assert langs["LICENSE"] == "txt" and langs["docs/guide.rst"] == "rst"
    assert langs["config/app.cfg"] == "ini" and "assets/blob.bin2" not in langs
    assert langs["nb/explore.ipynb"] == "python"
    rows = db.fetch_all("MATCH (f:File)-[:CONTAINS_CHUNK]->(c:Chunk) WHERE f.path = 'README.md' "
                        "RETURN c.line_start AS s, c.body AS b")
    assert rows and rows[0]["s"] == 1 and "zephyrine" in rows[0]["b"]
    fm = r.file_map("LICENSE")
    assert fm["entities"] == []


def test_indexed_search_finds_text_and_code(text_indexed):
    cfg, db, r, _e, _s = text_indexed
    assert r.search_backend() == "index"
    hits = r.search("magnetometer drift compensation", limit=5)
    assert any(h["label"] == "File" and h["file"] == "docs/guide.rst" for h in hits), hits
    kw = r.search("quokka_mode", limit=5)
    assert any(h["file"] == "config/app.cfg" for h in kw), kw
    code = r.search("orbit planner", limit=5)
    assert code and code[0]["name"] == "orbit_planner"
    assert all(h["label"] == "File" for h in r.search("zephyrine", kind="file", limit=5))
    assert all(h["label"] == "Function" for h in r.search("zephyrine", kind="function", limit=5))


def test_brute_search_fallback_without_indexes(text_indexed):
    cfg, db, r, emb, _s = text_indexed
    r2 = Retriever(db, emb, cfg=cfg)
    r2._search_backend_cache = "brute"      # a pre-v4 database has no indexes
    hits = r2.search("orbit planner", limit=5)
    assert hits and hits[0]["name"] == "orbit_planner"


def test_layout_and_tiles_persisted(text_indexed):
    cfg, db, r, _e, _s = text_indexed
    miss = db.fetch_all("MATCH (n:Function) WHERE n.x IS NULL RETURN count(n) AS c")[0]["c"]
    assert miss == 0
    st = json.loads((cfg.data_dir / "state.json").read_text())
    assert st["tiles"]["generation"] >= 1 and st["analytics"]["layout"] == "global"
    assert st["search_index"]["vector"]["Function"] == "ok"
    man = json.loads((cfg.data_dir / "tiles" / "manifest.json").read_text())
    assert man["counts"]["sym_nodes"] == db.count_symbols()


def test_incremental_keeps_layout_and_resolves_scoped(tmp_path_factory, text_indexed):
    cfg0, _db, _r, emb, _s = text_indexed
    root = tmp_path_factory.mktemp("text_repo_inc")
    shutil.copytree(cfg0.repo_root, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".docgraph"))
    cfg = load_config(root, history=False)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=emb)
    ix.index_all(incremental=False)
    pos0 = {x["p"]: (x["x"], x["y"]) for x in ix.db.fetch_all("MATCH (f:File) RETURN f.path AS p, f.x AS x, f.y AS y")}
    ix.db.close()
    # a new caller of an existing function, in a file that did not exist
    (root / "app" / "later.py").write_text("from app.core import helper_step\n\n\n"
                                           "def later_call():\n    return helper_step(4)\n", encoding="utf-8")
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=emb)
    ix.index_all(incremental=True)
    ix.db.close()
    ro = GraphDB(cfg.db_path, embedding_dim=384, read_only=True)
    try:
        pos1 = {x["p"]: (x["x"], x["y"]) for x in ro.fetch_all("MATCH (f:File) RETURN f.path AS p, f.x AS x, f.y AS y")}
        assert all(pos1[p] == pos0[p] for p in pos0), "unchanged files keep their position"
        assert pos1["app/later.py"][0] is not None
        calls = {(x["a"], x["b"]) for x in ro.fetch_all(
            "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.name AS a, b.name AS b")}
        assert ("later_call", "helper_step") in calls and ("run_it", "orbit_planner") in calls
        st = json.loads((cfg.data_dir / "state.json").read_text())
        assert st["analytics"]["layout"] == "incremental" and st["resolution_scope"]["scoped"]
        assert st["tiles"]["generation"] == 2
    finally:
        ro.close()


def test_big_graph_patch_path(tmp_path_factory, text_indexed):
    """Above full_recompute_max_nodes an incremental pass patches rank /
    communities / positions for the changed files only."""
    cfg0, _db, _r, emb, _s = text_indexed
    root = tmp_path_factory.mktemp("text_repo_big")
    shutil.copytree(cfg0.repo_root, root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".docgraph"))
    cfg = load_config(root, history=False, full_recompute_max_nodes=1, recompute_drift=1.0)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=emb)
    ix.index_all(incremental=False)
    ix.db.close()
    p = root / "app" / "core.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n\ndef extra_one():\n    return helper_step(1)\n",
                 encoding="utf-8")
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=emb)
    ix.index_all(incremental=True)
    ix.db.close()
    st = json.loads((cfg.data_dir / "state.json").read_text())
    assert st["analytics"]["stats"] == "patched" and st["analytics"]["layout"] == "incremental"
    ro = GraphDB(cfg.db_path, embedding_dim=384, read_only=True)
    try:
        row = ro.fetch_all("MATCH (f:Function) WHERE f.name = 'extra_one' RETURN f.x AS x, f.pagerank AS pr")
        assert row and row[0]["x"] is not None and row[0]["pr"] is not None
    finally:
        ro.close()


# ---------------------------------------------------------- HTTP / LOD API ---

@pytest.fixture(scope="module")
def text_client(text_indexed):
    from fastapi.testclient import TestClient
    from docgraph.server import make_app
    from docgraph.workspace import Workspace
    cfg = text_indexed[0]
    ws = Workspace([cfg])
    app = make_app(ws)
    with TestClient(app) as c:
        yield c
    ws.close()


def test_tile_endpoints(text_client):
    man = text_client.get("/api/tiles/manifest").json()
    assert man["ready"] and man["counts"]["sym_nodes"] > 0
    r = text_client.get("/api/tiles", params={"level": 0, "x": 0, "y": 0})
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    t = _decode(r.content)
    assert len(t["ids"]) > 0
    etag = r.headers["etag"]
    r2 = text_client.get("/api/tiles", params={"level": 0, "x": 0, "y": 0}, headers={"If-None-Match": etag})
    assert r2.status_code == 304
    b = text_client.get("/api/tiles/batch", params={"keys": "0/0/0,1/0/0,1/1/1"})
    assert b.status_code == 200 and struct.unpack_from("<I", b.content, 0)[0] == 3
    bb = man["content_bbox"]
    bx = text_client.get("/api/tiles/bbox", params={"x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3], "level": 2})
    assert bx.status_code == 200 and struct.unpack_from("<I", bx.content, 0)[0] >= 1
    assert text_client.get("/api/tiles/bbox", params={"x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3],
                                                      "level": 16}).status_code in (200, 400)
    loc = text_client.get("/api/tiles/locate", params={"ids": ",".join(str(i) for i in t["ids"][:3].tolist())}).json()
    assert loc["ready"] and len(loc["nodes"]) == min(3, len(t["ids"]))
    nid = int(t["ids"][0])
    nb = text_client.get("/api/node_neighbors", params={"id": nid}).json()
    assert all("x" in n for n in nb["nodes"] if n["id"] == nid)


def test_notebook_file_content_is_virtual_text(text_client):
    r = text_client.get("/api/file_content", params={"file": "nb/explore.ipynb"}).json()
    assert r.get("notebook") and "def load_frame" in r["content"]
