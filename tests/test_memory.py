"""Host memory: compact live structures behave like the plain ones, the
memory-mapped tile sidecar, the scaled Kuzu buffer pool, the small-table
embedding cache, and the idle graph unload (everything rebuilds lazily and
an index pass after it still leaves the same graph)."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from docgraph import procmem
from docgraph.config import load_config
from docgraph.db import GraphDB, buffer_pool_bytes, set_buffer_pool_mb
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.live import SymbolTable
from docgraph.retrieve import Retriever
from docgraph.workspace import Workspace
from tests.fw_fixture import materialize


def test_trim_and_memory_report():
    m = procmem.memory()
    assert m["rss_mb"] > 0 and m["private_mb"] > 0
    assert procmem.trim("test") is True
    assert procmem.trim("test", min_interval=3600) is False     # rate-limited


def test_mmap_npz_roundtrip(tmp_path):
    a = {"x": np.arange(10, dtype=np.float32), "ids": np.arange(5, dtype=np.int64) * 7,
         "blob": np.frombuffer(b"hello world", dtype=np.uint8), "empty": np.zeros(0, np.int32),
         "m": np.arange(12, dtype=np.int16).reshape(3, 4)}
    p = tmp_path / "t.npz"
    with open(p, "wb") as fh:
        np.savez(fh, **a)
    got = procmem.mmap_npz(p)
    assert got is not None and set(got) == set(a)
    for k, v in a.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape and np.array_equal(got[k], v)
    assert isinstance(got["x"], np.memmap)
    del got
    gc.collect()
    q = tmp_path / "c.npz"
    np.savez_compressed(q, **a)
    assert procmem.mmap_npz(q) is None                 # compressed: caller falls back


def test_buffer_pool_scaling(tmp_path):
    small = tmp_path / "db"
    small.mkdir()
    assert buffer_pool_bytes(small) == 256 << 20       # floor
    (small / "blob").write_bytes(b"\0" * (4 << 20))
    assert buffer_pool_bytes(small) == 256 << 20
    assert buffer_pool_bytes(small, 300) == 300 << 20
    try:
        set_buffer_pool_mb(128)
        assert buffer_pool_bytes(small) == 128 << 20
    finally:
        set_buffer_pool_mb(0)


def test_packed_symbol_table_matches_plain():
    st = SymbolTable()
    ents = [("Function", 7, "run", "a.py::run"), ("Class", 8, "A", "a.py::A"),
            ("Function", 9, "m", "a.py::A::m"), ("Variable", 10, "X", "a.py::X"),
            ("Function", 11, "odd name", "a.py::weird")]
    st.add_file("a.py", 1, ents)
    assert st.qname_index["a.py::A"] == ("Class", 8)
    assert st.qname_index.get("a.py::A::m") == ("Function", 9)
    assert st.qname_index.get("nope") is None and "a.py::X" in st.qname_index
    assert dict(st.qname_index.items())["a.py::X"] == ("Variable", 10)
    assert st.id_names.get(9) == "m" and st.id_names.get(11) == "odd name"
    assert st.id_names.get(99, "") == "" and 9 in st.method_ids
    assert st.by_file["a.py"][0] == ("Function", 7, "run", "a.py::run")
    names = st.remove_files(["a.py"])
    assert names == {"run", "A", "m", "X", "odd name"}
    assert not st.qname_index and st.id_names.get(9) is None and not st.by_file


def _full(root: Path) -> None:
    cfg = load_config(root)
    db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=Embedder(cfg.embedding_model))
    ix.index_all(incremental=False)
    ix.db.close()
    db.close()
    del ix, db
    gc.collect()


def test_local_embeddings_equal_db(fw_indexed):
    _cfg, db, _r, _s = fw_indexed
    r = Retriever(db, None)
    ids = [x["id"] for x in db.fetch_all("MATCH (n:Function) RETURN n.id AS id LIMIT 20")]
    got = r._local_embeddings("Function", ids + [10 ** 12])
    assert got is not None
    ref_ids, ref_mat = db.embeddings_for("Function", ids)
    order = {int(i): k for k, i in enumerate(ref_ids.tolist())}
    assert sorted(got[0].tolist()) == sorted(order)
    for i, row in zip(got[0].tolist(), got[1]):
        assert np.allclose(row, ref_mat[order[int(i)]])
    r2 = Retriever(db, None)
    r2.LOCAL_VEC_MAX_ROWS = 0                           # big table: read from Kuzu
    assert r2._local_embeddings("Function", ids) is None


def test_graph_unload_then_everything_rebuilds(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    materialize(work)
    _full(work)
    ws = Workspace([load_config(work)])
    try:
        ws.schedule_maintenance = lambda *a, **k: None
        slot = ws.resolve(work)
        r = slot.retriever
        before = r.trace("read_item", "_write")
        r.explore(seeds=["read_item"], hops=2, limit=10)
        assert getattr(r, "_mem_graph_cache", None) is not None
        f = work / "lib" / "tools.py"
        f.write_text(f.read_text(encoding="utf-8") + "\n\ndef unload_probe_zq(x):\n    return x\n",
                     encoding="utf-8")
        ws.index_pass(work, incremental=True, changed_paths=[str(f)])
        ws._persist_pool().submit(lambda: None).result(timeout=60)
        assert slot.live.symtab.qname_index
        # tile arrays are served memory-mapped once persisted
        assert slot.tile_store.get_manifest() is not None
        assert any(isinstance(v, np.memmap) for v in slot.tile_store.a.values())
        assert ws.unload_graph(slot)
        assert slot.maint["graph_loaded"] is False
        assert not slot.live.symtab.qname_index and not slot.live.cache
        assert slot.tile_store.manifest is None
        r2 = slot.retriever
        assert getattr(r2, "_mem_graph_cache", None) is None
        # everything comes back on demand
        assert r2.trace("read_item", "_write")["path"] == before["path"]
        assert r2.search("unload_probe_zq", limit=5)
        assert slot.tile_store.get_manifest() is not None
        f.write_text(f.read_text(encoding="utf-8") + "\n\ndef unload_probe_two_zq(x):\n    return x\n",
                     encoding="utf-8")
        st = ws.index_pass(work, incremental=True, changed_paths=[str(f)])
        assert st["changed"] == 1 and st["errors"] == 0
        rows = slot.db_ro.fetch_all("MATCH (n:Function) WHERE n.name STARTS WITH 'unload_probe' "
                                    "RETURN n.name AS nm ORDER BY nm")
        assert [x["nm"] for x in rows] == ["unload_probe_two_zq", "unload_probe_zq"]
        # the idle tick unloads on its own once the window has passed
        ws.graph_unload_after = 0.01
        slot.maint["graph_loaded"] = True
        for k in ("last_use", "last_pass", "maint_end", "started"):
            slot.maint[k] = 0.0
        ws._idle_memory_tick()
        assert slot.maint["graph_loaded"] is False
    finally:
        ws.close()
