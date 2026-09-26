"""Host-side incremental passes: Workspace.index_pass keeps the live state,
defers entity vectors / SIMILAR_TO / the tile patch, and maintenance lands
them -- ending in the same graph a CLI pass would produce."""
from __future__ import annotations

import gc
from pathlib import Path

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.workspace import Workspace
from tests.fw_fixture import materialize


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


def test_index_pass_defers_then_maintenance_lands(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    materialize(work)
    _full(work)
    ws = Workspace([load_config(work)])
    try:
        ws.schedule_maintenance = lambda *a, **k: None        # run it by hand below
        slot = ws.resolve(work)
        f = work / "lib" / "tools.py"
        f.write_text(f.read_text(encoding="utf-8")
                     + "\n\ndef brand_new_helper_zq(x):\n    return unique_thing() + x\n", encoding="utf-8")
        stats = ws.index_pass(work, incremental=True, changed_paths=[str(f)])
        assert stats["changed"] == 1 and stats["errors"] == 0, stats
        assert stats["scope"]["scoped"]
        # the new function's vector was deferred, not written
        q = slot.maint.get("vec_queue") or []
        assert q and any(label == "Function" for label, *_ in q), q
        fid = slot.db_ro.fetch_all("MATCH (n:Function) WHERE n.name = 'brand_new_helper_zq' RETURN n.id AS id")
        assert fid, "entity row written by the pass"
        fid = int(fid[0]["id"])
        assert not slot.db_ro.fetch_all(f"MATCH (v:FnVec) WHERE v.id = {fid} RETURN v.id")
        # keyword search sees it right away (live keyword index)
        hits = slot.retriever.search("brand_new_helper_zq", limit=5)
        assert any(h.get("name") == "brand_new_helper_zq" for h in hits), hits
        # the tile patch runs on the persist thread
        ws._persist_pool().submit(lambda: None).result(timeout=60)
        man = slot.tile_store.get_manifest()
        assert man and man.get("generation", 0) >= 2
        # maintenance: vectors land, SIMILAR_TO for the new entity, health precomputed
        slot.maint["health_keys"] = {(15, 0.5)}
        done = ws._maintenance(slot)
        assert done.get("vectors", 0) >= 1, done
        assert not slot.maint.get("vec_queue")
        assert slot.db_ro.fetch_all(f"MATCH (v:FnVec) WHERE v.id = {fid} RETURN v.id")
        assert not list((slot.cfg.data_dir / "pending_vectors").glob("*.npz"))
        h = slot.retriever.health()
        assert "hubs" in h and not h.get("refreshing")
        # a second pass on the same file: the warm state is reused (no symbol
        # table rebuild) and the earlier deferred vectors do not resurrect rows
        f.write_text(f.read_text(encoding="utf-8").replace("return unique_thing() + x",
                                                           "return unique_thing() - x"), encoding="utf-8")
        stats2 = ws.index_pass(work, incremental=True, changed_paths=[str(f)])
        assert stats2["changed"] == 1 and "symtab" not in {k for k, v in stats2["timings"].items() if v > 50}
        ws._maintenance(slot)
        rows = slot.db_ro.fetch_all("MATCH (n:Function) WHERE n.name = 'brand_new_helper_zq' RETURN n.id AS id")
        assert len(rows) == 1
        vec = slot.db_ro.fetch_all(f"MATCH (v:FnVec) WHERE v.id = {int(rows[0]['id'])} RETURN v.id")
        assert vec, "the re-created entity got its vector"
        assert not slot.db_ro.fetch_all(f"MATCH (v:FnVec) WHERE v.id = {fid} RETURN v.id"), "old vector removed"
    finally:
        ws.close()
