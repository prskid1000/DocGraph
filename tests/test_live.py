"""Live (proportional) incremental index: an incremental pass must leave
the graph exactly as a full index of the same tree would -- resolution,
OVERRIDES, TESTS, IMPORTS, framework HANDLES and the tile sidecar -- while
touching only what the change can reach. Plus the keyword index and the
same-file-edited-twice regression (Kuzu FTS crashed there)."""
from __future__ import annotations

import gc
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.live import LiveIndex
from tests.fw_fixture import materialize

# Edges whose content legitimately differs between an incremental pass and
# a full one (derived clustering / nearest-neighbour drift, documented).
SKIP_RELS = {"MEMBER_OF", "SIMILAR_TO"}


@pytest.fixture(scope="module")
def embedder():
    return Embedder(load_config(Path.cwd()).embedding_model)


def _index(root: Path, embedder, full: bool, live: LiveIndex | None = None,
           changed_paths=None) -> dict:
    cfg = load_config(root)
    db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=embedder, live=live)
    try:
        stats = ix.index_all(incremental=not full, changed_paths=changed_paths)
    finally:
        ix.db.close()
        db.close()
        del ix, db
        gc.collect()
    return stats


def _dump(root: Path) -> tuple[Counter, Counter]:
    """(nodes, edges) keyed by names, never ids."""
    cfg = load_config(root)
    db = GraphDB(cfg.db_path, read_only=True)
    try:
        key: dict[int, tuple] = {}
        nodes: Counter = Counter()
        spec = {"File": "n.path", "Class": "n.qname", "Function": "n.qname", "Variable": "n.qname",
                "Module": "n.name",
                "Route": "n.file + ':' + CAST(n.line AS STRING) + ':' + n.name",
                "Tool": "n.file + ':' + CAST(n.line AS STRING) + ':' + n.name",
                "Chunk": "n.file + ':' + CAST(n.line_start AS STRING) + ':' + n.parent_qname"}
        for label, expr in spec.items():
            if not db.has_table(label):
                continue
            for r in db.fetch_all(f"MATCH (n:{label}) RETURN n.id AS id, {expr} AS k"):
                key[int(r["id"])] = (label, r["k"])
                nodes[(label, r["k"])] += 1
        edges: Counter = Counter()
        rels = [r["name"] for r in db.fetch_all("CALL show_tables() RETURN *") if r.get("type") == "REL"]
        for rel in rels:
            if rel in SKIP_RELS:
                continue
            props = db.table_props(rel)
            extra = []
            for p in ("line", "confidence", "method", "count"):
                if p in props:
                    extra.append(f"r.{p} AS {p}")
            ret = ", ".join(["a.id AS a", "b.id AS b"] + extra)
            for r in db.fetch_all(f"MATCH (a)-[r:{rel}]->(b) RETURN {ret}"):
                ka, kb = key.get(int(r["a"])), key.get(int(r["b"]))
                if ka is None or kb is None:
                    continue
                vals = tuple(round(r[p], 3) if isinstance(r.get(p), float) else r.get(p)
                             for p in ("line", "confidence", "method", "count") if p in props)
                edges[(rel, ka, kb) + vals] += 1
        return nodes, edges
    finally:
        db.close()


def _tile_sets(root: Path) -> tuple[set, set]:
    from docgraph import tiles as T
    got = T.load(load_config(root).data_dir / "tiles")
    assert got is not None
    _man, a = got
    off, blob = a["sym_name_off"], a["sym_name_blob"]
    names = [bytes(blob[off[i]:off[i + 1]]).decode() for i in range(len(a["sym_id"]))]
    ids = a["sym_id"].tolist()
    name_of = dict(zip(ids, names))
    kinds = a["sym_kind"].tolist()
    nodes = {(k, n) for k, n in zip(kinds, names)}
    ea, eb = a["sym_ea"].astype(np.int64), a["sym_eb"].astype(np.int64)
    edges = {(int(k), name_of[ids[x]], name_of[ids[y]]) for x, y, k in
             zip(ea.tolist(), eb.tolist(), a["sym_ek"].tolist())
             if T.EDGE_KINDS[int(k)] not in SKIP_RELS}
    return nodes, edges


def _assert_same_as_full(work: Path, tmp: Path, embedder, tag: str) -> None:
    ref = tmp / f"ref_{tag}"
    if ref.exists():
        shutil.rmtree(ref)
    shutil.copytree(work, ref, ignore=shutil.ignore_patterns(".docgraph"))
    _index(ref, embedder, full=True)
    n1, e1 = _dump(work)
    n2, e2 = _dump(ref)
    assert n1 == n2, f"{tag}: nodes differ: +{dict(n1 - n2)} -{dict(n2 - n1)}"
    assert e1 == e2, f"{tag}: edges differ:\n incremental only {dict(e1 - e2)}\n full only {dict(e2 - e1)}"
    t1n, t1e = _tile_sets(work)
    t2n, t2e = _tile_sets(ref)
    assert t1n == t2n, f"{tag}: tile nodes differ"
    assert t1e == t2e, f"{tag}: tile edges differ: {sorted(t1e ^ t2e)[:10]}"


EDITS = [
    # 1. body edit: same candidate sets, callers re-link into the new ids
    ("body", lambda r: _sub(r / "app/helpers.py", "return str(x).strip()", "return str(x).strip().lower()")),
    # 2. a new definition of a name used elsewhere: callers re-resolve
    ("new-same-name", lambda r: _append(r / "app/helpers.py", "\n\ndef unique_thing():\n    return 'dup'\n")),
    # 3. delete a file: the ambiguous helper() becomes unique
    ("delete", lambda r: (r / "lib/helpers.py").unlink()),
    # 4. add a file with a subclass: INHERITS + OVERRIDES into unchanged code
    ("add-subclass", lambda r: _write(r / "lib/extra.py",
                                      "from app.models import Repo\n\n\nclass Special(Repo):\n"
                                      "    def save(self):\n        return helper()\n\n\n"
                                      "def test_special():\n    return Special().save()\n")),
    # 5. edit the parent: the unchanged subclass re-links INHERITS + OVERRIDES
    ("edit-parent", lambda r: _sub(r / "app/models.py", "return True", "return 1")),
    # 6. the same file again (Kuzu FTS used to crash deleting incremental rows)
    ("same-file-again", lambda r: _sub(r / "app/models.py", "return 1", "return 2")),
    # 7. a rename: every caller of the old name loses its edge
    ("rename", lambda r: _sub(r / "lib/tools.py", "def unique_thing():", "def unique_thing2():")),
]


def _sub(p: Path, a: str, b: str) -> None:
    s = p.read_text(encoding="utf-8")
    assert a in s, (p, a)
    p.write_text(s.replace(a, b), encoding="utf-8")


def _append(p: Path, text: str) -> None:
    p.write_text(p.read_text(encoding="utf-8") + text, encoding="utf-8")


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("host_like", [True, False], ids=["host-live", "cli-fresh"])
def test_incremental_equals_full(tmp_path, embedder, host_like):
    work = tmp_path / "work"
    work.mkdir()
    materialize(work)
    live = LiveIndex(load_config(work)) if host_like else None
    _index(work, embedder, full=True, live=live)
    for tag, edit in EDITS:
        edit(work)
        stats = _index(work, embedder, full=False, live=live)
        assert stats["errors"] == 0, (tag, stats)
        assert stats["changed"] + stats["deleted"] >= 1, (tag, stats)
        assert stats["scope"]["scoped"], stats
        _assert_same_as_full(work, tmp_path, embedder, tag)


def test_watcher_paths_fast_path(tmp_path, embedder):
    """changed_paths skips the walk; the result equals a walked pass."""
    work = tmp_path / "work"
    work.mkdir()
    materialize(work)
    live = LiveIndex(load_config(work))
    _index(work, embedder, full=True, live=live)
    _sub(work / "app/service.py", "return name.upper()", "return name.lower()")
    (work / "cyc/b.py").unlink()
    stats = _index(work, embedder, full=False, live=live,
                   changed_paths=[work / "app/service.py", work / "cyc/b.py"])
    assert stats["changed"] == 1 and stats["deleted"] == 1, stats
    assert "scan" in stats["timings"]
    _assert_same_as_full(work, tmp_path, embedder, "watcher")


def test_stage_timings_and_scope(tmp_path, embedder):
    work = tmp_path / "work"
    work.mkdir()
    materialize(work)
    _index(work, embedder, full=True)
    _sub(work / "app/helpers.py", "return 1", "return 3")
    stats = _index(work, embedder, full=False)
    t = stats["timings"]
    for stage in ("scan", "harvest", "delete", "parse_embed_insert", "resolve", "edge_write",
                  "tier4", "analytics", "persist"):
        assert stage in t, (stage, t)
    # only the changed file and the files naming its symbols were resolved
    assert stats["scope"]["files"].get("normal") == 1, stats["scope"]
    assert sum(stats["scope"]["files"].values()) < 10, stats["scope"]


# ---- keyword index -----------------------------------------------------------

def test_kwindex_matches_bruteforce(tmp_path):
    from docgraph import kwindex as K
    from docgraph.bm25 import BM25Index, tokenize
    rng = np.random.default_rng(3)
    vocab = [f"w{i}" for i in range(300)] + ["fetchAllRows", "parse_token", "cache"]
    texts = {i: " ".join(rng.choice(vocab, rng.integers(3, 30))) for i in range(1, 400)}
    kw = K.KeywordIndex(tmp_path)
    b = K.KeywordBuilder()
    b.add("Function", [(i, K.doc_terms(t)) for i, t in texts.items()])
    b.finish(kw)
    # incremental: remove 50, re-add 20 changed, add 30 new
    for i in range(1, 51):
        kw.remove("Function", [i])
        texts.pop(i)
    for i in range(51, 71):
        texts[i] = "cache parse_token " + texts[i]
        kw.remove("Function", [i])
        kw.add("Function", [(i, K.doc_terms(texts[i]))])
    for i in range(1000, 1030):
        texts[i] = " ".join(rng.choice(vocab, 12)) + " fetchAllRows"
        kw.add("Function", [(i, K.doc_terms(texts[i]))])
    kw.save()
    kw2 = K.KeywordIndex(tmp_path)
    assert kw2.load()
    ids = sorted(texts)
    ref = BM25Index([tokenize(texts[i]) for i in ids])
    for q in ("cache", "parse token", "fetch all rows w7", "w1 w2 w3"):
        toks = tokenize(q)
        got = kw2.topk("Function", toks, 10)
        want_scores = ref.scores(toks) if hasattr(ref, "scores") else None
        assert got, q
        if want_scores is not None:
            best = sorted(range(len(ids)), key=lambda j: -want_scores[j])[:5]
            assert {ids[j] for j in best} & {i for i, _ in got}, q
        # removed ids never come back
        assert not {i for i, _ in got} & set(range(1, 51)), q
    kw2.compact()
    kw3 = K.KeywordIndex(tmp_path)
    assert kw3.load()
    for q in ("cache", "fetch all rows"):
        a = kw2.topk("Function", tokenize(q), 10)
        c = kw3.topk("Function", tokenize(q), 10)
        assert [i for i, _ in a] == [i for i, _ in c], q


# ---- scan + sharded cache ------------------------------------------------------

def test_scan_cached_decisions_and_watcher_paths(tmp_path):
    from docgraph.config import MAX_FILE_BYTES
    from docgraph.live import ScanState, classify_paths, scan
    (tmp_path / "d").mkdir()
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (tmp_path / "d" / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("function x() {}\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    st = ScanState()
    first = scan(cfg, st, MAX_FILE_BYTES)
    assert set(first) == {"a.py", "d/b.py"}
    assert first["a.py"][1] > 0 and first["a.py"][2] > 0          # size, mtime_ns
    assert scan(cfg, st, MAX_FILE_BYTES) == first                  # cached decisions
    # watcher fast path: a new directory is walked, a deleted file is reported
    (tmp_path / "newdir").mkdir()
    (tmp_path / "newdir" / "c.py").write_text("def c():\n    pass\n", encoding="utf-8")
    (tmp_path / "d" / "b.py").unlink()
    present, gone = classify_paths(cfg, st, [tmp_path / "newdir", tmp_path / "d" / "b.py"], MAX_FILE_BYTES)
    assert set(present) == {"newdir/c.py"} and gone == {"d/b.py"}


def test_sharded_cache_roundtrip_and_dirty_shards(tmp_path):
    from docgraph.live import SHARDS, ShardedCache, shard_of
    c = ShardedCache(tmp_path / "cache")
    for i in range(50):
        c[f"f{i}.py"] = {"hash": f"h{i}", "size": i, "mtime": 1000 + i,
                         "entities": [{"qname": f"f{i}.py::x"}], "edges": []}
    assert c.save() == len({shard_of(f"f{i}.py") for i in range(50)})
    d = ShardedCache(tmp_path / "cache").load()
    assert len(d) == 50 and d.meta("f7.py") == (7, 1007, "h7")
    assert d["f7.py"]["entities"][0]["qname"] == "f7.py::x"
    d.set_stat("f7.py", 70, 2007)
    d.pop("f8.py")
    assert d.save() == len({shard_of("f7.py"), shard_of("f8.py")}) and not d.dirty
    e = ShardedCache(tmp_path / "cache").load()
    assert e.meta("f7.py") == (70, 2007, "h7") and "f8.py" not in e and len(e) == 49
    assert SHARDS & (SHARDS - 1) == 0


def test_kept_parse_pool_reused(tmp_path, embedder):
    """The host's live state keeps one parse pool between passes (more than
    INPROC_PARSE_MAX files) and the result equals a full index."""
    root = tmp_path / "pool"
    materialize(root)
    for k in range(12):
        (root / f"extra_{k}.py").write_text(f"def extra_{k}(x):\n    return x + {k}\n", encoding="utf-8")
    live = LiveIndex(load_config(root))
    live.keep_pool = True
    _index(root, embedder, full=True, live=live)
    changed = []
    for k in range(12):
        p = root / f"extra_{k}.py"
        p.write_text(p.read_text(encoding="utf-8") + f"\n\ndef more_{k}():\n    return extra_{k}(1)\n",
                     encoding="utf-8")
        changed.append(str(p))
    _index(root, embedder, full=False, live=live, changed_paths=changed)
    pool = live._pool
    assert pool is not None
    for k in range(12):
        p = root / f"extra_{k}.py"
        p.write_text(p.read_text(encoding="utf-8") + f"\n# edit {k}\n", encoding="utf-8")
    _index(root, embedder, full=False, live=live, changed_paths=changed)
    assert live._pool is pool                      # reused, not respawned
    live.close_pool()
    assert live._pool is None
    inc = _dump(root)
    shutil.rmtree(root / ".docgraph")
    _index(root, embedder, full=True)
    full = _dump(root)
    assert inc[0] == full[0]
    assert {k: v for k, v in inc[1].items() if k[0] not in SKIP_RELS} == \
        {k: v for k, v in full[1].items() if k[0] not in SKIP_RELS}
