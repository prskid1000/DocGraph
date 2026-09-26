"""The in-memory tool paths (CSR walks, the node payload table, explore's
array BFS, the compressed parse cache) against straightforward Cypher /
Python references on the framework fixture."""
from __future__ import annotations

from docgraph.retrieve import Retriever


def _ref_bfs(db, seeds, direction, depth, min_conf=0.0):
    """Reference: one 1-hop query per level, like the pre-CSR walk."""
    dist = {int(s): 0 for s in seeds}
    frontier = set(dist)
    for d in range(1, depth + 1):
        if not frontier:
            break
        side = "a" if direction == "out" else "b"
        rows = db.fetch_all(
            f"MATCH (a:Function)-[r:CALLS]->(b:Function) WHERE {side}.id IN $ids "
            f"AND coalesce(r.confidence, 1.0) >= $c RETURN a.id AS src, b.id AS dst",
            {"ids": list(frontier), "c": float(min_conf)})
        nxt = set()
        for r in rows:
            far = r["dst"] if direction == "out" else r["src"]
            if far not in dist:
                dist[far] = d
                nxt.add(far)
        frontier = nxt
    return dist


def _all_function_ids(db):
    return [r["id"] for r in db.fetch_all("MATCH (f:Function) RETURN f.id AS id ORDER BY id")]


def test_bfs_calls_matches_reference(fw_indexed):
    _cfg, db, r, _s = fw_indexed
    ids = _all_function_ids(db)
    assert ids
    for seed in ids:
        for direction in ("out", "in"):
            for conf in (0.0, 0.8):
                dist, parent, _e = r._bfs_calls([seed], direction, 4, conf)
                assert dist == _ref_bfs(db, [seed], direction, 4, conf), (seed, direction, conf)
                for node, p in parent.items():
                    assert dist[p] == dist[node] - 1


def test_bfs_edges_on_request(fw_indexed):
    _cfg, db, r, _s = fw_indexed
    seed = db.fetch_all("MATCH (f:Function) WHERE f.name = 'read_item' RETURN f.id AS id")[0]["id"]
    dist, _p, edges = r._bfs_calls([seed], "out", 3, with_edges=True)
    assert edges and all(e["src"] in dist and e["dst"] in dist for e in edges)
    assert all({"src", "dst", "conf", "method", "line"} <= set(e) for e in edges)
    assert r._bfs_calls([seed], "out", 3)[2] == []


def test_node_table_matches_db(fw_indexed):
    _cfg, db, _r, _s = fw_indexed
    ids = [x["id"] for x in db.fetch_all("MATCH (n:Function) RETURN n.id AS id")]
    ids += [x["id"] for x in db.fetch_all("MATCH (n:Class) RETURN n.id AS id")]
    via_db = Retriever(db, None)
    via_db.NODES_TABLE_MIN = 10 ** 9           # force the Cypher path
    via_table = Retriever(db, None)
    via_table.NODES_TABLE_MIN = 0              # force the in-memory table
    a, b = via_db._nodes(ids), via_table._nodes(ids + [10 ** 12])
    assert a.keys() == b.keys()
    for i in a:
        for k in ("id", "name", "qname", "file", "line", "line_end", "kind"):
            assert a[i][k] == b[i][k], (i, k)
        assert abs((a[i]["pagerank"] or 0.0) - (b[i]["pagerank"] or 0.0)) < 1e-9
        assert bool(a[i]["is_test"]) == bool(b[i]["is_test"])


def test_explore_ranking_and_subgraph(fw_indexed):
    _cfg, db, r, _s = fw_indexed
    out = r.explore(seeds=["read_item"], hops=3, limit=6)
    nodes = out["nodes"]
    assert nodes and nodes[0]["name"] == "read_item" and nodes[0]["distance"] == 0
    scores = [n["score"] for n in nodes]
    assert scores == sorted(scores, reverse=True)
    keep = {n["id"] for n in nodes}
    assert all(e["src"] in keep and e["dst"] in keep for e in out["edges"])
    assert out["edges_walked"] >= len(out["edges"])
    assert out["reached"] >= len(nodes)
    # distances agree with an undirected CALLS-only reference walk
    calls = r.explore(seeds=["read_item"], hops=2, limit=100, edges=("CALLS",))
    seed = db.fetch_all("MATCH (f:Function) WHERE f.name = 'read_item' RETURN f.id AS id")[0]["id"]
    ref = {seed: 0}
    frontier = {seed}
    for d in (1, 2):
        rows = db.fetch_all("MATCH (a)-[:CALLS]-(b) WHERE a.id IN $ids RETURN b.id AS b",
                            {"ids": list(frontier)})
        frontier = {x["b"] for x in rows if x["b"] not in ref}
        for x in frontier:
            ref[x] = d
    got = {n["id"]: n["distance"] for n in calls["nodes"]}
    assert got == {k: v for k, v in ref.items() if k in got}
    assert set(got) == set(ref)


def test_processes_flow_consistency(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    flows = r.processes(limit=10, max_chain_len=6)
    assert flows
    for f in flows:
        names = {c["qname"] for c in f["chain"]} | {f["entry"]["qname"]}
        assert all(e["src"] in names and e["dst"] in names for e in f["edges"])
        one = r.flow(f["id"], max_chain_len=6)
        assert one and one["entry"]["id"] == f["id"]


def test_sharded_cache_compressed_roundtrip(tmp_path):
    from docgraph.live import ShardedCache
    c = ShardedCache(tmp_path / "cache")
    entry = {"hash": "h1", "size": 10, "mtime": 5,
             "entities": [{"qname": "a.py::f", "body": "x" * 5000}], "edges": []}
    c["a.py"] = entry
    assert c["a.py"] == entry and c.meta("a.py") == (10, 5, "h1")
    assert len(c._raw["a.py"]) < 1000          # held compressed
    c.save()
    d = ShardedCache(tmp_path / "cache").load()
    assert d["a.py"] == entry and dict(d.items()) == {"a.py": entry}
