"""Schema-v3 features: resolution confidence, communities, routes/tools,
context, detect_changes, repo_map, trace, health, history, rename, SCIP,
embedding cache + file-hash tree, agent-setup, bench."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from docgraph.config import load_config
from docgraph.db import GraphDB, SCHEMA_VERSION
from docgraph.embed import Embedder
from docgraph.index import Indexer
from tests.fw_fixture import FW_FILES, line_of


def _calls(db, a: str, b: str) -> list[dict]:
    return db.fetch_all(
        "MATCH (x:Function)-[r:CALLS]->(y:Function) WHERE x.name = $a AND y.name = $b "
        "RETURN r.confidence AS c, r.method AS m, y.file AS f",
        {"a": a, "b": b})


# ---- 1. resolution cascade --------------------------------------------------

def test_confidence_tiers(fw_indexed):
    _cfg, db, _r, _s = fw_indexed
    fmt = _calls(db, "run_job", "format_name")
    assert fmt and fmt[0]["m"] == "import_map" and fmt[0]["c"] == pytest.approx(0.95)
    uniq = _calls(db, "caller_fn", "unique_thing")
    assert uniq and uniq[0]["m"] == "unique_global" and uniq[0]["c"] == pytest.approx(0.75)
    fuzzy = _calls(db, "caller_fn", "_check_value")
    assert fuzzy and fuzzy[0]["m"] == "fuzzy" and 0.3 <= fuzzy[0]["c"] <= 0.4
    save = _calls(db, "run_job", "save")
    assert save and save[0]["m"] == "import_suffix"
    write = _calls(db, "save", "_write")
    assert write and write[0]["m"] == "same_module"


def test_scip_edge_is_precise(fw_indexed):
    cfg, db, _r, _s = fw_indexed
    rows = _calls(db, "run_job", "local_step")
    assert rows and rows[0]["m"] == "scip" and rows[0]["c"] == pytest.approx(1.0)
    st = json.loads((cfg.data_dir / "state.json").read_text())
    assert st["scip"]["status"] == "ok" and st["scip"]["edges"] >= 1


def test_ambiguous_call_keeps_candidates(fw_indexed):
    _cfg, db, _r, _s = fw_indexed
    assert _calls(db, "caller_fn", "helper") == []
    cands = db.fetch_all(
        "MATCH (x:Function)-[r:CALLS_CANDIDATE]->(y:Function) WHERE x.name = 'caller_fn' "
        "RETURN y.file AS f, r.method AS m, r.confidence AS c")
    files = {c["f"] for c in cands}
    assert files == {"app/helpers.py", "lib/helpers.py"}
    assert all(c["m"] == "ambiguous" and c["c"] < 0.5 for c in cands)


def test_external_receiver_not_resolved(fw_indexed):
    _cfg, db, _r, _s = fw_indexed
    assert _calls(db, "caller_fn", "dumps") == []


def test_min_confidence_filters_call_graph(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    loose = {c["name"] for c in r.call_graph("caller_fn", depth=1)["calls"]}
    strict = {c["name"] for c in r.call_graph("caller_fn", depth=1, min_confidence=0.7)["calls"]}
    assert "_check_value" in loose and "_check_value" not in strict
    assert "unique_thing" in strict
    edges = r.call_graph("run_job", depth=1)["edges"]
    assert all("confidence" in e and "method" in e for e in edges)
    imp = r.impact_of("local_step", depth=3, min_confidence=0.9)
    assert "run_job" in {c["name"] for c in imp["callers"]}


def test_schema_version_recorded(fw_indexed):
    cfg, _db, r, _s = fw_indexed
    st = json.loads((cfg.data_dir / "state.json").read_text())
    assert st["schema_version"] == SCHEMA_VERSION
    info = r.index_info()
    assert info["reindex_required"] is False
    assert info["resolution"].get("import_map", 0) >= 1
    assert info["capabilities"]["calls_conf"] is True


# ---- 2. communities ---------------------------------------------------------

def test_clusters_persisted(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    data = r.list_clusters()
    assert data["clusters"], data
    c = data["clusters"][0]
    assert c["name"] and c["size"] >= 2 and 0.0 <= c["cohesion"] <= 1.0
    one = r.cluster(id=c["id"])
    assert one["found"] and one["members"]
    g = r.graph_dump(limit_nodes=200)
    assert any(n.get("cluster") is not None for n in g["nodes"])


# ---- 6. routes + tools -------------------------------------------------------

def test_route_map_frameworks(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    rm = r.route_map()
    routes = {x["name"]: x for x in rm["routes"]}
    assert routes["GET /items/{item_id}"]["handler"] == "read_item"
    assert routes["GET /items/{item_id}"]["framework"] == "fastapi"
    assert routes["POST /items"]["handler"] == "create_item"
    assert routes["GET /ping"]["handler"] == "handle_ping"
    assert routes["POST /echo"]["handler"] == "handle_echo"
    assert routes["GET,POST /hello"]["handler"] == "hello"
    assert routes["GET /users"]["handler"] == "listUsers"
    assert "POST /users" in routes
    assert not any("/fake" in n for n in routes), "docstring example must not be a route"
    tools = {t["name"]: t for t in rm["tools"]}
    assert tools["add_numbers"]["tool_kind"] == "tool"
    assert tools["res://greeting"]["tool_kind"] == "resource"
    assert tools["review"]["tool_kind"] == "prompt"


def test_api_impact_and_route_flows(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    imp = r.api_impact("GET /items/{item_id}")
    assert imp["found"] and imp["handler"]["name"] == "read_item"
    reach = {x["name"] for x in imp["reachable"]}
    assert {"run_job", "local_step"} <= reach
    assert any(t["name"] == "test_run_job" for t in imp["tests"])
    flows = r.processes(limit=50)
    kinds = {f["kind"] for f in flows}
    assert "route" in kinds
    f = next(f for f in flows if f["entry"]["name"] == "read_item")
    assert f["via"] == "GET /items/{item_id}" and f["edges"]
    assert r.flow(f["id"])["entry"]["name"] == "read_item"


# ---- 3. context ----------------------------------------------------------------

def test_context_360(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    c = r.context("run_job", tokens=3000)
    assert c["found"]
    assert "read_item" in {x["name"] for x in c["callers"]}
    assert {"format_name", "local_step"} <= {x["name"] for x in c["callees"]}
    assert "test_run_job" in {t["name"] for t in c["tests"]}
    assert any(f["kind"] == "route" for f in c["flows"])
    assert c["symbol"]["last_changed_commit"]
    assert "## Callers" in c["text"] and c["tokens"] <= 3000


def test_context_budget_trims(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    small = r.context("run_job", tokens=200)
    assert small["tokens"] <= 260  # the definition section is never dropped
    assert small["truncated"]
    missing = r.context("zz_nope_nothing")
    assert missing["found"] is False


# ---- 4. detect_changes ------------------------------------------------------------

def _diff_for_local_step() -> str:
    n = line_of("app/service.py", "return name.upper()")
    return (
        "diff --git a/app/service.py b/app/service.py\n"
        "--- a/app/service.py\n+++ b/app/service.py\n"
        f"@@ -{n},1 +{n},1 @@\n"
        "-    return name.upper()\n+    return name.lower()\n"
    )


def test_detect_changes_from_diff(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    out = r.detect_changes(diff=_diff_for_local_step())
    names = {c["name"] for c in out["changed_symbols"]}
    assert names == {"local_step"}
    assert {"run_job", "read_item"} <= {c["name"] for c in out["callers"]}
    assert "test_run_job" in {t["name"] for t in out["tests"]}
    assert "test_run_job" in out["test_command"]
    assert any(f["kind"] == "route" and f["name"] == "GET /items/{item_id}" for f in out["flows"])
    risk = out["changed_symbols"][0]["risk"]
    assert 0 <= risk["score"] <= 100 and risk["level"] in ("low", "medium", "high", "critical")
    assert {f["factor"] for f in risk["factors"]} >= {"centrality", "blast_radius", "test_gap"}
    assert out["overlay"]["changed"]


def test_detect_changes_git_worktree(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    out = r.detect_changes(ref="HEAD")
    assert "run_job" in {c["name"] for c in out["changed_symbols"]}


# ---- 5. repo_map ---------------------------------------------------------------------

def test_repo_map_budget_and_focus(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    big = r.repo_map(focus=["app/service.py"], tokens=2000)
    small = r.repo_map(focus=["app/service.py"], tokens=60)
    assert big["tokens"] <= 2000 and small["tokens"] <= 60
    assert small["symbols"] < big["symbols"]
    assert "app/service.py:" in big["text"] and "def run_job" in big["text"]
    assert big["ranking"] == "personalized"
    assert "test_run_job" not in big["text"]
    glob = r.repo_map(tokens=500)
    assert glob["ranking"] == "global" and glob["text"]


# ---- 7. trace -------------------------------------------------------------------------

def test_trace_path(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    t = r.trace("read_item", "_write")
    assert t["found"] and t["direction"] == "forward"
    assert [p["name"] for p in t["path"]] == ["read_item", "run_job", "save", "_write"]
    back = r.trace("_write", "read_item")
    assert back["found"] and back["direction"] == "reverse"
    assert r.trace("read_item", "nope_zz")["found"] is False


# ---- 8. health --------------------------------------------------------------------------

def test_health_report(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    h = r.health(limit=10)
    dead = {d["name"] for d in h["dead_code"]}
    assert "never_used_zz" in dead
    assert "run_job" not in dead and "add_numbers" not in dead and "setup" not in dead
    assert any({"cyc/a.py", "cyc/b.py"} <= set(c) for c in h["import_cycles"])
    assert h["summary"]["functions"] > 0 and isinstance(h["hubs"], list)


# ---- 14. history ---------------------------------------------------------------------

def test_symbol_history(fw_indexed):
    _cfg, _db, r, _s = fw_indexed
    h = r.symbol_history("run_job")
    s = h["symbols"][0]
    assert s["first_seen_commit"] and s["last_changed_commit"]
    assert s["first_seen_commit"] != s["last_changed_commit"]
    assert len(s["log"]) >= 2


# ---- 10. rename -----------------------------------------------------------------------

def test_rename_plan_never_writes(fw_indexed):
    cfg, _db, r, _s = fw_indexed
    before = (cfg.repo_root / "app" / "service.py").read_text(encoding="utf-8")
    plan = r.rename_plan("local_step", "local_stage")
    assert plan["dry_run"] is True and not plan.get("error")
    graph = [e for e in plan["edits"] if e["source"] == "graph"]
    lines = {e["line"] for e in graph}
    assert line_of("app/service.py", "def local_step") in lines
    assert line_of("app/service.py", "return local_step(name)") in lines
    assert all("local_stage" in e["after"] for e in plan["edits"])
    assert (cfg.repo_root / "app" / "service.py").read_text(encoding="utf-8") == before
    assert r.rename_plan("local_step", "not valid!")["error"]


def test_rename_apply_verifies_lines(tmp_path):
    from docgraph.rename import apply_plan
    (tmp_path / "m.py").write_text("def foo():\n    return 1\n\nfoo()\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    plan = {"edits": [
        {"file": "m.py", "line": 1, "before": "def foo():", "after": "def bar():", "source": "graph"},
        {"file": "m.py", "line": 4, "before": "foo()", "after": "bar()", "source": "graph"},
        {"file": "m.py", "line": 2, "before": "stale", "after": "x", "source": "graph"},
        {"file": "m.py", "line": 4, "before": "foo()", "after": "bar()", "source": "text"},
    ]}
    res = apply_plan(cfg, plan, sources=("graph",))
    assert res["applied"] == 2 and len(res["skipped"]) == 1
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "def bar():\n    return 1\n\nbar()\n"


# ---- 9. MCP resources / prompts + REST routes ------------------------------------------

@pytest.fixture(scope="module")
def fw_ws(fw_indexed):
    from docgraph.workspace import Workspace
    cfg, _db, _r, _s = fw_indexed
    ws = Workspace([cfg])
    yield ws
    ws.close()


def test_mcp_resources_and_prompts(fw_ws):
    from docgraph.mcp_tools import make_mcp
    mcp = make_mcp(fw_ws)

    async def go():
        res = {str(x.uri) for x in await mcp.list_resources()}
        tmpl = {t.uri_template for t in await mcp.list_resource_templates()}
        prompts = {p.name for p in await mcp.list_prompts()}
        schema = await mcp.read_resource("docgraph://schema")
        routes = await mcp.read_resource("docgraph://routes")
        arch = await mcp.render_prompt("architecture_map", {})
        impact = await mcp.render_prompt("detect_impact", {})
        ctx = await mcp.call_tool("context", {"symbol": "run_job", "tokens": 800})
        return res, tmpl, prompts, schema, routes, arch, impact, ctx

    res, tmpl, prompts, schema, routes, arch, impact, ctx = asyncio.run(go())
    assert {"docgraph://schema", "docgraph://clusters", "docgraph://flows", "docgraph://routes"} <= res
    assert {"docgraph://cluster/{cid}", "docgraph://flow/{fid}"} <= tmpl
    assert {"detect_impact", "architecture_map"} <= prompts
    assert "CALLS" in str(schema)
    assert "read_item" in str(routes)
    assert "graph LR" in str(arch)
    assert "Overall risk" in str(impact)
    assert "run_job" in str(ctx)


def test_rest_routes(fw_ws):
    from fastapi.testclient import TestClient
    from docgraph.server import make_app
    app = make_app(fw_ws)
    with TestClient(app) as c:
        assert c.get("/api/context", params={"symbol": "run_job"}).json()["found"]
        dc = c.post("/api/detect_changes", json={"diff": _diff_for_local_step()}).json()
        assert dc["changed_symbols"][0]["name"] == "local_step"
        assert c.get("/api/repo_map", params={"focus": "app/service.py", "tokens": 300}).json()["text"]
        cl = c.get("/api/clusters").json()["clusters"]
        assert c.get("/api/cluster", params={"id": cl[0]["id"]}).json()["found"]
        assert c.get("/api/cluster", params={"id": 999999999}).status_code == 404
        assert c.get("/api/routes").json()["routes"]
        assert c.get("/api/api_impact", params={"route": "GET /ping"}).json()["found"]
        assert c.get("/api/trace", params={"a": "read_item", "b": "_write"}).json()["found"]
        assert "dead_code" in c.get("/api/health").json()
        assert c.get("/api/symbol_history", params={"name": "run_job"}).json()["symbols"]
        plan = c.post("/api/rename", json={"symbol": "local_step", "new_name": "x_step"}).json()
        assert plan["dry_run"] is True and plan["edits"]
        # apply flag without dry_run=false never writes
        again = c.post("/api/rename", json={"symbol": "local_step", "new_name": "x_step",
                                            "apply": True}).json()
        assert again["dry_run"] is True
        assert c.get("/api/index_info").json()["current_schema_version"] == SCHEMA_VERSION
        flows = c.get("/api/processes").json()
        assert c.get("/api/flow", params={"id": flows[0]["id"]}).json()["entry"]
        cg = c.get("/api/call_graph", params={"name": "caller_fn", "min_confidence": 0.7}).json()
        assert "_check_value" not in {x["name"] for x in cg["calls"]}
        assert c.post("/api/rename", json={"symbol": "", "new_name": "x"}).status_code == 400


# ---- 12. embedding cache + file-hash tree (incremental on a mutable copy) ----------

@pytest.fixture(scope="module")
def fw_mutable(fw_repo, tmp_path_factory):
    root = tmp_path_factory.mktemp("docgraph_fw_mut")
    for p in fw_repo.rglob("*"):
        if ".docgraph" in p.parts or p.name == "index.scip":
            continue
        dst = root / p.relative_to(fw_repo)
        if p.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    return root


def _index(root: Path, incremental: bool) -> dict:
    cfg = load_config(root)
    db = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
    db.init_schema()
    ix = Indexer(cfg, db, embedder=Embedder(cfg.embedding_model))
    stats = ix.index_all(incremental=incremental)
    ix.db.close()
    db.close()
    return stats


def test_incremental_cache_merkle_history(fw_mutable):
    root = fw_mutable
    full = _index(root, incremental=False)
    assert full["errors"] == 0 and full["embedded"] > 0
    # no-op run: nothing re-read (stat reuse), nothing embedded
    noop = _index(root, incremental=True)
    assert noop["changed"] == 0 and noop["hashed"] == 0 and noop["hash_reused"] == full["files"]
    # move a file unchanged + edit another
    (root / "lib" / "tools.py").rename(root / "lib" / "tools_moved.py")
    svc = root / "app" / "service.py"
    svc.write_text(svc.read_text(encoding="utf-8").replace("name.upper()", "name.title()"),
                   encoding="utf-8")
    inc = _index(root, incremental=True)
    assert inc["changed"] == 2 and inc["deleted"] == 1
    assert inc["embed_cache_hits"] > 0, inc      # moved bodies were not re-embedded
    cfg = load_config(root)
    st = json.loads((cfg.data_dir / "state.json").read_text())
    removed = {r["qname"] for r in st["removed_symbols"]}
    assert "lib/tools.py::never_used_zz" in removed
    db = GraphDB(cfg.db_path, read_only=True)
    try:
        rows = db.fetch_all(
            "MATCH (a:Function)-[r:CALLS]->(b:Function) WHERE a.name = 'caller_fn' "
            "AND b.name = 'unique_thing' RETURN b.file AS f, r.method AS m")
        assert rows and rows[0]["f"] == "lib/tools_moved.py"
        hist = db.fetch_all("MATCH (f:Function) WHERE f.name = 'local_step' "
                            "RETURN f.last_changed_commit AS lc")
        assert hist[0]["lc"] == "uncommitted"
    finally:
        db.close()


# ---- unit tests: pure helpers -----------------------------------------------------------

def test_module_index_resolution():
    from docgraph.resolve import ModuleIndex
    mi = ModuleIndex(["src/pkg/mod.py", "src/pkg/__init__.py", "web/a.ts", "web/lib/index.js",
                      "app/x.py"])
    assert mi.resolve("pkg.mod", "app/x.py") == ["src/pkg/mod.py"]
    assert mi.resolve("./a", "web/b.ts") == ["web/a.ts"]
    assert mi.resolve("./lib", "web/b.ts") == ["web/lib/index.js"]
    assert mi.resolve(".mod", "src/pkg/other.py") == ["src/pkg/mod.py"]
    assert mi.resolve("os", "app/x.py") == []


def test_symbol_resolver_rules():
    from docgraph.resolve import SymbolResolver
    idx = {"go": [("Function", 1, "a.py"), ("Function", 2, "b.py")],
           "m": [("Function", 3, "c.py")], "get": [("Function", 4, "c.py")]}
    r = SymbolResolver(idx, {"x.py": {"b.py"}}, method_ids={3, 4},
                       external={"x.py": {"np"}})
    assert r.resolve("go", "x.py").target[1] == 2                     # import_suffix
    amb = r.resolve("go", "z.py")
    assert amb.target is None and len(amb.candidates) == 2              # ambiguous
    assert r.resolve("m", "x.py", recv="obj", attr=True).method == "unique_global"
    assert r.resolve("m", "x.py", recv="np", attr=True).method == "external"
    assert r.resolve("m", "x.py").target is None                        # bare call cannot hit a method
    assert r.resolve("get", "x.py", recv="d", attr=True).method == "common_name"


def test_frameworks_detect_snippets():
    from docgraph.frameworks import detect
    py = 'from fastapi import APIRouter\nr = APIRouter()\n@r.put("/x/{id}")\ndef upd(id):\n    pass\n'
    hits = detect(py, "python", [("function", "f.py::upd", 4, 5)])
    assert hits[0].name == "PUT /x/{id}" and hits[0].handler_qname == "f.py::upd"
    js = "router.delete('/a/:id', auth, ctrl.remove);\n"
    h = detect(js, "javascript", [])
    assert h[0].name == "DELETE /a/:id" and h[0].handler_name == "remove"
    assert detect("m.get('key')\n", "javascript", []) == []


def test_insights_diff_parse_and_budget():
    from docgraph.insights import fit_to_budget, parse_unified_diff, symbol_risk, trim_sections
    diff = ("--- a/q.sql\n+++ b/q.sql\n@@ -3,2 +3,1 @@\n--- old comment\n-x\n+y\n"
            "--- a/r.py\n+++ b/r.py\n@@ -10,0 +11,2 @@\n+a\n+b\n")
    files = parse_unified_diff(diff)
    assert [f["path"] for f in files] == ["q.sql", "r.py"]
    assert files[0]["removed"] == 2 and files[0]["added"] == 1
    assert files[1]["ranges"] == [(11, 12)]
    risk = symbol_risk(pagerank_pct=1.0, n_callers=100, n_routes=3, n_entries=0, n_tests=0,
                       changed_lines=100)
    assert risk["level"] == "critical" and risk["score"] == 100.0
    assert symbol_risk(pagerank_pct=0, n_callers=0, n_routes=0, n_entries=0, n_tests=5,
                       changed_lines=1)["level"] == "low"
    items = [{"i": i} for i in range(200)]
    text, n = fit_to_budget(items, 50, lambda xs: "\n".join(f"line {x['i']}" for x in xs))
    assert 0 < n < 200 and len(text) <= 200
    t, keep, trunc = trim_sections([("A", ["a"] * 5), ("B", ["b" * 40] * 50)], 60)
    assert keep["A"] >= 1 and "B" in trunc


def test_communities_two_cliques():
    from docgraph.communities import detect
    nodes = {i: {"label": "Function", "name": f"f{i}", "file": "a.py" if i < 4 else "b.py",
                 "pagerank": 0.1} for i in range(8)}
    edges = [(a, b, 1.0) for a in range(4) for b in range(4) if a < b]
    edges += [(a, b, 1.0) for a in range(4, 8) for b in range(4, 8) if a < b]
    edges.append((0, 4, 0.1))
    comms = detect(nodes, edges)
    assert len(comms) == 2 and {c.size for c in comms} == {4}
    assert all(c.cohesion > 0.8 for c in comms)
    assert {c.name.split(":")[0] for c in comms} == {"a", "b"}


def test_merkle_scan_reuses_stat(tmp_path):
    from docgraph import merkle
    (tmp_path / "d").mkdir()
    f1, f2 = tmp_path / "a.py", tmp_path / "d" / "b.py"
    f1.write_text("x")
    f2.write_text("y")
    files = [(f1, "a.py"), (f2, "d/b.py")]
    r1, st1 = merkle.scan(files, {})
    assert r1.hashed == 2
    out = tmp_path / "m.json"
    merkle.save(out, r1, st1)
    r2, _ = merkle.scan(files, merkle.load(out))
    assert r2.hashed == 0 and r2.reused == 2 and r2.root_unchanged
    assert r2.unchanged_dirs >= {"", "d"}
    f2.write_text("changed!")
    r3, _ = merkle.scan(files, merkle.load(out))
    assert r3.hashed == 1 and "d" not in r3.unchanged_dirs and not r3.root_unchanged


def test_history_span():
    from docgraph.history import UNCOMMITTED, symbol_span_history
    blame = [("aaa", 10), ("bbb", 30), ("ccc", 20), (UNCOMMITTED, 40)]
    h = symbol_span_history(blame, 1, 3)
    assert h["fc"] == "aaa" and h["lc"] == "bbb"
    assert symbol_span_history(blame, 2, 4)["lc"] == UNCOMMITTED


def test_scip_roundtrip_and_mapping():
    from docgraph.scip import Document, Occurrence, decode_index, edges_from_documents, encode_index
    docs = [Document(path="m.py", occurrences=[
        Occurrence(line=0, symbol="s/f().", roles=1),
        Occurrence(line=4, symbol="s/g().", roles=1),
        Occurrence(line=6, symbol="s/f().", roles=0),
    ])]
    back = decode_index(encode_index(docs))
    assert back[0].path == "m.py" and [o.line for o in back[0].occurrences] == [0, 4, 6]
    spans = [{"label": "Function", "id": 1, "file": "m.py", "s": 1, "e": 3},
             {"label": "Function", "id": 2, "file": "m.py", "s": 5, "e": 9}]
    assert edges_from_documents(back, "", spans) == [{"from_id": 2, "to_id": 1, "line": 7}]


def test_embed_profiles_no_download():
    from docgraph.embed import Embedder, dim_for_model, model_profile
    assert dim_for_model("nomic-ai/CodeRankEmbed") == 768
    assert dim_for_model("Qodo/Qodo-Embed-1-1.5B") == 1536
    assert model_profile("jinaai/jina-embeddings-v2-base-code")["trust_remote_code"] is True
    assert model_profile("BAAI/bge-small-en-v1.5") == {}
    e = Embedder("nomic-ai/CodeRankEmbed")
    assert e._prefixed(["q"], "query") == ["Represent this query for searching relevant code: q"]
    assert e._prefixed(["d"], "document") == ["d"]
    assert load_config(Path("."), embedding_model="Qodo/Qodo-Embed-1-1.5B").embedding_dim == 1536


# ---- 9. agent-setup ----------------------------------------------------------------------

def test_agent_setup_idempotent(tmp_path):
    from docgraph import agent_setup as a
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=False)
    (tmp_path / "AGENTS.md").write_text("# Mine\n\nkeep me\n", encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"model": "x", "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}}),
        encoding="utf-8")
    clusters = [{"id": 7, "name": "service: run_job", "size": 5, "cohesion": 0.8,
                 "top_members": ["run_job"], "files": ["app/service.py"]}]
    acts = a.plan(tmp_path, ["claude", "codex", "agy"], "http://127.0.0.1:5599", "fw", clusters)
    dry = a.apply(acts, dry_run=True)
    assert all(r["status"].startswith("would be") for r in dry)
    assert not (tmp_path / ".claude" / "skills").exists()
    first = a.apply(acts)
    assert {r["status"] for r in first} <= {"created", "updated"}
    second = a.apply(a.plan(tmp_path, ["claude", "codex", "agy"], "http://127.0.0.1:5599", "fw", clusters))
    assert {r["status"] for r in second} == {"unchanged"}
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["model"] == "x"
    pre = settings["hooks"]["PreToolUse"]
    assert len(pre) == 2 and "docgraph hook pre-edit" in pre[1]["hooks"][0]["command"]
    agents = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "keep me" in agents and agents.count(a.BEGIN) == 1
    assert (tmp_path / ".claude" / "skills" / "docgraph-area-service-run-job" / "SKILL.md").exists()
    hook = next(Path(r["path"]) for r in first if r["kind"] == "git_hook")
    assert a.GIT_BEGIN in hook.read_text(encoding="utf-8")


def test_hook_pre_edit_offline_is_silent():
    from docgraph import agent_setup as a
    assert a.pre_edit_hint({"tool_input": {"file_path": "x.py"}}, "http://127.0.0.1:9", None) is None
    assert "not reachable" in a.post_commit_notice("http://127.0.0.1:9", None)


# ---- 15. bench ----------------------------------------------------------------------------

def test_bench_questions_and_offline_runners(fw_repo):
    from docgraph import bench as b
    qs = b.load_questions()
    assert len(qs) >= 15
    assert {q["category"] for q in qs} >= {"where-defined", "callers", "impact", "flow",
                                           "architecture", "tests-to-run"}
    assert all(q["reference"] and q["keywords"] for q in qs)
    assert b.keyword_recall("run_job calls Local_Step", ["run_job", "local_step", "x"]) == pytest.approx(2 / 3)

    class Fake:
        name = "fake"

        def answer(self, q):
            return b.Answer(text=" ".join(q["keywords"]), tokens=10, tool_calls=1)

    rep = b.run(qs[:3], [Fake(), b.GrepRunner(fw_repo)])
    assert rep["summary"]["fake"]["recall"] == 1.0
    assert rep["summary"]["grep"]["n"] == 3
