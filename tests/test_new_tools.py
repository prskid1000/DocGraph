"""Tests for the 4 differentiator tools: explore, impact_of, test_impact, cypher."""
from __future__ import annotations


# --- explore -----------------------------------------------------------


def test_explore_basic(retriever):
    out = retriever.explore(seeds=["login"], hops=2, limit=20)
    assert "nodes" in out and "edges" in out
    assert out["nodes"], "explore should reach related nodes from login"
    # Distance 0 = the seed itself
    distances = {n["name"]: n["distance"] for n in out["nodes"]}
    assert distances.get("login") == 0


def test_explore_unknown_seed(retriever):
    out = retriever.explore(seeds=["nonexistent_symbol_xyz"], hops=2)
    assert out["nodes"] == []
    assert out["edges"] == []


def test_explore_empty_seeds(retriever):
    out = retriever.explore(seeds=[], hops=2)
    assert out == {"nodes": [], "edges": []}


def test_explore_hops_clamped(retriever):
    # hops=10 should clamp to 5; should not error
    out = retriever.explore(seeds=["login"], hops=10, limit=10)
    assert isinstance(out["nodes"], list)


# --- impact_of ---------------------------------------------------------


def test_impact_of_symbol(retriever):
    out = retriever.impact_of("login", depth=3, limit=50)
    assert "callers" in out
    assert "tests" in out
    # test_login and the api handler should appear among callers
    caller_names = {c.get("name") for c in out["callers"]}
    assert any(n in ("handle", "test_login", "make_handler", "login") for n in caller_names)


def test_impact_of_file(retriever):
    out = retriever.impact_of("src/auth.py", depth=2, limit=50)
    assert "importers" in out
    importer_paths = {i.get("file") for i in out["importers"]}
    assert any("api.py" in p for p in importer_paths)


def test_impact_of_unknown(retriever):
    out = retriever.impact_of("zzz_does_not_exist", depth=2)
    assert out["target"] == "zzz_does_not_exist"
    assert out["callers"] == [] and out["tests"] == []


# --- test_impact -------------------------------------------------------


def test_test_impact_symbol(retriever):
    rows = retriever.test_impact("login", limit=10)
    # We should find test_login via TESTS edges or via reverse CALLS*
    names = {r.get("name") for r in rows}
    assert "test_login" in names or any(n.startswith("test_") for n in names)


def test_test_impact_file(retriever):
    rows = retriever.test_impact("src/auth.py", limit=10)
    # Tests in tests/test_auth.py should surface
    files = {r.get("file") for r in rows}
    assert any("test_auth" in f for f in files) or any(r.get("name", "").startswith("test_") for r in rows)


def test_test_impact_no_matches(retriever):
    rows = retriever.test_impact("zzz_no_such_symbol", limit=10)
    assert rows == []


# --- cypher ------------------------------------------------------------


def test_cypher_read_works(retriever):
    out = retriever.cypher("MATCH (f:Function) RETURN f.name AS name", limit=5)
    assert out["rejected"] is None
    assert len(out["rows"]) <= 5
    assert out["rows"]


def test_cypher_blocks_create(retriever):
    out = retriever.cypher("CREATE (n:Function {name: 'evil'}) RETURN n", limit=5)
    assert out["rejected"] is not None
    assert out["rows"] == []


def test_cypher_blocks_delete(retriever):
    out = retriever.cypher("MATCH (n:Function) DELETE n", limit=5)
    assert out["rejected"] is not None


def test_cypher_blocks_set(retriever):
    out = retriever.cypher("MATCH (n:Function) SET n.pagerank = 1.0 RETURN n", limit=5)
    assert out["rejected"] is not None


def test_cypher_blocks_merge(retriever):
    out = retriever.cypher("MERGE (n:Function {name:'x'}) RETURN n", limit=5)
    assert out["rejected"] is not None


def test_cypher_string_literal_with_keyword_allowed(retriever):
    # "CREATE" appearing only inside a string literal must not block the query
    out = retriever.cypher(
        "MATCH (f:Function) WHERE f.name <> 'CREATE' RETURN f.name AS name",
        limit=5,
    )
    assert out["rejected"] is None


def test_cypher_invalid_query_returns_error(retriever):
    out = retriever.cypher("THIS IS NOT VALID CYPHER", limit=5)
    assert out["rejected"] is not None
    assert "query error" in out["rejected"]


def test_cypher_appends_limit_safety(retriever):
    # No LIMIT clause; should be appended
    out = retriever.cypher("MATCH (f:Function) RETURN f.name AS name", limit=2)
    assert out["rejected"] is None
    assert len(out["rows"]) <= 2


# --- personalized PageRank in search ---------------------------------


def test_search_with_focus_file(retriever):
    base = retriever.search("token", limit=10)
    focused = retriever.search("token", limit=10, focus_file="src/auth.py")
    # Focused results should not error and should include auth-related names
    assert focused
    auth_names = {r["name"] for r in focused if "auth" in r["file"]}
    # When biased to auth.py, auth-file results should appear in top-3
    top3_auth = sum(1 for r in focused[:3] if "auth" in r["file"])
    assert top3_auth >= 1


def test_search_with_focus_symbol(retriever):
    out = retriever.search("validate", limit=10, focus_symbol="login")
    assert out  # Should not crash; results returned
    # ppr field present on each result when focus is given
    assert "ppr" in out[0]
