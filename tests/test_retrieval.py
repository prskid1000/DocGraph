"""Tests for the original 6 retriever methods."""
from __future__ import annotations


def test_search_returns_relevant(retriever):
    results = retriever.search("authenticate user", limit=5)
    assert results
    names = [r["name"] for r in results]
    # Auth-related symbols should rank higher than utils
    assert any(n in ("Authenticator", "login", "_check_password", "_issue_token") for n in names)


def test_search_kind_filter_function(retriever):
    results = retriever.search("login", kind="function", limit=10)
    assert results
    assert all(r["label"] == "Function" for r in results)


def test_search_kind_filter_class(retriever):
    results = retriever.search("authenticator", kind="class", limit=5)
    assert results
    assert all(r["label"] == "Class" for r in results)


def test_definition_returns_body(retriever):
    rows = retriever.definition("login")
    assert rows
    assert any("password" in (r.get("body") or "").lower() for r in rows)


def test_references_finds_callers(retriever):
    rows = retriever.references("login")
    # api.py / test_login should reference login
    assert rows
    callers = {r.get("caller_name") for r in rows}
    assert any(c in ("handle", "test_login") for c in callers)


def test_call_graph(retriever):
    g = retriever.call_graph("login", depth=2)
    assert "calls" in g and "called_by" in g and "edges" in g
    # forward should include callees of login
    forward_names = {r["name"] for r in g["calls"]}
    assert any(n in ("_check_password", "_issue_token", "login") for n in forward_names)


def test_file_map(retriever):
    fm = retriever.file_map("src/auth.py")
    assert "entities" in fm
    names = {e["name"] for e in fm["entities"]}
    assert "Authenticator" in names
    assert "login" in names


def test_neighborhood(retriever):
    rows = retriever.neighborhood("login", limit=20)
    # Should find at least some related entities
    assert isinstance(rows, list)
