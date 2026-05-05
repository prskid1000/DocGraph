"""MCP tool surface tests.

Verifies that `make_mcp(workspace)` registers all base tools + list_roots
and that each one is invokable via FastMCP's `call_tool` path. Tools now
take a `root` enum argument; the single-root case auto-defaults so call
sites need not pass it.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from docgraph.mcp_tools import make_mcp
from docgraph.workspace import Workspace


EXPECTED_TOOLS = {
    "search", "definition", "references", "call_graph", "file_map", "neighborhood",
    "explore", "impact_of", "test_impact", "cypher",
    "git_changes", "git_blame", "git_recent",
    "rules_for",
    "list_roots",
}


@pytest.fixture(scope="module")
def mcp(indexed):
    cfg, _db, _e, _stats = indexed
    ws = Workspace([cfg])
    yield make_mcp(ws)
    ws.close()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def _structured(result) -> dict | list:
    """Pull JSON content out of a FastMCP ToolResult."""
    if hasattr(result, "structured_content") and result.structured_content is not None:
        sc = result.structured_content
        # FastMCP wraps non-dict returns under {"result": ...}
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    # Fall back to text content
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        try:
            return json.loads(text)
        except Exception:
            return text
    return result


def test_all_tools_registered(mcp):
    tools = _run(mcp.list_tools())
    names = {t.name for t in tools}
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS
    assert not missing, f"missing MCP tools: {missing}"
    # Extras are fine but worth surfacing
    assert not extra, f"unexpected MCP tools: {extra}"


def test_tool_count(mcp):
    tools = _run(mcp.list_tools())
    # 14 retriever-backed tools + list_roots
    assert len(tools) == 15


def test_list_roots_tool(mcp):
    tools = _run(mcp.list_tools())
    assert "list_roots" in {t.name for t in tools}
    out = _structured(_run(mcp.call_tool("list_roots", {})))
    assert isinstance(out, list)
    assert out
    assert {"slug", "path", "default"} <= set(out[0].keys())


def test_root_arg_is_enum_in_schema(mcp):
    """Verify the `root` parameter shows up as a JSON Schema enum so the
    LLM sees a closed value set rather than a free-form string."""
    tools = _run(mcp.list_tools())
    by_name = {t.name: t for t in tools}
    schema = by_name["search"].parameters
    props = schema.get("properties") or {}
    root_prop = props.get("root")
    assert root_prop is not None, f"search has no root prop: {props}"
    # Pydantic emits enum either inline or via $ref to defs; allow both.
    if "$ref" in root_prop:
        defs = schema.get("$defs") or schema.get("definitions") or {}
        ref_name = root_prop["$ref"].rsplit("/", 1)[-1]
        target = defs.get(ref_name) or {}
        assert "enum" in target, f"$ref'd schema missing enum: {target}"
    else:
        assert "enum" in root_prop, f"root prop missing enum: {root_prop}"
    # Single-root case: exactly one allowed value, also set as default.
    enum_vals = root_prop.get("enum") or []
    assert len(enum_vals) == 1, f"expected single-root enum, got {enum_vals}"
    assert root_prop.get("default") == enum_vals[0]


def test_search_tool(mcp):
    out = _structured(_run(mcp.call_tool("search", {"query": "authenticate", "limit": 3})))
    assert isinstance(out, list)
    assert out
    assert {"name", "score"} <= set(out[0].keys())


def test_definition_tool(mcp):
    out = _structured(_run(mcp.call_tool("definition", {"name": "Authenticator"})))
    assert isinstance(out, list)
    assert any(r.get("name") == "Authenticator" for r in out)


def test_references_tool(mcp):
    out = _structured(_run(mcp.call_tool("references", {"name": "Authenticator"})))
    assert isinstance(out, list)


def test_call_graph_tool(mcp):
    out = _structured(_run(mcp.call_tool("call_graph", {"name": "login", "depth": 2})))
    assert isinstance(out, dict)
    assert "calls" in out and "called_by" in out


def test_file_map_tool(mcp):
    out = _structured(_run(mcp.call_tool("file_map", {"file": "src/auth.py"})))
    assert isinstance(out, dict)
    assert "entities" in out


def test_neighborhood_tool(mcp):
    out = _structured(_run(mcp.call_tool("neighborhood", {"name": "Authenticator", "limit": 5})))
    assert isinstance(out, list)


def test_explore_tool(mcp):
    out = _structured(_run(mcp.call_tool(
        "explore", {"seeds": ["Authenticator", "login"], "hops": 2, "limit": 10}
    )))
    assert isinstance(out, dict)
    assert "nodes" in out and "edges" in out


def test_impact_of_tool(mcp):
    out = _structured(_run(mcp.call_tool("impact_of", {"target": "Authenticator", "depth": 2})))
    assert isinstance(out, dict)


def test_test_impact_tool(mcp):
    out = _structured(_run(mcp.call_tool("test_impact", {"target": "Authenticator"})))
    assert isinstance(out, list)


def test_cypher_tool_rejects_writes(mcp):
    out = _structured(_run(mcp.call_tool(
        "cypher", {"query": "CREATE (n:Function {name: 'x'})", "limit": 10}
    )))
    assert isinstance(out, dict)
    assert out.get("rejected")
    assert out.get("rows") == []


def test_cypher_tool_read_works(mcp):
    out = _structured(_run(mcp.call_tool(
        "cypher", {"query": "MATCH (n:Function) RETURN n.name AS name LIMIT 2", "limit": 2}
    )))
    assert isinstance(out, dict)
    assert "rows" in out
    assert len(out["rows"]) <= 2


def test_git_recent_tool(mcp):
    out = _structured(_run(mcp.call_tool("git_recent", {"limit": 3})))
    assert isinstance(out, list)
    assert out  # at least the initial commit from conftest._materialize_repo


def test_git_changes_tool(mcp):
    out = _structured(_run(mcp.call_tool("git_changes", {})))
    assert isinstance(out, dict)
    assert "files" in out


def test_git_blame_tool(mcp):
    out = _structured(_run(mcp.call_tool(
        "git_blame", {"file": "src/auth.py", "line_start": 1, "line_end": 2}
    )))
    assert isinstance(out, list)
    assert out
    assert "commit" in out[0]


def test_rules_for_tool(mcp):
    out = _structured(_run(mcp.call_tool("rules_for", {"file": "src/auth.py"})))
    assert isinstance(out, list)
    # No .mdc files in conftest repo, so list is empty (or contains AGENTS.md
    # if it ever gets seeded). Either way: a list, no error.


def test_unknown_tool_raises(mcp):
    with pytest.raises(Exception):
        _run(mcp.call_tool("does_not_exist", {}))
