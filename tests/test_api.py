"""HTTP API smoke tests.

Each FastAPI route in `server.py` is hit through `TestClient`. Coverage goals:
- every route returns 2xx with the expected shape
- `/api/file_content` enforces the repo-root sandbox + `.cursorignore` redaction
- `/api/cypher` POST body wiring is correct
- the wiring routes through `Retriever` with `cfg=` attached (git_*, rules_for need it)
"""
from __future__ import annotations

import gc
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from docgraph.config import load_config
from docgraph.server import make_app
from docgraph.workspace import Workspace


# --- App against the session-scoped indexed repo ----------------------


@pytest.fixture(scope="module")
def client(indexed):
    cfg, _db, _embedder, _stats = indexed
    # make_app opens its own read-only connections via the workspace.
    # The reader held by the `indexed` fixture coexists fine — Kuzu allows
    # multiple readers on the same DB file.
    ws = Workspace([cfg])
    app = make_app(ws)
    with TestClient(app) as c:
        yield c
    ws.close()


def test_index_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()


def test_search(client: TestClient):
    r = client.get("/api/search", params={"q": "authenticate user", "limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert rows, "search should return at least one hit on the seeded repo"
    assert {"name", "score"} <= set(rows[0].keys())


def test_search_with_focus_and_rerank(client: TestClient):
    # rerank=True should not 500 even if model can't load — Reranker fails
    # gracefully back to the bi-encoder ranker
    r = client.get("/api/search", params={
        "q": "login", "limit": 5,
        "focus_file": "src/api.py",
        "focus_symbol": "make_handler",
        "rerank": "true",
    })
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_definition(client: TestClient):
    r = client.get("/api/definition", params={"name": "Authenticator"})
    assert r.status_code == 200
    rows = r.json()
    assert any(row.get("name") == "Authenticator" for row in rows)


def test_references(client: TestClient):
    r = client.get("/api/references", params={"name": "Authenticator"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_call_graph(client: TestClient):
    r = client.get("/api/call_graph", params={"name": "login", "depth": 2})
    assert r.status_code == 200
    body = r.json()
    assert "calls" in body and "called_by" in body


def test_file_map(client: TestClient):
    r = client.get("/api/file_map", params={"file": "src/auth.py"})
    assert r.status_code == 200
    body = r.json()
    assert "entities" in body


def test_admin_index_runs_in_process(tmp_path: Path):
    """POST /api/admin/index should run an incremental reindex via the
    workspace's writer-lock dance and return a stats dict. Telecode's
    IndexRunner uses this when the host is alive to avoid spawning a
    parallel subprocess that would fight Kuzu's per-file lock.

    Uses an isolated tmp repo to avoid colliding with the session-scoped
    `indexed` fixture's reader connection."""
    import gc
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello(): return 1\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    # Build a baseline index so the workspace can open an RO handle.
    from docgraph.db import GraphDB
    from docgraph.embed import Embedder
    from docgraph.index import Indexer
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    Indexer(cfg, db, embedder=embedder).index_all(incremental=False)
    db.close()
    gc.collect()

    ws = Workspace([cfg])
    app = make_app(ws)
    try:
        with TestClient(app) as c:
            r = c.post("/api/admin/index", json={"full": False})
            assert r.status_code == 200, r.text
            body = r.json()
            assert "stats" in body
            assert body.get("full") is False
            assert "slug" in body
    finally:
        ws.close()


def test_neighborhood(client: TestClient):
    r = client.get("/api/neighborhood", params={"name": "Authenticator", "limit": 5})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_explore(client: TestClient):
    r = client.get("/api/explore", params={"seeds": "Authenticator,login", "hops": 2, "limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "edges" in body


def test_impact_of(client: TestClient):
    r = client.get("/api/impact_of", params={"target": "Authenticator", "depth": 2})
    assert r.status_code == 200
    body = r.json()
    # impact_of returns a dict bundling callers / importers / co_changed / tests
    assert isinstance(body, dict)


def test_test_impact(client: TestClient):
    r = client.get("/api/test_impact", params={"target": "Authenticator"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_cypher_read_only(client: TestClient):
    r = client.post("/api/cypher", json={"query": "MATCH (n:Function) RETURN n.name AS name LIMIT 3", "limit": 3})
    assert r.status_code == 200
    body = r.json()
    assert "rows" in body
    assert len(body["rows"]) <= 3


def test_cypher_rejects_writes(client: TestClient):
    r = client.post("/api/cypher", json={"query": "CREATE (n:Function {name: 'bad'})", "limit": 10})
    assert r.status_code == 200
    body = r.json()
    # Write rejected; rejected reason surfaced; no rows
    assert body.get("rejected")
    assert body.get("rows") == []


def test_git_recent(client: TestClient):
    r = client.get("/api/git_recent", params={"limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert rows  # the conftest fixture commits at least once
    assert "subject" in rows[0]


def test_git_changes_clean_tree(client: TestClient):
    r = client.get("/api/git_changes")
    assert r.status_code == 200
    body = r.json()
    assert "files" in body and "entities" in body


def test_git_blame(client: TestClient):
    r = client.get("/api/git_blame", params={"file": "src/auth.py", "line_start": 1, "line_end": 3})
    assert r.status_code == 200
    rows = r.json()
    assert rows
    assert "commit" in rows[0]


def test_rules_for(client: TestClient):
    # No .mdc files seeded; should still return a list (likely empty or
    # always-on AGENTS.md if present)
    r = client.get("/api/rules_for", params={"file": "src/auth.py"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_search_docs_empty(client: TestClient):
    r = client.get("/api/search_docs", params={"q": "anything", "limit": 5})
    assert r.status_code == 200
    # No docs ingested in the conftest repo → empty list
    assert r.json() == []


def test_graph_dump(client: TestClient):
    r = client.get("/api/graph", params={"limit_nodes": 100})
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "edges" in body
    assert body["nodes"]


def test_stats(client: TestClient):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert "Function" in body and "Class" in body


def test_file_content_serves_repo_file(client: TestClient):
    r = client.get("/api/file_content", params={"file": "src/auth.py"})
    assert r.status_code == 200
    body = r.json()
    assert "Authenticator" in body["content"]
    assert not body.get("redacted")


def test_file_content_404_outside(client: TestClient):
    r = client.get("/api/file_content", params={"file": "nonexistent/path.py"})
    # Either 403 (escapes repo) or 404 (within repo but missing) — both fine
    assert r.status_code in (403, 404)


def test_file_content_rejects_traversal(client: TestClient):
    r = client.get("/api/file_content", params={"file": "../../etc/passwd"})
    assert r.status_code in (403, 404)


# --- .cursorignore redaction ------------------------------------------


def test_sse_subscribers_list_exists(client: TestClient):
    """`/api/events` is registered and `app.state.subscribers` is wired up
    for the watcher to push into. We avoid actually consuming the stream
    here because httpx TestClient doesn't play well with long-lived SSE
    connections (its portal serializes the request)."""
    routes = {r.path for r in client.app.router.routes if hasattr(r, "path")}
    assert "/api/events" in routes
    assert isinstance(client.app.state.subscribers, list)


def test_broadcast_no_subscribers_safe():
    """`broadcast` must be a no-op when no subscriber is listening — the
    watcher calls it after every reindex regardless of whether the UI is
    open."""
    from fastapi import FastAPI

    from docgraph.server import broadcast

    fake = FastAPI()
    fake.state.subscribers = []
    # Should not raise
    broadcast(fake, "reindex_done", {"ts": 1})


def test_broadcast_enqueues_for_subscriber():
    """When a queue is registered, broadcast pushes the event payload into it."""
    import asyncio

    from fastapi import FastAPI

    from docgraph.server import broadcast

    fake = FastAPI()
    q: asyncio.Queue = asyncio.Queue()
    fake.state.subscribers = [q]
    broadcast(fake, "reindex_done", {"ts": 42})
    assert q.qsize() == 1
    payload = q.get_nowait()
    assert payload == {"event": "reindex_done", "data": {"ts": 42}}


def test_file_content_redacts_ai_blocked(tmp_path: Path):
    """Wholly isolated repo — verifies the redaction path in api_file_content."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "secrets.py").write_text("API_KEY = 'shh'\n", encoding="utf-8")
    (tmp_path / "public.py").write_text("def hello(): pass\n", encoding="utf-8")
    (tmp_path / ".cursorignore").write_text("secrets.py\n", encoding="utf-8")

    cfg = load_config(tmp_path)
    # Build an empty index (db file) so make_app is happy
    from docgraph.db import GraphDB
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    del db
    gc.collect()

    ws = Workspace([cfg])
    app = make_app(ws)
    with TestClient(app) as c:
        r = c.get("/api/file_content", params={"file": "secrets.py"})
        assert r.status_code == 200
        body = r.json()
        assert body.get("redacted") is True
        assert "[redacted" in body["content"]

        # Sibling non-blocked file is served raw
        r2 = c.get("/api/file_content", params={"file": "public.py"})
        assert r2.status_code == 200
        assert r2.json().get("redacted") is not True
