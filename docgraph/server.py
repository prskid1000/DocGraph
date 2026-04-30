"""FastAPI server: hosts the graph UI + JSON API + mounts MCP over HTTP.

For stdio MCP (Cursor / Claude Desktop), use `docgraph mcp` instead.

DB swap protocol (used by `watch --serve`): `make_app` accepts an optional
pre-opened `GraphDB`. The `app.state.db_holder` wraps the DB plus a
threading.Lock so the watcher can atomically swap to a fresh post-reindex
GraphDB without racing in-flight API requests. SSE subscribers at
`/api/events` get pinged after each swap so the UI can re-fetch the graph.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from threading import Lock
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.retrieve import Retriever

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


class DBHolder:
    """Mutable wrapper around GraphDB + Retriever. The watcher swaps both at
    once after each reindex; API handlers acquire the lock briefly per query
    so they always see a coherent (db, retriever) pair."""

    def __init__(self, db: GraphDB, retriever: Retriever) -> None:
        self.db = db
        self.retriever = retriever
        self.lock = Lock()

    def swap(self, db: GraphDB, retriever: Retriever) -> None:
        with self.lock:
            old = self.db
            self.db = db
            self.retriever = retriever
        try:
            old.close()
        except Exception:
            pass


def make_app(cfg: Config, db: GraphDB | None = None) -> FastAPI:
    app = FastAPI(
        title="DocGraph",
        version="2.0.0",
    )

    if db is None:
        db = GraphDB(cfg.db_path, read_only=True)
    from docgraph.embed import GPU_PROVIDERS
    embedder = Embedder(
        cfg.embedding_model,
        providers=list(GPU_PROVIDERS) if cfg.gpu else None,
    )
    retriever = Retriever(db, embedder, cfg=cfg)
    holder = DBHolder(db, retriever)
    app.state.db_holder = holder
    app.state.subscribers = []  # list[asyncio.Queue]
    app.state.embedder = embedder

    def _r() -> Retriever:
        with holder.lock:
            return holder.retriever

    def _db() -> GraphDB:
        with holder.lock:
            return holder.db

    # --- UI ---
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    # --- JSON API ---
    @app.get("/api/search")
    async def api_search(
        q: str, kind: str | None = None, limit: int = 10,
        focus_file: str | None = None, focus_symbol: str | None = None,
        rerank: bool = False,
    ):
        return _r().search(
            q, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol, rerank=rerank,
        )

    @app.get("/api/definition")
    async def api_definition(name: str, file: str | None = None):
        return _r().definition(name, file=file)

    @app.get("/api/references")
    async def api_references(name: str):
        return _r().references(name)

    @app.get("/api/call_graph")
    async def api_call_graph(name: str, depth: int = 2):
        return _r().call_graph(name, depth=depth)

    @app.get("/api/file_map")
    async def api_file_map(file: str):
        return _r().file_map(file)

    @app.get("/api/neighborhood")
    async def api_neighborhood(name: str, limit: int = 10):
        return _r().neighborhood(name, limit=limit)

    @app.get("/api/explore")
    async def api_explore(seeds: str, hops: int = 3, limit: int = 25):
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
        return _r().explore(seeds=seed_list, hops=hops, limit=limit)

    @app.get("/api/impact_of")
    async def api_impact_of(target: str, depth: int = 3, limit: int = 50):
        return _r().impact_of(target, depth=depth, limit=limit)

    @app.get("/api/test_impact")
    async def api_test_impact(target: str, limit: int = 25):
        return _r().test_impact(target, limit=limit)

    @app.post("/api/cypher")
    async def api_cypher(payload: dict):
        return _r().cypher(payload.get("query", ""), limit=int(payload.get("limit", 100)))

    @app.get("/api/git_changes")
    async def api_git_changes(ref: str | None = None):
        return _r().git_changes(ref=ref)

    @app.get("/api/git_blame")
    async def api_git_blame(file: str, line_start: int = 1, line_end: int | None = None):
        return _r().git_blame(file, line_start=line_start, line_end=line_end)

    @app.get("/api/git_recent")
    async def api_git_recent(file: str | None = None, limit: int = 20):
        return _r().git_recent(file=file, limit=limit)

    @app.get("/api/rules_for")
    async def api_rules_for(file: str):
        return _r().rules_for(file)

    @app.get("/api/search_docs")
    async def api_search_docs(q: str, limit: int = 10):
        return _r().search_docs(q, limit=limit)

    @app.get("/api/graph")
    async def api_graph(limit_nodes: int = 2000):
        return _r().graph_dump(limit_nodes=limit_nodes)

    @app.get("/api/stats")
    async def api_stats():
        d = _db()
        rows = d.fetch_all("CALL show_tables() RETURN *")
        out: dict = {"tables": rows, "repo": str(cfg.repo_root)}
        for label in ("File", "Function", "Class", "Variable", "Module"):
            r = d.fetch_all(f"MATCH (n:{label}) RETURN count(n) AS c")
            out[label] = r[0]["c"] if r else 0
        return out

    # --- SSE: live reindex events ---
    @app.get("/api/events")
    async def api_events(request: Request) -> StreamingResponse:
        """Server-Sent Events stream. The watcher (in `docgraph watch --serve`
        mode) pushes a `reindex_done` event after each successful reindex
        so the UI can refresh its graph + stats without polling."""
        queue: asyncio.Queue[dict] = asyncio.Queue()
        app.state.subscribers.append(queue)

        async def gen() -> AsyncIterator[bytes]:
            try:
                # Initial hello so EventSource resolves immediately
                yield b": ready\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Heartbeat keeps the connection alive through proxies
                        yield b": keepalive\n\n"
                        continue
                    name = evt.get("event", "message")
                    data = json.dumps(evt.get("data", {}))
                    yield f"event: {name}\ndata: {data}\n\n".encode("utf-8")
            finally:
                try:
                    app.state.subscribers.remove(queue)
                except ValueError:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/file_content")
    async def api_file_content(file: str):
        full = cfg.path_for(file).resolve()
        allowed = False
        for root, _ in cfg.roots_with_prefix():
            try:
                full.relative_to(root.resolve())
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise HTTPException(403, "outside repo")
        if not full.exists():
            raise HTTPException(404)
        if cfg.ai_blocked_logical(file):
            return {"file": file, "content": "[redacted by .cursorignore]", "redacted": True}
        return {"file": file, "content": full.read_text(encoding="utf-8", errors="replace")}

    return app


def broadcast(app: FastAPI, event_name: str, data: dict | None = None) -> None:
    """Push an event to every active SSE subscriber. Safe to call from a
    non-asyncio thread — uses `loop.call_soon_threadsafe` to schedule the
    queue.put on the FastAPI event loop."""
    payload = {"event": event_name, "data": data or {}}
    subs = list(getattr(app.state, "subscribers", []))
    if not subs:
        return
    loop = getattr(app.state, "loop", None)
    for q in subs:
        try:
            if loop is not None:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            else:
                q.put_nowait(payload)
        except Exception:
            pass
