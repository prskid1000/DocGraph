"""FastAPI server: hosts the graph UI + JSON API + mounts MCP over HTTP.

For stdio MCP (Cursor / Claude Desktop), use `docgraph mcp` instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.retrieve import Retriever

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


def make_app(cfg: Config) -> FastAPI:
    app = FastAPI(
        title="DocGraph",
        default_response_class=ORJSONResponse,
        version="2.0.0",
    )

    db = GraphDB(cfg.db_path, read_only=True)
    embedder = Embedder(cfg.embedding_model)
    retriever = Retriever(db, embedder)

    # --- UI ---
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    # --- JSON API ---
    @app.get("/api/search")
    async def api_search(q: str, kind: str | None = None, limit: int = 10):
        return retriever.search(q, kind=kind, limit=limit)

    @app.get("/api/definition")
    async def api_definition(name: str, file: str | None = None):
        return retriever.definition(name, file=file)

    @app.get("/api/references")
    async def api_references(name: str):
        return retriever.references(name)

    @app.get("/api/call_graph")
    async def api_call_graph(name: str, depth: int = 2):
        return retriever.call_graph(name, depth=depth)

    @app.get("/api/file_map")
    async def api_file_map(file: str):
        return retriever.file_map(file)

    @app.get("/api/neighborhood")
    async def api_neighborhood(name: str, limit: int = 10):
        return retriever.neighborhood(name, limit=limit)

    @app.get("/api/explore")
    async def api_explore(seeds: str, hops: int = 3, limit: int = 25):
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
        return retriever.explore(seeds=seed_list, hops=hops, limit=limit)

    @app.get("/api/impact_of")
    async def api_impact_of(target: str, depth: int = 3, limit: int = 50):
        return retriever.impact_of(target, depth=depth, limit=limit)

    @app.get("/api/test_impact")
    async def api_test_impact(target: str, limit: int = 25):
        return retriever.test_impact(target, limit=limit)

    @app.post("/api/cypher")
    async def api_cypher(payload: dict):
        return retriever.cypher(payload.get("query", ""), limit=int(payload.get("limit", 100)))

    @app.get("/api/graph")
    async def api_graph(limit_nodes: int = 2000):
        return retriever.graph_dump(limit_nodes=limit_nodes)

    @app.get("/api/stats")
    async def api_stats():
        rows = db.fetch_all("CALL show_tables() RETURN *")
        out: dict = {"tables": rows, "repo": str(cfg.repo_root)}
        for label in ("File", "Function", "Class", "Variable", "Module"):
            r = db.fetch_all(f"MATCH (n:{label}) RETURN count(n) AS c")
            out[label] = r[0]["c"] if r else 0
        return out

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
        return {"file": file, "content": full.read_text(encoding="utf-8", errors="replace")}

    return app
