"""FastAPI host server: graph UI + JSON API + MCP HTTP endpoint.

Multi-root: every API route accepts `?root=<slug>` (closed enum built
from the workspace's registered slugs). When omitted, the default root
(first registered) is used. The web UI's repo picker dropdown is
populated from `GET /api/roots`.

Reindex SSE events on `/api/events` carry `{repo_slug, ts}` so a
multi-root UI can refresh only the affected slot.

Note: this module deliberately does NOT use `from __future__ import
annotations`. Each route's `root` parameter is annotated with the
dynamically-built enum class (a local in `make_app`); deferred
annotations would prevent FastAPI/Pydantic's `TypeAdapter` from
resolving `RootSlug` at request time.
"""
import asyncio
import enum
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from docgraph.workspace import Workspace, RootSlot

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"


def _root_enum(workspace: Workspace) -> type[enum.Enum]:
    """Same dynamic enum as mcp_tools._root_enum, but for FastAPI query params.
    FastAPI's parameter handling renders `Enum` subclasses as OpenAPI enums."""
    members = {s.upper().replace("-", "_"): s for s in workspace.slugs()}
    if not members:
        raise ValueError("Workspace has no roots")
    return enum.Enum("RootSlug", members, type=str)  # type: ignore[arg-type]


def make_app(workspace: Workspace) -> FastAPI:
    app = FastAPI(title="DocGraph", version="2.2.0")
    RootSlug = _root_enum(workspace)
    DEFAULT = RootSlug(workspace.default_slug())

    app.state.workspace = workspace
    app.state.subscribers = []  # list[asyncio.Queue]

    def _slot(root) -> RootSlot:
        if root is None:
            return workspace.resolve(None)
        slug = root.value if hasattr(root, "value") else str(root)
        return workspace.resolve(slug)

    def _r(root):
        return _slot(root).retriever

    # --- UI ---
    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    # --- Roots discovery ---
    @app.get("/api/roots")
    async def api_roots():
        return workspace.list()

    # --- JSON API (every route accepts ?root=<slug>) ---
    @app.get("/api/search")
    async def api_search(
        q: str, kind: str | None = None, limit: int = 10,
        focus_file: str | None = None, focus_symbol: str | None = None,
        rerank: bool = False, root: RootSlug = DEFAULT,
    ):
        return _r(root).search(
            q, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol, rerank=rerank,
        )

    @app.get("/api/definition")
    async def api_definition(name: str, file: str | None = None,
                              root: RootSlug = DEFAULT):
        return _r(root).definition(name, file=file)

    @app.get("/api/references")
    async def api_references(name: str, root: RootSlug = DEFAULT):
        return _r(root).references(name)

    @app.get("/api/call_graph")
    async def api_call_graph(name: str, depth: int = 2, root: RootSlug = DEFAULT):
        return _r(root).call_graph(name, depth=depth)

    @app.get("/api/file_map")
    async def api_file_map(file: str, root: RootSlug = DEFAULT):
        return _r(root).file_map(file)

    @app.get("/api/neighborhood")
    async def api_neighborhood(name: str, limit: int = 10, root: RootSlug = DEFAULT):
        return _r(root).neighborhood(name, limit=limit)

    @app.get("/api/explore")
    async def api_explore(seeds: str, hops: int = 3, limit: int = 25,
                           root: RootSlug = DEFAULT):
        seed_list = [s.strip() for s in seeds.split(",") if s.strip()]
        return _r(root).explore(seeds=seed_list, hops=hops, limit=limit)

    @app.get("/api/impact_of")
    async def api_impact_of(target: str, depth: int = 3, limit: int = 50,
                             root: RootSlug = DEFAULT):
        return _r(root).impact_of(target, depth=depth, limit=limit)

    @app.get("/api/test_impact")
    async def api_test_impact(target: str, limit: int = 25, root: RootSlug = DEFAULT):
        return _r(root).test_impact(target, limit=limit)

    @app.get("/api/processes")
    async def api_processes(limit: int = 25, max_chain_len: int = 8,
                             root: RootSlug = DEFAULT):
        return _r(root).processes(limit=limit, max_chain_len=max_chain_len)

    @app.get("/api/wiki/list")
    async def api_wiki_list(root: RootSlug = DEFAULT):
        from docgraph.wiki import list_wiki
        return list_wiki(_slot(root).cfg)

    @app.get("/api/wiki/page")
    async def api_wiki_page(slug: str, root: RootSlug = DEFAULT):
        from docgraph.wiki import get_wiki_page
        page = get_wiki_page(_slot(root).cfg, slug)
        if not page:
            raise HTTPException(404, "wiki page not found")
        return page

    @app.post("/api/wiki/build")
    async def api_wiki_build(payload: dict | None = None, root: RootSlug = DEFAULT):
        from docgraph.wiki import build_wiki
        slot = _slot(root)
        p = payload or {}
        only = p.get("module") if isinstance(p, dict) else None
        force = bool(p.get("force")) if isinstance(p, dict) else False
        try:
            depth = int(p.get("depth", 12)) if isinstance(p, dict) else 12
        except (TypeError, ValueError):
            depth = 12
        pages = await asyncio.to_thread(
            build_wiki, slot.cfg, slot.db_ro, None, only, None, force, depth,
        )
        return {"built": len(pages), "modules": [pg.module for pg in pages]}

    @app.post("/api/cypher")
    async def api_cypher(payload: dict, root: RootSlug = DEFAULT):
        return _r(root).cypher(payload.get("query", ""), limit=int(payload.get("limit", 100)))

    @app.get("/api/git_changes")
    async def api_git_changes(ref: str | None = None, root: RootSlug = DEFAULT):
        return _r(root).git_changes(ref=ref)

    @app.get("/api/git_blame")
    async def api_git_blame(file: str, line_start: int = 1,
                             line_end: int | None = None, root: RootSlug = DEFAULT):
        return _r(root).git_blame(file, line_start=line_start, line_end=line_end)

    @app.get("/api/git_recent")
    async def api_git_recent(file: str | None = None, limit: int = 20,
                              root: RootSlug = DEFAULT):
        return _r(root).git_recent(file=file, limit=limit)

    @app.get("/api/rules_for")
    async def api_rules_for(file: str, root: RootSlug = DEFAULT):
        return _r(root).rules_for(file)

    @app.get("/api/search_docs")
    async def api_search_docs(q: str, limit: int = 10, root: RootSlug = DEFAULT):
        return _r(root).search_docs(q, limit=limit)

    @app.get("/api/graph")
    async def api_graph(limit_nodes: int = 10000, root: RootSlug = DEFAULT):
        return _r(root).graph_dump(limit_nodes=limit_nodes)

    @app.get("/api/files")
    async def api_files(root: RootSlug = DEFAULT):
        return _r(root).files_dump()

    @app.get("/api/node_neighbors")
    async def api_node_neighbors(id: int, hops: int = 1, root: RootSlug = DEFAULT):
        return _r(root).node_neighbors(int(id), hops=hops)

    @app.get("/api/stats")
    async def api_stats(root: RootSlug = DEFAULT):
        slot = _slot(root)
        d = slot.db_ro
        rows = d.fetch_all("CALL show_tables() RETURN *")
        out: dict = {"tables": rows, "repo": str(slot.cfg.repo_root)}
        for label in ("File", "Function", "Class", "Variable", "Module"):
            r = d.fetch_all(f"MATCH (n:{label}) RETURN count(n) AS c")
            out[label] = r[0]["c"] if r else 0
        return out

    # --- SSE: live reindex events ---
    @app.get("/api/events")
    async def api_events(request: Request) -> StreamingResponse:
        """SSE stream. Watcher tasks broadcast `reindex_done` after each
        per-root reindex; payload carries `{repo_slug, ts, events}`."""
        queue: asyncio.Queue[dict] = asyncio.Queue()
        app.state.subscribers.append(queue)

        async def gen() -> AsyncIterator[bytes]:
            try:
                yield b": ready\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
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

    # --- Admin: in-process index ---
    # Lets external supervisors (telecode) trigger a reindex without
    # spawning a separate `docgraph index` subprocess that would fight
    # the host for the Kuzu writer lock. The host briefly closes its RO
    # handle for the affected root, takes the writer, runs an incremental
    # (or full) index pass via `Indexer.index_all`, then reopens RO.
    @app.post("/api/admin/index")
    async def api_admin_index(payload: dict | None = None,
                               root: RootSlug = DEFAULT):
        from docgraph.index import Indexer
        from docgraph.embed import Embedder, GPU_PROVIDERS
        slot = _slot(root)
        full = bool((payload or {}).get("full", False))

        def _do() -> dict:
            writer = workspace.take_writer(slot.cfg.repo_root)
            try:
                if full:
                    writer.wipe(keep_schema=False)
                writer.init_schema()
                embedder = Embedder(
                    slot.cfg.embedding_model,
                    providers=list(GPU_PROVIDERS) if slot.cfg.gpu else None,
                )
                indexer = Indexer(slot.cfg, writer, embedder=embedder)
                stats = indexer.index_all(incremental=not full)
                # Indexer.index_all(incremental=False) swaps `self.db` for a
                # fresh GraphDB after wipe — close that one, not the original.
                try:
                    indexer.db.close()
                except Exception:
                    pass
                return stats
            finally:
                workspace.release_writer(slot.cfg.repo_root)

        try:
            stats = await asyncio.to_thread(_do)
        except Exception as exc:
            log.exception("api_admin_index failed: %s", exc)
            raise HTTPException(500, f"index failed: {exc}") from exc
        import time as _time
        workspace.mark_indexed(slot.cfg.repo_root, _time.time())
        broadcast(app, "reindex_done", {
            "repo_slug": slot.slug, "ts": _time.time(), "events": -1,
        })
        return {"slug": slot.slug, "full": full, "stats": stats}

    @app.get("/api/file_content")
    async def api_file_content(file: str, root: RootSlug = DEFAULT):
        slot = _slot(root)
        cfg = slot.cfg
        full = cfg.path_for(file).resolve()
        allowed = False
        for r, _ in cfg.roots_with_prefix():
            try:
                full.relative_to(r.resolve())
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
