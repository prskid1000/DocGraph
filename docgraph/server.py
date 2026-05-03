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
import contextlib
import enum
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from docgraph.cancel import OperationCancelled
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
    # Build the FastMCP server up-front so we can chain its lifespan
    # (initializes the streamable-HTTP session manager) into FastAPI's.
    # Without this, any POST /mcp/ would 500 with "Session terminated".
    from docgraph.mcp_tools import make_mcp
    mcp_server = make_mcp(workspace)
    mcp_http = mcp_server.http_app(path="/")  # Starlette-with-lifespan

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        async with mcp_http.lifespan(_app):
            yield

    app = FastAPI(title="DocGraph", version="2.2.0", lifespan=_lifespan)
    RootSlug = _root_enum(workspace)
    DEFAULT = RootSlug(workspace.default_slug())

    app.state.workspace = workspace
    app.state.subscribers = []  # list[asyncio.Queue]
    app.state.mcp = mcp_server

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
        rerank: bool | None = None, root: RootSlug = DEFAULT,
    ):
        slot = _slot(root)
        # Per-call ?rerank= wins; otherwise fall back to cfg.rerank_default
        # (set via DOCGRAPH_RERANK_DEFAULT env var or telecode tray toggle).
        use_rerank = rerank if rerank is not None else bool(getattr(slot.cfg, "rerank_default", False))
        return slot.retriever.search(
            q, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol, rerank=use_rerank,
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
        workspace.reset_cancel(slot.cfg.repo_root)
        token = workspace.cancel_token_for(slot.cfg.repo_root)
        try:
            pages = await asyncio.to_thread(
                build_wiki, slot.cfg, slot.db_ro, None, only, None, force, depth,
                token,
            )
        except OperationCancelled:
            workspace.reset_cancel(slot.cfg.repo_root)
            raise HTTPException(499, "wiki build cancelled") from None
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
        # Reset before kicking off so a stale cancel from a previous run
        # doesn't immediately abort.
        workspace.reset_cancel(slot.cfg.repo_root)
        token = workspace.cancel_token_for(slot.cfg.repo_root)

        def _do() -> tuple[dict, str]:
            # Indexer.index_all() prints Rich progress bars. We capture
            # them so the response can return both the stats dict AND a
            # plain-text transcript the API caller (telecode) can write
            # to its own log file. Capturing also avoids any cp1252-
            # related crashes if the host runs in a non-utf-8 console.
            import io, contextlib
            sink = io.StringIO()
            writer = workspace.take_writer(slot.cfg.repo_root)
            try:
                if full:
                    writer.wipe(keep_schema=False)
                writer.init_schema()
                # Reuse the workspace's embedder pool. DirectML / CUDA
                # don't take kindly to two ONNX sessions sharing one GPU
                # device; building a fresh `Embedder` here would race
                # the host's existing read-path embedder.
                embedder = workspace._embedder_for(slot.cfg)
                indexer = Indexer(slot.cfg, writer, embedder=embedder)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    stats = indexer.index_all(incremental=not full,
                                               cancel_token=token)
                # Indexer.index_all(incremental=False) swaps `self.db` for a
                # fresh GraphDB after wipe — close that one, not the original.
                try:
                    indexer.db.close()
                except Exception:
                    pass
                return stats, sink.getvalue()
            finally:
                workspace.release_writer(slot.cfg.repo_root)

        try:
            stats, captured = await asyncio.to_thread(_do)
        except OperationCancelled:
            # 499 = "Client Closed Request" (nginx convention). Telecode
            # treats this as a clean cancel rather than a failure.
            workspace.reset_cancel(slot.cfg.repo_root)
            raise HTTPException(499, "index cancelled") from None
        except Exception as exc:
            log.exception("api_admin_index failed: %s", exc)
            raise HTTPException(500, f"index failed: {exc}") from exc
        import time as _time
        workspace.mark_indexed(slot.cfg.repo_root, _time.time())
        broadcast(app, "reindex_done", {
            "repo_slug": slot.slug, "ts": _time.time(), "events": -1,
        })
        return {"slug": slot.slug, "full": full, "stats": stats, "log": captured}

    # Admin: cancel the currently-running long-op (index / wiki) for a
    # root. Cooperative — the long op polls a per-root token at safe
    # checkpoints and raises `OperationCancelled` when it fires. Idempotent.
    @app.post("/api/admin/cancel")
    async def api_admin_cancel(root: RootSlug = DEFAULT):
        slot = _slot(root)
        workspace.request_cancel(slot.cfg.repo_root)
        return {"slug": slot.slug, "cancel_requested": True}

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

    # Admin: wipe a root's index. Same writer-lock dance as /api/admin/index.
    @app.post("/api/admin/clear")
    async def api_admin_clear(root: RootSlug = DEFAULT):
        slot = _slot(root)

        def _do() -> dict:
            writer = workspace.take_writer(slot.cfg.repo_root)
            try:
                writer.wipe(keep_schema=False)
                writer.init_schema()
            finally:
                workspace.release_writer(slot.cfg.repo_root)
            # Also drop the per-file delta cache so the next index pass
            # is treated as a full rebuild rather than an incremental no-op.
            cache = slot.cfg.data_dir / "cache.json"
            try:
                if cache.exists():
                    cache.unlink()
            except OSError:
                pass
            wiki_dir = slot.cfg.data_dir / "wiki"
            try:
                if wiki_dir.exists():
                    import shutil as _sh
                    _sh.rmtree(wiki_dir, ignore_errors=True)
            except OSError:
                pass
            return {"slug": slot.slug, "cleared": True}

        try:
            out = await asyncio.to_thread(_do)
        except Exception as exc:
            log.exception("api_admin_clear failed: %s", exc)
            raise HTTPException(500, f"clear failed: {exc}") from exc
        broadcast(app, "reindex_done", {
            "repo_slug": slot.slug, "ts": __import__("time").time(), "events": -1,
        })
        return out

    # External docs (`@Docs` parity). Add takes the writer; list reads from
    # the slot's existing RO handle; remove takes the writer briefly.
    @app.get("/api/docs/list")
    async def api_docs_list(root: RootSlug = DEFAULT):
        from docgraph.docs import list_docs
        slot = _slot(root)
        return await asyncio.to_thread(list_docs, slot.cfg)

    @app.post("/api/docs/add")
    async def api_docs_add(payload: dict, root: RootSlug = DEFAULT):
        from docgraph.docs import add_doc
        slot = _slot(root)
        url = (payload or {}).get("url", "").strip()
        if not url:
            raise HTTPException(400, "url is required")

        workspace.reset_cancel(slot.cfg.repo_root)
        token = workspace.cancel_token_for(slot.cfg.repo_root)

        def _do() -> dict:
            writer = workspace.take_writer(slot.cfg.repo_root)
            try:
                return add_doc(slot.cfg, url, db=writer, cancel_token=token)
            finally:
                workspace.release_writer(slot.cfg.repo_root)

        try:
            out = await asyncio.to_thread(_do)
        except OperationCancelled:
            workspace.reset_cancel(slot.cfg.repo_root)
            raise HTTPException(499, "docs add cancelled") from None
        except Exception as exc:
            log.exception("api_docs_add failed: %s", exc)
            raise HTTPException(500, f"docs add failed: {exc}") from exc
        if isinstance(out, dict) and out.get("error"):
            raise HTTPException(400, out["error"])
        return out

    @app.post("/api/docs/remove")
    async def api_docs_remove(payload: dict, root: RootSlug = DEFAULT):
        from docgraph.docs import remove_doc
        slot = _slot(root)
        url = (payload or {}).get("url", "").strip()
        if not url:
            raise HTTPException(400, "url is required")

        def _do() -> int:
            writer = workspace.take_writer(slot.cfg.repo_root)
            try:
                return remove_doc(slot.cfg, url, db=writer)
            finally:
                workspace.release_writer(slot.cfg.repo_root)

        try:
            removed = await asyncio.to_thread(_do)
        except Exception as exc:
            log.exception("api_docs_remove failed: %s", exc)
            raise HTTPException(500, f"docs remove failed: {exc}") from exc
        return {"url": url, "removed_chunks": removed}

    # Mount the FastMCP HTTP app under /mcp. Telecode's bridge does
    # POST http://host:port/mcp; Starlette redirects /mcp → /mcp/ which
    # the mcp.client.streamable_http transport (httpx-backed) follows.
    app.mount("/mcp", mcp_http)

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
