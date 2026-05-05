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
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from docgraph.cancel import OperationCancelled
from docgraph.db import DatabaseBusy
from docgraph.locks import BusyTimeout
from docgraph.workspace import Workspace, RootSlot

log = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent / "ui"

@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    root: str = ""
    status: str = "running" # running, completed, failed, cancelled
    result: dict | None = None
    error: str | None = None
    log: str = ""
    progress: dict | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    cancel_token: Any = None

class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def create_job(self, jtype: str, root: str, token: Any) -> Job:
        j = Job(type=jtype, root=root, cancel_token=token)
        self.jobs[j.id] = j
        return j

    def get_job(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def finish_job(self, job_id: str, status: str, result: dict | None = None, error: str | None = None, log: str = ""):
        j = self.jobs.get(job_id)
        if j:
            j.status = status
            j.result = result
            j.error = error
            j.log = log
            j.end_time = time.time()

    def update_job_progress(self, job_id: str, progress: dict) -> None:
        j = self.jobs.get(job_id)
        if j:
            j.progress = progress

job_manager = JobManager()


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
        # Cache the running loop so sync callers (the watcher thread) can
        # bridge their take_writer/release_writer into DBLock via
        # run_coroutine_threadsafe. Must happen before the watcher is
        # spawned so there's no window where the loop is missing.
        workspace.attach_loop(asyncio.get_running_loop())
        async with mcp_http.lifespan(_app):
            yield

    app = FastAPI(title="DocGraph", version="2.2.0", lifespan=_lifespan)
    RootSlug = _root_enum(workspace)
    DEFAULT = RootSlug(workspace.default_slug())

    app.state.workspace = workspace
    app.state.subscribers = []  # list[asyncio.Queue]
    app.state.mcp = mcp_server

    # Reads racing with a writer hit DatabaseBusy (db_ro.conn is None while
    # the writer holds Kuzu's exclusive file lock). Surface as 503 +
    # Retry-After so clients poll instead of crashing on AttributeError.
    @app.exception_handler(DatabaseBusy)
    async def _busy_handler(_request: Request, exc: DatabaseBusy):
        return JSONResponse(
            status_code=503,
            content={"error": "database_busy", "detail": str(exc)},
            headers={"Retry-After": "2"},
        )

    @app.exception_handler(BusyTimeout)
    async def _busy_timeout_handler(_request: Request, exc: BusyTimeout):
        return JSONResponse(
            status_code=503,
            content={"error": "lock_timeout", "detail": str(exc)},
            headers={"Retry-After": "2"},
        )

    # Paths that don't touch the graph DB — skip the read gate so a
    # status poll (e.g. /api/jobs/<id> while an index is running) doesn't
    # block on the very lock the caller is waiting to clear. Also skips
    # SSE (would deadlock — consumer waits for events emitted AFTER the
    # write completes) and the chat endpoint (LLM-only, no DB).
    _GATE_SKIP_PREFIXES = (
        "/api/jobs/", "/api/admin/cancel", "/api/locks",
        "/api/roots", "/api/llm_config", "/api/events",
        "/api/chat", "/api/file_content",
    )

    # Read gate. Every GET /api/* request that targets a specific root
    # blocks briefly if a writer is currently held — the watcher reindex /
    # API index / wiki op finishes (typically <2s for incrementals,
    # <30s for wiki). Gate-times-out → 503 + Retry-After. While a request
    # is in flight we increment reads_in_flight so a queued writer waits
    # for clean drain instead of cancelling mid-query.
    @app.middleware("http")
    async def _read_gate(request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        # Only gate GET /api/* on graph-touching routes. POST/DELETE writes
        # have their own queueing via workspace.take_writer_async.
        if method != "GET" or not path.startswith("/api/"):
            return await call_next(request)
        if any(path.startswith(p) for p in _GATE_SKIP_PREFIXES):
            return await call_next(request)
        slug = request.query_params.get("root")
        try:
            slot = workspace.resolve(slug) if slug else workspace.default()
        except KeyError:
            return await call_next(request)
        try:
            await slot.lock.wait_idle(timeout=workspace.lock_timeouts.read_wait)
        except BusyTimeout as exc:
            return JSONResponse(
                status_code=503,
                content={"error": "lock_timeout", "detail": str(exc),
                         "lock": _lock_status_dict(slot)},
                headers={"Retry-After": "2"},
            )
        await slot.lock.enter_read()
        try:
            return await call_next(request)
        finally:
            await slot.lock.leave_read()

    def _lock_status_dict(slot: RootSlot) -> dict:
        s = slot.lock.status()
        return {
            "slug": slot.slug,
            "held": s.held,
            "holder_label": s.holder_label,
            "holder_age": s.holder_age,
            "queue_depth": s.queue_depth,
            "queued_labels": s.queued_labels,
            "reads_in_flight": s.reads_in_flight,
        }

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

    # Lock observability — show what's holding writers / queue depths.
    # Useful for debugging "why is /api/stats slow" or "why is the UI
    # showing 503 retries"; the UI's status badge reads this.
    @app.get("/api/locks")
    async def api_locks():
        out = []
        for slot in (workspace.resolve(s) for s in workspace.slugs()):
            out.append(_lock_status_dict(slot))
        return {"locks": out, "timeouts": {
            "read_wait": workspace.lock_timeouts.read_wait,
            "write_wait": workspace.lock_timeouts.write_wait,
            "wiki_write": workspace.lock_timeouts.wiki_write,
            "watcher_write": workspace.lock_timeouts.watcher_write,
            "force_free_after": workspace.lock_timeouts.force_free_after,
        }}

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
        results = slot.retriever.search(
            q, kind=kind, limit=limit,
            focus_file=focus_file, focus_symbol=focus_symbol, rerank=use_rerank,
        )
        log.info(
            f"Search: query='{q}' kind={kind} limit={limit} rerank={use_rerank} "
            f"root={slot.slug} -> {len(results)} hits"
        )
        return results

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
        fetch_links_wiki = bool(p.get("fetch_links", True)) if isinstance(p, dict) else True
        force_fetch_wiki = bool(p.get("force_fetch_links", False)) if isinstance(p, dict) else False
        default_depth = int(getattr(slot.cfg, "wiki_depth", 12) or 12)
        try:
            depth = int(p.get("depth", default_depth)) if isinstance(p, dict) else default_depth
        except (TypeError, ValueError):
            depth = default_depth
        workspace.reset_cancel(slot.cfg.repo_root)
        token = workspace.cancel_token_for(slot.cfg.repo_root)

        job = job_manager.create_job("wiki", str(slot.cfg.repo_root), token)

        # Per-module progress → SSE so telecode mirrors CLI status into
        # docgraph_wiki.log. Payload: {repo_slug, phase, current, total, module, ts}.
        import time as _time_progress
        def _progress_cb(phase: str, current: int = 0, total: int = 0,
                         module: str = "") -> None:
            broadcast(app, "wiki_progress", {
                "job_id": job.id,
                "repo_slug": slot.slug,
                "phase": phase,
                "current": int(current),
                "total": int(total),
                "module": module,
                "ts": _time_progress.time(),
            })

        async def _run_job():
            try:
                pages = await asyncio.to_thread(
                    build_wiki, slot.cfg, slot.db_ro, None, only, None, force, depth,
                    token, _progress_cb, fetch_links_wiki, force_fetch_wiki,
                )
                result = {"built": len(pages), "modules": [pg.module for pg in pages]}
                job_manager.finish_job(job.id, "completed", result=result)
            except OperationCancelled:
                workspace.reset_cancel(slot.cfg.repo_root)
                job_manager.finish_job(job.id, "cancelled", error="wiki build cancelled")
            except Exception as exc:
                log.exception("api_wiki_build failed: %s", exc)
                job_manager.finish_job(job.id, "failed", error=str(exc))

        asyncio.create_task(_run_job())
        return {"built": -1, "modules": [], "job_id": job.id, "status": "running"}


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
        # Edge counts per REL table. show_tables() rows have a `type` column
        # equal to "REL" for relationship tables; iterate those and count.
        edges_by_type: dict[str, int] = {}
        total_edges = 0
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("type", "")).upper() != "REL":
                continue
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            try:
                r = d.fetch_all(f"MATCH ()-[r:{name}]->() RETURN count(r) AS c")
                c = int(r[0]["c"]) if r else 0
            except Exception:
                c = 0
            edges_by_type[name] = c
            total_edges += c
        out["edges"] = total_edges
        out["edges_by_type"] = edges_by_type
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

    @app.get("/api/jobs")
    async def api_get_jobs(root: str | None = None):
        jobs = list(job_manager.jobs.values())
        if root:
            jobs = [j for j in jobs if j.root == root]
        return [{"id": j.id, "type": j.type, "root": j.root, "status": j.status, "start_time": j.start_time, "end_time": j.end_time} for j in jobs]

    @app.get("/api/jobs/{job_id}")
    async def api_get_job(job_id: str):
        j = job_manager.get_job(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        return {
            "id": j.id,
            "type": j.type,
            "root": j.root,
            "status": j.status,
            "result": j.result,
            "error": j.error,
            "log": j.log,
            "progress": j.progress,
            "start_time": j.start_time,
            "end_time": j.end_time
        }

    @app.post("/api/jobs/{job_id}/cancel")
    async def api_cancel_job(job_id: str):
        j = job_manager.get_job(job_id)
        if not j:
            raise HTTPException(404, "job not found")
        if j.cancel_token:
            j.cancel_token.request()
        return {"id": j.id, "cancel_requested": True}

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
        fetch_links = bool((payload or {}).get("fetch_links", True))
        force_fetch = bool((payload or {}).get("force_fetch_links", False))
        # Reset before kicking off so a stale cancel from a previous run
        # doesn't immediately abort.
        workspace.reset_cancel(slot.cfg.repo_root)
        token = workspace.cancel_token_for(slot.cfg.repo_root)

        job = job_manager.create_job("index", str(slot.cfg.repo_root), token)

        # Phase progress → SSE so telecode (or any subscriber) can mirror
        # the same status the CLI prints. Sent on the existing `index_progress`
        # event with `{repo_slug, phase, current, total, ts}`.
        import time as _time_progress
        def _progress_cb(phase: str, current: int = 0, total: int = 0) -> None:
            broadcast(app, "index_progress", {
                "job_id": job.id,
                "repo_slug": slot.slug,
                "phase": phase,
                "current": int(current),
                "total": int(total),
                "ts": _time_progress.time(),
            })
            # Also store progress on the Job so /api/jobs/<id> can return it
            try:
                job_manager.update_job_progress(job.id, {
                    "phase": phase,
                    "current": int(current),
                    "total": int(total),
                    "ts": _time_progress.time(),
                })
            except Exception:
                pass

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
                writer.init_schema()
                embedder = workspace._embedder_for(slot.cfg)
                indexer = Indexer(slot.cfg, writer, embedder=embedder)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    stats = indexer.index_all(incremental=not full,
                                               cancel_token=token,
                                               progress_cb=_progress_cb,
                                               fetch_links=fetch_links,
                                               force_fetch=force_fetch)
                try:
                    indexer.db.close()
                except Exception:
                    pass
                return stats, sink.getvalue()
            finally:
                workspace.release_writer(slot.cfg.repo_root)

        async def _run_job():
            try:
                stats, captured = await asyncio.to_thread(_do)
                job_manager.finish_job(job.id, "completed", result=stats, log=captured)
                import time as _time
                workspace.mark_indexed(slot.cfg.repo_root, _time.time())
                broadcast(app, "reindex_done", {
                    "job_id": job.id, "repo_slug": slot.slug, "ts": _time.time(), "events": -1,
                })
            except OperationCancelled:
                workspace.reset_cancel(slot.cfg.repo_root)
                job_manager.finish_job(job.id, "cancelled", error="index cancelled")
            except Exception as exc:
                log.exception("api_admin_index failed: %s", exc)
                job_manager.finish_job(job.id, "failed", error=str(exc))

        asyncio.create_task(_run_job())
        return {"slug": slot.slug, "full": full, "job_id": job.id, "status": "running"}

    # Admin: cancel the currently-running long-op (index / wiki) for a
    # root. Cooperative — the long op polls a per-root token at safe
    # checkpoints and raises `OperationCancelled` when it fires. Idempotent.
    # ── Extra local paths CRUD ──────────────────────────────────────────

    @app.get("/api/repos")
    async def api_repos_list(root: RootSlug = DEFAULT):
        import json as _json
        slot = _slot(root)
        repos_file = slot.cfg.data_dir / "repos.json"
        if not repos_file.exists():
            return []
        try:
            return _json.loads(repos_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    @app.post("/api/repos")
    async def api_repos_add(payload: dict, root: RootSlug = DEFAULT):
        import json as _json
        slot = _slot(root)
        p = str((payload or {}).get("path", "") or "").strip()
        if not p:
            raise HTTPException(400, "path required")
        repos_file = slot.cfg.data_dir / "repos.json"
        try:
            raw = _json.loads(repos_file.read_text(encoding="utf-8")) if repos_file.exists() else []
        except Exception:
            raw = []
        resolved = str(Path(p).resolve())
        if resolved not in raw:
            raw.append(resolved)
            repos_file.write_text(_json.dumps(raw))
        return {"paths": raw}

    @app.delete("/api/repos")
    async def api_repos_remove(path: str, root: RootSlug = DEFAULT):
        import json as _json
        slot = _slot(root)
        repos_file = slot.cfg.data_dir / "repos.json"
        try:
            raw = _json.loads(repos_file.read_text(encoding="utf-8")) if repos_file.exists() else []
        except Exception:
            raw = []
        resolved = str(Path(path).resolve())
        raw = [p for p in raw if p != resolved]
        repos_file.write_text(_json.dumps(raw))
        return {"paths": raw}

    # ── External links CRUD ─────────────────────────────────────────────

    @app.get("/api/links")
    async def api_links_list(root: RootSlug = DEFAULT):
        from docgraph.links import load_links
        from dataclasses import asdict
        slot = _slot(root)
        return [asdict(lk) for lk in load_links(slot.cfg.data_dir)]

    @app.post("/api/links")
    async def api_links_upsert(payload: dict, root: RootSlug = DEFAULT):
        from docgraph.links import upsert_link
        slot = _slot(root)
        url = str(payload.get("url", "")).strip()
        if not url:
            raise HTTPException(400, "url required")
        depth = int(payload.get("depth", 1))
        ttl_hours = float(payload.get("ttl_hours", 24.0))
        links = upsert_link(slot.cfg.data_dir, url, depth, ttl_hours)
        return {"links": len(links)}

    @app.delete("/api/links")
    async def api_links_remove(url: str, root: RootSlug = DEFAULT):
        from docgraph.links import remove_link
        slot = _slot(root)
        links = remove_link(slot.cfg.data_dir, url)
        return {"links": len(links)}

    @app.post("/api/links/fetch")
    async def api_links_fetch(payload: dict | None = None, root: RootSlug = DEFAULT):
        """Trigger a standalone fetch job (no re-index)."""
        from docgraph.fetch import fetch_all
        slot = _slot(root)
        p = payload or {}
        force = bool(p.get("force", False))
        only_url: str | None = p.get("url") or None

        job = job_manager.create_job("fetch", str(slot.cfg.repo_root), None)

        async def _run():
            try:
                results = await asyncio.to_thread(
                    fetch_all, slot.cfg.data_dir, force, only_url
                )
                job_manager.finish_job(job.id, "completed", result=results)
            except Exception as exc:
                log.exception("api_links_fetch failed: %s", exc)
                job_manager.finish_job(job.id, "failed", error=str(exc))

        asyncio.create_task(_run())
        return {"job_id": job.id, "status": "running"}

    # ────────────────────────────────────────────────────────────────────

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

    # Admin: wipe a root's index. rmtrees the entire .docgraph/ — same as
    # the `docgraph clear` CLI — then opens a fresh empty schema. The
    # workspace handles the close-RO → rmtree → reopen-RO dance so other
    # in-flight reads against this slot don't crash.
    @app.post("/api/admin/clear")
    async def api_admin_clear(root: RootSlug = DEFAULT):
        slot = _slot(root)
        # If a watcher / index / wiki is mid-flight it's holding the writer
        # and clear_data() would refuse. Signal cancel + wait briefly for
        # the writer to release before giving up with 503.
        if slot.db_writer is not None:
            workspace.request_cancel(slot.cfg.repo_root)
            deadline = time.time() + 10.0
            while time.time() < deadline and slot.db_writer is not None:
                await asyncio.sleep(0.2)
            if slot.db_writer is not None:
                raise HTTPException(
                    status_code=503,
                    detail="clear blocked: writer still held after 10s — retry",
                    headers={"Retry-After": "5"},
                )
            workspace.reset_cancel(slot.cfg.repo_root)

        def _do() -> dict:
            workspace.clear_data(slot.cfg.repo_root)
            return {"slug": slot.slug, "cleared": True}

        try:
            out = await asyncio.to_thread(_do)
        except Exception as exc:
            log.exception("api_admin_clear failed: %s", exc)
            raise HTTPException(500, f"clear failed: {exc}") from exc
        broadcast(app, "reindex_done", {
            "repo_slug": slot.slug, "ts": time.time(), "events": -1,
        })
        return out

    # LLM chat — uses the same host/port/model/format the indexer uses for
    # docstring + wiki augmentation (Config.llm_*). Off when llm_model is
    # unset. The UI's right-panel Chat tab probes /api/llm_config first.
    @app.get("/api/llm_config")
    async def api_llm_config(root: RootSlug = DEFAULT):
        cfg = _slot(root).cfg
        # Features are enabled via their respective toggles; the model always
        # has a system default (qwen3.6-35b) if the user leaves it blank.
        return {
            "configured": True,
            "host": getattr(cfg, "llm_host", "localhost"),
            "port": int(getattr(cfg, "llm_port", 1235)),
            "model": getattr(cfg, "llm_model", "") or "qwen3.6-35b",
            "format": getattr(cfg, "llm_format", "openai") or "openai",
            "max_tokens": int(getattr(cfg, "llm_max_tokens", 512) or 512),
            "max_tokens_chat": int(getattr(cfg, "llm_max_tokens_chat", 0) or 0),
            "has_key": bool(getattr(cfg, "llm_api_key", "") or ""),
            "rerank_default": bool(getattr(cfg, "rerank_default", False)),
        }

    @app.post("/api/chat")
    async def api_chat(payload: dict, root: RootSlug = DEFAULT):
        cfg = _slot(root).cfg
        messages = payload.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        # Attached node context becomes a system-message preamble so the
        # model sees the snippet without the user having to paste it.
        ctx = payload.get("context") or None
        if ctx and isinstance(ctx, dict):
            name = str(ctx.get("name") or "")
            file = str(ctx.get("file") or "")
            lang = str(ctx.get("language") or "")
            snippet = str(ctx.get("snippet") or "")[:4000]
            summary = str(ctx.get("summary") or "")
            summary_part = f"\nAI Summary: {summary}" if summary else ""
            sys_note = (
                f"You are answering questions about a code entity in the user's repo.\n"
                f"Entity: `{name}` ({file}){summary_part}\n"
                f"```{lang}\n{snippet}\n```"
            )
            messages = [{"role": "system", "content": sys_note}] + list(messages)
        # Chat doesn't cap output by default — the docstring/wiki budgets
        # are for fixed-length use cases, not free-form Q&A. Cap order:
        #   1. payload.max_tokens (per-request override from the UI)
        #   2. cfg.llm_max_tokens_chat (host CLI flag / settings)
        #   3. None  → omit max_tokens for openai (server decides);
        #              anthropic falls back to 8192 since its API requires it.
        explicit_cap = payload.get("max_tokens")
        if not explicit_cap:
            cfg_cap = int(getattr(cfg, "llm_max_tokens_chat", 0) or 0)
            if cfg_cap > 0:
                explicit_cap = cfg_cap
        from docgraph.llm import LLMClient, LLMConfig
        client = LLMClient(LLMConfig(
            host=cfg.llm_host, port=int(cfg.llm_port), model=cfg.llm_model,
            format=cfg.llm_format,
            api_key=getattr(cfg, "llm_api_key", "") or None,
            # max_tokens here is only used as the Anthropic fallback below.
            max_tokens=int(explicit_cap or 8192),
            timeout=int(getattr(cfg, "llm_timeout", 60) or 60),
        ))
        try:
            if client.cfg.format == "anthropic":
                body = {
                    "model": client.cfg.model,
                    "messages": messages,
                    "max_tokens": client.cfg.max_tokens,  # required by Anthropic
                    "temperature": 0.4,
                }
                data = await asyncio.to_thread(client._post, client.cfg.endpoint, body)
                content = ""
                for b in data.get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "text":
                        content = (b.get("text") or "").strip()
                        break
            else:
                body = {
                    "model": client.cfg.model,
                    "messages": messages,
                    "temperature": 0.4,
                    "stream": False,
                    "reasoning_effort": "none",
                }
                if explicit_cap:
                    body["max_tokens"] = int(explicit_cap)
                data = await asyncio.to_thread(client._post, client.cfg.endpoint, body)
                try:
                    content = (data["choices"][0]["message"]["content"] or "").strip()
                except (KeyError, IndexError, TypeError):
                    content = ""
        except Exception as exc:
            raise HTTPException(502, f"LLM call failed: {exc}") from exc
        return {"content": content, "model": client.cfg.model}

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
