"""File watcher: per-root awatch tasks that incrementally reindex on change.

After this rewrite, watching is workspace-scoped. The single `docgraph
host` process can watch any subset of its registered roots — each root
gets its own asyncio task holding its own writer connection on its own
.docgraph/graph.kuzu (Kuzu's per-file locks make this safe).

A workspace-wide `asyncio.Semaphore(1)` serializes reindexes across roots
so two CPU-bound passes don't fight; an unbounded semaphore would let
them run in parallel but the indexer is heavy enough that fairness is
better than aggregate throughput here.

After each per-root reindex:
  1. Close the writer.
  2. Reopen the workspace's read-only slot for that root (via
     `Workspace.release_writer`) so subsequent reads see the new data.
  3. Broadcast SSE `reindex_done {repo_slug, ts, events}` so the live
     UI refreshes only the affected slot.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from rich.console import Console
from watchfiles import Change, awatch

from docgraph.config import Config, MAX_FILE_BYTES
# (embedder is sourced from the workspace pool, not constructed here)
from docgraph.index import Indexer
from docgraph.parse import detect_language
from docgraph.workspace import Workspace, slug_for_root

_console = Console()
log = logging.getLogger(__name__)


# ── path-level filter (unchanged) ───────────────────────────────────────────

def _is_relevant(cfg: Config, path: Path) -> bool:
    abs_path = path.resolve()
    matched_root: Path | None = None
    rel: str = ""
    for root, _prefix in cfg.roots_with_prefix():
        try:
            rel = str(abs_path.relative_to(root.resolve())).replace("\\", "/")
            matched_root = root
            break
        except ValueError:
            continue
    if matched_root is None:
        return False
    if cfg.is_ignored(rel, root=matched_root):
        return False
    if path.exists():
        if path.is_dir():
            return False
        if detect_language(path) is None:
            return False
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return False
        except OSError:
            return False
    else:
        if detect_language(path) is None:
            return False
    return True


class _WatchFilter:
    """watchfiles filter — drops ignored / non-source paths pre-debounce."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def __call__(self, change: Change, path: str) -> bool:
        return _is_relevant(self.cfg, Path(path))


# ── per-root watcher coroutine ──────────────────────────────────────────────

async def _watch_one(
    workspace: Workspace,
    root: Path,
    debounce_ms: int,
    serialize: asyncio.Semaphore,
    on_reindex: callable | None = None,
) -> None:
    """Watch one root forever (until cancelled). Holds the writer for that
    root for the duration of each reindex pass; releases between passes so
    the read-only slot can serve queries."""
    slot = workspace.resolve(root)
    cfg = slot.cfg
    workspace.mark_watching(root, True)

    # Baseline incremental — just to make sure the on-disk DB is current.
    try:
        await asyncio.to_thread(_baseline_reindex, workspace, root)
    except Exception as exc:
        log.exception("baseline reindex failed for %s: %s", root, exc)

    roots = [str(r) for r, _ in cfg.roots_with_prefix()]
    _console.print(
        f"[green]Watching[/] {slot.slug} ({len(roots)} fs root) "
        f"debounce={debounce_ms}ms"
    )

    try:
        async for changes in awatch(
            *roots,
            step=debounce_ms,
            recursive=True,
            watch_filter=_WatchFilter(cfg),
            stop_event=None,
        ):
            relevant = [Path(p) for _, p in changes]
            if not relevant:
                continue
            _console.rule(
                f"[cyan]Reindex {slot.slug}[/] — {len(relevant)} event(s)",
                style="cyan",
            )
            async with serialize:
                try:
                    await asyncio.to_thread(_reindex, workspace, root)
                except Exception as exc:
                    _console.print(f"[red]Reindex {slot.slug} failed:[/] {exc}")
                    continue
            workspace.mark_indexed(root, time.time())
            if on_reindex is not None:
                try:
                    on_reindex(slot.slug, len(relevant))
                except Exception as exc:
                    log.exception("on_reindex callback failed: %s", exc)
    except asyncio.CancelledError:
        pass
    finally:
        workspace.mark_watching(root, False)


def _baseline_reindex(workspace: Workspace, root: Path) -> None:
    """Run an incremental index pass on `root`. Used to bring the on-disk
    DB up to date before the watcher starts emitting deltas."""
    slot = workspace.resolve(root)
    writer = workspace.take_writer(root)
    try:
        # Use the workspace-pooled embedder (not a fresh standalone one) so
        # in-process mode shares a single model + idle-unload is single-source,
        # and so daemon routing (Embedder.embed → daemon) applies uniformly.
        embedder = workspace.embedder_for(slot.cfg)
        indexer = Indexer(slot.cfg, writer, embedder=embedder)
        indexer.index_all(incremental=True)
    finally:
        workspace.release_writer(root)


def _reindex(workspace: Workspace, root: Path) -> None:
    """Same as baseline; kept as a separate function so future logic can
    diverge (e.g. a faster delta-only pass)."""
    _baseline_reindex(workspace, root)


# ── public entry points ─────────────────────────────────────────────────────

async def watch_workspace(
    workspace: Workspace,
    roots: list[Path],
    debounce_ms: int = 500,
) -> None:
    """Foreground multi-root watcher. No web/MCP — pure reindex loop."""
    serialize = asyncio.Semaphore(1)
    tasks = [
        asyncio.create_task(_watch_one(workspace, r, debounce_ms, serialize, None))
        for r in roots
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def watch_and_serve_workspace(
    workspace: Workspace,
    app,
    roots: list[Path],
    *,
    host: str = "127.0.0.1",
    port: int = 5500,
    debounce_ms: int = 500,
    verbose: bool = False,
) -> None:
    """Run the FastAPI host AND per-root watchers on one event loop."""
    import uvicorn
    from docgraph.server import broadcast

    app.state.loop = asyncio.get_running_loop()
    config = uvicorn.Config(
        app, host=host, port=port,
        log_level="info" if verbose else "warning",
        # NB: lifespan must be enabled — the FastAPI app's lifespan
        # initializes FastMCP's streamable-HTTP session manager. Setting
        # this to "off" makes /mcp 500 on every request.
        lifespan="on", timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.1)
    _console.print(
        f"[green]DocGraph host[/] http://{host}:{port}/  "
        f"[dim](watching {len(roots)} root{'s' if len(roots) > 1 else ''})[/]"
    )

    serialize = asyncio.Semaphore(1)

    def _emit(slug: str, n_events: int) -> None:
        broadcast(app, "reindex_done", {
            "repo_slug": slug, "ts": time.time(), "events": n_events,
        })

    watcher_tasks = [
        asyncio.create_task(_watch_one(workspace, r, debounce_ms, serialize, _emit))
        for r in roots
    ]

    try:
        await server_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.should_exit = True
        for t in watcher_tasks:
            t.cancel()
        await asyncio.gather(*watcher_tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
