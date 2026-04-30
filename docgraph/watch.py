"""File watcher: triggers incremental reindex on changes.

Uses watchfiles (Rust notify under the hood). Filters out paths that hit
.gitignore / .docgraphignore / unsupported languages before debouncing,
so a `git checkout` of 5k files in node_modules doesn't blow up.

Two modes:
- `watch_repo` — classic: holds a single writer connection for the
  watcher's lifetime. `docgraph serve` / `docgraph mcp` against the same
  DB will fail to acquire their locks.
- `watch_and_serve` — unified: runs the watcher AND a FastAPI server in
  one process so they can share a single Database lock. After each
  reindex, we close+reopen the DB (Kuzu writer-visibility quirk) and push
  an SSE `reindex_done` event so the live UI refreshes itself.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from rich.console import Console
from watchfiles import Change, awatch, watch

from docgraph.config import Config, MAX_FILE_BYTES
from docgraph.db import GraphDB
from docgraph.embed import Embedder, GPU_PROVIDERS
from docgraph.index import Indexer
from docgraph.parse import detect_language

_console = Console()

log = logging.getLogger(__name__)


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
    # For deletes, the file no longer exists; we still want to track them.
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
        # Best-effort: only react to deletions of language-supported extensions
        if detect_language(path) is None:
            return False
    return True


def watch_repo(cfg: Config, debounce_ms: int = 500) -> None:
    """Block forever, reindexing on changes. Ctrl-C to stop."""
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(
        cfg.embedding_model,
        providers=list(GPU_PROVIDERS) if cfg.gpu else None,
    )
    indexer = Indexer(cfg, db, embedder=embedder)

    _console.rule(f"[bold cyan]Baseline index[/] — {cfg.repo_root}")
    indexer.index_all(incremental=True)

    roots = [str(r) for r, _ in cfg.roots_with_prefix()]
    _console.print(
        f"[green]Watching[/] {len(roots)} root(s), debounce={debounce_ms}ms. "
        "[dim]Ctrl-C to stop.[/]"
    )
    try:
        for changes in watch(
            *roots,
            step=debounce_ms,
            recursive=True,
            watch_filter=_WatchFilter(cfg),
        ):
            relevant = [Path(p) for _, p in changes]
            if not relevant:
                continue
            _console.rule(
                f"[bold cyan]Reindex[/] — {len(relevant)} fs event(s)",
                style="cyan",
            )
            try:
                indexer.index_all(incremental=True)
            except Exception as e:  # noqa: BLE001
                _console.print(f"[red]Reindex failed:[/] {e}")
                continue
    except KeyboardInterrupt:
        _console.print("[yellow]watcher stopped[/]")


async def watch_and_serve(
    cfg: Config,
    debounce_ms: int = 500,
    host: str = "127.0.0.1",
    port: int = 5500,
) -> None:
    """Run the watcher AND the web/JSON API in a single process.

    Single Kuzu writer, swapped for a fresh read-only handle after each
    reindex (writer-visibility quirk: a Connection that just performed
    writes won't see them on subsequent fetch_all queries until the
    Database is reopened). The API and the UI receive consistent reads
    via `DBHolder` + an SSE `reindex_done` event after each swap.
    """
    import uvicorn

    from docgraph.embed import Embedder, GPU_PROVIDERS
    from docgraph.retrieve import Retriever
    from docgraph.server import broadcast, make_app

    # Bootstrap: writer DB for the baseline reindex. Closed afterward and
    # replaced with a read-only handle that the API will share.
    writer_db = GraphDB(cfg.db_path, embedding_dim=384)
    writer_db.init_schema()
    embedder = Embedder(
        cfg.embedding_model,
        providers=list(GPU_PROVIDERS) if cfg.gpu else None,
    )
    indexer = Indexer(cfg, writer_db, embedder=embedder)

    _console.rule(f"[bold cyan]Baseline index[/] — {cfg.repo_root}")
    indexer.index_all(incremental=True)
    writer_db.close()

    ro_db = GraphDB(cfg.db_path, embedding_dim=384, read_only=True)
    app = make_app(cfg, db=ro_db)
    app.state.loop = asyncio.get_running_loop()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Give uvicorn a tick to bind the port before we print the URL.
    await asyncio.sleep(0.1)
    _console.print(
        f"[green]Serving[/] http://{host}:{port}/  "
        f"[dim](live UI auto-redraws on reindex via /api/events)[/]"
    )
    roots = [str(r) for r, _ in cfg.roots_with_prefix()]
    _console.print(
        f"[green]Watching[/] {len(roots)} root(s), debounce={debounce_ms}ms. "
        "[dim]Ctrl-C to stop.[/]"
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
                f"[bold cyan]Reindex[/] - {len(relevant)} fs event(s)",
                style="cyan",
            )
            try:
                # 1. Swap API to a writer DB so the read-only handle is
                #    closed first (Kuzu file lock disallows two modes).
                holder = app.state.db_holder
                old_ro = holder.db
                with holder.lock:
                    try:
                        old_ro.close()
                    except Exception:
                        pass
                    writer = GraphDB(cfg.db_path, embedding_dim=384)
                    writer.init_schema()
                    holder.db = writer
                    holder.retriever = Retriever(writer, embedder, cfg=cfg)

                # 2. Reindex (sync, on a thread so SSE keepalives still flow)
                indexer = Indexer(cfg, writer, embedder=embedder)
                await asyncio.to_thread(indexer.index_all, True)

                # 3. Close writer, reopen read-only so the freshly-written
                #    rows become visible.
                with holder.lock:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    new_ro = GraphDB(cfg.db_path, embedding_dim=384, read_only=True)
                    holder.db = new_ro
                    holder.retriever = Retriever(new_ro, embedder, cfg=cfg)

                # 4. Tell the UI to refresh
                broadcast(app, "reindex_done", {"ts": time.time(), "events": len(relevant)})
            except Exception as e:  # noqa: BLE001
                _console.print(f"[red]Reindex failed:[/] {e}")
                continue
    except (KeyboardInterrupt, asyncio.CancelledError):
        _console.print("[yellow]watcher stopped[/]")
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(server_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


class _WatchFilter:
    """watchfiles filter that drops irrelevant paths before they're emitted.

    watchfiles passes (Change, str) to its filter. Returning False skips the
    event entirely, which is what we want for ignore'd paths and unsupported
    languages — otherwise a git checkout in a vendored dir would generate
    thousands of events that all get debounced to a single reindex anyway.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def __call__(self, change: Change, path: str) -> bool:
        return _is_relevant(self.cfg, Path(path))
