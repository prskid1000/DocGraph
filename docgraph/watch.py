"""File watcher: triggers incremental reindex on changes.

Uses watchfiles (Rust notify under the hood). Filters out paths that hit
.gitignore / .docgraphignore / unsupported languages before debouncing,
so a `git checkout` of 5k files in node_modules doesn't blow up.

Holds a single writer connection for the watcher's lifetime — that means
`docgraph serve` / `docgraph mcp` cannot run against the same DB at the
same time (they'd take a read lock that blocks our writer).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from watchfiles import Change, watch

from docgraph.config import Config, MAX_FILE_BYTES
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.parse import detect_language

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
    embedder = Embedder(cfg.embedding_model)
    indexer = Indexer(cfg, db, embedder=embedder)

    log.info(f"Initial incremental index of {cfg.repo_root}")
    stats = indexer.index_all(incremental=True)
    log.info(
        f"  baseline: {stats['changed']} changed, {stats['deleted']} deleted, "
        f"{stats['elapsed']:.2f}s"
    )

    roots = [str(r) for r, _ in cfg.roots_with_prefix()]
    log.info(f"Watching {roots} (debounce={debounce_ms}ms). Ctrl-C to stop.")
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
            t0 = time.perf_counter()
            try:
                stats = indexer.index_all(incremental=True)
            except Exception as e:  # noqa: BLE001
                log.error(f"Reindex failed: {e}")
                continue
            log.info(
                f"reindex: {len(relevant)} fs events → "
                f"{stats['changed']} changed, {stats['deleted']} deleted "
                f"in {time.perf_counter() - t0:.2f}s"
            )
    except KeyboardInterrupt:
        log.info("watcher stopped")


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
