"""Workspace — a registry of roots a single host process serves.

Each root is one indexed `.docgraph/graph.kuzu` directory. The workspace
holds a per-root `RootSlot` with a long-lived read-only Kuzu connection
and a Retriever wired to it. The watcher (when active for a root) gets
its own writer handle on demand and swaps the read-only slot atomically
after each reindex.

Lookup rules for the `root` argument that every tool / API route accepts:
    1. None / "" → default root (first registered).
    2. Exact match against any registered absolute root path.
    3. Match against root **slug** (last path segment, lowercased).
    4. Match against any root that is a path-prefix of the supplied value
       (so an agent passing a file path picks the right root automatically).
Raises `KeyError` if nothing matches and the value is non-empty.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from docgraph.cancel import CancelToken
from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder, GPU_PROVIDERS
from docgraph.locks import DBLock, LockTimeouts
from docgraph.retrieve import Retriever

log = logging.getLogger(__name__)


def slug_for_root(root: Path | str) -> str:
    """Stable short slug from a root path. Matches telecode's slug_for_path."""
    name = os.path.basename(os.path.normpath(str(root))) or "root"
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
    return safe.lower() or "root"


@dataclass
class RootSlot:
    cfg: Config
    db_ro: GraphDB
    retriever: Retriever
    slug: str
    watching: bool = False
    last_indexed_at: float | None = None
    # Writer is owned by an active watcher; None when no watcher is running
    # for this root. Kept here so the host can introspect it.
    db_writer: GraphDB | None = field(default=None, repr=False)
    # Per-root async lock manager. Writers queue here; readers wait for
    # idle. One per RootSlot today; with the group refactor it'll move
    # to DBHandle (one per shared db_path).
    lock: DBLock = field(default_factory=lambda: DBLock(name="root"), repr=False)


class Workspace:
    """Ordered registry of roots. Thread-safe lookup + RO-DB swap.

    The first registered root is the default. Per-root resources are
    opened upfront in `__init__` and closed in `close()`. An embedder
    pool is shared across roots that use the same embedding model
    (de-duped further inside `embed.py::_MODEL_CACHE`).
    """

    def __init__(self, configs: list[Config], lock_timeouts: LockTimeouts | None = None) -> None:
        if not configs:
            raise ValueError("Workspace needs at least one Config")
        self._lock = threading.Lock()
        self._slots: dict[Path, RootSlot] = {}
        self._order: list[Path] = []
        self._embedders: dict[tuple[str, bool], Embedder] = {}
        # One cooperative-cancel token per root, shared across whatever
        # long-op is currently running for that root (index / wiki /
        # docs-add — only one runs at a time per root because they all
        # take the writer lock).
        self._cancel_tokens: dict[Path, CancelToken] = {}
        # Lock timeouts (read gate / writer queue / wiki) — surfaced via
        # CLI flags + telecode settings. The host caches the running
        # event loop on startup so sync callers (the watcher thread) can
        # bridge into the async DBLock via run_coroutine_threadsafe.
        self.lock_timeouts = lock_timeouts or LockTimeouts()
        self._loop: asyncio.AbstractEventLoop | None = None
        for cfg in configs:
            self._add_locked(cfg)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once on host startup. Sync callers (watcher thread)
        bridge their take_writer into this loop's DBLock.acquire_write
        via run_coroutine_threadsafe. Async callers ignore it."""
        self._loop = loop

    # ── construction helpers ────────────────────────────────────────────
    def embedder_for(self, cfg: Config) -> Embedder:
        """Return a workspace-pooled `Embedder` for this Config. Multiple
        roots with the same `(embedding_model, gpu)` share one ONNX
        session — important under GPU because DirectML / CUDA don't take
        kindly to two sessions racing for the same device."""
        key = (cfg.embedding_model, cfg.gpu)
        emb = self._embedders.get(key)
        if emb is None:
            from docgraph.embed import resolve_providers
            emb = Embedder(
                cfg.embedding_model,
                providers=resolve_providers(cfg.gpu),
            )
            self._embedders[key] = emb
        return emb

    # Internal alias for back-compat — older call sites can still use this.
    _embedder_for = embedder_for

    def _add_locked(self, cfg: Config) -> RootSlot:
        root = cfg.repo_root.resolve()
        if root in self._slots:
            return self._slots[root]
        if not cfg.db_path.exists():
            # Fresh root — initialize an empty graph DB so the host can
            # serve queries (returning empty results) until the user
            # triggers an index. Writer is closed before opening RO so
            # Kuzu releases the file lock on Windows.
            log.info("Initializing empty graph DB for unindexed root %s", root)
            cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
            db_w = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
            db_w.init_schema()
            db_w.close()
        db_ro = GraphDB(cfg.db_path, read_only=True)
        embedder = self._embedder_for(cfg)
        retriever = Retriever(db_ro, embedder, cfg=cfg)
        slug = slug_for_root(root)
        slot = RootSlot(
            cfg=cfg, db_ro=db_ro, retriever=retriever,
            slug=slug,
            lock=DBLock(name=slug),
        )
        self._slots[root] = slot
        self._order.append(root)
        self._cancel_tokens[root] = CancelToken()
        return slot

    # ── lookup ──────────────────────────────────────────────────────────
    def default(self) -> RootSlot:
        with self._lock:
            return self._slots[self._order[0]]

    def resolve(self, root: str | Path | None) -> RootSlot:
        """Resolve a root by path / slug / file-prefix / default. Raises
        `KeyError` if a non-empty argument matches nothing."""
        with self._lock:
            if not root:
                return self._slots[self._order[0]]
            # 1) exact path
            try:
                p = Path(str(root)).resolve()
                if p in self._slots:
                    return self._slots[p]
            except (OSError, ValueError):
                p = None
            # 2) slug
            s = str(root).lower().strip()
            for r, slot in self._slots.items():
                if slot.slug == s:
                    return slot
            # 3) path-prefix (file inside a registered root)
            if p is not None:
                for r, slot in self._slots.items():
                    try:
                        p.relative_to(r)
                        return slot
                    except ValueError:
                        continue
            raise KeyError(f"No registered root matches {root!r}")

    def slugs(self) -> list[str]:
        """Ordered list of root slugs. First entry is the default."""
        with self._lock:
            return [self._slots[r].slug for r in self._order]

    def default_slug(self) -> str:
        with self._lock:
            return self._slots[self._order[0]].slug

    def list(self) -> list[dict]:
        """Snapshot of registered roots; safe to serialize to JSON."""
        with self._lock:
            out = []
            for i, r in enumerate(self._order):
                slot = self._slots[r]
                out.append({
                    "slug": slot.slug,
                    "path": str(r),
                    "default": i == 0,
                    "watching": slot.watching,
                    "last_indexed_at": slot.last_indexed_at,
                })
            return out

    def roots(self) -> list[Path]:
        with self._lock:
            return list(self._order)

    # ── watcher integration ─────────────────────────────────────────────
    async def take_writer_async(self, root: str | Path, label: str = "api",
                                 timeout: float | None = None) -> GraphDB:
        """Async writer acquisition. Queues behind active writer + drains
        readers. `label` flows into LockStatus so /api/locks shows who's
        holding (e.g. 'api:index', 'api:wiki', 'watch')."""
        slot = self.resolve(root)
        if timeout is None:
            timeout = self.lock_timeouts.for_label(label)
        await slot.lock.acquire_write(label, timeout=timeout)
        try:
            with self._lock:
                # Closing RO is required so Kuzu releases the file lock
                # for the writer instance we're about to open.
                slot.db_ro.close()
                slot.db_writer = GraphDB(
                    slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim,
                )
                return slot.db_writer
        except Exception:
            await slot.lock.release_write()
            raise

    async def release_writer_async(self, root: str | Path) -> None:
        slot = self.resolve(root)
        with self._lock:
            if slot.db_writer is not None:
                try:
                    slot.db_writer.close()
                except Exception:
                    log.exception("failed closing writer for %s", slot.cfg.repo_root)
                slot.db_writer = None
            slot.db_ro = GraphDB(slot.cfg.db_path, read_only=True)
            embedder = self._embedder_for(slot.cfg)
            slot.retriever = Retriever(slot.db_ro, embedder, cfg=slot.cfg)
        await slot.lock.release_write()

    def take_writer(self, root: str | Path, label: str = "api",
                    timeout: float | None = None) -> GraphDB:
        """Sync writer acquisition. The watcher thread calls this from
        outside the event loop; we bridge into DBLock.acquire_write via
        run_coroutine_threadsafe so the queue stays consistent across
        async + sync callers. If no loop is attached (e.g. CLI subprocess
        with no host) we fall back to the legacy "raise if held" path."""
        slot = self.resolve(root)
        loop = self._loop
        if loop is not None and loop.is_running():
            if timeout is None:
                timeout = self.lock_timeouts.for_label(label)
            fut = asyncio.run_coroutine_threadsafe(
                slot.lock.acquire_write(label, timeout=timeout), loop,
            )
            # Pad the wait by 1s so BusyTimeout from inside the coroutine
            # surfaces before our own threadsafe-future timeout.
            wait = (timeout if timeout != float("inf") else None)
            wait = (wait + 1.0) if wait is not None else None
            fut.result(timeout=wait)
            with self._lock:
                slot.db_ro.close()
                slot.db_writer = GraphDB(
                    slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim,
                )
                return slot.db_writer
        # No event loop attached — single-shot CLI usage. Keep prior
        # contract: refuse if already held, otherwise grant immediately.
        with self._lock:
            if slot.db_writer is not None:
                raise RuntimeError(f"writer already taken for {slot.cfg.repo_root}")
            slot.db_ro.close()
            slot.db_writer = GraphDB(slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim)
            return slot.db_writer

    def release_writer(self, root: str | Path) -> None:
        """Sync release. Mirrors take_writer — bridges into the async
        lock when a loop is attached, otherwise legacy behaviour."""
        slot = self.resolve(root)
        loop = self._loop
        with self._lock:
            if slot.db_writer is not None:
                try:
                    slot.db_writer.close()
                except Exception:
                    log.exception("failed closing writer for %s", slot.cfg.repo_root)
                slot.db_writer = None
            slot.db_ro = GraphDB(slot.cfg.db_path, read_only=True)
            embedder = self._embedder_for(slot.cfg)
            slot.retriever = Retriever(slot.db_ro, embedder, cfg=slot.cfg)
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(slot.lock.release_write(), loop)

    def clear_data(self, root: str | Path) -> None:
        """Wipe a root's `.docgraph/` (graph DB, cache, wiki, state) and
        reopen a fresh empty schema. Closes the slot's RO handle, removes
        the directory, recreates an empty DB, and reopens RO. The CLI's
        `docgraph clear` does the same thing — just with the host alive."""
        import gc
        import shutil
        slot = self.resolve(root)
        with self._lock:
            if slot.db_writer is not None:
                raise RuntimeError(f"writer is currently held for {slot.cfg.repo_root}")
            try:
                slot.db_ro.close()
            except Exception:
                log.exception("failed closing RO before clear for %s", slot.cfg.repo_root)
            # Kuzu's COPY-FROM internals can hold extra refs that survive
            # close on Windows; force a GC pass before rmtree to release.
            gc.collect()
            data_dir = slot.cfg.data_dir
            if data_dir.exists():
                shutil.rmtree(data_dir, ignore_errors=False)
            data_dir.mkdir(parents=True, exist_ok=True)
            tmp = GraphDB(slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim)
            try:
                tmp.init_schema()
            finally:
                tmp.close()
            slot.db_ro = GraphDB(slot.cfg.db_path, read_only=True)
            embedder = self._embedder_for(slot.cfg)
            slot.retriever = Retriever(slot.db_ro, embedder, cfg=slot.cfg)

    def mark_watching(self, root: str | Path, watching: bool) -> None:
        slot = self.resolve(root)
        with self._lock:
            slot.watching = bool(watching)

    def mark_indexed(self, root: str | Path, ts: float) -> None:
        slot = self.resolve(root)
        with self._lock:
            slot.last_indexed_at = float(ts)

    # ── cancellation ────────────────────────────────────────────────────
    def cancel_token_for(self, root: str | Path) -> CancelToken:
        """Per-root cooperative-cancel token. The long op polls
        `token.raise_if_set()` at safe checkpoints; another request
        flips it via `request_cancel()`."""
        slot = self.resolve(root)
        with self._lock:
            return self._cancel_tokens[slot.cfg.repo_root.resolve()]

    def request_cancel(self, root: str | Path) -> None:
        """Set the cancel flag for `root`'s currently running long op.
        Idempotent — flipping a no-op token is harmless."""
        self.cancel_token_for(root).request()

    def reset_cancel(self, root: str | Path) -> None:
        """Clear the cancel flag. Call before kicking off a new long op
        so a stale prior cancel doesn't immediately abort the new run."""
        self.cancel_token_for(root).reset()

    # ── lifecycle ───────────────────────────────────────────────────────
    def close(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                try:
                    if slot.db_writer is not None:
                        slot.db_writer.close()
                except Exception:
                    pass
                try:
                    slot.db_ro.close()
                except Exception:
                    pass
            self._slots.clear()
            self._order.clear()
            self._embedders.clear()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
