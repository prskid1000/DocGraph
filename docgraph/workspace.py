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
import time
from dataclasses import dataclass, field
from pathlib import Path

from docgraph import procmem as _procmem
from docgraph.cancel import CancelToken
from docgraph.config import Config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.locks import DBLock, LockTimeouts
from docgraph.rerank import Reranker
from docgraph.retrieve import Retriever

log = logging.getLogger(__name__)


def _host_live(cfg):
    """The host's per-root live state: it also keeps a parse pool warm."""
    from docgraph.live import LiveIndex
    live = LiveIndex(cfg)
    live.keep_pool = True
    return live


_RETRIEVER_CACHES = ("_mem_graph_cache", "_node_table_cache", "_trace_graph_cache",
                     "_explore_adj_cache", "_symbol_pr_cache", "_processes_memo",
                     "_graph_dump_memo", "_clusters_memo", "_pr_sorted", "_local_vec_cache")


def _drop_retriever_caches(r) -> None:
    d = getattr(r, "__dict__", {})
    for name in _RETRIEVER_CACHES:
        d.pop(name, None)
    if hasattr(r, "_ranker"):
        r._ranker = None
    if hasattr(r, "_bm25"):
        r._bm25 = {}


def _lower_thread_priority() -> None:
    """Background maintenance yields the CPU to request handling."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetThreadPriority(k32.GetCurrentThread(), -2)   # THREAD_PRIORITY_LOWEST
    except Exception:
        try:
            os.nice(5)
        except Exception:
            pass


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
    # Long-lived incremental state (live.LiveIndex): cache shards, symbol
    # table, import / ref indexes, keyword index, tile arrays. An index pass
    # patches it; nothing is re-read per pass.
    live: object = field(default=None, repr=False)
    # Tile sidecar served from memory (tiles.TileStore); the indexer installs
    # each new generation directly, persistence runs behind it.
    tile_store: object = field(default=None, repr=False)
    # Background maintenance bookkeeping (see Workspace._maintenance).
    maint: dict = field(default_factory=dict, repr=False)


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
        self._embedders: dict[tuple[str, bool, bool], Embedder] = {}
        # Cross-encoder rerankers, pooled the same way as embedders so
        # multiple roots with identical (model, gpu, torch_compile) share
        # one torch session. Tracked centrally so the idle unloader can
        # iterate them.
        self._rerankers: dict[tuple[str, bool, bool], Reranker] = {}
        # One cooperative-cancel token per root, shared across whatever
        # long-op is currently running for that root (index / wiki).
        self._cancel_tokens: dict[Path, CancelToken] = {}
        # Per-class idle-unload windows (seconds). 0 = disabled. Set from
        # CLI via `host --embed-idle-unload-sec` / `--rerank-idle-unload-sec`;
        # the background task is started by `start_idle_unloader_async`
        # when the lifespan boots.
        self.embed_unload_after: float = 0.0
        self.rerank_unload_after: float = 0.0
        # host --graph-idle-unload-sec (see unload_graph)
        self.graph_unload_after: float = float(getattr(configs[0], "graph_unload_after", 0.0) or 0.0)
        self._idle_task: asyncio.Task | None = None
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
        roots with the same `(embedding_model, gpu, torch_compile)` share
        one torch session — important under GPU because two CUDA contexts
        racing for the same device is the fast lane to OOM."""
        key = (cfg.embedding_model, cfg.gpu, cfg.embed_torch_compile)
        emb = self._embedders.get(key)
        if emb is None:
            # "cuda" is only a request here: importing torch to check it
            # would cost ~1.2 s of host startup; the Embedder downgrades to
            # CPU on its first (background warm-up) load when CUDA is absent.
            emb = Embedder(
                cfg.embedding_model,
                device="cuda" if cfg.gpu else None,
                torch_compile=cfg.embed_torch_compile,
            )
            self._embedders[key] = emb
        return emb

    # Internal alias for back-compat — older call sites can still use this.
    _embedder_for = embedder_for

    def reranker_for(self, cfg: Config) -> Reranker:
        """Workspace-pooled cross-encoder. Multiple roots with identical
        `(rerank_model, rerank_gpu, rerank_torch_compile)` share one
        session — same rationale as `embedder_for`. Lazy-created on first
        call. Held centrally so the idle unloader can evict them as a
        group."""
        from docgraph.embed import resolve_device
        model = cfg.rerank_model or ""
        key = (model, bool(cfg.rerank_gpu), bool(cfg.rerank_torch_compile))
        r = self._rerankers.get(key)
        if r is None:
            r = Reranker(
                model_name=(model or None),
                device=resolve_device(cfg.rerank_gpu),
                torch_compile=cfg.rerank_torch_compile,
            )
            self._rerankers[key] = r
        return r

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
        try:
            db_ro = GraphDB(cfg.db_path, read_only=True)
        except RuntimeError as exc:
            # A host killed mid-write leaves uncommitted shadow pages that
            # Kuzu can only replay in read-write mode. Recover by opening RW
            # once (replays + checkpoints), closing, then reopening RO.
            if "shadow pages" not in str(exc).lower():
                raise
            log.warning("Replaying shadow pages for %s (recovering from unclean shutdown)", root)
            db_rw = GraphDB(cfg.db_path, embedding_dim=cfg.embedding_dim)
            db_rw.close()
            db_ro = GraphDB(cfg.db_path, read_only=True)
        embedder = self._embedder_for(cfg)
        retriever = Retriever(db_ro, embedder, cfg=cfg, workspace=self)
        slug = slug_for_root(root)
        state_path = cfg.data_dir / "state.json"
        try:
            import json as _json
            _state = _json.loads(state_path.read_text()) if state_path.exists() else {}
            _last_indexed = _state.get("last_indexed_at") or None
            if _last_indexed is not None:
                _last_indexed = float(_last_indexed)
        except Exception:
            _last_indexed = None
        from docgraph.tiles import TileStore
        slot = RootSlot(
            cfg=cfg, db_ro=db_ro, retriever=retriever,
            slug=slug,
            last_indexed_at=_last_indexed,
            lock=DBLock(name=slug),
            live=_host_live(cfg),
            tile_store=TileStore(cfg.data_dir / "tiles"),
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
                return self._open_writer(slot)
        except Exception:
            await slot.lock.release_write()
            raise

    async def release_writer_async(self, root: str | Path) -> None:
        slot = self.resolve(root)
        try:
            with self._lock:
                self._close_writer(slot, keep_warm=True)
        finally:
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
                return self._open_writer(slot)
        # No event loop attached — single-shot CLI usage. Keep prior
        # contract: refuse if already held, otherwise grant immediately.
        with self._lock:
            if slot.maint.get("writer_taken"):
                raise RuntimeError(f"writer already taken for {slot.cfg.repo_root}")
            return self._open_writer(slot)

    def release_writer(self, root: str | Path) -> None:
        """Sync release. Mirrors take_writer — bridges into the async
        lock when a loop is attached, otherwise legacy behaviour."""
        slot = self.resolve(root)
        loop = self._loop
        with self._lock:
            try:
                self._close_writer(slot, keep_warm=loop is not None and loop.is_running())
            finally:
                # Always hand the writer lock back, even if the reopen failed —
                # otherwise every later writer waits out its timeout forever.
                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(slot.lock.release_write(), loop)

    # The read-write handle of the last pass keeps serving reads (it sees
    # every committed write; per-thread connections run reads concurrently)
    # for this long, so a burst of watcher passes reuses one warm buffer pool
    # instead of paying close / reopen / cold pages each time. After that it
    # is swapped back to a read-only handle, which lets other processes (the
    # CLI's `docgraph stats`, a second reader) open the database again.
    RW_KEEP_SEC = 120.0

    def _open_writer(self, slot) -> GraphDB:
        """Under self._lock with the root's write lock held."""
        slot.maint["writer_taken"] = True
        slot.maint["rw_gen"] = int(slot.maint.get("rw_gen", 0)) + 1
        if slot.db_writer is not None:          # warm handle from the last pass
            return slot.db_writer
        try:
            slot.db_ro.close()
            slot.db_writer = GraphDB(slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim)
        except Exception:
            slot.maint["writer_taken"] = False
            raise
        return slot.db_writer

    def _close_writer(self, slot, keep_warm: bool) -> None:
        """Under self._lock: end a write session -- keep the handle serving
        reads (keep_warm) or close it and reopen read-only."""
        slot.maint["writer_taken"] = False
        if getattr(slot.db_writer, "bulk", False):
            keep_warm = False          # never keep the bulk-sized buffer pool around
        if keep_warm and self.RW_KEEP_SEC > 0 and slot.db_writer is not None \
                and slot.db_writer.conn is not None:
            slot.db_ro = slot.db_writer
            self._new_retriever(slot)
            gen = slot.maint.get("rw_gen")
            t = threading.Timer(self.RW_KEEP_SEC, self._cooldown, args=(slot, gen))
            t.daemon = True
            t.start()
            return
        if slot.db_writer is not None:
            try:
                slot.db_writer.close()
            except Exception:
                log.exception("failed closing writer for %s", slot.cfg.repo_root)
            slot.db_writer = None
        slot.db_ro = self._reopen_ro(slot)
        self._new_retriever(slot)

    def _cooldown(self, slot, gen) -> None:
        """Swap an idle warm read-write handle back to read-only."""
        if slot.maint.get("rw_gen") != gen or slot.db_writer is None:
            return
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                slot.lock.acquire_write("rw-cooldown", timeout=30.0), loop)
            fut.result(timeout=31.0)
        except Exception:
            return
        try:
            with self._lock:
                if slot.maint.get("rw_gen") == gen and slot.db_writer is not None \
                        and not slot.maint.get("writer_taken"):
                    self._close_writer(slot, keep_warm=False)
        except Exception:
            log.warning("rw cooldown failed for %s", slot.slug, exc_info=True)
        finally:
            asyncio.run_coroutine_threadsafe(slot.lock.release_write(), loop)

    def _new_retriever(self, slot) -> None:
        """Fresh Retriever on the reopened RO handle. It shares the live
        keyword index (kept current in-process by the indexer) and inherits
        the precomputed health report."""
        old = slot.retriever
        r = Retriever(slot.db_ro, self._embedder_for(slot.cfg), cfg=slot.cfg, workspace=self)
        live = slot.live
        if live is not None and getattr(live, "kw", None) is not None and live.kw.loaded:
            r.kw = live.kw
        r.pending_vectors = lambda label, i, _s=slot: self._pending_vector(_s, label, i)
        # the health report of the previous generation is served (marked
        # "refreshing") until the background recompute lands
        prev = getattr(old, "_health_cache", None) or getattr(old, "_health_stale", None)
        if prev:
            r._health_stale = dict(prev)
        # memoised per-generation results the old retriever had built are
        # rebuilt by maintenance for the new one (first call stays fast)
        warm = slot.maint.setdefault("warm", set())
        for k in (getattr(old, "_processes_memo", None) or {}):
            warm.add(("processes",) + tuple(k))
        for k in (getattr(old, "_graph_dump_memo", None) or {}):
            warm.add(("graph_dump", int(k)))
        if getattr(old, "_mem_graph_cache", None) is not None:
            warm.add(("mem_graph",))
        if getattr(old, "_node_table_cache", None) is not None:
            warm.add(("node_table",))
        if getattr(old, "_trace_graph_cache", None) is not None:
            warm.add(("trace_graph",))
        for rel in (getattr(old, "_explore_adj_cache", None) or {}):
            warm.add(("explore_adj", rel))
        while len(warm) > 24:
            warm.pop()
        slot.retriever = r

    @staticmethod
    def _reopen_ro(slot) -> GraphDB:
        """Open the read-only handle after a write session. A handle that is
        still being torn down can hold the Windows file lock briefly, so
        collect garbage and retry a few times before giving up."""
        import gc
        last: Exception | None = None
        for attempt in range(5):
            try:
                return GraphDB(slot.cfg.db_path, read_only=True)
            except RuntimeError as exc:
                if "lock" not in str(exc).lower():
                    raise
                last = exc
                gc.collect()
                time.sleep(0.2 * (attempt + 1))
        raise last  # type: ignore[misc]

    def clear_data(self, root: str | Path) -> None:
        """Wipe a root's `.docgraph/` (graph DB, cache, wiki, state) and
        reopen a fresh empty schema. Closes the slot's RO handle, removes
        the directory, recreates an empty DB, and reopens RO. The CLI's
        `docgraph clear` does the same thing — just with the host alive."""
        import gc
        import shutil
        slot = self.resolve(root)
        with self._lock:
            if slot.maint.get("writer_taken"):
                raise RuntimeError(f"writer is currently held for {slot.cfg.repo_root}")
            if slot.db_writer is not None:          # warm handle serving reads
                try:
                    slot.db_writer.close()
                except Exception:
                    pass
                slot.db_writer = None
            try:
                slot.db_ro.close()
            except Exception:
                log.exception("failed closing RO before clear for %s", slot.cfg.repo_root)
            # Kuzu's COPY-FROM internals can hold extra refs that survive
            # close on Windows; force a GC pass before rmtree to release.
            gc.collect()
            data_dir = slot.cfg.data_dir
            # Preserve user configuration that survives a clear.
            _PRESERVE = ("repos.json", "links.json")
            saved: dict[str, bytes] = {}
            for name in _PRESERVE:
                p = data_dir / name
                if p.exists():
                    try:
                        saved[name] = p.read_bytes()
                    except Exception:
                        pass
            if data_dir.exists():
                shutil.rmtree(data_dir, ignore_errors=False)
            data_dir.mkdir(parents=True, exist_ok=True)
            for name, content in saved.items():
                try:
                    (data_dir / name).write_bytes(content)
                except Exception:
                    pass
            tmp = GraphDB(slot.cfg.db_path, embedding_dim=slot.cfg.embedding_dim)
            try:
                tmp.init_schema()
            finally:
                tmp.close()
            slot.db_ro = GraphDB(slot.cfg.db_path, read_only=True)
            from docgraph.tiles import TileStore
            if slot.live is not None:
                slot.live.close_pool()
            slot.live = _host_live(slot.cfg)
            slot.tile_store = TileStore(slot.cfg.data_dir / "tiles")
            embedder = self._embedder_for(slot.cfg)
            slot.retriever = Retriever(slot.db_ro, embedder, cfg=slot.cfg, workspace=self)

    # -- index passes + background maintenance ---------------------------
    def _tile_sink(self, slot):
        """Indexer hook: serve a new tile generation from memory at once and
        persist it on the (single) persist thread; a generation superseded
        before its turn is skipped."""
        from docgraph import tiles as _tiles

        def sink(man: dict, arrays: dict) -> None:
            gen = int(man.get("generation", 0))

            def job() -> None:
                # through the same FIFO as the patches: a rebuild supersedes
                # every patch queued before it
                slot.live.tiles = (man, arrays)
                slot.tile_store.install(man, arrays)
                slot.maint["tiles_gen"] = gen
                slot.maint.pop("tiles_stale", None)
                cb = self.on_tiles_ready
                if cb is not None:
                    try:
                        cb(slot.slug, gen)
                    except Exception:
                        pass
                try:
                    _tiles.write(slot.cfg.data_dir / "tiles", arrays, man)
                    self._remap_tiles(slot, gen)
                except Exception:
                    log.warning("tile persist failed for %s", slot.slug, exc_info=True)
            self._persist_pool().submit(job)
        return sink

    def _remap_tiles(self, slot, gen: int) -> None:
        """A generation just persisted: serve it from the memory-mapped file
        instead of the in-RAM arrays the pass built (same bytes)."""
        from docgraph import tiles as _tiles
        try:
            loaded = _tiles.load(slot.cfg.data_dir / "tiles")
        except Exception:
            return
        if loaded is None or int(loaded[0].get("generation", -1)) != int(gen):
            return
        live = slot.live
        cur = live.tiles
        if cur is not None and int(cur[0].get("generation", -1)) != int(gen):
            return                                   # a newer one is in memory
        live.tiles = loaded
        slot.tile_store.install(*loaded)

    def _tile_patcher(self, slot):
        """Indexer hook: apply an incremental tile patch on the persist
        thread (FIFO, so patches and rebuilds land in pass order), serve it
        from memory, then persist. A failed patch marks the sidecar stale;
        maintenance rebuilds it from the graph."""
        from docgraph import tiles as _tiles

        def submit(delta, gen: int) -> None:
            slot.maint["tiles_pending"] = int(slot.maint.get("tiles_pending", 0)) + 1

            def job() -> None:
                live = slot.live
                if slot.maint.get("tiles_stale"):
                    # a rebuild from the graph is queued: it covers this pass
                    slot.maint["tiles_pending"] = max(0, int(slot.maint.get("tiles_pending", 1)) - 1)
                    return
                try:
                    if live.tiles is None:
                        # share the arrays the store already serves (one copy)
                        st = slot.tile_store
                        if st.manifest is not None and st.a:
                            live.tiles = (st.manifest, st.a)
                        else:
                            live.tiles = _tiles.load(slot.cfg.data_dir / "tiles")
                    if live.tiles is None or "sym_file" not in live.tiles[1]:
                        raise RuntimeError("no patchable tile sidecar")
                    man, arrays = live.tiles
                    if int(man.get("generation", 0)) >= gen:
                        return
                    new_arrays, new_man = _tiles.patch(arrays, man, delta, gen)
                    live.tiles = (new_man, new_arrays)
                    slot.tile_store.install(new_man, new_arrays)
                    slot.maint["tiles_gen"] = gen
                    cb = self.on_tiles_ready
                    if cb is not None:
                        try:
                            cb(slot.slug, gen)
                        except Exception:
                            pass
                    if int(slot.maint.get("tiles_pending", 1)) <= 1:
                        _tiles.write(slot.cfg.data_dir / "tiles", new_arrays, new_man)
                        self._remap_tiles(slot, gen)
                except Exception as exc:
                    log.warning("tile patch for %s failed (%s): full rebuild scheduled", slot.slug, exc)
                    slot.maint["tiles_stale"] = True
                    self.schedule_maintenance(slot.cfg.repo_root)
                finally:
                    slot.maint["tiles_pending"] = max(0, int(slot.maint.get("tiles_pending", 1)) - 1)
            self._persist_pool().submit(job)
        return submit

    # set by the server: on_tiles_ready(slug, generation) -> SSE "tiles_ready"
    on_tiles_ready = None

    def _prime_live(self, slot) -> None:
        from docgraph.config import MAX_FILE_BYTES
        from docgraph.live import scan as live_scan
        live = slot.live
        try:
            import json as _json
            st = _json.loads((slot.cfg.data_dir / "state.json").read_text())
        except Exception:
            return
        from docgraph.db import SCHEMA_VERSION
        if st.get("schema_version") != SCHEMA_VERSION:
            return                                  # the next pass rebuilds anyway
        with live.lock:
            if slot.maint.get("writer_taken"):
                return
            live.ensure_cache()
            if not live.cache:
                return
            live.ensure_symtab(slot.db_ro)
            live.ensure_resolve()
            live.ensure_kw()
            if not live.scan_ready:
                live_scan(slot.cfg, live.scan_state, MAX_FILE_BYTES)
                live.scan_ready = True
                live.last_walk = time.time()

    def start_warmup(self) -> None:
        """Background warm-up after host start: load the embedding model
        (from the local cache, no network round trips), the keyword index
        and the tile arrays, and touch every vector index once. Never
        blocks startup; failures only log."""
        if getattr(self, "_warm_started", False):
            return
        self._warm_started = True

        def run() -> None:
            _lower_thread_priority()
            for root in self.roots():
                try:
                    slot = self.resolve(root)
                    t0 = time.perf_counter()
                    emb = self.embedder_for(slot.cfg)
                    qv = emb.embed_query("warm up")
                    # prime the live index state (cache, symbol table, import
                    # and reference indexes, scan decisions) so the first
                    # watcher / API pass is already proportional
                    self._prime_live(slot)
                    r = slot.retriever
                    kw = r._kw_index() if hasattr(r, "_kw_index") else None
                    if kw is not None and slot.live is not None and slot.live.kw is None:
                        slot.live.kw = kw
                    db = slot.db_ro
                    for label in ("Function", "Class", "Chunk"):
                        try:
                            db.vector_topk(label, qv, 5)
                        except Exception:
                            pass
                    try:
                        slot.tile_store.ready()          # memory-mapped, cheap
                    except Exception:
                        pass
                    # The in-memory call graph / node table are NOT built
                    # here: the first tool that needs them builds them
                    # (~0.3 s on 200k functions) and idle unload drops them.
                    log.info("warm-up of %s done in %.1fs", slot.slug, time.perf_counter() - t0)
                except Exception as exc:  # noqa: BLE001
                    log.info("warm-up skipped for %s: %s", root, exc)
            # Loading torch + the first CUDA encode maps ~1.4 GB of DLL and
            # kernel images that are touched once and never again: give the
            # working set back now.
            _procmem.trim("warm-up")
        threading.Thread(target=run, name="docgraph-warmup", daemon=True).start()
        self.start_graph_unloader()

    # -- idle memory ------------------------------------------------------
    GRAPH_CHECK_SEC = 15.0
    # a root with no tool call / pass for this long gets a working-set trim
    # (nothing is dropped; touched pages fault back cheaply)
    LIGHT_TRIM_IDLE_SEC = 60.0
    # a working set above this is trimmed by the idle thread even while the
    # host is busy, at most every WS_CAP_INTERVAL_SEC: pages still in use
    # fault back from the standby list (a slower call or two), the rest --
    # freed heap, stale Kuzu pages, torch images -- stays out
    WS_SOFT_CAP_MB = 1024
    WS_CAP_INTERVAL_SEC = 60.0

    def touch(self, slot) -> None:
        """A tool / API call used this root's retriever."""
        slot.maint["last_use"] = time.time()
        slot.maint["graph_loaded"] = True

    def _last_activity(self, slot) -> float:
        m = slot.maint
        return max(float(m.get("last_use", 0.0)), float(m.get("last_pass", 0.0)),
                   float(m.get("maint_end", 0.0)), float(m.get("started", 0.0)))

    def start_graph_unloader(self) -> None:
        """Background thread: light working-set trim after LIGHT_TRIM_IDLE_SEC
        of inactivity, and the full graph unload after `graph_unload_after`."""
        if getattr(self, "_graph_unloader", None) is not None:
            return
        now = time.time()
        for slot in self._slots.values():
            slot.maint.setdefault("started", now)

        def run() -> None:
            _lower_thread_priority()
            while not getattr(self, "_closing", False):
                time.sleep(self.GRAPH_CHECK_SEC)
                try:
                    self._idle_memory_tick()
                except Exception as exc:  # noqa: BLE001 - never kill the host
                    log.debug("idle memory tick failed: %s", exc)
        t = threading.Thread(target=run, name="docgraph-graph-unloader", daemon=True)
        self._graph_unloader = t
        t.start()

    def _idle_memory_tick(self) -> None:
        now = time.time()
        thr = float(getattr(self, "graph_unload_after", 0.0) or 0.0)
        trimmed = False
        for slot in list(self._slots.values()):
            m = slot.maint
            if m.get("running") or m.get("writer_taken"):
                continue
            idle = now - self._last_activity(slot)
            if thr > 0 and idle >= thr and m.get("graph_loaded", True):
                self.unload_graph(slot)
                trimmed = True
            elif idle >= self.LIGHT_TRIM_IDLE_SEC and m.get("trimmed_at", 0.0) < self._last_activity(slot):
                m["trimmed_at"] = now
                if not trimmed:
                    _procmem.trim("idle")
                    trimmed = True
        if not trimmed and _procmem.memory().get("rss_mb", 0) > self.WS_SOFT_CAP_MB:
            _procmem.trim("soft cap", min_interval=self.WS_CAP_INTERVAL_SEC)

    def unload_graph(self, slot) -> bool:
        """Drop the in-memory structures of an idle root -- retriever caches
        (call-graph CSRs, node table, trace / explore graphs, memos, the PPR
        graph), tile arrays, the incremental live state and the parse pool --
        reopen the read handle (releases Kuzu's buffer pool) and trim the
        working set. Everything is rebuilt lazily: the next tool call builds
        what it needs, the next index pass reloads the live state."""
        live = slot.live
        if live is not None and not live.lock.acquire(blocking=False):
            return False
        try:
            if slot.maint.get("running") or slot.maint.get("writer_taken"):
                return False
            if live is not None:
                kw = live.kw
                live.close_pool()
                live.reset()
                live.kw = kw                      # search keeps its keyword index
            try:
                slot.tile_store.unload()
            except Exception:
                pass
            self._release_read_handle(slot)
            slot.maint.pop("warm", None)          # nothing to prewarm after this
            slot.maint["graph_loaded"] = False
        finally:
            if live is not None:
                live.lock.release()
        _procmem.trim("graph unload")
        log.info("idle unload of %s done: %s", slot.slug, _procmem.memory())
        return True

    def _release_read_handle(self, slot) -> None:
        """Fresh read handle + retriever (drops every per-generation cache
        and Kuzu's buffer pool), under the root's write lock so no request
        sees a closed handle."""
        loop = self._loop
        if loop is None or not loop.is_running():
            _drop_retriever_caches(slot.retriever)
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                slot.lock.acquire_write("graph-unload", timeout=10.0), loop)
            fut.result(timeout=11.0)
        except Exception:
            _drop_retriever_caches(slot.retriever)
            return
        try:
            with self._lock:
                if slot.maint.get("writer_taken"):
                    _drop_retriever_caches(slot.retriever)
                    return
                if slot.db_writer is not None:
                    self._close_writer(slot, keep_warm=False)
                else:
                    old = slot.db_ro
                    try:
                        old.close()
                    except Exception:
                        pass
                    slot.db_ro = self._reopen_ro(slot)
                    self._new_retriever(slot)
                _drop_retriever_caches(slot.retriever)
        finally:
            asyncio.run_coroutine_threadsafe(slot.lock.release_write(), loop)

    def _persist_pool(self):
        pool = getattr(self, "_persist", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor
            pool = self._persist = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docgraph-persist")
        return pool

    def index_pass(self, root: str | Path, *, incremental: bool = True,
                   changed_paths: list | None = None, cancel_token=None, progress_cb=None,
                   fetch_links: bool = True, force_fetch: bool = False,
                   label: str = "api:index") -> dict:
        """One index pass on the root's writer with its long-lived live
        state. Returns the indexer's stats. Schedules background
        maintenance (deferred global analytics, bulk SIMILAR_TO, keyword
        compaction, the health report) after the writer is released."""
        from docgraph.index import Indexer
        slot = self.resolve(root)
        slot.maint["last_pass"] = time.time()
        slot.maint["graph_loaded"] = True
        writer = self.take_writer(root, label=label)
        indexer = None
        try:
            if not slot.maint.get("schema_ok"):
                writer.init_schema()
                slot.maint["schema_ok"] = True
            embedder = self.embedder_for(slot.cfg)
            with slot.live.lock:
                indexer = Indexer(slot.cfg, writer, embedder=embedder, live=slot.live,
                                  tile_sink=self._tile_sink(slot), defer_global=True,
                                  tile_patcher=self._tile_patcher(slot))
                indexer.defer_vectors = True
                stats = indexer.index_all(incremental=incremental, changed_paths=changed_paths,
                                          cancel_token=cancel_token, progress_cb=progress_cb,
                                          fetch_links=fetch_links, force_fetch=force_fetch)
            slot.maint["last_pass"] = time.time()
            if not incremental:
                # a full pass leaves hundreds of MB of freed Arrow batches /
                # parse results in the heaps: hand them back
                _procmem.trim("full index")
            return stats
        except BaseException:
            # a failed / cancelled pass may leave the in-memory state ahead
            # of the DB: rebuild it from disk on the next pass
            try:
                slot.live.reset()
            except Exception:
                pass
            raise
        finally:
            if indexer is not None and indexer.db is not writer:
                # a full reindex wiped and reopened the database: its new
                # handle becomes the root's (warm) writer
                with self._lock:
                    slot.db_writer = indexer.db
            if indexer is not None:
                try:
                    indexer.db.vector_sink = None
                    writer.vector_sink = None
                except Exception:
                    pass
                self._queue_vectors(slot, getattr(indexer, "pending_vectors", None) or {})
            self.release_writer(root)
            self.schedule_maintenance(root)

    # -- deferred entity vectors --------------------------------------------
    def _pending_vector(self, slot, label: str, i: int):
        idx = slot.maint.get("vec_index")
        q = slot.maint.get("vec_queue") or []
        if idx is None or idx[0] != len(q) or idx[1] is not (q[-1] if q else None):
            m: dict = {}
            for lab, ids, mat, _p in q:
                for j, v in enumerate(ids.tolist()):
                    m[(lab, int(v))] = mat[j]
            idx = (len(q), q[-1] if q else None, m)
            slot.maint["vec_index"] = idx
        return idx[2].get((label, int(i)))

    # ~5 ms per HNSW insertion: a slice holds the writer ~0.25 s, the most
    # a watcher pass arriving meanwhile waits
    VEC_BATCH = 48

    def _vec_dir(self, slot) -> Path:
        return slot.cfg.data_dir / "pending_vectors"

    def _queue_vectors(self, slot, pending: dict) -> None:
        """Keep the vectors an index pass deferred (in memory + on disk, so
        a host restart before maintenance does not lose them)."""
        import numpy as np
        if not pending:
            return
        q = slot.maint.setdefault("vec_queue", [])
        d = self._vec_dir(slot)
        d.mkdir(parents=True, exist_ok=True)
        for label, rows in pending.items():
            if not rows:
                continue
            ids = np.array([int(r["id"]) for r in rows], dtype=np.int64)
            mat = np.array([np.asarray(r["embedding"], dtype=np.float32) for r in rows], dtype=np.float32)
            seq = int(slot.maint.get("vec_seq", 0)) + 1
            slot.maint["vec_seq"] = seq
            path = d / f"{label}-{int(time.time() * 1000)}-{seq}.npz"
            try:
                with open(path, "wb") as fh:
                    np.savez(fh, ids=ids, mat=mat)
            except OSError:
                path = None
            q.append((label, ids, mat, path))

    def _load_vec_files(self, slot) -> None:
        """Vectors deferred by a previous host run (maintenance start)."""
        import numpy as np
        if slot.maint.get("vec_loaded"):
            return
        slot.maint["vec_loaded"] = True
        d = self._vec_dir(slot)
        if not d.exists():
            return
        known = {str(t[3]) for t in slot.maint.get("vec_queue", []) if t[3] is not None}
        for f in sorted(d.glob("*.npz")):
            if str(f) in known:
                continue
            try:
                with np.load(f) as z:
                    slot.maint.setdefault("vec_queue", []).append(
                        (f.name.split("-", 1)[0], z["ids"], z["mat"], f))
            except Exception:
                try:
                    f.unlink()
                except OSError:
                    pass

    def _flush_vectors(self, slot, done: dict) -> None:
        """Insert queued vectors in small writer batches, so reads interleave
        with the HNSW insertion; ids deleted meanwhile are skipped."""
        q = slot.maint.get("vec_queue") or []
        n = 0
        while q:
            label, ids, mat, path = q[0]
            db_ro = slot.db_ro
            try:
                with db_ro.thread_conn():
                    alive = {int(r["id"]) for s0 in range(0, len(ids), 20_000)
                             for r in db_ro.fetch_all(f"MATCH (n:{label}) WHERE n.id IN $ids RETURN n.id AS id",
                                                      {"ids": [int(i) for i in ids[s0:s0 + 20_000]]})}
            except Exception:
                slot.maint["again"] = True
                return
            rows = [{"id": int(i), "embedding": mat[j]} for j, i in enumerate(ids.tolist()) if int(i) in alive]
            for s0 in range(0, len(rows), self.VEC_BATCH):
                part = rows[s0:s0 + self.VEC_BATCH]
                w = self.take_writer(slot.cfg.repo_root, label="maintenance")
                try:
                    w.insert_vectors(label, part)
                    n += len(part)
                finally:
                    self.release_writer(slot.cfg.repo_root)
            q.pop(0)
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass
        if n:
            done["vectors"] = n

    MAINT_DELAY_SEC = 1.0
    # Background work waits until no index pass has run for this long (a
    # burst of watcher passes is not interrupted by the health recompute).
    MAINT_QUIET_SEC = 3.0
    KW_COMPACT_DELTA = 4_000

    def schedule_maintenance(self, root: str | Path, delay: float | None = None) -> None:
        """Start the background maintenance thread for a root (once; a
        request while it runs makes it loop once more)."""
        slot = self.resolve(root)
        with self._lock:
            if slot.maint.get("running"):
                slot.maint["again"] = True
                return
            slot.maint["running"] = True
            slot.maint["again"] = False
        t = threading.Thread(target=self._maintenance_loop, args=(slot, delay),
                             name=f"docgraph-maint-{slot.slug}", daemon=True)
        t.start()

    def note_health_use(self, root: str | Path, key: tuple) -> None:
        """health() was asked for with these parameters: keep the report
        precomputed after every reindex (at most a few parameter sets)."""
        slot = self.resolve(root)
        keys = slot.maint.setdefault("health_keys", set())
        if key not in keys and len(keys) < 4:
            keys.add(key)

    def maintenance_status(self, root: str | Path) -> dict:
        slot = self.resolve(root)
        m = slot.maint
        return {"running": bool(m.get("running")), "last": m.get("last"), "error": m.get("error")}

    def _maintenance_loop(self, slot, delay: float | None) -> None:
        _lower_thread_priority()
        try:
            while True:
                time.sleep(self.MAINT_DELAY_SEC if delay is None else delay)
                while time.time() - float(slot.maint.get("last_pass", 0.0)) < self.MAINT_QUIET_SEC:
                    time.sleep(0.5)
                try:
                    slot.maint["last"] = self._maintenance(slot)
                    slot.maint.pop("error", None)
                except Exception as exc:  # noqa: BLE001 - never kill the host
                    log.warning("maintenance for %s failed: %s", slot.slug, exc, exc_info=True)
                    slot.maint["error"] = str(exc)
                slot.maint["maint_end"] = time.time()
                with self._lock:
                    if not slot.maint.get("again"):
                        slot.maint["running"] = False
                        return
                    slot.maint["again"] = False
        except BaseException:
            slot.maint["running"] = False
            raise

    def _maintenance(self, slot) -> dict:
        """Work an index pass handed off, computed OUTSIDE the writer on the
        read-only handle (a writer taking the file lock meanwhile makes the
        reads fail -> retried after that pass); only the final writes take
        the writer, and only if no index pass ran in between (generation
        check). Also precomputes the health report so the first /api/health
        after a reindex is instant."""
        import json as _json
        from docgraph.index import Indexer
        live = slot.live
        done: dict = {"at": time.time()}
        t0 = time.perf_counter()
        # 0. entity vectors deferred by index passes (before SIMILAR_TO)
        self._load_vec_files(slot)
        if slot.maint.get("vec_queue"):
            self._flush_vectors(slot, done)
        # 1. keyword index compaction (in memory + files, no graph writes)
        kw = getattr(live, "kw", None)
        if kw is not None and kw.loaded and kw.delta_size() > self.KW_COMPACT_DELTA:
            with live.lock:
                kw.compact()
            done["kw_compacted"] = True
        try:
            state = _json.loads((slot.cfg.data_dir / "state.json").read_text())
        except Exception:
            state = {}
        need_global = bool(state.get("analytics_pending"))
        tiles_only = bool(slot.maint.get("tiles_stale")) and not need_global
        sim_pending = state.get("similar_pending") or {}
        if need_global or sim_pending or tiles_only:
            gen0 = live.generation
            db_ro = slot.db_ro
            ix = Indexer(slot.cfg, db_ro, embedder=self.embedder_for(slot.cfg), live=live,
                         tile_sink=self._tile_sink(slot), defer_global=True)
            ix._next_id = int(live.next_id or 0) or ix._next_id
            plan = None
            sim_rows: dict[str, list[dict]] = {}
            def superseded(_stage: str = "") -> None:
                # a pass that started meanwhile makes this result stale (the
                # generation check below would drop it): stop at the next
                # stage boundary instead of competing with the pass for the
                # GIL for seconds
                if live.generation != gen0 or slot.maint.get("writer_taken"):
                    raise RuntimeError("superseded by an index pass")
            try:
                with db_ro.thread_conn():
                    if need_global:
                        want_tiles = bool(getattr(slot.cfg, "tiles", True))
                        plan = ix._global_compute(db_ro, True, want_tiles, want_tiles, emit=superseded)
                    elif tiles_only:
                        plan = ix._global_compute(db_ro, False, False, True, emit=superseded)
                    for label, ids in sim_pending.items():
                        superseded()
                        sim_rows[label] = ix._similar_rows(label, [int(i) for i in ids], db=db_ro)
            except Exception as exc:  # the RO handle was closed by a writer
                log.info("maintenance compute for %s interrupted (%s); retrying later", slot.slug, exc)
                slot.maint["again"] = True
                return dict(done, interrupted=True)
            done["compute_s"] = round(time.perf_counter() - t0, 3)
            if live.generation != gen0:
                slot.maint["again"] = True
                return dict(done, stale=True)
            writer = self.take_writer(slot.cfg.repo_root, label="maintenance")
            try:
                if live.generation != gen0:
                    slot.maint["again"] = True
                    return dict(done, stale=True)
                with live.lock:
                    ix.db = writer
                    state = ix._load_state()
                    for label, rows in sim_rows.items():
                        if rows:
                            writer.insert_edges("SIMILAR_TO", label, label, rows, validate=True)
                    state["similar_pending"] = {}
                    if plan is not None:
                        n_comm = ix._global_apply(plan, state)
                        if n_comm is not None:
                            state["communities"] = n_comm
                        if plan.get("stats"):
                            state["analytics_drift"] = 0
                            state["analytics_pending"] = False
                        if plan.get("relayout"):
                            state["layout_complete"] = True
                    state["maintenance"] = dict(done, at=time.time())
                    ix._save_state(state)
                    live.next_id = ix._next_id
                    live.generation += 1
                    # positions / ranks / communities changed under the tile
                    # sidecar and the in-memory ranks: reload lazily
                    done["global"] = plan is not None
                    done["similar"] = {k: len(v) for k, v in sim_rows.items()}
            finally:
                self.release_writer(slot.cfg.repo_root)
        # 2. health report for the current generation, off the request path
        try:
            r = slot.retriever
            for key in sorted(slot.maint.get("health_keys") or ()):
                if key not in (getattr(r, "_health_cache", None) or {}):
                    r.health(key[0], key[1], refresh=True)
                    done["health"] = True
        except Exception as exc:
            log.debug("health precompute failed: %s", exc)
        try:
            r = slot.retriever
            for key in sorted(slot.maint.get("warm") or (), key=str):
                if key[0] == "mem_graph" and getattr(r, "_mem_graph_cache", None) is None:
                    r._mem_graph()
                elif key[0] == "processes":
                    r.processes(key[1], key[2], key[3])
                elif key[0] == "graph_dump":
                    r.graph_dump(key[1])
                elif key[0] == "node_table":
                    r._node_table()
                elif key[0] == "trace_graph":
                    r._trace_graph()
                elif key[0] == "explore_adj":
                    r._explore_adj(key[1])
                    r._symbol_pagerank()
            done["warm"] = len(slot.maint.get("warm") or ())
        except Exception as exc:
            log.debug("memo prewarm failed: %s", exc)
        done["seconds"] = round(time.perf_counter() - t0, 3)
        return done

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

    # ── model load state ──────────────────────────────────────────────
    def models_status(self) -> dict:
        """Snapshot of pooled embedder + reranker load state. Designed
        for status dashboards: returns load flag, idle age, configured
        unload window, and the configured model name per class. Cheap —
        no IO, just reads from in-memory pools.

        Each entry: `{loaded, last_used_sec, idle_for_sec, unload_after,
        model}`. `last_used_sec=0` when the model has never been touched
        (or has been since unloaded)."""
        import time as _time
        with self._lock:
            embedders = list(self._embedders.items())
            rerankers = list(self._rerankers.items())
        now = _time.monotonic()

        def _one(model_name: str, gpu: bool, loaded: bool, last: float,
                 threshold: float) -> dict:
            idle_for = (now - last) if (loaded and last > 0) else 0.0
            return {
                "loaded":         loaded,
                "model":          model_name,
                "gpu":            bool(gpu),
                "idle_for_sec":   round(idle_for, 1),
                "unload_after":   float(threshold or 0.0),
            }

        embed_entries: list[dict] = []
        for (model, gpu, _torch_compile), emb in embedders:
            embed_entries.append(_one(
                model_name=model, gpu=gpu,
                loaded=emb.is_loaded(),
                last=emb.last_used(),
                threshold=self.embed_unload_after,
            ))
        rerank_entries: list[dict] = []
        for (model, gpu, _torch_compile), r in rerankers:
            rerank_entries.append(_one(
                model_name=model or "default", gpu=gpu,
                loaded=r.is_loaded(),
                last=r.last_used(),
                threshold=self.rerank_unload_after,
            ))
        return {
            "embed": embed_entries,
            "rerank": rerank_entries,
            "embed_unload_after":  float(self.embed_unload_after or 0.0),
            "rerank_unload_after": float(self.rerank_unload_after or 0.0),
        }

    # ── idle unloader ──────────────────────────────────────────────────
    def start_idle_unloader_async(self, check_interval: float = 30.0) -> None:
        """Launch the periodic eviction task on the running loop.

        No-op when both thresholds are <= 0 (nothing to evict) or the
        task is already running. The task polls every `check_interval`
        seconds, asking each pooled Embedder / Reranker whether its
        `last_used` is older than its respective threshold, and calling
        `unload()` on the ones that qualify. Loading is lazy on the
        next call, so eviction is transparent to callers.

        Safe to call from `make_app`'s lifespan — no work happens until
        models are actually loaded *and* go idle.
        """
        if self.embed_unload_after <= 0 and self.rerank_unload_after <= 0:
            return
        if self._idle_task is not None and not self._idle_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_task = loop.create_task(
            self._idle_unloader_loop(check_interval),
            name="docgraph-idle-unloader",
        )

    async def _idle_unloader_loop(self, check_interval: float) -> None:
        import time as _time
        while True:
            try:
                await asyncio.sleep(check_interval)
                embed_thr = self.embed_unload_after
                rerank_thr = self.rerank_unload_after
                if embed_thr <= 0 and rerank_thr <= 0:
                    continue  # disabled mid-flight
                now = _time.monotonic()
                # Snapshot under the lock; eviction itself is on the
                # model's own lock so we can release ours first.
                with self._lock:
                    embedders = list(self._embedders.values())
                    rerankers = list(self._rerankers.values())
                pairs = (
                    [(m, embed_thr) for m in embedders] +
                    [(m, rerank_thr) for m in rerankers]
                )
                for m, threshold in pairs:
                    if threshold <= 0 or not m.is_loaded():
                        continue
                    last = m.last_used()
                    if last <= 0:
                        continue  # never used → leave the warm spot alone
                    if now - last >= threshold:
                        try:
                            m.unload()
                        except Exception as exc:
                            log.warning("idle unload failed for %r: %s", m, exc)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # pragma: no cover — defensive
                log.warning("idle unloader loop: %s", exc)

    # ── lifecycle ───────────────────────────────────────────────────────
    def close(self) -> None:
        self._closing = True
        # Cancel the idle unloader first so it doesn't race with the
        # embedder cache being emptied beneath it.
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None
        with self._lock:
            for slot in self._slots.values():
                try:
                    if slot.live is not None:
                        slot.live.close_pool()
                except Exception:
                    pass
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
            self._rerankers.clear()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
