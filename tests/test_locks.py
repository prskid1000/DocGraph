"""Edge cases for `docgraph.locks.DBLock` — the per-DB writer queue +
read gate that mediates Kuzu's exclusive write lock.

These tests exercise the lock primitive directly (no real Kuzu DB),
plus a couple of integration tests that hit the read-gate middleware via
TestClient. They cover:
  - first-acquire grants immediately (regression for the bug where the
    very first writer queued forever because release_write was the only
    path that called _maybe_grant_next).
  - FIFO ordering across queued writers.
  - reads pass when idle, wait when writer held, and time out on
    BusyTimeout if the writer doesn't release in time.
  - reads-in-flight drain before a queued writer is granted.
  - independent DBLocks (different roots) don't block each other.
  - timeout cleans up the queued slot so subsequent acquires aren't
    poisoned.
  - status() snapshot is accurate during each phase.

Async coroutines are driven via `asyncio.run(...)` since the project
doesn't carry a pytest-asyncio dep.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docgraph.config import load_config
from docgraph.db import GraphDB
from docgraph.embed import Embedder
from docgraph.index import Indexer
from docgraph.locks import BusyTimeout, DBLock, LockTimeouts
from docgraph.server import make_app
from docgraph.workspace import Workspace


def _run(coro):
    """Run an async coroutine in a fresh event loop. Each test gets its
    own loop so a test that leaves a coroutine pending can't poison the
    next one."""
    return asyncio.run(coro)


# ── primitive: DBLock ────────────────────────────────────────────────


def test_first_acquire_grants_immediately():
    """Regression: the very first acquire used to queue forever because
    _maybe_grant_next was only called on release. acquire_write now
    kicks the queue itself."""
    async def go():
        lock = DBLock(name="t")
        await asyncio.wait_for(lock.acquire_write("first", timeout=2.0), timeout=2.5)
        s = lock.status()
        assert s.held is True
        assert s.holder_label == "first"
        await lock.release_write()
    _run(go())


def test_writers_queue_fifo():
    """Two writers issued back-to-back: second waits for first, then
    runs. holder_label flips through both."""
    async def go():
        lock = DBLock(name="t")
        await lock.acquire_write("A", timeout=1.0)
        assert lock.status().holder_label == "A"

        second_done = asyncio.Event()

        async def second():
            await lock.acquire_write("B", timeout=5.0)
            second_done.set()

        task = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        # B is queued, A still held
        assert lock.status().queue_depth == 1
        assert lock.status().queued_labels == ["B"]
        assert lock.status().holder_label == "A"

        await lock.release_write()
        await asyncio.wait_for(second_done.wait(), timeout=2.0)
        assert lock.status().holder_label == "B"
        await lock.release_write()
        await task
    _run(go())


def test_writer_queue_timeout_cleans_up():
    """If a queued writer times out, its slot must be removed so the
    next acquire isn't blocked behind a dead future."""
    async def go():
        lock = DBLock(name="t")
        await lock.acquire_write("hold", timeout=1.0)

        with pytest.raises(BusyTimeout):
            await lock.acquire_write("victim", timeout=0.2)

        # Queue empty after timeout cleanup.
        assert lock.status().queue_depth == 0

        # A new acquire should still succeed once the holder releases.
        await lock.release_write()
        await lock.acquire_write("next", timeout=1.0)
        assert lock.status().holder_label == "next"
        await lock.release_write()
    _run(go())


def test_read_passes_when_idle():
    async def go():
        lock = DBLock(name="t")
        await asyncio.wait_for(lock.wait_idle(timeout=0.5), timeout=1.0)
    _run(go())


def test_read_waits_for_writer_then_passes():
    """Read blocks while writer is held; passes once released."""
    async def go():
        lock = DBLock(name="t")
        await lock.acquire_write("W", timeout=1.0)

        read_passed = asyncio.Event()

        async def reader():
            await lock.wait_idle(timeout=2.0)
            read_passed.set()

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.05)
        assert not read_passed.is_set()

        await lock.release_write()
        await asyncio.wait_for(read_passed.wait(), timeout=1.0)
        await task
    _run(go())


def test_read_gate_timeout_raises_busy():
    async def go():
        lock = DBLock(name="t")
        await lock.acquire_write("forever", timeout=1.0)
        with pytest.raises(BusyTimeout):
            await lock.wait_idle(timeout=0.2)
        await lock.release_write()
    _run(go())


def test_reads_in_flight_delay_writer_grant():
    """A queued writer must wait for active reads to drain before being
    granted — this is what prevents the writer from yanking the file
    lock out from under a mid-query reader."""
    async def go():
        lock = DBLock(name="t")
        await lock.enter_read()
        assert lock.status().reads_in_flight == 1

        granted = asyncio.Event()

        async def writer():
            await lock.acquire_write("W", timeout=2.0)
            granted.set()

        task = asyncio.create_task(writer())
        await asyncio.sleep(0.05)
        assert not granted.is_set()
        assert lock.status().held is False

        await lock.leave_read()
        await asyncio.wait_for(granted.wait(), timeout=1.0)
        assert lock.status().held is True
        await lock.release_write()
        await task
    _run(go())


def test_status_snapshot_accuracy():
    async def go():
        lock = DBLock(name="snap")
        s = lock.status()
        assert s.held is False
        assert s.queue_depth == 0
        assert s.reads_in_flight == 0

        await lock.acquire_write("idx", timeout=1.0)
        s = lock.status()
        assert s.held is True
        assert s.holder_label == "idx"
        assert s.holder_age is not None and s.holder_age >= 0

        async def second():
            await lock.acquire_write("wiki", timeout=2.0)
            await lock.release_write()

        task = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        s = lock.status()
        assert s.queue_depth == 1
        assert s.queued_labels == ["wiki"]

        await lock.release_write()
        await task
        s = lock.status()
        assert s.held is False
        assert s.queue_depth == 0
    _run(go())


def test_independent_locks_dont_block():
    """Two roots = two DBLocks. A writer on root A must not delay reads
    or writers on root B — this is the whole point of per-root locks
    in multi-root mode (and per-DBHandle once groups land)."""
    async def go():
        lock_a = DBLock(name="A")
        lock_b = DBLock(name="B")
        await lock_a.acquire_write("idx_a", timeout=1.0)
        await asyncio.wait_for(lock_b.wait_idle(timeout=0.2), timeout=0.5)
        await asyncio.wait_for(lock_b.acquire_write("idx_b", timeout=0.5), timeout=1.0)
        assert lock_b.status().held is True
        await lock_b.release_write()
        await lock_a.release_write()
    _run(go())


def test_lock_timeouts_for_label():
    """LockTimeouts.for_label routes wiki labels to the longer wiki
    timeout, watcher labels to infinite, others to write_wait."""
    t = LockTimeouts(read_wait=1.0, write_wait=10.0, wiki_write=99.0,
                     watcher_write=0.0)
    assert t.for_label("api:index") == 10.0
    assert t.for_label("api:wiki:build") == 99.0
    assert t.for_label("wiki") == 99.0
    assert t.for_label("watch") == float("inf")
    assert t.for_label("watcher") == float("inf")


# ── integration: read gate + workspace + server ─────────────────────


def _setup_repo(tmp_path: Path, name: str = "lockrepo"):
    repo = tmp_path / name
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("def hello(): return 1\n", encoding="utf-8")
    cfg = load_config(repo)
    db = GraphDB(cfg.db_path, embedding_dim=384)
    db.init_schema()
    embedder = Embedder(cfg.embedding_model)
    Indexer(cfg, db, embedder=embedder).index_all(incremental=False)
    db.close()
    return cfg


def test_api_locks_endpoint_exposes_status(tmp_path: Path):
    cfg = _setup_repo(tmp_path)
    ws = Workspace([cfg])
    app = make_app(ws)
    with TestClient(app) as client:
        r = client.get("/api/locks")
        assert r.status_code == 200
        data = r.json()
        assert "locks" in data and "timeouts" in data
        assert data["locks"][0]["slug"] == "lockrepo"
        assert data["locks"][0]["held"] is False
        assert data["timeouts"]["read_wait"] == ws.lock_timeouts.read_wait


def test_read_gate_skips_jobs_endpoint(tmp_path: Path):
    """While an index is running, /api/jobs/<id> must NOT block on the
    read gate (it's in-memory state, not a graph query). Regression for
    a parallel-jobs hang we saw during initial integration."""
    import time as _t
    cfg = _setup_repo(tmp_path)
    ws = Workspace([cfg])
    app = make_app(ws)
    with TestClient(app) as client:
        r = client.post("/api/admin/index", json={"full": True})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        for _ in range(20):
            rr = client.get(f"/api/jobs/{job_id}")
            assert rr.status_code == 200
            if rr.json()["status"] in ("completed", "failed"):
                break
            _t.sleep(0.2)
        else:
            pytest.fail("index never finished")
