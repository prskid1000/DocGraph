"""Per-DB writer queue + read gate.

Kuzu enforces an exclusive write lock per DB file: only one writer at a
time and no concurrent readers. Without coordination, every read that
races a writer crashed with `AttributeError: 'NoneType' object has no
attribute 'execute'` (because `Workspace.take_writer` closes the RO
handle so the writer can grab the file lock).

`DBLock` mediates this:

- Writers queue. `acquire_write(label, timeout)` blocks until the
  current writer releases or times out. Watcher reindex / API index /
  wiki all go through here.
- Readers don't take the lock — they pass through when no writer is
  active. When a writer IS active, `wait_idle(timeout)` blocks until
  the writer drains. Times out → caller raises HTTP 503.
- A writer is granted only after `_reads_in_flight` drains to 0, so
  a reader currently mid-query can't crash mid-flight.
- Force-free safeguard: if a writer holds for longer than
  `force_free_after`, log it; caller (the host's housekeeping task)
  decides whether to cancel. We don't yank the lock out from under
  the holder — that risks corrupting Kuzu state — only flip the
  cancel token and report the stuck holder.

Status is exposed via `status()` so a `/api/locks` endpoint can show
"indexing — held 12s by api:wiki" instead of users seeing opaque 503s.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class BusyTimeout(RuntimeError):
    """Raised when a writer queue or read gate exceeds its timeout. The
    server's exception handler maps this to HTTP 503 + Retry-After."""


@dataclass
class _QueuedWriter:
    label: str
    enqueued_at: float
    fut: asyncio.Future


@dataclass
class LockStatus:
    """Snapshot of a DBLock for /api/locks observability."""
    held: bool
    holder_label: str | None
    holder_started: float | None
    holder_age: float | None
    queue_depth: int
    queued_labels: list[str]
    reads_in_flight: int


@dataclass
class DBLock:
    """One per Kuzu DB file. Async-only; callers must use `acquire_write`
    / `release_write` and `wait_idle` from the asyncio event loop. The
    watcher takes the writer from a thread via `asyncio.run_coroutine_threadsafe`
    indirection in workspace.take_writer."""
    name: str = "db"
    _writer_held: bool = field(default=False, init=False)
    _holder_label: str | None = field(default=None, init=False)
    _holder_started: float | None = field(default=None, init=False)
    _writer_queue: deque = field(default_factory=deque, init=False)
    _cv: asyncio.Condition = field(default_factory=asyncio.Condition, init=False)
    _reads_in_flight: int = field(default=0, init=False)

    # ── writer side ────────────────────────────────────────────────────
    async def acquire_write(self, label: str, timeout: float = 60.0) -> None:
        """Wait until this DB's writer slot is free, then claim it.

        Implementation: enqueue a future; the head of the queue is
        granted when (a) no writer is held and (b) no reads are in
        flight. Times out → BusyTimeout, the caller's slot is removed
        from the queue so it doesn't poison subsequent waits.
        """
        loop = asyncio.get_running_loop()
        my = _QueuedWriter(label=label, enqueued_at=loop.time(), fut=loop.create_future())
        async with self._cv:
            self._writer_queue.append(my)
            self._cv.notify_all()
        # Kick the queue. Without this the very first acquire never
        # progresses (release_write is what normally grants the head).
        asyncio.create_task(self._maybe_grant_next())
        try:
            await asyncio.wait_for(my.fut, timeout)
        except asyncio.TimeoutError:
            async with self._cv:
                try:
                    self._writer_queue.remove(my)
                except ValueError:
                    pass
                self._cv.notify_all()
            raise BusyTimeout(
                f"writer queue timeout on {self.name} after {timeout}s "
                f"(holder={self._holder_label})"
            )

    async def release_write(self) -> None:
        async with self._cv:
            self._writer_held = False
            self._holder_label = None
            self._holder_started = None
            self._cv.notify_all()
        # Pump the queue from a background task so the releaser doesn't
        # hold the CV across the wait-for-readers-drain dance.
        asyncio.create_task(self._maybe_grant_next())

    async def _maybe_grant_next(self) -> None:
        """Grant the head of the writer queue if conditions allow.
        Conditions: no writer held, no reads in flight, queue non-empty."""
        async with self._cv:
            # Drain readers first.
            while self._reads_in_flight > 0:
                await self._cv.wait()
            if self._writer_held or not self._writer_queue:
                return
            head = self._writer_queue.popleft()
            self._writer_held = True
            self._holder_label = head.label
            self._holder_started = time.time()
            if not head.fut.done():
                head.fut.set_result(None)
            self._cv.notify_all()

    # ── reader side ────────────────────────────────────────────────────
    async def wait_idle(self, timeout: float = 5.0) -> None:
        """Block until no writer is held, with timeout. Reads should call
        this before touching db_ro; on success they MUST call enter_read()
        before the query and leave_read() after, so a queued writer waits
        for them to drain.
        """
        async with self._cv:
            if not self._writer_held:
                return
            try:
                await asyncio.wait_for(
                    self._cv.wait_for(lambda: not self._writer_held),
                    timeout,
                )
            except asyncio.TimeoutError:
                raise BusyTimeout(
                    f"read gate timeout on {self.name} after {timeout}s "
                    f"(holder={self._holder_label})"
                )

    async def enter_read(self) -> None:
        async with self._cv:
            self._reads_in_flight += 1

    async def leave_read(self) -> None:
        async with self._cv:
            if self._reads_in_flight > 0:
                self._reads_in_flight -= 1
            if self._reads_in_flight == 0:
                self._cv.notify_all()
        if self._reads_in_flight == 0 and self._writer_queue and not self._writer_held:
            asyncio.create_task(self._maybe_grant_next())

    # ── observability ──────────────────────────────────────────────────
    def status(self) -> LockStatus:
        now = time.time()
        return LockStatus(
            held=self._writer_held,
            holder_label=self._holder_label,
            holder_started=self._holder_started,
            holder_age=(now - self._holder_started) if self._holder_started else None,
            queue_depth=len(self._writer_queue),
            queued_labels=[q.label for q in self._writer_queue],
            reads_in_flight=self._reads_in_flight,
        )


@dataclass
class LockTimeouts:
    """Per-host lock timeouts. Surface via CLI flags + telecode settings."""
    read_wait: float = 5.0          # reader gate timeout
    write_wait: float = 60.0        # API index writer queue timeout
    wiki_write: float = 180.0       # wiki builds run longer
    watcher_write: float = 0.0      # 0 = wait forever (watch always queues)
    force_free_after: float = 300.0 # log + cancel a writer holding > 5 min

    def for_label(self, label: str) -> float:
        if label.startswith("api:wiki") or label.startswith("wiki"):
            return self.wiki_write
        if label.startswith("watch") or label.startswith("watcher"):
            return self.watcher_write or float("inf")
        return self.write_wait
