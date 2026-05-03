"""Cooperative cancellation for long-running host operations.

The host runs index / wiki / docs-add via `await asyncio.to_thread(...)`,
so the underlying threadpool worker doesn't notice when the asyncio task
is cancelled (e.g. the client disconnects). This module provides a
threadsafe `CancelToken` the long ops poll at safe checkpoints, plus a
`OperationCancelled` exception they raise when the token fires.

Wiring:
    - Workspace owns one token per root (`workspace.cancel_token_for(root)`).
    - HTTP route handlers reset the token before kicking off, supply it
      to the long op, and translate `OperationCancelled` into HTTP 499.
    - `POST /api/admin/cancel?root=<slug>` flips the token from another
      request — that's how telecode actually halts a running pass.

Cancellation is always cooperative: the long op decides where it's safe
to bail. We sprinkle `token.raise_if_set()` at the top of major loops
and between batches, never inside Kuzu COPY FROM or in the middle of an
embedding batch (would corrupt state).
"""
from __future__ import annotations

import threading


class OperationCancelled(Exception):
    """Raised by long-running ops when their cancel token fires."""


class CancelToken:
    __slots__ = ("_lock", "_set")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._set = False

    def request(self) -> None:
        with self._lock:
            self._set = True

    def reset(self) -> None:
        with self._lock:
            self._set = False

    def is_set(self) -> bool:
        with self._lock:
            return self._set

    def raise_if_set(self) -> None:
        if self.is_set():
            raise OperationCancelled()
