"""Tests for the optional embedding daemon.

These spin up the daemon in a background thread on a unique loopback port,
exercise the wire protocol (ping / embed / shutdown), then tear it down.
No subprocess — the daemon is a thread inside this test process so we can
assert state crisply.
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from docgraph import daemon as dmn


def _free_port() -> int:
    """Bind on port 0, read the chosen port, close. The OS keeps the port in
    TIME_WAIT briefly — daemon binds with SO_REUSEADDR so this is fine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def running_daemon(monkeypatch):
    """Start a daemon thread on a free port, yield the lock info, stop on teardown."""
    port = _free_port()
    # Sandbox the lock file so we never clobber the user's real one
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(dmn, "LOCK_DIR", tmp)
    monkeypatch.setattr(dmn, "LOCK_PATH", tmp / "daemon.lock")

    t = threading.Thread(
        target=dmn.run_daemon,
        kwargs={"port": port, "model_name": "BAAI/bge-small-en-v1.5", "gpu": False},
        daemon=True,
    )
    t.start()
    # Wait for lock file (means socket is up + model loaded)
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if (tmp / "daemon.lock").exists():
            break
        time.sleep(0.1)
    else:
        pytest.fail("daemon did not start within 60s")

    info = dmn._read_lock()
    assert info is not None
    yield info
    # Teardown
    dmn.stop_daemon()
    t.join(timeout=5.0)


def test_daemon_ping(running_daemon):
    info = running_daemon
    resp = dmn._send_recv(
        info["host"], int(info["port"]), {"op": "ping"}, timeout=3.0
    )
    assert resp is not None
    assert resp.get("ok") is True
    assert resp.get("dim") == 384


def test_daemon_embed_roundtrip(running_daemon):
    arr = dmn.embed_via_daemon(["hello world", "another sentence"])
    assert arr is not None
    assert arr.shape == (2, 384)
    assert arr.dtype == np.float32
    # Vectors should differ (not all zeros) and be different from each other
    assert np.any(arr[0] != 0)
    assert not np.array_equal(arr[0], arr[1])


def test_daemon_is_running_detects(running_daemon):
    info = dmn.is_running()
    assert info is not None
    assert int(info["port"]) == int(running_daemon["port"])


def test_daemon_status_op(running_daemon):
    info = running_daemon
    resp = dmn._send_recv(
        info["host"], int(info["port"]), {"op": "status"}, timeout=3.0
    )
    assert resp is not None and resp.get("ok") is True
    assert resp.get("model") == "BAAI/bge-small-en-v1.5"
    assert "rerank_model" in resp
    # Per-model loaded flags present (embedder may already be warm from a
    # prior embed test sharing the process cache, so don't assert the value).
    assert "loaded" in resp["embed"]
    assert "loaded" in resp["rerank"]


def test_daemon_rerank_roundtrip(running_daemon):
    info = running_daemon
    docs = ["open and read a file", "unrelated banana text"]
    # Hit the raw op so we can distinguish "wired but model can't load in
    # this env" (some transformers versions reject the jina reranker's
    # attn_implementation) from a real plumbing failure. Generous timeout:
    # the first call triggers a one-time cross-encoder download + load.
    resp = dmn._send_recv(
        info["host"], int(info["port"]),
        {"op": "rerank", "query": "how to read a file", "documents": docs},
        timeout=300.0,
    )
    assert resp is not None, "daemon did not respond to rerank op"
    if "error" in resp:
        pytest.skip(f"reranker model unavailable in this env: {resp['error'][:80]}")
    scores = resp["scores"]
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    # The client helper should agree with the raw op.
    assert dmn.rerank_via_daemon("how to read a file", docs, timeout=60.0) is not None


def test_daemon_idle_exit(monkeypatch, tmp_path):
    """With idle_exit_sec set, the daemon self-exits once idle — and the
    lock file is cleared so the next caller can respawn it."""
    port = _free_port()
    monkeypatch.setattr(dmn, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(dmn, "LOCK_PATH", tmp_path / "daemon.lock")
    t = threading.Thread(
        target=dmn.run_daemon,
        kwargs={"port": port, "model_name": "BAAI/bge-small-en-v1.5",
                "gpu": False, "idle_exit_sec": 1.0},
        daemon=True,
    )
    t.start()
    deadline = time.time() + 30.0
    while time.time() < deadline and not (tmp_path / "daemon.lock").exists():
        time.sleep(0.05)
    assert (tmp_path / "daemon.lock").exists()
    # Never used + idle_exit_sec=1.0 → should exit within a few seconds.
    t.join(timeout=10.0)
    assert not t.is_alive()
    assert not (tmp_path / "daemon.lock").exists()


def test_daemon_clears_stale_lock(monkeypatch, tmp_path):
    """is_running() should clean up a lock file pointing at a dead port."""
    monkeypatch.setattr(dmn, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(dmn, "LOCK_PATH", tmp_path / "daemon.lock")
    # Write a lock pointing at a port nobody's listening on
    import json
    (tmp_path / "daemon.lock").write_text(
        json.dumps({"host": "127.0.0.1", "port": _free_port(), "pid": 1})
    )
    assert dmn.is_running() is None
    # And the stale lock should be gone
    assert not (tmp_path / "daemon.lock").exists()
