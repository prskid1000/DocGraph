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
