"""Optional cross-CLI embedding daemon.

A long-lived TCP loopback server that holds a single ONNX session in memory.
Clients (CLI invocations, MCP server, tests) can ask it to embed text instead
of paying the ~1s ONNX session load on every fresh process.

Wire protocol — newline-delimited JSON, one request per line:

    request:  {"op": "embed", "texts": [str, ...]}
    response: {"embeddings": [[float, ...], ...]}

    request:  {"op": "ping"}
    response: {"ok": true, "model": str, "dim": int}

    request:  {"op": "shutdown"}
    response: {"ok": true}

The daemon binds to 127.0.0.1 only — no exposure outside the host. Port + PID
are written to `~/.docgraph/daemon.lock` so any docgraph process on the box
can find it. Default port is 5577 (above the web UI's 5500 to avoid clashes).

Opt-in. `Embedder` consults the lock file at construction time; if the daemon
isn't running, the embedder loads its own ONNX session as before. Failures
during a `daemon-routed` embed call (refused connection, malformed reply)
fall back transparently to in-process embedding — never fails the request.
"""
from __future__ import annotations

import io
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5577
LOCK_DIR = Path.home() / ".docgraph"
LOCK_PATH = LOCK_DIR / "daemon.lock"


def _read_lock() -> dict | None:
    if not LOCK_PATH.exists():
        return None
    try:
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_lock(payload: dict) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _clear_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _send_recv(host: str, port: int, payload: dict, timeout: float = 30.0) -> dict | None:
    """Send one JSON request and read one JSON response. Length-prefixed (4-byte
    big-endian uint32) so a payload's newlines don't break framing."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            data = json.dumps(payload).encode("utf-8")
            s.sendall(struct.pack("!I", len(data)) + data)
            header = _recv_exact(s, 4)
            if not header:
                return None
            (n,) = struct.unpack("!I", header)
            body = _recv_exact(s, n)
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except (OSError, ValueError) as e:
        log.debug(f"daemon RPC failed: {e}")
        return None


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = io.BytesIO()
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            return b""
        buf.write(chunk)
        remaining -= len(chunk)
    return buf.getvalue()


def is_running() -> dict | None:
    """Return the daemon's lock payload if it's reachable, else None.
    Probes the lock-recorded port with a `ping` op; cleans up a stale lock
    if the port is dead."""
    info = _read_lock()
    if not info:
        return None
    host = info.get("host", DEFAULT_HOST)
    port = int(info.get("port", DEFAULT_PORT))
    resp = _send_recv(host, port, {"op": "ping"}, timeout=2.0)
    if resp and resp.get("ok"):
        return info
    # Stale lock — daemon died without cleanup. Remove it so future starts
    # don't think the port is in use.
    _clear_lock()
    return None


def embed_via_daemon(texts: list[str], timeout: float = 60.0) -> np.ndarray | None:
    """Try to embed via the daemon. Returns None if daemon isn't reachable
    or the protocol fails — caller should fall back to in-process embed.
    Vectors come back as float32 ndarray of shape (N, dim)."""
    info = is_running()
    if not info:
        return None
    resp = _send_recv(
        info.get("host", DEFAULT_HOST),
        int(info.get("port", DEFAULT_PORT)),
        {"op": "embed", "texts": list(texts)},
        timeout=timeout,
    )
    if not resp or "embeddings" not in resp:
        return None
    try:
        arr = np.asarray(resp["embeddings"], dtype=np.float32)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


# ----- Server side -------------------------------------------------------


def _serve_one(conn: socket.socket, ctx: dict) -> bool:
    """Handle one request on `conn`. Returns False if the daemon should exit."""
    try:
        header = _recv_exact(conn, 4)
        if not header:
            return True
        (n,) = struct.unpack("!I", header)
        body = _recv_exact(conn, n)
        if not body:
            return True
        req = json.loads(body.decode("utf-8"))
    except Exception as e:
        log.warning(f"daemon: bad request frame: {e}")
        return True

    op = req.get("op")
    if op == "ping":
        _send(conn, {"ok": True, "model": ctx["model"], "dim": ctx["dim"]})
        return True
    if op == "shutdown":
        _send(conn, {"ok": True})
        return False
    if op == "embed":
        texts = req.get("texts") or []
        try:
            # Bypass Embedder.embed() — that wrapper's daemon-detection path
            # would route this call right back to us. Hit fastembed directly.
            model = ctx["embedder"]._ensure()
            vecs = [list(map(float, v)) for v in model.embed(list(texts), batch_size=256)]
            payload: dict[str, Any] = {"embeddings": vecs}
        except Exception as e:
            payload = {"error": str(e)}
        _send(conn, payload)
        return True
    _send(conn, {"error": f"unknown op: {op}"})
    return True


def _send(conn: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    try:
        conn.sendall(struct.pack("!I", len(data)) + data)
    except OSError:
        pass


def run_daemon(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_name: str = "BAAI/bge-small-en-v1.5",
    gpu: bool = False,
) -> int:
    """Start the daemon in the calling process. Blocks until shutdown.
    Returns 0 on clean exit, non-zero on bind failure."""
    from docgraph.embed import Embedder, resolve_providers

    embedder = Embedder(
        model_name=model_name,
        providers=resolve_providers(gpu),
    )
    embedder._ensure()  # warm up before we accept clients

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
    except OSError as e:
        print(f"docgraph daemon: bind failed on {host}:{port} — {e}", file=sys.stderr)
        return 2
    srv.listen(8)
    srv.settimeout(0.5)  # so the accept loop can poll the shutdown flag

    _write_lock({
        "host": host,
        "port": port,
        "pid": os.getpid(),
        "model": model_name,
        "gpu": bool(gpu),
        "started": time.time(),
    })

    ctx = {"embedder": embedder, "model": model_name, "dim": embedder.dim}
    shutdown = threading.Event()

    def serve(conn: socket.socket) -> None:
        try:
            keep = _serve_one(conn, ctx)
            if not keep:
                shutdown.set()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    log.info(f"docgraph daemon listening on {host}:{port} (model={model_name}, gpu={gpu})")
    try:
        while not shutdown.is_set():
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=serve, args=(conn,), daemon=True)
            t.start()
    finally:
        try:
            srv.close()
        except Exception:
            pass
        _clear_lock()
    return 0


def stop_daemon() -> bool:
    """Ask the running daemon to exit. Returns True on success."""
    info = _read_lock()
    if not info:
        return False
    resp = _send_recv(
        info.get("host", DEFAULT_HOST),
        int(info.get("port", DEFAULT_PORT)),
        {"op": "shutdown"},
        timeout=2.0,
    )
    # Even on a None response, the daemon may have already cleared its lock
    # and exited — treat lock disappearance as success.
    for _ in range(20):
        if not LOCK_PATH.exists():
            return True
        time.sleep(0.1)
    _clear_lock()
    return resp is not None and resp.get("ok", False)
