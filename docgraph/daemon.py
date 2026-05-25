"""Optional cross-CLI embedding daemon.

A long-lived TCP loopback server that holds a single torch /
sentence-transformers session in memory. Clients (CLI invocations, MCP
server, tests) can ask it to embed text instead of paying the ~2-3 s model
load + (optional) torch.compile capture on every fresh process.

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
isn't running, the embedder loads its own torch session as before. Failures
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


def rerank_via_daemon(
    query: str, documents: list[str], timeout: float = 60.0,
) -> list[float] | None:
    """Try to score (query, doc) pairs via the daemon's cross-encoder.
    Returns one float per document, or None if the daemon isn't reachable
    or lacks a reranker — caller should fall back to in-process."""
    info = is_running()
    if not info:
        return None
    resp = _send_recv(
        info.get("host", DEFAULT_HOST),
        int(info.get("port", DEFAULT_PORT)),
        {"op": "rerank", "query": query, "documents": list(documents)},
        timeout=timeout,
    )
    if not resp or "scores" not in resp:
        return None
    try:
        return [float(s) for s in resp["scores"]]
    except Exception:
        return None


def daemon_status(timeout: float = 2.0) -> dict | None:
    """Return the daemon's rich status (per-model loaded state, models,
    idle thresholds) via the `status` op, or None if unreachable."""
    info = _read_lock()
    if not info:
        return None
    return _send_recv(
        info.get("host", DEFAULT_HOST),
        int(info.get("port", DEFAULT_PORT)),
        {"op": "status"},
        timeout=timeout,
    )


def ensure_daemon(
    *,
    model_name: str,
    rerank_model: str = "",
    gpu: bool = False,
    dtype: str = "auto",
    embed_torch_compile: bool = False,
    rerank_torch_compile: bool = False,
    embed_idle_unload_sec: float = 0.0,
    rerank_idle_unload_sec: float = 0.0,
    idle_exit_sec: float = 0.0,
    port: int = DEFAULT_PORT,
    wait_sec: float = 30.0,
) -> dict | None:
    """Return the running daemon's lock info, spawning one (detached) with
    this config if none is up. Used for lazy respawn: when the daemon
    self-exits on idle to free the CUDA context, the next embed/rerank
    caller brings it back. Returns None if spawn fails (caller falls back
    to in-process). Never raises."""
    if _IN_DAEMON:
        return None  # never route/spawn from within the daemon itself
    info = is_running()
    if info:
        return info
    try:
        cmd = [
            sys.executable, "-m", "docgraph.cli", "daemon", "start", "--detach",
            "--port", str(port), "--model", model_name, "--dtype", dtype,
            "--embed-idle-unload-sec", str(embed_idle_unload_sec),
            "--rerank-idle-unload-sec", str(rerank_idle_unload_sec),
            "--idle-exit-sec", str(idle_exit_sec),
        ]
        if rerank_model:
            cmd += ["--rerank-model", rerank_model]
        if gpu:
            cmd.append("--gpu")
        if embed_torch_compile:
            cmd.append("--embed-torch-compile")
        if rerank_torch_compile:
            cmd.append("--rerank-torch-compile")
        import subprocess
        if sys.platform.startswith("win"):
            DETACHED = 0x00000008
            CREATE_NEW_GROUP = 0x00000200
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(cmd, creationflags=DETACHED | CREATE_NEW_GROUP | CREATE_NO_WINDOW, close_fds=True)
        else:
            subprocess.Popen(cmd, start_new_session=True, close_fds=True)
    except Exception as exc:
        log.warning("ensure_daemon: spawn failed: %s", exc)
        return None
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        info = is_running()
        if info:
            return info
        time.sleep(0.2)
    log.warning("ensure_daemon: daemon did not come up within %.0fs", wait_sec)
    return None


# ----- Client routing config --------------------------------------------

# Set True inside `run_daemon` so the embed/rerank client paths in
# embed.py / rerank.py never try to route a call back into this very
# process (which would recurse / deadlock).
_IN_DAEMON = False

# Full daemon spec registered by the host at startup when `--embed-daemon`
# is on. None = daemon-client mode disabled (in-process embedding). Holds
# everything `ensure_daemon` needs so whichever caller (embed or rerank)
# first needs a model spawns a fully-configured daemon.
_CLIENT_SPEC: dict | None = None


def configure_client(spec: dict | None) -> None:
    """Register (or clear) the daemon client spec. Called once by the host."""
    global _CLIENT_SPEC
    _CLIENT_SPEC = spec


def client_enabled() -> bool:
    """True when this process should route embed/rerank to a daemon."""
    return _CLIENT_SPEC is not None and not _IN_DAEMON


def ensure_client_daemon() -> dict | None:
    """Ensure the configured daemon is up (spawning lazily). Returns lock
    info or None. No-op when client mode is disabled."""
    if not client_enabled():
        return None
    return ensure_daemon(**_CLIENT_SPEC)


def client_model() -> str | None:
    return (_CLIENT_SPEC or {}).get("model_name")


def client_rerank_model() -> str | None:
    return (_CLIENT_SPEC or {}).get("rerank_model")


# ----- Server side -------------------------------------------------------


def _serve_one(conn: socket.socket, ctx: dict) -> bool:
    """Handle one request on `conn`. Returns False if the daemon should exit.

    All model inference (embed + rerank) runs under `ctx["lock"]` so
    concurrent clients — e.g. a watcher reindex and an interactive search —
    queue through the one warm session instead of racing on the GPU."""
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
    if op == "status":
        embedder = ctx["embedder"]
        reranker = ctx["reranker"]
        _send(conn, {
            "ok": True,
            "model": ctx["model"],
            "rerank_model": ctx["rerank_model"],
            "dim": ctx["dim"],
            "gpu": ctx["gpu"],
            "embed": {
                "loaded": embedder.is_loaded(),
                "idle_unload_sec": ctx["embed_idle_unload_sec"],
            },
            "rerank": {
                "loaded": reranker.is_loaded(),
                "idle_unload_sec": ctx["rerank_idle_unload_sec"],
            },
            "idle_exit_sec": ctx["idle_exit_sec"],
        })
        return True
    if op == "shutdown":
        _send(conn, {"ok": True})
        return False
    if op == "embed":
        texts = req.get("texts") or []
        try:
            ctx["last_embed"] = time.monotonic()
            import torch
            with ctx["lock"]:
                # Bypass Embedder.embed() — that wrapper's daemon-detection
                # path would route this call right back to us. Hit the
                # underlying SentenceTransformer directly.
                model = ctx["embedder"]._ensure()
                with torch.no_grad():
                    arr = model.encode(
                        list(texts), batch_size=64,
                        convert_to_numpy=True, normalize_embeddings=True,
                        show_progress_bar=False,
                    )
            vecs = [list(map(float, v)) for v in arr]
            payload: dict[str, Any] = {"embeddings": vecs}
        except Exception as e:
            payload = {"error": str(e)}
        _send(conn, payload)
        return True
    if op == "rerank":
        query = req.get("query") or ""
        documents = req.get("documents") or []
        try:
            ctx["last_rerank"] = time.monotonic()
            with ctx["lock"]:
                # Reranker.score() owns its own _ensure + CPU-fallback
                # recovery; the lock just serializes GPU work across ops.
                scores = ctx["reranker"].score(query, list(documents))
            payload = {"scores": [float(s) for s in scores]}
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
    rerank_model: str = "",
    dtype: str = "auto",
    embed_torch_compile: bool = False,
    rerank_torch_compile: bool = False,
    embed_idle_unload_sec: float = 0.0,
    rerank_idle_unload_sec: float = 0.0,
    idle_exit_sec: float = 0.0,
) -> int:
    """Start the daemon in the calling process. Blocks until shutdown.
    Returns 0 on clean exit (incl. idle-exit), non-zero on bind failure.

    Owns one embedder + one reranker for the whole host. Both load
    **lazily** on first request — a freshly (re)spawned daemon does no GPU
    work until something actually needs it, which is what makes idle-exit
    loop-safe. Two-stage idle (when the thresholds are set):
      1. `*_idle_unload_sec`  → unload that model's weights (frees VRAM),
                                reload lazily on the next request.
      2. `idle_exit_sec`      → once both models are unloaded and the daemon
                                has been idle this long, exit the process to
                                release the ~300 MB CUDA context. The next
                                embed/rerank caller respawns it via
                                `ensure_daemon`.
    """
    global _IN_DAEMON
    _IN_DAEMON = True
    from docgraph.embed import Embedder, resolve_device, dim_for_model
    from docgraph.rerank import Reranker, DEFAULT_RERANK_MODEL

    device = resolve_device(gpu)
    embedder = Embedder(
        model_name=model_name, device=device, dtype=dtype,
        torch_compile=embed_torch_compile,
    )
    reranker = Reranker(
        model_name=rerank_model or None, device=device, dtype=dtype,
        torch_compile=rerank_torch_compile,
    )
    # Lazy: do NOT warm up here — load on first request.

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
    except OSError as e:
        print(f"docgraph daemon: bind failed on {host}:{port} — {e}", file=sys.stderr)
        return 2
    srv.listen(8)
    srv.settimeout(0.5)  # so the accept loop can poll the shutdown flag

    resolved_rerank = rerank_model or DEFAULT_RERANK_MODEL
    _write_lock({
        "host": host,
        "port": port,
        "pid": os.getpid(),
        "model": model_name,
        "rerank_model": resolved_rerank,
        "gpu": bool(gpu),
        "started": time.time(),
    })

    now0 = time.monotonic()
    ctx: dict[str, Any] = {
        "embedder": embedder,
        "reranker": reranker,
        "model": model_name,
        "rerank_model": resolved_rerank,
        "dim": dim_for_model(model_name),
        "gpu": bool(gpu),
        "lock": threading.Lock(),   # serializes all inference (the queue)
        "embed_idle_unload_sec": embed_idle_unload_sec,
        "rerank_idle_unload_sec": rerank_idle_unload_sec,
        "idle_exit_sec": idle_exit_sec,
        "last_embed": now0,
        "last_rerank": now0,
    }
    shutdown = threading.Event()

    def idle_monitor() -> None:
        """Two-stage idle: unload weights, then exit to free the context."""
        while not shutdown.wait(2.0):
            now = time.monotonic()
            if (embed_idle_unload_sec > 0 and embedder.is_loaded()
                    and now - ctx["last_embed"] >= embed_idle_unload_sec):
                embedder.unload()
            if (rerank_idle_unload_sec > 0 and reranker.is_loaded()
                    and now - ctx["last_rerank"] >= rerank_idle_unload_sec):
                reranker.unload()
            if (idle_exit_sec > 0
                    and not embedder.is_loaded() and not reranker.is_loaded()):
                idle_for = now - max(ctx["last_embed"], ctx["last_rerank"])
                if idle_for >= idle_exit_sec:
                    log.info(
                        "docgraph daemon: idle %.0fs with models unloaded — "
                        "exiting to free CUDA context", idle_for,
                    )
                    shutdown.set()

    monitor = threading.Thread(target=idle_monitor, daemon=True)
    monitor.start()

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

    log.info(
        "docgraph daemon listening on %s:%s (model=%s, rerank=%s, gpu=%s)",
        host, port, model_name, resolved_rerank, gpu,
    )
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
