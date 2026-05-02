"""Thin stdio ↔ HTTP MCP proxy.

Used by `docgraph mcp <path> --transport stdio` when a `docgraph host`
is already running. Lets editors (Cursor, Claude Desktop) keep their
stdio launch shape while sharing a single workspace + DB connections
with whatever else is using the host.

Two-phase:
  1) Probe the host for `/api/roots`. If it doesn't respond fast, return
     False so the caller falls back to single-process stdio.
  2) Resolve `scope_root` against the host's registered roots; refuse if
     not present (strict by default — see CLAUDE.md).
  3) Open the FastMCP HTTP client to host's `/mcp/` and run a stdio
     bridge that forwards every JSON-RPC request → HTTP and streams
     responses back to stdout.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import urllib.request

log = logging.getLogger(__name__)


def _probe_host(host_url: str, timeout_sec: float = 0.5) -> list[dict] | None:
    """Hit `/api/roots` and parse the response. Returns the roots list on
    success, None on any failure (timeout, non-JSON, connection refused)."""
    try:
        url = host_url.rstrip("/") + "/api/roots"
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("host probe failed at %s: %s", host_url, exc)
        return None


def run_stdio_proxy(host_url: str, scope_root: Path) -> bool:
    """Try to start the proxy. Returns True if the proxy ran (and exited
    on stdin EOF or signal). Returns False if no host was found at
    `host_url`, so the caller can fall back to single-process stdio.

    Strict scoping: `scope_root` must exactly match (by absolute path
    OR slug) one of the host's registered roots. Otherwise we error
    out and return True without proxying — the caller treats this as
    "intentional refusal", not "fall through to standalone".
    """
    roots = _probe_host(host_url)
    if roots is None:
        return False

    target = Path(scope_root).resolve()
    target_slug = None
    for r in roots:
        if Path(r["path"]).resolve() == target:
            target_slug = r["slug"]
            break
        if r["slug"].lower() == str(scope_root).lower():
            target_slug = r["slug"]
            break
    if target_slug is None:
        # The host is up but doesn't know about this path. Surface the
        # config mismatch loudly rather than starting a duplicate
        # workspace that the user didn't ask for.
        msg = {
            "jsonrpc": "2.0",
            "method": "notifications/error",
            "params": {
                "message": (
                    f"docgraph host at {host_url} has no root matching "
                    f"{scope_root}. Add it to telecode's docgraph.roots "
                    f"and restart, or pass --standalone to run a "
                    f"single-process stdio server."
                ),
            },
        }
        import sys
        sys.stderr.write(json.dumps(msg) + "\n")
        sys.stderr.flush()
        return True

    asyncio.run(_pipe_stdio(host_url, target_slug))
    return True


async def _pipe_stdio(host_url: str, default_slug: str) -> None:
    """Bridge stdin↔HTTP-MCP. Reads JSON-RPC frames from stdin (one per
    line), forwards them via the FastMCP client to the host, streams
    responses back to stdout. Injects `default_slug` into any tool-call
    request that omits a `root` argument."""
    try:
        from fastmcp.client import Client
    except Exception as exc:
        raise RuntimeError(
            "fastmcp.Client not available; cannot run stdio proxy"
        ) from exc

    mcp_endpoint = host_url.rstrip("/") + "/mcp/"
    async with Client(mcp_endpoint) as client:
        loop = asyncio.get_event_loop()
        import sys
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = req.get("method", "")
            params = req.get("params", {}) or {}
            req_id = req.get("id")

            try:
                if method == "tools/list":
                    tools = await client.list_tools()
                    response = {
                        "jsonrpc": "2.0", "id": req_id,
                        "result": {"tools": [t.model_dump() for t in tools]},
                    }
                elif method == "tools/call":
                    name = params.get("name", "")
                    args = dict(params.get("arguments", {}) or {})
                    # Inject default root for tools that take one but
                    # the editor didn't pass one.
                    args.setdefault("root", default_slug)
                    result = await client.call_tool(name, args)
                    response = {
                        "jsonrpc": "2.0", "id": req_id,
                        "result": result.model_dump(),
                    }
                elif method == "initialize":
                    # Hand back a minimal capabilities reply; the host's
                    # real capabilities are discovered via tools/list.
                    response = {
                        "jsonrpc": "2.0", "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "docgraph-stdio-proxy",
                                "version": "0.1.0",
                            },
                        },
                    }
                else:
                    response = {
                        "jsonrpc": "2.0", "id": req_id,
                        "error": {"code": -32601, "message": f"unknown method {method}"},
                    }
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32603, "message": str(exc)},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
